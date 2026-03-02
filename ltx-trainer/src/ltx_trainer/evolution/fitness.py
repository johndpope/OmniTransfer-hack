"""
AR rollout fitness evaluator for SCD evolution.

Evaluates multi-frame autoregressive generation quality against ground truth
using both latent-space and (optionally) pixel-space metrics:

Latent-space (always active):
  1. Flow matching velocity MSE — per-step prediction quality
  2. Latent reconstruction MSE — per-frame generation quality
  3. Temporal coherence gap — AR drift relative to GT

Pixel-space (optional, requires VAE decoder on secondary GPU):
  4. LPIPS — perceptual similarity (requires lpips package)
  5. SSIM — structural similarity

Multi-sample batch evaluation: evaluates the same perturbation against
multiple dataset samples and averages, reducing noise in the fitness signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer import logger


@dataclass
class FitnessNormalization:
    """Per-component scale factors so each metric contributes ~1.0 at baseline.

    Computed from baseline evaluation; applied in fitness total computation.
    Without normalization, fm_loss (~5.0) dominates recon (~2.0) and tcoh (~0.8),
    meaning the ES gradient mostly optimizes for flow matching and ignores
    temporal coherence and perceptual quality.
    """

    fm_scale: float = 1.0
    recon_scale: float = 1.0
    tcoh_scale: float = 1.0
    lpips_scale: float = 1.0
    ssim_scale: float = 1.0

    @classmethod
    def from_baseline(cls, baseline: "FitnessResult") -> "FitnessNormalization":
        """Compute normalization so each raw metric maps to ~1.0."""
        return cls(
            fm_scale=1.0 / max(baseline.fm_loss, 1e-6),
            recon_scale=1.0 / max(baseline.latent_recon, 1e-6),
            tcoh_scale=1.0 / max(baseline.temporal_coh, 1e-6),
            lpips_scale=1.0 / max(baseline.pixel_lpips, 1e-6) if baseline.pixel_lpips > 0 else 1.0,
            ssim_scale=1.0 / max(baseline.pixel_ssim, 1e-6) if baseline.pixel_ssim > 0 else 1.0,
        )


@dataclass
class FitnessResult:
    """Combined fitness score from AR rollout evaluation."""

    total: float  # Combined score (higher = better)
    fm_loss: float  # Flow matching velocity MSE (lower = better)
    latent_recon: float  # x_0_hat vs GT latent MSE (lower = better)
    temporal_coh: float  # Temporal coherence gap (lower = better)
    pixel_lpips: float = 0.0  # LPIPS perceptual distance (lower = better)
    pixel_ssim: float = 1.0  # SSIM score (higher = better)

    def __repr__(self) -> str:
        parts = [
            f"total={self.total:.4f}",
            f"fm={self.fm_loss:.4f}",
            f"recon={self.latent_recon:.4f}",
            f"tcoh={self.temporal_coh:.4f}",
        ]
        if self.pixel_lpips > 0:
            parts.append(f"lpips={self.pixel_lpips:.4f}")
        if self.pixel_ssim < 1.0:
            parts.append(f"ssim={self.pixel_ssim:.4f}")
        return f"Fitness({', '.join(parts)})"


class ARRolloutEvaluator:
    """Evaluates SCD decoder quality via autoregressive rollout against GT.

    Supports dual-GPU: transformer on cuda:0, VAE decoder on cuda:1.
    Supports batch evaluation: average fitness over multiple samples per perturbation.
    """

    def __init__(
        self,
        scd_model: torch.nn.Module,
        patchifier: object,  # VideoLatentPatchifier
        get_positions_for_frame_fn: callable,
        tokens_per_frame: int,
        latent_h: int,
        latent_w: int,
        latent_channels: int,
        num_inference_steps: int = 15,
        ar_frames: int = 4,
        w_fm_loss: float = 0.5,
        w_latent_recon: float = 0.3,
        w_temporal_coherence: float = 0.2,
        w_pixel_lpips: float = 0.0,
        w_pixel_ssim: float = 0.0,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        distilled: bool = False,
        # Dual-GPU / pixel metrics
        vae_decoder: torch.nn.Module | None = None,
        vae_device: torch.device | str | None = None,
        lpips_net: torch.nn.Module | None = None,
    ):
        self.scd_model = scd_model
        self.patchifier = patchifier
        self.get_positions_for_frame = get_positions_for_frame_fn
        self.tokens_per_frame = tokens_per_frame
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.latent_channels = latent_channels
        # Distilled model has a fixed 8-step schedule
        if distilled:
            self.num_inference_steps = 8
        else:
            self.num_inference_steps = num_inference_steps
        self.ar_frames = ar_frames

        # Fitness weights
        self.w_fm = w_fm_loss
        self.w_recon = w_latent_recon
        self.w_tcoh = w_temporal_coherence
        self.w_lpips = w_pixel_lpips
        self.w_ssim = w_pixel_ssim
        self.use_pixel_metrics = w_pixel_lpips > 0 or w_pixel_ssim > 0

        self.device = torch.device(device)
        self.dtype = dtype
        self.distilled = distilled

        # Fitness normalization (set after first baseline evaluation)
        self.normalization: FitnessNormalization | None = None

        # Dual-GPU: VAE decoder on secondary GPU
        self.vae_decoder = vae_decoder
        self.vae_device = torch.device(vae_device) if vae_device else None
        self.lpips_net = lpips_net

        if self.use_pixel_metrics and self.vae_decoder is None:
            logger.warning(
                "Pixel-space metrics requested but no VAE decoder provided. "
                "Falling back to latent-space only."
            )
            self.w_lpips = 0.0
            self.w_ssim = 0.0
            self.use_pixel_metrics = False

        # Precompute sigma schedule (using LTX2Scheduler)
        self._sigmas: Tensor | None = None

    def _get_sigmas(self) -> Tensor:
        """Lazily compute sigma schedule matching training/inference distribution."""
        if self._sigmas is not None:
            return self._sigmas

        if self.distilled:
            # Distilled model uses a predefined 8-step schedule (from ltx-pipelines).
            # This schedule is heavily front-loaded: steps 1-4 are tiny deltas,
            # steps 5-8 are large jumps, matching the teacher's trajectory.
            DISTILLED_SIGMA_VALUES = [
                1.0, 0.99375, 0.9875, 0.98125, 0.975,
                0.909375, 0.725, 0.421875, 0.0,
            ]
            self._sigmas = torch.tensor(
                DISTILLED_SIGMA_VALUES, device=self.device, dtype=self.dtype,
            )
        else:
            from ltx_core.components.schedulers import LTX2Scheduler

            # Use frames=1 because SCD denoises one frame at a time.
            # The LTX2Scheduler applies a token-count-dependent sigma shift,
            # so the dummy latent shape must match the single-frame decode.
            dummy_latent = torch.empty(1, 1, 1, self.latent_h, self.latent_w)
            scheduler = LTX2Scheduler()
            self._sigmas = scheduler.execute(
                steps=self.num_inference_steps, latent=dummy_latent,
            ).to(device=self.device, dtype=self.dtype)

        return self._sigmas

    def _decode_latent_to_pixels(self, latent: Tensor) -> Tensor:
        """Decode latent to RGB pixels on VAE device.

        Args:
            latent: [1, C, 1, H, W] latent frame.

        Returns:
            [1, 3, pixel_H, pixel_W] RGB image tensor on vae_device.
        """
        lat = latent.to(self.vae_device, self.dtype)
        with torch.inference_mode():
            pixels = self.vae_decoder(lat)
        # Handle unpatchify if needed (some decoders return raw patch output)
        if pixels.dim() == 5:
            pixels = pixels[:, :, 0]  # Take first temporal frame: [1, 3, pH, pW]
        return pixels.clamp(-1, 1)

    def _compute_ssim(self, img1: Tensor, img2: Tensor) -> float:
        """Compute SSIM between two image tensors.

        Simple implementation without external dependencies.
        """
        # Convert to [0, 1] range
        a = (img1.float() + 1) / 2
        b = (img2.float() + 1) / 2

        c1 = 0.01**2
        c2 = 0.03**2

        mu1 = torch.nn.functional.avg_pool2d(a, 11, stride=1, padding=5)
        mu2 = torch.nn.functional.avg_pool2d(b, 11, stride=1, padding=5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2

        sigma1_sq = torch.nn.functional.avg_pool2d(a * a, 11, stride=1, padding=5) - mu1_sq
        sigma2_sq = torch.nn.functional.avg_pool2d(b * b, 11, stride=1, padding=5) - mu2_sq
        sigma12 = torch.nn.functional.avg_pool2d(a * b, 11, stride=1, padding=5) - mu12

        ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean().item()

    @torch.inference_mode()
    def evaluate(
        self,
        gt_latents: Tensor,
        prompt_embeds: Tensor,
        prompt_mask: Tensor | None,
        seed: int = 0,
    ) -> FitnessResult:
        """Run AR rollout and compute fitness vs ground truth.

        Args:
            gt_latents: Ground truth video latents [1, C, F, H, W] where F >= ar_frames.
            prompt_embeds: Text embeddings [1, seq_len, dim].
            prompt_mask: Attention mask for prompt [1, seq_len] or None.
            seed: Random seed for noise generation.

        Returns:
            FitnessResult with combined and component scores.
        """
        from ltx_core.components.patchifiers import VideoLatentShape
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.model.transformer.scd_model import KVCache

        sigmas = self._get_sigmas()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        kv_cache = KVCache.empty()
        kv_cache.is_cache_step = True

        prev_enc_features: Tensor | None = None
        generated: list[Tensor] = []

        total_fm_loss = 0.0
        total_recon_loss = 0.0
        gt_cos_sims: list[float] = []
        gen_cos_sims: list[float] = []
        total_lpips = 0.0
        total_ssim = 0.0

        output_shape = VideoLatentShape(
            batch=1,
            channels=self.latent_channels,
            frames=1,
            height=self.latent_h,
            width=self.latent_w,
        )

        for f in range(self.ar_frames):
            # ── ENCODER: process previous frame (clean, sigma=0) ──
            if f == 0:
                enc_latent = torch.zeros(
                    1, self.latent_channels, 1, self.latent_h, self.latent_w,
                    device=self.device, dtype=self.dtype,
                )
            else:
                enc_latent = generated[-1]

            patchified_enc = self.patchifier.patchify(enc_latent)
            enc_modality = Modality(
                enabled=True,
                latent=patchified_enc,
                timesteps=torch.zeros(
                    1, self.tokens_per_frame, device=self.device, dtype=self.dtype
                ),
                positions=self.get_positions_for_frame(f),
                context=prompt_embeds,
                context_mask=prompt_mask,
            )

            enc_out, _ = self.scd_model.forward_encoder(
                video=enc_modality,
                audio=None,
                perturbations=None,
                kv_cache=kv_cache,
                tokens_per_frame=self.tokens_per_frame,
            )
            current_enc = enc_out.x.detach()

            # Shift-by-1: decoder uses PREVIOUS encoder features
            if prev_enc_features is not None:
                dec_enc_ctx = prev_enc_features
            else:
                dec_enc_ctx = torch.zeros(
                    1, self.tokens_per_frame, current_enc.shape[-1],
                    device=self.device, dtype=self.dtype,
                )
            prev_enc_features = current_enc

            # ── DECODER: denoise from noise -> clean frame ──
            x_t = torch.randn(
                1, self.latent_channels, 1, self.latent_h, self.latent_w,
                device=self.device, dtype=self.dtype, generator=generator,
            )

            dec_positions = self.get_positions_for_frame(f)

            fm_loss_frame = 0.0
            for step in range(self.num_inference_steps):
                sigma = sigmas[step]
                sigma_next = sigmas[step + 1]

                noisy_patch = self.patchifier.patchify(x_t)
                ts = torch.full(
                    (1, self.tokens_per_frame), sigma.item(),
                    device=self.device, dtype=self.dtype,
                )
                dec_modality = Modality(
                    enabled=True,
                    latent=noisy_patch,
                    timesteps=ts,
                    positions=dec_positions,
                    context=prompt_embeds,
                    context_mask=prompt_mask,
                )

                velocity, _ = self.scd_model.forward_decoder(
                    video=dec_modality,
                    encoder_features=dec_enc_ctx,
                    audio=None,
                    perturbations=None,
                )

                # FM velocity MSE against GT-derived true velocity
                gt_frame = gt_latents[:, :, f : f + 1, :, :]
                gt_patch = self.patchifier.patchify(gt_frame)

                if sigma.item() > 1e-6:
                    noise_est = (noisy_patch - (1.0 - sigma) * gt_patch) / sigma
                    v_true = gt_patch - noise_est
                    fm_loss_step = (velocity - v_true).pow(2).mean().item()
                    fm_loss_frame += fm_loss_step

                # Euler ODE step
                vel_unpatch = self.patchifier.unpatchify(velocity, output_shape)
                x_t = x_t + (sigma_next - sigma) * vel_unpatch

            generated.append(x_t.detach())
            total_fm_loss += fm_loss_frame / self.num_inference_steps

            # Latent reconstruction MSE
            gt_frame = gt_latents[:, :, f : f + 1, :, :]
            total_recon_loss += (x_t - gt_frame).pow(2).mean().item()

            # Pixel-space metrics (VAE decode on secondary GPU)
            if self.use_pixel_metrics and self.vae_decoder is not None:
                gen_pixels = self._decode_latent_to_pixels(x_t)
                gt_pixels = self._decode_latent_to_pixels(gt_frame)

                if self.w_lpips > 0 and self.lpips_net is not None:
                    lpips_val = self.lpips_net(gen_pixels, gt_pixels).item()
                    total_lpips += lpips_val

                if self.w_ssim > 0:
                    ssim_val = self._compute_ssim(gen_pixels, gt_pixels)
                    total_ssim += ssim_val

            # Temporal coherence: cosine similarity between consecutive frames
            if f > 0:
                gt_prev = gt_latents[:, :, f - 1 : f, :, :].flatten()
                gt_curr = gt_latents[:, :, f : f + 1, :, :].flatten()
                gt_cos = torch.nn.functional.cosine_similarity(
                    gt_prev.unsqueeze(0), gt_curr.unsqueeze(0)
                ).item()
                gt_cos_sims.append(gt_cos)

                gen_prev = generated[-2].flatten()
                gen_curr = generated[-1].flatten()
                gen_cos = torch.nn.functional.cosine_similarity(
                    gen_prev.unsqueeze(0), gen_curr.unsqueeze(0)
                ).item()
                gen_cos_sims.append(gen_cos)

        # Aggregate fitness
        n = self.ar_frames
        avg_fm = total_fm_loss / n
        avg_recon = total_recon_loss / n
        avg_lpips = total_lpips / n if self.use_pixel_metrics else 0.0
        avg_ssim = total_ssim / n if self.use_pixel_metrics else 1.0

        if gt_cos_sims:
            temporal_gap = sum(
                abs(g - gt) for g, gt in zip(gen_cos_sims, gt_cos_sims)
            ) / len(gt_cos_sims)
        else:
            temporal_gap = 0.0

        # Combined fitness (higher = better):
        # Negate losses (lower is better → higher negative = worse → higher total = better)
        # SSIM is already higher=better, so add it directly
        #
        # With normalization: each raw metric is scaled so baseline ≈ 1.0,
        # ensuring all objectives contribute equally to the ES gradient.
        norm = self.normalization
        if norm is not None:
            total = -(
                self.w_fm * (avg_fm * norm.fm_scale)
                + self.w_recon * (avg_recon * norm.recon_scale)
                + self.w_tcoh * (temporal_gap * norm.tcoh_scale)
                + self.w_lpips * (avg_lpips * norm.lpips_scale)
            )
            if self.w_ssim > 0:
                total += self.w_ssim * (avg_ssim * norm.ssim_scale)
        else:
            total = -(
                self.w_fm * avg_fm
                + self.w_recon * avg_recon
                + self.w_tcoh * temporal_gap
                + self.w_lpips * avg_lpips
            )
            if self.w_ssim > 0:
                total += self.w_ssim * avg_ssim

        return FitnessResult(
            total=total,
            fm_loss=avg_fm,
            latent_recon=avg_recon,
            temporal_coh=temporal_gap,
            pixel_lpips=avg_lpips,
            pixel_ssim=avg_ssim,
        )

    @torch.inference_mode()
    def evaluate_batch(
        self,
        samples: list[dict[str, Tensor]],
        seed_base: int = 0,
    ) -> FitnessResult:
        """Evaluate fitness over multiple samples and average.

        Each sample dict has keys: "latent", "prompt_embeds", "prompt_mask".
        This reduces noise in the fitness signal.
        """
        results: list[FitnessResult] = []
        for i, sample in enumerate(samples):
            r = self.evaluate(
                gt_latents=sample["latent"],
                prompt_embeds=sample["prompt_embeds"],
                prompt_mask=sample.get("prompt_mask"),
                seed=seed_base + i * 7919,
            )
            results.append(r)

        n = len(results)
        return FitnessResult(
            total=sum(r.total for r in results) / n,
            fm_loss=sum(r.fm_loss for r in results) / n,
            latent_recon=sum(r.latent_recon for r in results) / n,
            temporal_coh=sum(r.temporal_coh for r in results) / n,
            pixel_lpips=sum(r.pixel_lpips for r in results) / n,
            pixel_ssim=sum(r.pixel_ssim for r in results) / n,
        )
