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

GPU-batched evaluation: batches CFG conditional+unconditional passes AND
multiple samples in a single decoder forward for maximum throughput.
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
    Supports GPU-batched evaluation: batches CFG passes and multiple samples
    in a single decoder forward call for maximum throughput.
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
        guidance_scale: float = 1.0,
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

        # CFG (Classifier-Free Guidance)
        self.guidance_scale = guidance_scale
        self.use_cfg = guidance_scale > 1.0

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

    def _compute_fitness_score(
        self,
        avg_fm: float,
        avg_recon: float,
        temporal_gap: float,
        avg_lpips: float,
        avg_ssim: float,
    ) -> float:
        """Compute combined fitness score from component metrics."""
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
        return total

    @torch.inference_mode()
    def evaluate(
        self,
        gt_latents: Tensor,
        prompt_embeds: Tensor,
        prompt_mask: Tensor | None,
        seed: int = 0,
    ) -> FitnessResult:
        """Run AR rollout and compute fitness vs ground truth (single sample).

        Args:
            gt_latents: Ground truth video latents [1, C, F, H, W] where F >= ar_frames.
            prompt_embeds: Text embeddings [1, seq_len, dim].
            prompt_mask: Attention mask for prompt [1, seq_len] or None.
            seed: Random seed for noise generation.

        Returns:
            FitnessResult with combined and component scores.
        """
        # Delegate to batched version with a single sample
        return self.evaluate_batch(
            samples=[{
                "latent": gt_latents,
                "prompt_embeds": prompt_embeds,
                "prompt_mask": prompt_mask,
            }],
            seed_base=seed,
        )

    @torch.inference_mode()
    def evaluate_batch(
        self,
        samples: list[dict[str, Tensor]],
        seed_base: int = 0,
    ) -> FitnessResult:
        """Evaluate fitness over multiple samples with GPU-batched decoder.

        Batches both CFG (cond+uncond) and multiple samples into single decoder
        forward calls. Encoder stays sequential (KV-cache is per-sample).

        Each sample dict has keys: "latent", "prompt_embeds", "prompt_mask".
        """
        from ltx_core.components.patchifiers import VideoLatentShape
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.model.transformer.scd_model import KVCache

        sigmas = self._get_sigmas()
        N = len(samples)  # Number of samples to batch
        C = self.latent_channels
        H, W = self.latent_h, self.latent_w
        tpf = self.tokens_per_frame

        # CFG multiplier: 2 if using CFG (cond + uncond), 1 otherwise
        cfg_mult = 2 if self.use_cfg else 1

        output_shape = VideoLatentShape(
            batch=1, channels=C, frames=1, height=H, width=W,
        )

        # Per-sample state (encoder is sequential due to KV-cache)
        per_sample_enc_features: list[Tensor | None] = [None] * N
        per_sample_generated: list[list[Tensor]] = [[] for _ in range(N)]
        per_sample_kv_cache: list = []
        per_sample_generators: list[torch.Generator] = []

        for i in range(N):
            kv = KVCache.empty()
            kv.is_cache_step = True
            per_sample_kv_cache.append(kv)
            per_sample_generators.append(
                torch.Generator(device=self.device).manual_seed(seed_base + i * 7919)
            )

        # Accumulators per sample
        fm_losses = [0.0] * N
        recon_losses = [0.0] * N
        gt_cos_per_sample: list[list[float]] = [[] for _ in range(N)]
        gen_cos_per_sample: list[list[float]] = [[] for _ in range(N)]
        lpips_per_sample = [0.0] * N
        ssim_per_sample = [0.0] * N

        for f in range(self.ar_frames):
            # ── ENCODER: sequential per sample (KV-cache is per-sample) ──
            enc_features_list: list[Tensor] = []
            for i in range(N):
                if f == 0:
                    enc_latent = torch.zeros(
                        1, C, 1, H, W, device=self.device, dtype=self.dtype,
                    )
                else:
                    enc_latent = per_sample_generated[i][-1]

                patchified_enc = self.patchifier.patchify(enc_latent)
                enc_modality = Modality(
                    enabled=True,
                    latent=patchified_enc,
                    timesteps=torch.zeros(1, tpf, device=self.device, dtype=self.dtype),
                    positions=self.get_positions_for_frame(f),
                    context=samples[i]["prompt_embeds"],
                    context_mask=samples[i].get("prompt_mask"),
                )

                enc_out, _ = self.scd_model.forward_encoder(
                    video=enc_modality, audio=None, perturbations=None,
                    kv_cache=per_sample_kv_cache[i],
                    tokens_per_frame=tpf,
                )
                current_enc = enc_out.x.detach()

                # Shift-by-1: decoder uses PREVIOUS encoder features
                if per_sample_enc_features[i] is not None:
                    enc_features_list.append(per_sample_enc_features[i])
                else:
                    enc_features_list.append(torch.zeros(
                        1, tpf, current_enc.shape[-1],
                        device=self.device, dtype=self.dtype,
                    ))
                per_sample_enc_features[i] = current_enc

            # ── DECODER: GPU-batched across samples (+ CFG) ──
            # Initialize noise per sample
            x_t_list = []
            for i in range(N):
                x_t_list.append(torch.randn(
                    1, C, 1, H, W, device=self.device, dtype=self.dtype,
                    generator=per_sample_generators[i],
                ))

            dec_positions = self.get_positions_for_frame(f)

            fm_loss_frame = [0.0] * N
            for step in range(self.num_inference_steps):
                sigma = sigmas[step]
                sigma_next = sigmas[step + 1]

                # Patchify all samples
                noisy_patches = [self.patchifier.patchify(x) for x in x_t_list]

                # Build batched decoder input: [N * cfg_mult, tokens, dim]
                # Layout: [sample0_cond, sample1_cond, ..., sampleN_cond,
                #          sample0_uncond, sample1_uncond, ..., sampleN_uncond]
                batched_latent = torch.cat(noisy_patches, dim=0)  # [N, tpf, patch_dim]
                batched_ts = torch.full(
                    (N, tpf), sigma.item(), device=self.device, dtype=self.dtype,
                )
                batched_positions = dec_positions.expand(N, -1, -1, -1)
                batched_context = torch.cat(
                    [s["prompt_embeds"] for s in samples], dim=0,
                )  # [N, seq, dim]
                batched_mask = None
                if samples[0].get("prompt_mask") is not None:
                    batched_mask = torch.cat(
                        [s["prompt_mask"] for s in samples], dim=0,
                    )
                batched_enc_ctx = torch.cat(enc_features_list, dim=0)  # [N, tpf, hidden]

                if self.use_cfg:
                    # Append unconditional: same latent/ts/positions/enc_ctx, zero context
                    null_context = torch.zeros_like(batched_context)
                    null_mask = torch.zeros_like(batched_mask) if batched_mask is not None else None

                    batched_latent = torch.cat([batched_latent, batched_latent], dim=0)
                    batched_ts = torch.cat([batched_ts, batched_ts], dim=0)
                    batched_positions = torch.cat([batched_positions, batched_positions], dim=0)
                    batched_context = torch.cat([batched_context, null_context], dim=0)
                    if batched_mask is not None and null_mask is not None:
                        batched_mask = torch.cat([batched_mask, null_mask], dim=0)
                    batched_enc_ctx = torch.cat([batched_enc_ctx, batched_enc_ctx], dim=0)

                # Single batched decoder forward: [N * cfg_mult, tpf, dim]
                batched_modality = Modality(
                    enabled=True,
                    latent=batched_latent,
                    timesteps=batched_ts,
                    positions=batched_positions,
                    context=batched_context,
                    context_mask=batched_mask,
                )

                velocity_batched, _ = self.scd_model.forward_decoder(
                    video=batched_modality,
                    encoder_features=batched_enc_ctx,
                    audio=None,
                    perturbations=None,
                )

                # Split and apply CFG
                if self.use_cfg:
                    vel_cond = velocity_batched[:N]      # [N, tpf, dim]
                    vel_uncond = velocity_batched[N:]     # [N, tpf, dim]
                    velocity_batched = vel_uncond + self.guidance_scale * (vel_cond - vel_uncond)
                # velocity_batched is now [N, tpf, dim]

                # Per-sample: FM loss + Euler step
                for i in range(N):
                    velocity_i = velocity_batched[i : i + 1]  # [1, tpf, dim]

                    gt_frame = samples[i]["latent"][:, :, f : f + 1, :, :]
                    gt_patch = self.patchifier.patchify(gt_frame)

                    if sigma.item() > 1e-6:
                        noise_est = (noisy_patches[i] - (1.0 - sigma) * gt_patch) / sigma
                        v_true = gt_patch - noise_est
                        fm_loss_step = (velocity_i - v_true).pow(2).mean().item()
                        fm_loss_frame[i] += fm_loss_step

                    vel_unpatch = self.patchifier.unpatchify(velocity_i, output_shape)
                    x_t_list[i] = x_t_list[i] + (sigma_next - sigma) * vel_unpatch

            # Per-sample: accumulate metrics for this frame
            for i in range(N):
                per_sample_generated[i].append(x_t_list[i].detach())
                fm_losses[i] += fm_loss_frame[i] / self.num_inference_steps

                gt_frame = samples[i]["latent"][:, :, f : f + 1, :, :]
                recon_losses[i] += (x_t_list[i] - gt_frame).pow(2).mean().item()

                # Pixel-space metrics
                if self.use_pixel_metrics and self.vae_decoder is not None:
                    gen_pixels = self._decode_latent_to_pixels(x_t_list[i])
                    gt_pixels = self._decode_latent_to_pixels(gt_frame)

                    if self.w_lpips > 0 and self.lpips_net is not None:
                        lpips_per_sample[i] += self.lpips_net(gen_pixels, gt_pixels).item()
                    if self.w_ssim > 0:
                        ssim_per_sample[i] += self._compute_ssim(gen_pixels, gt_pixels)

                # Temporal coherence
                if f > 0:
                    gt_prev = samples[i]["latent"][:, :, f - 1 : f, :, :].flatten()
                    gt_curr = samples[i]["latent"][:, :, f : f + 1, :, :].flatten()
                    gt_cos = torch.nn.functional.cosine_similarity(
                        gt_prev.unsqueeze(0), gt_curr.unsqueeze(0),
                    ).item()
                    gt_cos_per_sample[i].append(gt_cos)

                    gen_prev = per_sample_generated[i][-2].flatten()
                    gen_curr = per_sample_generated[i][-1].flatten()
                    gen_cos = torch.nn.functional.cosine_similarity(
                        gen_prev.unsqueeze(0), gen_curr.unsqueeze(0),
                    ).item()
                    gen_cos_per_sample[i].append(gen_cos)

        # Aggregate: average across samples
        n_frames = self.ar_frames
        results: list[FitnessResult] = []
        for i in range(N):
            avg_fm = fm_losses[i] / n_frames
            avg_recon = recon_losses[i] / n_frames
            avg_lpips = lpips_per_sample[i] / n_frames if self.use_pixel_metrics else 0.0
            avg_ssim = ssim_per_sample[i] / n_frames if self.use_pixel_metrics else 1.0

            if gt_cos_per_sample[i]:
                temporal_gap = sum(
                    abs(g - gt)
                    for g, gt in zip(gen_cos_per_sample[i], gt_cos_per_sample[i])
                ) / len(gt_cos_per_sample[i])
            else:
                temporal_gap = 0.0

            total = self._compute_fitness_score(
                avg_fm, avg_recon, temporal_gap, avg_lpips, avg_ssim,
            )
            results.append(FitnessResult(
                total=total,
                fm_loss=avg_fm,
                latent_recon=avg_recon,
                temporal_coh=temporal_gap,
                pixel_lpips=avg_lpips,
                pixel_ssim=avg_ssim,
            ))

        # Average across samples
        n = len(results)
        return FitnessResult(
            total=sum(r.total for r in results) / n,
            fm_loss=sum(r.fm_loss for r in results) / n,
            latent_recon=sum(r.latent_recon for r in results) / n,
            temporal_coh=sum(r.temporal_coh for r in results) / n,
            pixel_lpips=sum(r.pixel_lpips for r in results) / n,
            pixel_ssim=sum(r.pixel_ssim for r in results) / n,
        )
