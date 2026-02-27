"""Text-to-video training strategy.
This strategy implements standard text-to-video generation training where:
- Only target latents are used (no reference videos)
- Standard noise application and loss computation
- Supports first frame conditioning
- Optionally supports joint audio-video training
"""

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class TextToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for text-to-video training strategy."""

    name: Literal["text_to_video"] = "text_to_video"

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )

    log_reconstructions: bool = Field(
        default=False,
        description="Whether to log reconstruction visualizations to W&B",
    )

    reconstruction_log_interval: int = Field(
        default=200,
        description="Steps between reconstruction visualization logging",
    )


class TextToVideoStrategy(TrainingStrategy):
    """Text-to-video training strategy.
    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Supports first frame conditioning
    - Optionally supports joint audio-video training when with_audio=True
    """

    config: TextToVideoConfig

    def __init__(self, config: TextToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Text-to-video configuration
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """
        Text-to-video training requires latents and text conditions.
        When with_audio is True, also requires audio latents.
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for text-to-video training."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        video_latents = latents["latents"]

        # Get video dimensions (assume same for all batch elements)
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS with backward compatibility
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # Create conditioning mask (first frame conditioning)
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Compute video targets (velocity prediction)
        video_targets = video_noise - video_latents

        # Create per-token timesteps
        video_timesteps = self._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        # Generate video positions using ltx_core's native implementation
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Video loss mask: True for tokens we want to compute loss on (non-conditioning tokens)
        video_loss_mask = ~video_conditioning_mask

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        model_inputs = ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            shared_noise=video_noise,
            shared_sigmas=sigmas,
        )

        # Store raw latents for reconstruction visualization (pre-patchified [B, C, F, H, W])
        model_inputs._raw_video_latents = batch["latents"]["latents"]

        return model_inputs

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint audio-video training.
        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as video)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Get audio latents - dataset provides uniform non-patchified format [B, C, T, F]
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]

        # Patchify audio latents: [B, C, T, F] -> [B, T, C*F]
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)

        # Apply noise to audio (same sigma as video)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (1 - sigmas_expanded) * audio_latents + sigmas_expanded * audio_noise

        # Compute audio targets
        audio_targets = audio_noise - audio_latents

        # Audio timesteps: all tokens use the sampled sigma (no conditioning mask)
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        # Generate audio positions
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Audio loss mask: all tokens contribute to loss (no conditioning)
        audio_loss_mask = torch.ones(batch_size, audio_seq_len, dtype=torch.bool, device=device)

        return audio_modality, audio_targets, audio_loss_mask

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for video and optionally audio."""
        # Video loss
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        video_loss = video_loss.mean()

        # If no audio, return video loss only
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        # Audio loss (no conditioning mask)
        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean()

        # Combined loss
        return video_loss + audio_loss

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: ModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction visualizations to W&B.

        Recovers the predicted clean latent from the velocity prediction using
        flow matching: x_0_hat = noise - v_hat, then VAE-decodes both ground
        truth and prediction for side-by-side comparison.

        Args:
            video_pred: Model velocity prediction [B, seq_len, C]
            inputs: ModelInputs with raw latents and noise
            step: Current training step
            vae_decoder: VAE decoder for pixel-space visualization
            prefix: W&B metric prefix

        Returns:
            Dictionary of logged W&B metrics
        """
        try:
            import wandb
        except ImportError:
            return {}

        if wandb.run is None or not self.config.log_reconstructions:
            return {}

        raw_latents = getattr(inputs, "_raw_video_latents", None)
        if raw_latents is None:
            logger.warning("No raw latents stored for reconstruction")
            return {}

        b, c, f, h, w = raw_latents.shape
        noise = inputs.shared_noise  # [B, seq_len, C] (patchified)

        # Flow matching recovery: v = noise - clean → clean_hat = noise - v_hat
        pred_clean = noise - video_pred  # [B, seq_len, C]

        # Unpatchify: [B, seq_len, C] → [B, C, F, H, W]
        pred_clean_spatial = pred_clean.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
        gt_latents = raw_latents

        # VAE-decode to pixel space if decoder available
        if vae_decoder is not None:
            try:
                decoder_device = next(vae_decoder.parameters()).device
                decoder_dtype = next(vae_decoder.parameters()).dtype

                with torch.inference_mode():
                    gt_decoded = vae_decoder(
                        gt_latents[:1].to(device=decoder_device, dtype=decoder_dtype)
                    )
                    pred_decoded = vae_decoder(
                        pred_clean_spatial[:1].to(device=decoder_device, dtype=decoder_dtype)
                    )

                gt_decoded = gt_decoded.float().clamp(-1, 1) * 0.5 + 0.5
                pred_decoded = pred_decoded.float().clamp(-1, 1) * 0.5 + 0.5

                # For T2I (1 frame), use frame 0; for T2V use middle frame
                mid_f = gt_decoded.shape[2] // 2
                gt_frame = gt_decoded[0, :, mid_f].cpu()
                pred_frame = pred_decoded[0, :, mid_f].cpu()

                import torchvision.utils as vutils

                grid = vutils.make_grid([gt_frame, pred_frame], nrow=2, padding=4)

                log_dict = {
                    f"{prefix}/reconstruction": wandb.Image(
                        grid.permute(1, 2, 0).numpy(),
                        caption=f"Step {step} | Left: Ground Truth | Right: Prediction",
                    ),
                }

                # Log first and last frames for multi-frame data
                if gt_decoded.shape[2] > 1:
                    gt_first = gt_decoded[0, :, 0].cpu()
                    pred_first = pred_decoded[0, :, 0].cpu()
                    gt_last = gt_decoded[0, :, -1].cpu()
                    pred_last = pred_decoded[0, :, -1].cpu()

                    grid_first = vutils.make_grid([gt_first, pred_first], nrow=2, padding=4)
                    grid_last = vutils.make_grid([gt_last, pred_last], nrow=2, padding=4)

                    log_dict[f"{prefix}/reconstruction_first"] = wandb.Image(
                        grid_first.permute(1, 2, 0).numpy(),
                        caption=f"Step {step} | First frame | GT vs Pred",
                    )
                    log_dict[f"{prefix}/reconstruction_last"] = wandb.Image(
                        grid_last.permute(1, 2, 0).numpy(),
                        caption=f"Step {step} | Last frame | GT vs Pred",
                    )

                # Return log_dict — caller (trainer) handles wandb.log with proper step
                logger.debug(f"Logged T2V reconstruction images at step {step}")
                return log_dict

            except Exception as e:
                logger.warning(f"Failed to decode reconstruction: {e}")

        # Fallback: latent-space pseudo-RGB visualization
        mid_f = f // 2
        gt_vis = raw_latents[0, :3, mid_f].cpu().float()
        pred_vis = pred_clean_spatial[0, :3, mid_f].cpu().float()

        def normalize(x: Tensor) -> Tensor:
            x = x - x.min()
            return x / (x.max() + 1e-8)

        import torchvision.utils as vutils

        grid = vutils.make_grid([normalize(gt_vis), normalize(pred_vis)], nrow=2, padding=4)

        log_dict = {
            f"{prefix}/reconstruction_latent": wandb.Image(
                grid.permute(1, 2, 0).numpy(),
                caption=f"Step {step} | Latent pseudo-RGB | GT vs Pred",
            ),
        }
        # Return log_dict — caller (trainer) handles wandb.log with proper step
        return log_dict
