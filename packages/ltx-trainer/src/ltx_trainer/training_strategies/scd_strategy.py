"""Separable Causal Diffusion (SCD) training strategy for LTX-2.

This strategy splits the forward pass into encoder and decoder phases:
- Encoder: Runs once with timestep=0 (clean signal) and causal frame mask
- Decoder: Runs N times with actual noise levels, conditioned on shifted encoder features

The key insight from the SCD paper is that causal temporal reasoning (encoder) is
separable from multi-step denoising (decoder), enabling significant speedup.
"""

import random
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    build_frame_causal_mask,
    shift_encoder_features,
)
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class SCDTrainingConfig(TrainingStrategyConfigBase):
    """Configuration for SCD training strategy."""

    name: Literal["scd"] = "scd"

    encoder_layers: int = Field(
        default=32,
        description="Number of transformer layers for the encoder (remaining go to decoder). "
        "Default 32 follows the SCD paper's ~2:1 encoder:decoder ratio for a 48-layer model.",
        ge=1,
    )

    decoder_input_combine: str = Field(
        default="token_concat",
        description="How to combine encoder features with decoder input. "
        "Options: 'token_concat' (best from SCD paper), 'add', 'token_concat_with_proj'.",
    )

    clean_context_ratio: float = Field(
        default=0.0,
        description="Fraction of frames (beyond the first) kept clean as additional context. "
        "0.0 means only the first frame is always clean context.",
        ge=0.0,
        le=1.0,
    )

    decoder_multi_batch: int = Field(
        default=1,
        description="Number of decoder passes per encoder pass. Higher values amortize "
        "encoder cost and provide more training signal. Default 1 for simplicity.",
        ge=1,
        le=4,
    )

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )


class SCDTrainingStrategy(TrainingStrategy):
    """SCD training strategy for LTX-2.

    This strategy implements the Separable Causal Diffusion training paradigm:
    1. Encoder pass with causal mask and timestep=0 (clean features)
    2. Shift encoder output by 1 frame (frame t-1 context → frame t decoder)
    3. Decoder pass with actual noise levels and velocity prediction
    """

    config: SCDTrainingConfig

    def __init__(self, config: SCDTrainingConfig):
        super().__init__(config)
        self._scd_model: LTXSCDModel | None = None

    def set_scd_model(self, model: LTXSCDModel) -> None:
        """Set the SCD model wrapper. Called by the trainer after model creation."""
        self._scd_model = model

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
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
        """Prepare SCD training inputs with split encoder/decoder passes.

        The flow:
        1. Patchify latents [B, C, F, H, W] → [B, seq_len, C]
        2. Build frame-level causal mask for encoder
        3. Encoder pass: timestep=0 for all tokens → encoder_features
        4. Shift encoder features by 1 frame (frame t-1 → frame t)
        5. Decoder pass: actual sigma timestep + shifted encoder features → velocity

        If the SCD model wrapper is not set, falls back to standard training
        (encoder + decoder as a single pass through the full model).
        """
        # Get pre-encoded latents [B, C, F, H, W]
        latents = batch["latents"]
        video_latents = latents["latents"]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify: [B, C, F, H, W] → [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}, using first: {fps[0].item()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        tokens_per_frame = video_seq_len // num_frames

        # Create conditioning mask (first frame conditioning)
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        # Sample noise and sigmas for decoder
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # Conditioning tokens use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Velocity target: v = noise - clean
        video_targets = video_noise - video_latents

        # Generate positions
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # === SCD-specific: Split encoder/decoder if SCD model is available ===
        if self._scd_model is not None:
            return self._prepare_scd_inputs(
                video_latents=video_latents,
                noisy_video=noisy_video,
                video_noise=video_noise,
                video_targets=video_targets,
                video_positions=video_positions,
                video_prompt_embeds=video_prompt_embeds,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                video_conditioning_mask=video_conditioning_mask,
                sigmas=sigmas,
                batch_size=batch_size,
                video_seq_len=video_seq_len,
                tokens_per_frame=tokens_per_frame,
                num_frames=num_frames,
                device=device,
                dtype=dtype,
                batch=batch,
            )

        # === Fallback: Standard training (no SCD split) ===
        video_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

        video_modality = Modality(
            enabled=True,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        video_loss_mask = ~video_conditioning_mask

        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
            shared_noise=video_noise,
            shared_sigmas=sigmas,
        )

    def _prepare_scd_inputs(
        self,
        video_latents: Tensor,
        noisy_video: Tensor,
        video_noise: Tensor,
        video_targets: Tensor,
        video_positions: Tensor,
        video_prompt_embeds: Tensor,
        audio_prompt_embeds: Tensor | None,
        prompt_attention_mask: Tensor,
        video_conditioning_mask: Tensor,
        sigmas: Tensor,
        batch_size: int,
        video_seq_len: int,
        tokens_per_frame: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        batch: dict[str, Any],
    ) -> ModelInputs:
        """Prepare inputs with SCD encoder/decoder split.

        This runs the encoder pass internally and returns ModelInputs configured
        so the trainer's forward pass only runs the decoder.
        """
        # --- ENCODER PASS ---
        # Encoder sees CLEAN latents with timestep=0
        encoder_timesteps = torch.zeros(
            batch_size, video_seq_len, device=device, dtype=dtype
        )

        encoder_modality = Modality(
            enabled=True,
            latent=video_latents,  # Clean latents for encoder
            timesteps=encoder_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Run encoder pass with causal mask (tokens_per_frame enables the mask)
        encoder_video_args, encoder_audio_args = self._scd_model.forward_encoder(
            video=encoder_modality,
            audio=None,
            perturbations=None,
            tokens_per_frame=tokens_per_frame,
        )

        # Get encoder features and shift by 1 frame
        # NOTE: Do NOT detach — gradients must flow through encoder during training
        encoder_features = encoder_video_args.x  # [B, seq_len, D]
        shifted_features = shift_encoder_features(
            encoder_features, tokens_per_frame, num_frames
        )

        # --- DECODER PASS ---
        # Decoder sees NOISY latents with actual sigma timestep
        decoder_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

        # Create decoder modality with noisy latents
        # The SCD model's forward_decoder will handle combining with encoder features
        decoder_modality = Modality(
            enabled=True,
            latent=noisy_video,
            timesteps=decoder_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        video_loss_mask = ~video_conditioning_mask

        # Store encoder features and SCD model ref in a way the trainer can access
        # We attach them to the ModelInputs via a custom attribute
        model_inputs = ModelInputs(
            video=decoder_modality,
            audio=None,
            video_targets=video_targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            shared_noise=video_noise,
            shared_sigmas=sigmas,
        )

        # Attach SCD-specific data for the trainer's forward pass
        # Encoder features are NOT detached so gradients flow back through encoder
        model_inputs._encoder_features = shifted_features
        model_inputs._scd_model = self._scd_model
        model_inputs._encoder_audio_args = encoder_audio_args

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for SCD training.

        Velocity prediction target: v = noise - clean
        Loss is masked to exclude conditioning frames.
        """
        # Video loss
        video_loss = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1).float()
        video_loss = video_loss.mul(video_loss_mask).div(video_loss_mask.mean())
        video_loss = video_loss.mean()

        # Audio loss if enabled
        if not self.config.with_audio or audio_pred is None or inputs.audio_targets is None:
            return video_loss

        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean()
        return video_loss + audio_loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Include SCD-specific metadata in checkpoints."""
        return {
            "scd_encoder_layers": self.config.encoder_layers,
            "scd_decoder_input_combine": self.config.decoder_input_combine,
        }
