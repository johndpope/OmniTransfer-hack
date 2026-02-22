"""EditCtrl + SCD training strategy for LTX-2.

Extends the SCD training strategy with EditCtrl-style video editing:
- Generates/loads edit masks and converts to token space
- Runs LocalContextModule on sparse masked source tokens
- Optionally runs GlobalContextEmbedder on downsampled background
- Computes loss only on masked (edit) regions
- (Optional) HRR text enhancement for richer conditioning + cross-caption signal

Two training phases:
  Phase 1: Local context only (LocalContextModule + EditCtrl LoRA)
  Phase 2: Local + Global context (adds GlobalContextEmbedder)
"""

from __future__ import annotations

import sys
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.editctrl_modules import (
    GlobalContextEmbedder,
    LocalContextModule,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    shift_encoder_features,
)
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
)
from ltx_trainer.training_strategies.scd_strategy import (
    SCDTrainingConfig,
    SCDTrainingStrategy,
)

# Add mask utils path
sys.path.insert(0, "/home/johndpope/Documents/GitHub/sparse-causal-diffusion")
from scd.utils.mask_utils import (
    dilate_token_mask,
    gather_masked_tokens,
    generate_random_token_masks,
    prepare_background_latents,
    scatter_masked_tokens,
)


class EditCtrlSCDConfig(SCDTrainingConfig):
    """Configuration for EditCtrl + SCD training strategy."""

    name: Literal["editctrl_scd"] = "editctrl_scd"

    # EditCtrl-specific params
    editctrl_phase: int = Field(
        default=1,
        description="Training phase: 1=local context only, 2=local+global context",
        ge=1,
        le=2,
    )

    local_num_blocks: int = Field(
        default=4,
        description="Number of transformer blocks in LocalContextModule",
        ge=1,
        le=8,
    )

    mask_dilation_latent: int = Field(
        default=2,
        description="Dilation of edit mask in latent token space (provides boundary context)",
        ge=0,
    )

    global_context_num_tokens: int = Field(
        default=256,
        description="Number of background tokens for GlobalContextEmbedder",
    )

    mask_min_area: float = Field(
        default=0.05,
        description="Minimum mask area fraction for synthetic masks",
        ge=0.01,
        le=0.5,
    )

    mask_max_area: float = Field(
        default=0.6,
        description="Maximum mask area fraction for synthetic masks",
        ge=0.1,
        le=0.95,
    )

    freeze_base_lora: bool = Field(
        default=True,
        description="Freeze the base SCD LoRA weights (only train EditCtrl LoRA + modules)",
    )

    gradient_checkpointing_local: bool = Field(
        default=True,
        description="Enable gradient checkpointing on LocalContextModule blocks",
    )

    use_hrr: bool = Field(
        default=False,
        description="Enable HRR text enhancement for richer conditioning + cross-caption signal",
    )

    cross_caption_loss_weight: float = Field(
        default=0.1,
        description="Weight for cross-caption edit loss (requires HRR + paired dataset)",
        ge=0.0,
    )


class EditCtrlSCDTrainingStrategy(SCDTrainingStrategy):
    """EditCtrl + SCD training strategy.

    Extends SCD with mask-based editing:
    1. Inherits encoder pass from SCD (causal mask, timestep=0)
    2. Generates synthetic edit masks in token space
    3. Runs LocalContextModule on sparse source tokens at masked positions
    4. (Phase 2) Runs GlobalContextEmbedder on downsampled background
    5. Passes local_control + global_context to SCD decoder
    6. Computes loss only on masked (edit) region tokens
    """

    config: EditCtrlSCDConfig

    def __init__(self, config: EditCtrlSCDConfig):
        super().__init__(config)
        self._local_context_module: LocalContextModule | None = None
        self._global_embedder: GlobalContextEmbedder | None = None
        self._hrr_enhancer: nn.Module | None = None

    def set_local_context_module(self, module: LocalContextModule) -> None:
        """Set the LocalContextModule. Called by trainer after instantiation."""
        self._local_context_module = module

    def set_global_embedder(self, module: GlobalContextEmbedder) -> None:
        """Set the GlobalContextEmbedder. Called by trainer for phase 2."""
        self._global_embedder = module

    def set_hrr_enhancer(self, enhancer: nn.Module) -> None:
        """Set the HRR text enhancer. Called by trainer when HRR is enabled."""
        self._hrr_enhancer = enhancer

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return EditCtrl module parameters for the optimizer."""
        params = []
        if self._local_context_module is not None:
            params.extend(p for p in self._local_context_module.parameters() if p.requires_grad)
        if self._global_embedder is not None:
            params.extend(p for p in self._global_embedder.parameters() if p.requires_grad)
        if self._hrr_enhancer is not None:
            params.extend(p for p in self._hrr_enhancer.parameters() if p.requires_grad)
        return params

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare EditCtrl + SCD training inputs.

        Overrides the SCD strategy to add:
        1. Synthetic mask generation in token space
        2. Noisy latent construction (noise at masked, clean at unmasked)
        3. LocalContextModule forward pass on sparse masked source tokens
        4. (Phase 2) GlobalContextEmbedder forward pass
        5. Edit-region-only loss mask

        The encoder pass is still handled by the parent SCD logic.
        """
        # Get pre-encoded latents
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify: [B, C, F, H, W] → [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # === HRR text enhancement ===
        hrr_input_embeds = None
        edit_prompt_embeds = None
        edit_prompt_mask = None
        hrr_edit_mask = None

        if self._hrr_enhancer is not None:
            # Store pre-HRR embeddings for entropy regularization
            hrr_input_embeds = video_prompt_embeds.detach()
            # Enhance source prompt with HRR
            video_prompt_embeds = self._hrr_enhancer(video_prompt_embeds)

            # Cross-caption: enhance edit prompt if available
            if "edit_conditions" in batch:
                edit_cond = batch["edit_conditions"]
                edit_embeds_raw = edit_cond.get("video_prompt_embeds", edit_cond.get("prompt_embeds"))
                if edit_embeds_raw is not None:
                    edit_prompt_mask = edit_cond.get("prompt_attention_mask")
                    edit_prompt_embeds = self._hrr_enhancer(edit_embeds_raw)

                    # Compute HRR routing divergence (edit mask)
                    if hasattr(self._hrr_enhancer, "get_edit_mask"):
                        hrr_edit_mask = self._hrr_enhancer.get_edit_mask(
                            hrr_input_embeds, edit_embeds_raw
                        )

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        tokens_per_frame = video_seq_len // num_frames

        # === Generate synthetic edit masks in token space ===
        edit_mask = generate_random_token_masks(
            batch_size=batch_size,
            seq_len=video_seq_len,
            tokens_per_frame=tokens_per_frame,
            min_area=self.config.mask_min_area,
            max_area=self.config.mask_max_area,
            device=device,
        )  # [B, seq_len] boolean

        # Dilate mask for boundary context
        dilated_mask = dilate_token_mask(
            edit_mask,
            tokens_per_frame=tokens_per_frame,
            dilation=self.config.mask_dilation_latent,
        )  # [B, seq_len] boolean

        # === Sample noise and construct noisy latents ===
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        # Noisy at masked positions, clean at unmasked
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise
        edit_mask_expanded = edit_mask.unsqueeze(-1)  # [B, seq_len, 1]
        noisy_video = torch.where(edit_mask_expanded, noisy_video, video_latents)

        # Velocity target
        video_targets = video_noise - video_latents

        # Video positions
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        # === SCD Encoder Pass (reuse parent logic) ===
        if self._scd_model is None:
            raise RuntimeError("EditCtrl requires SCD model wrapper. Set via set_scd_model().")

        # Encoder sees clean latents with timestep=0
        encoder_timesteps = torch.zeros(batch_size, video_seq_len, device=device, dtype=dtype)
        encoder_modality = Modality(
            enabled=True,
            latent=video_latents,
            timesteps=encoder_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        encoder_video_args, encoder_audio_args = self._scd_model.forward_encoder(
            video=encoder_modality, audio=None, perturbations=None,
            tokens_per_frame=tokens_per_frame,
        )

        encoder_features = encoder_video_args.x
        shifted_features = shift_encoder_features(encoder_features, tokens_per_frame, num_frames)

        # === LocalContextModule forward pass ===
        local_control = None
        if self._local_context_module is not None:
            # Gather sparse source tokens at dilated mask positions
            sparse_tokens, sparse_lengths = gather_masked_tokens(video_latents, dilated_mask)

            # Get timestep embedding from the base model's AdaLN
            # We use the decoder timestep (actual sigma) for the local module
            timestep_emb = self._scd_model.base_model.adaln_single(
                sigmas.view(-1, 1).expand(-1, 1),
                None,  # class_labels
            )  # [B, 1, inner_dim] approximately

            # Reshape if needed — adaln_single returns [B, num_ada_params, dim]
            if timestep_emb.dim() == 3 and timestep_emb.shape[1] > 1:
                # Take just the timestep portion [B, 1, dim]
                timestep_emb = timestep_emb[:, :1, :]

            local_control = self._local_context_module(
                source_tokens=sparse_tokens,
                mask_indices=dilated_mask,
                text_context=video_prompt_embeds,
                text_mask=prompt_attention_mask,
                timestep_emb=timestep_emb,
                seq_len=video_seq_len,
            )

        # === GlobalContextEmbedder forward pass (Phase 2) ===
        global_context = None
        if self._global_embedder is not None and self.config.editctrl_phase >= 2:
            bg_tokens = prepare_background_latents(
                source_latents=video_latents,
                edit_mask=edit_mask,
                target_num_tokens=self.config.global_context_num_tokens,
            )
            global_context = self._global_embedder(bg_tokens)

        # === Build decoder modality ===
        decoder_timesteps = self._create_per_token_timesteps(
            torch.zeros_like(edit_mask),  # No first-frame conditioning for EditCtrl
            sigmas.squeeze(),
        )

        decoder_modality = Modality(
            enabled=True,
            latent=noisy_video,
            timesteps=decoder_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Loss mask: only compute loss on edit (masked) region tokens
        video_loss_mask = edit_mask

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

        # Attach SCD + EditCtrl data for the trainer's forward pass
        model_inputs._encoder_features = shifted_features
        model_inputs._scd_model = self._scd_model
        model_inputs._encoder_audio_args = encoder_audio_args
        model_inputs._raw_video_latents = batch["latents"]["latents"]

        # Attach EditCtrl-specific data
        model_inputs._local_control = local_control
        model_inputs._global_context = global_context
        model_inputs._edit_mask = edit_mask

        # Attach HRR-specific data (for cross-caption loss + entropy reg)
        if self._hrr_enhancer is not None:
            model_inputs._hrr_input_embeds = hrr_input_embeds
            model_inputs._edit_prompt_embeds = edit_prompt_embeds
            model_inputs._edit_prompt_mask = edit_prompt_mask
            model_inputs._hrr_edit_mask = hrr_edit_mask

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for EditCtrl training.

        Loss is computed only on edit (masked) region tokens and normalized
        by the number of masked tokens (not total tokens).

        When HRR is enabled, also adds:
        - Entropy regularization (encourages diverse routing)
        - Gate encouragement (pushes gate open so HRR contributes)
        """
        edit_mask = getattr(inputs, "_edit_mask", inputs.video_loss_mask)

        # Velocity prediction loss
        video_loss = (video_pred - inputs.video_targets).pow(2)  # [B, seq_len, C]

        # Mask: only edit region
        mask_float = edit_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        masked_loss = video_loss * mask_float

        # Normalize by masked token count (avoid division by zero)
        num_masked = mask_float.sum().clamp(min=1.0)
        loss = masked_loss.sum() / num_masked

        # === HRR auxiliary losses ===
        if self._hrr_enhancer is not None and inputs._hrr_input_embeds is not None:
            from ltx_trainer.config import HRRConfig  # noqa: PLC0415

            # Get HRR config from trainer config (accessed via strategy config parent)
            entropy_weight = self.config.cross_caption_loss_weight * 0.01  # Scale down
            gate_weight = self.config.cross_caption_loss_weight * 0.01

            # Entropy regularization: encourage diverse routing
            if hasattr(self._hrr_enhancer, "get_routing_entropy"):
                entropy = self._hrr_enhancer.get_routing_entropy(inputs._hrr_input_embeds)
                # Negative entropy = encourage higher entropy (more diverse routing)
                loss = loss - entropy_weight * entropy

            # Gate encouragement: push gate toward opening
            if hasattr(self._hrr_enhancer, "gate"):
                gate_val = torch.sigmoid(self._hrr_enhancer.gate)
                loss = loss - gate_weight * gate_val

        return loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Include EditCtrl metadata in checkpoints."""
        meta = super().get_checkpoint_metadata()
        meta.update({
            "editctrl_phase": self.config.editctrl_phase,
            "editctrl_local_num_blocks": self.config.local_num_blocks,
            "editctrl_mask_dilation": self.config.mask_dilation_latent,
            "editctrl_global_num_tokens": self.config.global_context_num_tokens,
        })
        return meta

    def get_strategy_state_dict(self) -> dict[str, Tensor]:
        """Save EditCtrl module weights alongside LoRA checkpoint."""
        state = {}
        if self._local_context_module is not None:
            for k, v in self._local_context_module.state_dict().items():
                state[f"strategy.local_context_module.{k}"] = v
        if self._global_embedder is not None:
            for k, v in self._global_embedder.state_dict().items():
                state[f"strategy.global_embedder.{k}"] = v
        if self._hrr_enhancer is not None:
            for k, v in self._hrr_enhancer.state_dict().items():
                state[f"strategy.hrr_enhancer.{k}"] = v
        return state

    def load_strategy_state_dict(
        self, state_dict: dict[str, Tensor]
    ) -> tuple[list[str], list[str]]:
        """Load EditCtrl module weights from checkpoint."""
        loaded = []
        skipped = []

        # Extract local context module weights
        lcm_prefix = "strategy.local_context_module."
        lcm_dict = {
            k[len(lcm_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(lcm_prefix)
        }
        if lcm_dict and self._local_context_module is not None:
            self._local_context_module.load_state_dict(lcm_dict, strict=False)
            loaded.extend(lcm_dict.keys())
        elif lcm_dict:
            skipped.extend(lcm_dict.keys())

        # Extract global embedder weights
        ge_prefix = "strategy.global_embedder."
        ge_dict = {
            k[len(ge_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(ge_prefix)
        }
        if ge_dict and self._global_embedder is not None:
            self._global_embedder.load_state_dict(ge_dict, strict=False)
            loaded.extend(ge_dict.keys())
        elif ge_dict:
            skipped.extend(ge_dict.keys())

        # Extract HRR enhancer weights
        hrr_prefix = "strategy.hrr_enhancer."
        hrr_dict = {
            k[len(hrr_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(hrr_prefix)
        }
        if hrr_dict and self._hrr_enhancer is not None:
            self._hrr_enhancer.load_state_dict(hrr_dict, strict=False)
            loaded.extend(hrr_dict.keys())
            logger.info(f"Loaded HRR enhancer: {len(hrr_dict)} tensors")
        elif hrr_dict:
            skipped.extend(hrr_dict.keys())

        return loaded, skipped
