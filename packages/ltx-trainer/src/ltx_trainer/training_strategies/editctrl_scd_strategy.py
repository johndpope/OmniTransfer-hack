"""EditCtrl + SCD training strategy for LTX-2.

Extends the SCD training strategy with EditCtrl-style video editing:
- Generates/loads edit masks and converts to token space
- Runs LocalContextModule on sparse masked source tokens
- Optionally runs GlobalContextEmbedder on downsampled background
- Computes loss only on masked (edit) regions

Two training phases:
  Phase 1: Local context only (LocalContextModule + EditCtrl LoRA)
  Phase 2: Local + Global context (adds GlobalContextEmbedder)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Literal

import torch
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
    SemanticMaskLibrary,
    dilate_token_mask,
    gather_masked_tokens,
    generate_random_token_masks,
    generate_semantic_masks,
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

    local_inner_dim: int = Field(
        default=0,
        description="Inner dim for LocalContextModule (0 = use transformer inner_dim 4096)",
    )

    local_heads: int = Field(
        default=0,
        description="Number of attention heads in LCM (0 = auto: inner_dim / 64)",
    )

    local_dim_head: int = Field(
        default=64,
        description="Dimension per attention head in LCM",
    )

    background_fill_value: float = Field(
        default=0.5,
        description="Value to fill masked (edit) region in background latents (paper: 0.5)",
        ge=0.0,
        le=1.0,
    )

    mask_channel_concat: bool = Field(
        default=True,
        description="Concatenate edit mask as extra channel to LCM input (paper: C=[E(V_b),V_m↓])",
    )

    local_control_injection: str = Field(
        default="per_layer",
        description="How to inject local control: 'pre_decoder' (once before blocks) or 'per_layer' (after FFN at selected blocks)",
    )

    local_control_layers: list[int] | None = Field(
        default=None,
        description="Which decoder block indices to inject local control at (None = all blocks). Only used when local_control_injection='per_layer'",
    )

    freeze_base_lora: bool = Field(
        default=True,
        description="Freeze the base SCD LoRA weights (only train EditCtrl LoRA + modules)",
    )

    gradient_checkpointing_local: bool = Field(
        default=True,
        description="Enable gradient checkpointing on LocalContextModule blocks",
    )

    # TMA (Task-adaptive Multimodal Alignment) params
    use_tma: bool = Field(
        default=False,
        description="Enable TMA with pre-computed Qwen VL features for semantic guidance",
    )

    tma_mllm_hidden_dim: int = Field(
        default=3584,
        description="Qwen VL hidden dimension (7B=3584, 3B=2048)",
    )

    tma_output_dim: int = Field(
        default=3840,
        description="TMA output dim — must match Gemma embedding dim (3840 for LTX-2)",
    )

    tma_num_queries: int = Field(
        default=8,
        description="Number of learnable MetaQuery tokens per task type",
    )

    tma_connector_layers: int = Field(
        default=3,
        description="Number of layers in the TMA connector MLP",
    )

    tma_features_dir: str = Field(
        default="qwen_vl_features",
        description="Subdirectory name for cached Qwen VL features under data root",
    )

    # Mask source settings
    mask_source: str = Field(
        default="random",
        description="Mask generation mode: 'random' (rectangles/ellipses), "
                    "'semantic' (per-sample Qwen VL object masks), 'mixed' (blend both)",
    )

    semantic_mask_dir: str | None = Field(
        default=None,
        description="Path to pre-computed per-sample semantic masks (output of compute_semantic_masks.py). "
                    "Each {idx:03d}.pt contains 'masks' [N_objects, H, W] and 'objects' list.",
    )

    semantic_mask_ratio: float = Field(
        default=0.5,
        description="Fraction of samples using semantic masks in 'mixed' mode",
        ge=0.0,
        le=1.0,
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
        self._tma: torch.nn.Module | None = None
        self._mask_library: SemanticMaskLibrary | None = None
        self._per_sample_masks: dict[int, dict] | None = None

        # Load per-sample semantic masks from compute_semantic_masks.py output
        if config.mask_source in ("semantic", "mixed") and config.semantic_mask_dir:
            mask_dir = config.semantic_mask_dir
            self._per_sample_masks = {}
            count = 0
            for pt_file in sorted(Path(mask_dir).glob("*.pt")):
                try:
                    idx = int(pt_file.stem)
                    data = torch.load(pt_file, weights_only=False, map_location="cpu")
                    if data.get("num_objects", 0) > 0:
                        self._per_sample_masks[idx] = data
                        count += 1
                except (ValueError, RuntimeError):
                    continue
            logger.info(
                f"EditCtrl mask_source={config.mask_source}, "
                f"loaded {count} per-sample semantic masks from {mask_dir}, "
                f"ratio={config.semantic_mask_ratio}"
            )

    def set_local_context_module(self, module: LocalContextModule) -> None:
        """Set the LocalContextModule. Called by trainer after instantiation."""
        self._local_context_module = module

    def set_global_embedder(self, module: GlobalContextEmbedder) -> None:
        """Set the GlobalContextEmbedder. Called by trainer for phase 2."""
        self._global_embedder = module

    def set_tma(self, module: torch.nn.Module) -> None:
        """Set the TMA module. Called by trainer when use_tma=True."""
        self._tma = module

    def get_data_sources(self) -> dict[str, str]:
        """Return data source subdirectory mapping.

        When TMA is enabled, adds qwen_vl_features as an additional data source
        so the dataloader loads pre-computed Qwen features alongside latents/conditions.
        """
        sources = super().get_data_sources() if hasattr(super(), "get_data_sources") else {}
        if self.config.use_tma:
            sources[self.config.tma_features_dir] = "qwen_vl_features"
        return sources

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return EditCtrl + TMA module parameters for the optimizer."""
        params = []
        if self._local_context_module is not None:
            params.extend(p for p in self._local_context_module.parameters() if p.requires_grad)
        if self._global_embedder is not None:
            params.extend(p for p in self._global_embedder.parameters() if p.requires_grad)
        if self._tma is not None:
            params.extend(p for p in self._tma.parameters() if p.requires_grad)
        return params

    def _generate_per_sample_masks(
        self,
        batch: dict[str, Any],
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        tokens_per_frame: int,
        video_seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Generate edit masks from per-sample Qwen VL object detections.

        For each sample in the batch:
        1. Look up its index in the pre-computed semantic masks
        2. Pick a random detected object that fits area constraints
        3. Resize its bounding box mask to the current latent grid
        4. Broadcast across all frames (static scene = same mask per frame)

        Falls back to random token masks for samples without detections
        or when in 'mixed' mode and the coin flip says random.
        """
        import torch.nn.functional as F

        edit_mask = torch.zeros(batch_size, video_seq_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # In mixed mode, randomly choose semantic vs random
            if self.config.mask_source == "mixed" and random.random() >= self.config.semantic_mask_ratio:
                # Random mask for this sample
                m = generate_random_token_masks(
                    batch_size=1, seq_len=video_seq_len,
                    tokens_per_frame=tokens_per_frame,
                    min_area=self.config.mask_min_area,
                    max_area=self.config.mask_max_area,
                    device=device, height=height, width=width,
                )
                edit_mask[b] = m[0]
                continue

            # Try to get per-sample mask data
            # Dataset stores idx at top level of batch (see datasets.py line 243)
            sample_idx = None
            if "idx" in batch:
                idx_val = batch["idx"]
                sample_idx = idx_val[b].item() if isinstance(idx_val, Tensor) else int(idx_val)
            elif "latents" in batch and "idx" in batch["latents"]:
                sample_idx = batch["latents"]["idx"][b].item()

            mask_data = None
            if sample_idx is not None and self._per_sample_masks is not None:
                mask_data = self._per_sample_masks.get(sample_idx)

            if mask_data is None:
                # No mask data for this sample — try random index from available
                if self._per_sample_masks:
                    fallback_idx = random.choice(list(self._per_sample_masks.keys()))
                    mask_data = self._per_sample_masks[fallback_idx]

            if mask_data is None or mask_data.get("num_objects", 0) == 0:
                # Fallback to random mask
                m = generate_random_token_masks(
                    batch_size=1, seq_len=video_seq_len,
                    tokens_per_frame=tokens_per_frame,
                    min_area=self.config.mask_min_area,
                    max_area=self.config.mask_max_area,
                    device=device, height=height, width=width,
                )
                edit_mask[b] = m[0]
                continue

            # Pick a random object from detected objects
            object_masks = mask_data["masks"]  # [N_objects, stored_h, stored_w]
            objects_list = mask_data["objects"]
            n_objects = object_masks.shape[0]

            # Filter by area constraints
            valid_indices = []
            for i in range(n_objects):
                bbox = objects_list[i]["bbox_norm"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if self.config.mask_min_area <= area <= self.config.mask_max_area:
                    valid_indices.append(i)

            if not valid_indices:
                # No objects in area range — use any object
                valid_indices = list(range(n_objects))

            obj_idx = random.choice(valid_indices)
            obj_mask = object_masks[obj_idx]  # [stored_h, stored_w]
            obj_name = objects_list[obj_idx]["name"]

            # Resize to current latent dims [height, width]
            obj_mask_resized = F.interpolate(
                obj_mask.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, stored_h, stored_w]
                size=(height, width),
                mode="nearest",
            ).squeeze()  # [height, width]

            # Broadcast across all frames → [num_frames, height, width]
            frame_mask = obj_mask_resized.unsqueeze(0).expand(num_frames, -1, -1)

            # Flatten to token sequence [seq_len]
            token_mask = frame_mask.reshape(-1) > 0.5  # [num_frames * height * width]
            edit_mask[b] = token_mask.to(device)

        return edit_mask

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

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # === TMA: Prepend Qwen VL semantic tokens to text context ===
        if self._tma is not None and "qwen_vl_features" in batch:
            qwen_data = batch["qwen_vl_features"]
            # qwen_features: [B, seq_len_qwen, hidden_dim] — variable-length Qwen outputs
            qwen_features = qwen_data["qwen_features"].to(device=device, dtype=dtype)
            # Use task index 0 (editing) for all samples
            task_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            # TMA: cross-attend MetaQueries over Qwen features → [B, num_queries, output_dim]
            tma_context = self._tma(qwen_features, task_indices)  # [B, 8, 3840]

            # Prepend TMA tokens to text context
            video_prompt_embeds = torch.cat([tma_context, video_prompt_embeds], dim=1)
            tma_mask = torch.ones(
                batch_size, tma_context.shape[1],
                dtype=prompt_attention_mask.dtype, device=device,
            )
            prompt_attention_mask = torch.cat([tma_mask, prompt_attention_mask], dim=1)

        tokens_per_frame = video_seq_len // num_frames

        # === Generate edit masks in token space ===
        # Try per-sample semantic masks (from compute_semantic_masks.py / Qwen VL detections)
        use_semantic = (
            self._per_sample_masks is not None
            and len(self._per_sample_masks) > 0
            and self.config.mask_source in ("semantic", "mixed")
        )
        if use_semantic:
            edit_mask = self._generate_per_sample_masks(
                batch, batch_size, num_frames, height, width,
                tokens_per_frame, video_seq_len, device,
            )
        else:
            edit_mask = generate_random_token_masks(
                batch_size=batch_size,
                seq_len=video_seq_len,
                tokens_per_frame=tokens_per_frame,
                min_area=self.config.mask_min_area,
                max_area=self.config.mask_max_area,
                device=device,
                height=height,
                width=width,
            )  # [B, seq_len] boolean

        # Dilate mask for boundary context
        dilated_mask = dilate_token_mask(
            edit_mask,
            tokens_per_frame=tokens_per_frame,
            dilation=self.config.mask_dilation_latent,
            height=height,
            width=width,
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
            # Optionally concatenate edit mask channel to source tokens (paper: C = [E(V_b), V_m↓])
            # This tells LCM which tokens are edit-region (1) vs boundary context (0)
            if getattr(self._local_context_module, 'mask_channel_concat', False):
                edit_mask_channel = edit_mask.unsqueeze(-1).to(dtype=dtype)  # [B, seq_len, 1] — match latent dtype
                lcm_input = torch.cat([video_latents, edit_mask_channel], dim=-1)  # [B, seq_len, C+1]
            else:
                lcm_input = video_latents

            # Gather sparse source tokens at dilated mask positions
            sparse_tokens, sparse_lengths = gather_masked_tokens(lcm_input, dilated_mask)

            # Get timestep embedding from the base model's AdaLN
            # adaln_single returns (shift_scale_gate [B, 6*D], embedded_timestep [B, D])
            # LocalContextModule's timestep_proj handles dim reduction internally
            _, embedded_timestep = self._scd_model.base_model.adaln_single(
                sigmas.flatten(),  # [B] — adaln expects 1D timesteps
                None,  # hidden_dtype
            )
            timestep_emb = embedded_timestep.unsqueeze(1)  # [B, 1, base_model_dim]

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
                fill_value=self.config.background_fill_value,
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

        return loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Include EditCtrl + TMA metadata in checkpoints."""
        meta = super().get_checkpoint_metadata()
        meta.update({
            "editctrl_phase": self.config.editctrl_phase,
            "editctrl_local_num_blocks": self.config.local_num_blocks,
            "editctrl_mask_dilation": self.config.mask_dilation_latent,
            "editctrl_global_num_tokens": self.config.global_context_num_tokens,
            "editctrl_background_fill": self.config.background_fill_value,
            "editctrl_mask_channel_concat": self.config.mask_channel_concat,
            "editctrl_local_control_injection": self.config.local_control_injection,
            "editctrl_local_control_layers": self.config.local_control_layers,
            "use_tma": self.config.use_tma,
        })
        if self.config.use_tma:
            meta.update({
                "tma_mllm_hidden_dim": self.config.tma_mllm_hidden_dim,
                "tma_output_dim": self.config.tma_output_dim,
                "tma_num_queries": self.config.tma_num_queries,
                "tma_connector_layers": self.config.tma_connector_layers,
            })
        return meta

    def get_strategy_state_dict(self) -> dict[str, Tensor]:
        """Save EditCtrl + TMA module weights alongside LoRA checkpoint."""
        state = {}
        if self._local_context_module is not None:
            for k, v in self._local_context_module.state_dict().items():
                state[f"strategy.local_context_module.{k}"] = v
        if self._global_embedder is not None:
            for k, v in self._global_embedder.state_dict().items():
                state[f"strategy.global_embedder.{k}"] = v
        if self._tma is not None:
            for k, v in self._tma.state_dict().items():
                state[f"strategy.tma.{k}"] = v
        return state

    def load_strategy_state_dict(
        self, state_dict: dict[str, Tensor]
    ) -> tuple[list[str], list[str]]:
        """Load EditCtrl + TMA module weights from checkpoint."""
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

        # Extract TMA weights
        tma_prefix = "strategy.tma."
        tma_dict = {
            k[len(tma_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(tma_prefix)
        }
        if tma_dict and self._tma is not None:
            self._tma.load_state_dict(tma_dict, strict=False)
            loaded.extend(tma_dict.keys())
            logger.info(f"Loaded TMA weights: {len(tma_dict)} tensors")
        elif tma_dict:
            skipped.extend(tma_dict.keys())

        return loaded, skipped
