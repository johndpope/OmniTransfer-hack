"""OmniTransfer Training Strategy for LTX-2.

This module implements the complete OmniTransfer training strategy that combines:
1. Reference Latent Construction (Section 4.1)
2. Task-aware Positional Bias (Section 4.2)
3. Reference-decoupled Causal Learning (Section 4.3)
4. Task-adaptive Multimodal Alignment (Section 4.4)

Training follows a multi-stage approach:
Quote: "The training process is divided into three sequential stages with distinct
optimization objectives. In the first stage, we train the DiT blocks with TPB and
RCL for 10,000 steps. In the second stage, we freeze the DiT blocks and train only
the TMA connector for 2,000 steps. In the third stage, we jointly fine-tune all
components for 5,000 steps." (Section 5.1)

Includes W&B integration for logging:
- Training metrics (loss, learning rate)
- Reconstruction visualizations (reference, target, prediction)
- Video comparisons at specified intervals

References: OmniTransfer paper (arXiv:2601.14250v1, Jan 20, 2026)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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
from ltx_trainer.omnitransfer.components import (
    OmniTransferTask,
    TaskAwarePositionalBias,
    TaskAwarePositionalBiasConfig,
)
from ltx_trainer.omnitransfer.latent_constructor import (
    ConstructedLatents,
    ReferenceLatentConstructor,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class OmniTransferStage(Enum):
    """Training stages for OmniTransfer.

    Quote: "training process is divided into three sequential stages" (Section 5.1)
    """
    # Stage 1: Train DiT blocks with TPB/RCL (10k steps)
    IN_CONTEXT = "in_context"
    # Stage 2: Freeze DiT, train TMA connector (2k steps)
    CONNECTOR = "connector"
    # Stage 3: Joint fine-tuning (5k steps)
    JOINT = "joint"


class OmniTransferConfig(TrainingStrategyConfigBase):
    """Configuration for OmniTransfer training strategy.

    This configuration controls all aspects of OmniTransfer training including
    task type, component enables, and stage-specific settings.
    """

    name: Literal["omnitransfer"] = "omnitransfer"

    # Task configuration
    task_type: str = Field(
        default="motion_transfer",
        description="Type of transfer task. Options: motion_transfer, pose_reenactment, "
        "action_customization, style_transfer, identity_preservation, scene_composition",
    )

    # Component enables
    enable_tpb: bool = Field(
        default=True,
        description="Enable Task-aware Positional Bias (TPB) for task-specific "
        "RoPE offsets. Quote: 'Task-aware Positional Bias applies distinct "
        "positional biases based on the task type' (Section 4.2)",
    )

    enable_rcl: bool = Field(
        default=True,
        description="Enable Reference-decoupled Causal Learning (RCL) for "
        "separate ref/target attention branches. Quote: 'RCL decouples the "
        "reference and target branches in attention computation' (Section 4.3)",
    )

    enable_tma: bool = Field(
        default=True,
        description="Enable Task-adaptive Multimodal Alignment (TMA) for "
        "MLLM-guided semantic features. Quote: 'TMA leverages a multimodal LLM "
        "to provide semantic guidance' (Section 4.4)",
    )

    # First frame conditioning
    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    # Data directories
    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Directory name for reference video latents",
    )

    # TPB configuration
    tpb_max_pos: list[int] = Field(
        default=[20, 2048, 2048],
        description="Maximum position values [time, height, width] for RoPE normalization",
    )

    tpb_theta: float = Field(
        default=10000.0,
        description="Theta parameter for RoPE frequency computation",
    )

    # RCL configuration
    rcl_ref_timestep: float = Field(
        default=0.0,
        description="Fixed timestep for reference branch (t=0 means noise-free). "
        "Quote: 'reference branch adopts a fixed t=0' (Section 4.3)",
    )

    # TMA configuration
    tma_num_queries: int = Field(
        default=8,
        description="Number of MetaQueries per task for TMA aggregation",
    )

    tma_connector_layers: int = Field(
        default=3,
        description="Number of MLP layers in TMA connector. "
        "Quote: 'projected through a three-layer MLP connector' (Section 4.4)",
    )

    # Loss weighting
    target_loss_weight: float = Field(
        default=1.0,
        description="Weight for target reconstruction loss",
    )

    ref_preservation_loss_weight: float = Field(
        default=0.0,
        description="Optional weight for reference preservation regularization",
    )

    # W&B Visualization configuration
    log_reconstructions: bool = Field(
        default=True,
        description="Whether to log reconstruction visualizations to W&B",
    )

    reconstruction_log_interval: int = Field(
        default=500,
        description="Steps between reconstruction visualization logging",
    )

    num_frames_to_visualize: int = Field(
        default=8,
        description="Number of frames to include in visualization grids",
    )

    max_samples_per_log: int = Field(
        default=4,
        description="Maximum batch samples to log per visualization step",
    )

    log_video_comparisons: bool = Field(
        default=True,
        description="Whether to log video comparisons (more expensive)",
    )

    video_log_interval: int = Field(
        default=2000,
        description="Steps between video comparison logging (less frequent than images)",
    )

    save_reconstructions_locally: bool = Field(
        default=False,
        description="Whether to save reconstruction images locally",
    )

    local_reconstruction_dir: str | None = Field(
        default=None,
        description="Directory for local reconstruction saves",
    )

    @property
    def task(self) -> OmniTransferTask:
        """Get the OmniTransferTask enum from string config."""
        task_map = {
            "motion_transfer": OmniTransferTask.MOTION_TRANSFER,
            "pose_reenactment": OmniTransferTask.POSE_REENACTMENT,
            "action_customization": OmniTransferTask.ACTION_CUSTOMIZATION,
            "style_transfer": OmniTransferTask.STYLE_TRANSFER,
            "identity_preservation": OmniTransferTask.IDENTITY_PRESERVATION,
            "scene_composition": OmniTransferTask.SCENE_COMPOSITION,
        }
        return task_map.get(self.task_type, OmniTransferTask.MOTION_TRANSFER)


@dataclass
class OmniTransferModelInputs(ModelInputs):
    """Extended ModelInputs for OmniTransfer with additional metadata.

    Extends base ModelInputs to include OmniTransfer-specific information
    needed for loss computation, component coordination, and visualization.
    """
    # Task information
    task: OmniTransferTask | None = None

    # Reference/target split information
    ref_positions: Tensor | None = None  # [B, 3, ref_seq_len, 2]
    tgt_positions: Tensor | None = None  # [B, 3, tgt_seq_len, 2]

    # Target dimensions for TPB offset computation
    target_width: int | None = None
    target_frames: int | None = None

    # TMA features if computed
    tma_features: Tensor | None = None

    # Raw latents for reconstruction visualization (not patchified)
    # These are stored for W&B logging of source/target/prediction comparisons
    ref_latent_raw: Tensor | None = None      # [B, C, F, H, W] - reference video latent
    tgt_latent_raw: Tensor | None = None      # [B, C, F, H, W] - target video latent (clean)
    tgt_latent_noisy: Tensor | None = None    # [B, C, F, H, W] - noisy target for visualization
    noise: Tensor | None = None               # [B, C, F, H, W] - noise used for training
    sigmas: Tensor | None = None              # [B] - noise levels used

    # Prompts for logging
    prompts: list[str] | None = None


class OmniTransferStrategy(TrainingStrategy):
    """OmniTransfer training strategy for unified spatio-temporal video transfer.

    This strategy implements the complete OmniTransfer training pipeline:
    1. Constructs reference (noise-free) and target (noised) latents
    2. Applies Task-aware Positional Bias to reference positions
    3. Uses Reference-decoupled Causal Learning for attention
    4. Optionally integrates Task-adaptive Multimodal Alignment

    Quote: "OmniTransfer comprises three key components: 1) Task-aware Positional
    Bias that applies distinct positional biases for different task types,
    2) Reference-decoupled Causal Learning that separates reference and target
    branches for improved efficiency, 3) Task-adaptive Multimodal Alignment that
    provides task-specific semantic guidance." (Section 4)
    """

    config: OmniTransferConfig

    def __init__(self, config: OmniTransferConfig):
        """Initialize OmniTransfer strategy.

        Args:
            config: OmniTransfer configuration
        """
        super().__init__(config)

        # Initialize latent constructor
        self._latent_constructor = ReferenceLatentConstructor(
            latent_channels=128,  # LTX-2 video latent channels
            default_task=config.task,
        )

        # Initialize TPB if enabled
        if config.enable_tpb:
            self._tpb = TaskAwarePositionalBias(
                dim=4096,  # LTX-2 model dim
                num_heads=32,
                config=TaskAwarePositionalBiasConfig(
                    max_pos=config.tpb_max_pos,
                    theta=config.tpb_theta,
                ),
            )
        else:
            self._tpb = None

        logger.info(
            f"Initialized OmniTransfer strategy: task={config.task_type}, "
            f"TPB={config.enable_tpb}, RCL={config.enable_rcl}, TMA={config.enable_tma}"
        )

    def get_data_sources(self) -> dict[str, str]:
        """OmniTransfer requires latents, conditions, and reference latents.

        Returns:
            Dictionary mapping data directory names to output keys
        """
        return {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_latents",
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> OmniTransferModelInputs:
        """Prepare inputs for OmniTransfer training.

        This method:
        1. Extracts reference and target latents from batch
        2. Constructs latents with task-specific masks
        3. Applies noise to target (reference stays at t=0)
        4. Computes task-aware positions for TPB
        5. Concatenates ref+target for model input

        Quote: "The reference branch adopts a fixed t = 0, meaning it remains
        noise-free throughout the diffusion process." (Section 4.3)

        Args:
            batch: Raw batch data containing latents, conditions, ref_latents
            timestep_sampler: Sampler for generating timesteps

        Returns:
            OmniTransferModelInputs ready for model forward pass
        """
        # Extract latents
        latents_info = batch["latents"]
        target_latents = latents_info["latents"]  # [B, C, F, H, W]
        ref_latents_info = batch["ref_latents"]
        ref_latents = ref_latents_info["latents"]  # [B, C, F, H, W]

        # Get dimensions
        num_frames = latents_info["num_frames"][0].item()
        height = latents_info["height"][0].item()
        width = latents_info["width"][0].item()

        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()

        # Handle FPS
        fps = latents_info.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values in batch. Found: {fps.tolist()}, using first: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        device = target_latents.device
        dtype = target_latents.dtype

        # Sample timestep/sigma for target
        # Reference uses fixed t=0 per RCL design
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents)

        # Construct latents using ReferenceLatentConstructor
        constructed = self._latent_constructor.construct(
            ref_video_latent=ref_latents,
            tgt_video_latent=target_latents,
            task=self.config.task,
            noise=noise,
            sigma=sigmas,
            first_frame_conditioning=True,
            first_frame_conditioning_prob=self.config.first_frame_conditioning_p,
        )

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        ref_latents_patched = self._video_patchifier.patchify(constructed.ref_latent)
        tgt_latents_patched = self._video_patchifier.patchify(constructed.tgt_latent)

        ref_seq_len = ref_latents_patched.shape[1]
        tgt_seq_len = tgt_latents_patched.shape[1]

        # Create per-token timesteps
        # Reference: all zeros (t=0, noise-free)
        # Target: sampled sigma (except conditioning tokens)
        ref_timesteps = torch.full(
            (batch_size, ref_seq_len),
            self.config.rcl_ref_timestep,
            device=device, dtype=dtype
        )

        # Create target conditioning mask (first frame if applicable)
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=tgt_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )

        tgt_timesteps = self._create_per_token_timesteps(
            conditioning_mask=target_conditioning_mask,
            sampled_sigma=sigmas.squeeze(),
        )

        # Concatenate reference and target latents
        combined_latents = torch.cat([ref_latents_patched, tgt_latents_patched], dim=1)

        # Concatenate timesteps
        combined_timesteps = torch.cat([ref_timesteps, tgt_timesteps], dim=1)

        # Generate positions for reference and target
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        tgt_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Apply Task-aware Positional Bias if enabled
        # Quote: "we add an offset Δ along the spatial/temporal dimension" (Section 4.2)
        if self._tpb is not None:
            biased_ref_positions = self._tpb.apply_task_bias(
                ref_positions=ref_positions,
                task=self.config.task,
                target_width=width,
                target_frames=num_frames,
            )
        else:
            biased_ref_positions = ref_positions

        # Concatenate positions
        combined_positions = torch.cat([biased_ref_positions, tgt_positions], dim=2)

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            timesteps=combined_timesteps,
            positions=combined_positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Compute targets for loss (velocity prediction)
        # v = noise - clean
        targets = noise - target_latents
        targets_patched = self._video_patchifier.patchify(targets)

        # Loss mask: only compute loss on non-conditioning target tokens
        ref_loss_mask = torch.zeros(batch_size, ref_seq_len, dtype=torch.bool, device=device)
        tgt_loss_mask = ~target_conditioning_mask
        video_loss_mask = torch.cat([ref_loss_mask, tgt_loss_mask], dim=1)

        # Extract prompts from conditions for visualization
        prompts = batch.get("prompts", None)
        if prompts is None:
            # Try to get from conditions metadata
            prompts = conditions.get("prompts", [""] * batch_size)

        return OmniTransferModelInputs(
            video=video_modality,
            audio=None,  # OmniTransfer currently video-only
            video_targets=targets_patched,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
            task=self.config.task,
            ref_positions=ref_positions,
            tgt_positions=tgt_positions,
            target_width=width,
            target_frames=num_frames,
            tma_features=None,  # TMA features computed in model forward
            # Store raw latents for W&B reconstruction visualization
            ref_latent_raw=constructed.ref_latent.detach(),
            tgt_latent_raw=constructed.tgt_clean.detach(),
            tgt_latent_noisy=constructed.tgt_latent.detach(),
            noise=noise.detach(),
            sigmas=sigmas.detach(),
            prompts=prompts if isinstance(prompts, list) else [prompts] * batch_size,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: OmniTransferModelInputs,
    ) -> Tensor:
        """Compute OmniTransfer training loss.

        The loss is computed only on the target portion (not reference)
        following the RCL design where reference is noise-free and loss-free.

        Quote: "The reference branch... remains noise-free throughout the diffusion
        process... loss is computed only on the target tokens." (Section 4.3)

        Args:
            video_pred: Model prediction [B, ref_seq_len + tgt_seq_len, C]
            _audio_pred: Audio prediction (unused for video-only)
            inputs: OmniTransferModelInputs with targets and masks

        Returns:
            Scalar loss tensor
        """
        # Extract target portion of prediction
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]

        # Get target portion of loss mask
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

        # Compute MSE loss (velocity prediction)
        loss = (target_pred - inputs.video_targets).pow(2)

        # Apply loss mask
        loss_mask = target_loss_mask.unsqueeze(-1).float()

        # Avoid division by zero
        mask_sum = loss_mask.sum()
        if mask_sum > 0:
            loss = loss.mul(loss_mask).sum() / mask_sum
        else:
            loss = loss.mean()

        # Apply target loss weight
        loss = loss * self.config.target_loss_weight

        # Optional reference preservation loss (regularization)
        if self.config.ref_preservation_loss_weight > 0:
            ref_pred = video_pred[:, :ref_seq_len, :]
            # Reference should predict zero velocity (no change)
            ref_loss = ref_pred.pow(2).mean()
            loss = loss + self.config.ref_preservation_loss_weight * ref_loss

        return loss

    def compute_predicted_clean_latent(
        self,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
    ) -> Tensor:
        """Compute predicted clean latent from model velocity prediction.

        For flow matching, the model predicts velocity v = noise - clean.
        We can recover the predicted clean latent as: clean_pred = noise - v_pred

        Args:
            video_pred: Model prediction [B, ref_seq_len + tgt_seq_len, C]
            inputs: OmniTransferModelInputs with noise and sigmas

        Returns:
            Predicted clean target latent [B, C, F, H, W]
        """
        ref_seq_len = inputs.ref_seq_len

        # Extract target portion of prediction
        target_pred_patched = video_pred[:, ref_seq_len:, :]  # [B, tgt_seq_len, C]

        # Unpatchify to get back to [B, C, F, H, W]
        # Need to infer shape from stored raw latents
        batch_size = inputs.tgt_latent_raw.shape[0]
        channels = inputs.tgt_latent_raw.shape[1]
        frames = inputs.tgt_latent_raw.shape[2]
        height = inputs.tgt_latent_raw.shape[3]
        width = inputs.tgt_latent_raw.shape[4]

        # Reshape from [B, F*H*W, C] to [B, C, F, H, W]
        target_pred = target_pred_patched.view(batch_size, frames, height, width, channels)
        target_pred = target_pred.permute(0, 4, 1, 2, 3)  # [B, C, F, H, W]

        # For velocity prediction: v = noise - clean
        # So: clean_pred = noise - v_pred
        predicted_clean = inputs.noise - target_pred

        return predicted_clean

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction visualizations to W&B.

        Creates and logs comparison grids showing:
        - Reference (source) video frames
        - Target (ground truth) video frames
        - Prediction (model output) video frames

        Args:
            video_pred: Model prediction tensor
            inputs: OmniTransferModelInputs with raw latents
            step: Current training step
            vae_decoder: Optional VAE decoder for pixel-space visualization
            prefix: W&B metric prefix

        Returns:
            Dictionary of logged metrics
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return {}

        if not self.config.log_reconstructions:
            return {}

        # Import visualization module
        from ltx_trainer.omnitransfer.visualization import (
            OmniTransferVisualizer,
            ReconstructionSample,
            decode_latents_for_visualization,
        )

        # Initialize visualizer
        visualizer = OmniTransferVisualizer(
            log_to_wandb=True,
            log_interval=self.config.reconstruction_log_interval,
            num_frames_to_log=self.config.num_frames_to_visualize,
            save_local=self.config.save_reconstructions_locally,
            local_save_dir=Path(self.config.local_reconstruction_dir)
            if self.config.local_reconstruction_dir else None,
        )

        # Compute predicted clean latents
        predicted_clean = self.compute_predicted_clean_latent(video_pred, inputs)

        # Get latents for visualization
        ref_latents = inputs.ref_latent_raw
        tgt_latents = inputs.tgt_latent_raw
        pred_latents = predicted_clean

        # Decode to pixel space if VAE decoder provided
        if vae_decoder is not None:
            with torch.inference_mode():
                ref_decoded = decode_latents_for_visualization(ref_latents, vae_decoder)
                tgt_decoded = decode_latents_for_visualization(tgt_latents, vae_decoder)
                pred_decoded = decode_latents_for_visualization(pred_latents, vae_decoder)
        else:
            # Use latents directly (less interpretable but still useful)
            # Normalize for visualization
            def normalize_latent(x):
                x = x - x.min()
                x = x / (x.max() + 1e-8)
                return x

            ref_decoded = normalize_latent(ref_latents)
            tgt_decoded = normalize_latent(tgt_latents)
            pred_decoded = normalize_latent(pred_latents)

        # Log batch reconstructions
        num_samples = min(self.config.max_samples_per_log, ref_decoded.shape[0])
        tasks = [self.config.task] * num_samples
        prompts = inputs.prompts[:num_samples] if inputs.prompts else [""] * num_samples

        log_dict = visualizer.log_batch_reconstructions(
            references=ref_decoded[:num_samples],
            targets=tgt_decoded[:num_samples],
            predictions=pred_decoded[:num_samples],
            tasks=tasks,
            prompts=prompts,
            step=step,
            losses=None,  # Could compute per-sample losses
            max_samples=num_samples,
            prefix=prefix,
        )

        # Log video comparisons less frequently
        if (self.config.log_video_comparisons and
                step % self.config.video_log_interval == 0):
            sample = ReconstructionSample(
                reference=ref_decoded[0],
                target=tgt_decoded[0],
                prediction=pred_decoded[0],
                task=self.config.task,
                prompt=prompts[0] if prompts else "",
                step=step,
            )
            video_logs = visualizer.log_video_comparison(sample, prefix=prefix)
            log_dict.update(video_logs)

        return log_dict

    def get_wandb_metrics(
        self,
        loss: Tensor,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
        step: int,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """Get metrics dictionary for W&B logging.

        Args:
            loss: Training loss tensor
            video_pred: Model prediction
            inputs: Model inputs
            step: Training step
            learning_rate: Current learning rate

        Returns:
            Dictionary of metrics for W&B
        """
        metrics = {
            "train/loss": loss.item(),
            "train/task": self.config.task.value,
            "train/step": step,
        }

        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate

        # Add per-token loss statistics
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]
        per_token_loss = (target_pred - inputs.video_targets).pow(2).mean(dim=-1)

        metrics["train/target_loss_mean"] = per_token_loss.mean().item()
        metrics["train/target_loss_std"] = per_token_loss.std().item()

        # Add sigma statistics
        if inputs.sigmas is not None:
            metrics["train/sigma_mean"] = inputs.sigmas.mean().item()
            metrics["train/sigma_std"] = inputs.sigmas.std().item()

        return metrics


def get_omnitransfer_training_schedule(
    stage: OmniTransferStage,
    base_lr: float = 1e-5,
) -> dict[str, Any]:
    """Get training configuration for each OmniTransfer stage.

    Quote: "In the first stage, we train the DiT blocks with TPB and RCL for
    10,000 steps. In the second stage, we freeze the DiT blocks and train only
    the TMA connector for 2,000 steps. In the third stage, we jointly fine-tune
    all components for 5,000 steps." (Section 5.1)

    Args:
        stage: Training stage
        base_lr: Base learning rate

    Returns:
        Dictionary with stage-specific training configuration
    """
    schedules = {
        OmniTransferStage.IN_CONTEXT: {
            "steps": 10000,
            "lr": base_lr,
            "freeze_modules": ["tma"],
            "train_modules": ["dit", "tpb", "rcl"],
            "description": "Stage 1: Train DiT blocks with TPB and RCL",
        },
        OmniTransferStage.CONNECTOR: {
            "steps": 2000,
            "lr": base_lr,
            "freeze_modules": ["dit", "tpb", "rcl"],
            "train_modules": ["tma"],
            "description": "Stage 2: Train TMA connector only",
        },
        OmniTransferStage.JOINT: {
            "steps": 5000,
            "lr": base_lr * 0.5,  # Lower LR for joint fine-tuning
            "freeze_modules": [],
            "train_modules": ["dit", "tpb", "rcl", "tma"],
            "description": "Stage 3: Joint fine-tuning of all components",
        },
    }
    return schedules.get(stage, schedules[OmniTransferStage.IN_CONTEXT])
