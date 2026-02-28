"""OmniTransfer Training Callback for W&B Integration.

This module provides a callback that integrates with the LTX-2 trainer
to automatically log OmniTransfer reconstruction visualizations.

The callback:
1. Logs training metrics (loss, learning rate) at specified intervals
2. Logs reconstruction visualizations (reference, target, prediction)
3. Creates comparison grids and videos for training monitoring
4. Tracks per-task and per-stage metrics

Usage:
    from ltx_trainer.omnitransfer.training_callback import OmniTransferTrainingCallback

    callback = OmniTransferTrainingCallback(
        strategy=omnitransfer_strategy,
        vae_decoder=vae_decoder,
    )

    # In training loop:
    callback.on_train_step_end(step, loss, video_pred, inputs, lr)
"""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer import logger
from ltx_trainer.omnitransfer.strategy import OmniTransferModelInputs, OmniTransferStrategy
from ltx_trainer.omnitransfer.visualization import (
    OmniTransferVisualizer,
    OmniTransferWandBCallback,
    ReconstructionSample,
    decode_latents_for_visualization,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class OmniTransferTrainingCallback:
    """Callback for OmniTransfer training with W&B visualization.

    Integrates with the trainer to log:
    - Training metrics at each step
    - Reconstruction visualizations at specified intervals
    - Video comparisons periodically
    - Stage transition events
    """

    def __init__(
        self,
        strategy: OmniTransferStrategy,
        vae_decoder: torch.nn.Module | None = None,
        project: str = "ltx2-omnitransfer",
        entity: str | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
        config_dict: dict | None = None,
    ):
        """Initialize the callback.

        Args:
            strategy: OmniTransfer training strategy
            vae_decoder: VAE decoder for pixel-space visualization
            project: W&B project name
            entity: W&B entity
            run_name: W&B run name
            tags: W&B tags
            config_dict: Configuration to log to W&B
        """
        self.strategy = strategy
        self.vae_decoder = vae_decoder
        self.config = strategy.config

        # Initialize W&B callback
        self.wandb_callback = OmniTransferWandBCallback(
            project=project,
            entity=entity,
            config=config_dict,
            tags=tags or [self.config.task_type, "omnitransfer"],
            log_interval=100,
            reconstruction_interval=self.config.reconstruction_log_interval,
            num_frames_to_log=self.config.num_frames_to_visualize,
            max_samples_per_log=self.config.max_samples_per_log,
            save_local=self.config.save_reconstructions_locally,
            local_save_dir=Path(self.config.local_reconstruction_dir)
            if self.config.local_reconstruction_dir else None,
        )

        # Initialize visualizer
        self.visualizer = OmniTransferVisualizer(
            log_to_wandb=True,
            log_interval=self.config.reconstruction_log_interval,
            num_frames_to_log=self.config.num_frames_to_visualize,
            save_local=self.config.save_reconstructions_locally,
            local_save_dir=Path(self.config.local_reconstruction_dir)
            if self.config.local_reconstruction_dir else None,
        )

        self._initialized = False
        self._run_name = run_name

    def on_train_begin(self):
        """Called at the start of training."""
        self.wandb_callback.init_wandb(self._run_name)
        self._initialized = True
        logger.info("OmniTransfer W&B callback initialized")

    def on_train_end(self):
        """Called at the end of training."""
        self.wandb_callback.finish()
        self._initialized = False

    def on_train_step_end(
        self,
        step: int,
        loss: Tensor,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
        learning_rate: float | None = None,
        audio_pred: Tensor | None = None,
    ):
        """Called at the end of each training step.

        Args:
            step: Current training step
            loss: Training loss
            video_pred: Model video prediction
            inputs: Model inputs
            learning_rate: Current learning rate
            audio_pred: Model audio prediction (unused)
        """
        if not self._initialized:
            self.on_train_begin()

        # Log metrics
        if self.wandb_callback.should_log_metrics(step):
            metrics = self.strategy.get_wandb_metrics(
                loss=loss,
                video_pred=video_pred,
                inputs=inputs,
                step=step,
                learning_rate=learning_rate,
            )
            self.wandb_callback.log_metrics(metrics, step)

        # Log reconstructions
        if self.wandb_callback.should_log_reconstructions(step):
            self._log_reconstructions(video_pred, inputs, step)

    def _log_reconstructions(
        self,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
        step: int,
    ):
        """Log reconstruction visualizations.

        Args:
            video_pred: Model prediction
            inputs: Model inputs with raw latents
            step: Training step
        """
        if not self.config.log_reconstructions:
            return

        try:
            # Compute predicted clean latent
            predicted_clean = self.strategy.compute_predicted_clean_latent(video_pred, inputs)

            # Get reference and target latents
            ref_latents = inputs.ref_latent_raw
            tgt_latents = inputs.tgt_latent_raw

            # Decode if VAE available
            if self.vae_decoder is not None:
                with torch.inference_mode():
                    ref_decoded = decode_latents_for_visualization(
                        ref_latents, self.vae_decoder, chunk_size=1
                    )
                    tgt_decoded = decode_latents_for_visualization(
                        tgt_latents, self.vae_decoder, chunk_size=1
                    )
                    pred_decoded = decode_latents_for_visualization(
                        predicted_clean, self.vae_decoder, chunk_size=1
                    )
            else:
                # Normalize latents for visualization
                ref_decoded = self._normalize_for_vis(ref_latents)
                tgt_decoded = self._normalize_for_vis(tgt_latents)
                pred_decoded = self._normalize_for_vis(predicted_clean)

            # Get tasks and prompts
            num_samples = min(self.config.max_samples_per_log, ref_decoded.shape[0])
            tasks = [self.config.task] * num_samples
            prompts = inputs.prompts[:num_samples] if inputs.prompts else [""] * num_samples

            # Log batch reconstructions
            self.wandb_callback.log_reconstructions(
                references=ref_decoded[:num_samples],
                targets=tgt_decoded[:num_samples],
                predictions=pred_decoded[:num_samples],
                tasks=tasks,
                prompts=prompts,
                step=step,
                prefix="train",
            )

            # Log video comparisons less frequently
            if (self.config.log_video_comparisons and
                    step % self.config.video_log_interval == 0):
                sample = ReconstructionSample(
                    reference=ref_decoded[0],
                    target=tgt_decoded[0],
                    prediction=pred_decoded[0],
                    task=self.config.task,
                    prompt=prompts[0],
                    step=step,
                )
                self.visualizer.log_video_comparison(sample, prefix="train")

        except Exception as e:
            logger.warning(f"Failed to log reconstructions at step {step}: {e}")

    def _normalize_for_vis(self, latents: Tensor) -> Tensor:
        """Normalize latents for visualization when no VAE available."""
        # Take first 3 channels for RGB visualization
        if latents.shape[1] > 3:
            latents = latents[:, :3]

        # Normalize to [0, 1]
        latents = latents - latents.min()
        latents = latents / (latents.max() + 1e-8)

        return latents

    def on_validation_step(
        self,
        step: int,
        ref_video: Tensor,
        tgt_video: Tensor,
        pred_video: Tensor,
        prompt: str,
    ):
        """Log validation sample.

        Args:
            step: Training step
            ref_video: Reference video [C, F, H, W]
            tgt_video: Target video [C, F, H, W]
            pred_video: Predicted video [C, F, H, W]
            prompt: Text prompt
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        sample = ReconstructionSample(
            reference=ref_video,
            target=tgt_video,
            prediction=pred_video,
            task=self.config.task,
            prompt=prompt,
            step=step,
        )

        self.visualizer.log_reconstruction(sample, prefix="val")
        self.visualizer.log_video_comparison(sample, prefix="val")


def create_omnitransfer_callback(
    strategy: OmniTransferStrategy,
    vae_decoder: torch.nn.Module | None = None,
    wandb_config: dict | None = None,
) -> OmniTransferTrainingCallback:
    """Factory function to create OmniTransfer callback.

    Args:
        strategy: OmniTransfer training strategy
        vae_decoder: Optional VAE decoder
        wandb_config: W&B configuration dict with keys:
            - project: Project name
            - entity: Entity name
            - run_name: Run name
            - tags: List of tags

    Returns:
        Configured OmniTransferTrainingCallback
    """
    wandb_config = wandb_config or {}

    return OmniTransferTrainingCallback(
        strategy=strategy,
        vae_decoder=vae_decoder,
        project=wandb_config.get("project", "ltx2-omnitransfer"),
        entity=wandb_config.get("entity"),
        run_name=wandb_config.get("run_name"),
        tags=wandb_config.get("tags"),
        config_dict=wandb_config.get("config"),
    )
