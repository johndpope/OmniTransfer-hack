"""OmniTransfer Visualization and W&B Logging.

This module provides visualization utilities for OmniTransfer training,
including side-by-side comparisons of:
- Reference (source) video/frames
- Target (ground truth) video/frames
- Prediction (model output) video/frames

These visualizations are logged to Weights & Biases for training monitoring.

Quote: "To evaluate the effectiveness of our method, we visualize the
transfer results comparing reference inputs, ground truth targets, and
model predictions." (Section 5, OmniTransfer paper)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ltx_trainer import logger
from ltx_trainer.omnitransfer.components import OmniTransferTask


@dataclass
class ReconstructionSample:
    """Container for a reconstruction sample with all components.

    For I2V mode, we have 4 components:
    - target_image: Static image to animate [C, 1, H, W] - the INPUT image
    - reference: Reference video for motion/effect source [C, F, H, W]
    - target: Ground truth output video [C, F, H, W] - what we want to produce
    - prediction: Model prediction [C, F, H, W]

    Attributes:
        reference: Reference video tensor [C, F, H, W] or [F, H, W, C]
        target: Ground truth target tensor [C, F, H, W] or [F, H, W, C]
        prediction: Model prediction tensor [C, F, H, W] or [F, H, W, C]
        target_image: Optional target image for I2V mode [C, 1, H, W]
        task: The transfer task type
        prompt: Text prompt used
        step: Training step number
        loss: Loss value for this sample (optional)
    """
    reference: torch.Tensor
    target: torch.Tensor
    prediction: torch.Tensor
    task: OmniTransferTask
    prompt: str
    step: int
    target_image: torch.Tensor | None = None  # For I2V mode
    loss: float | None = None

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Convert tensors to numpy arrays in [F, H, W, C] format, range [0, 255].

        Returns:
            Tuple of (reference, target, prediction, target_image) as numpy arrays.
            target_image is None if not in I2V mode.
        """
        def process(t: torch.Tensor) -> np.ndarray:
            if t.dim() == 4 and t.shape[0] in [1, 3, 4, 128]:
                # [C, F, H, W] -> [F, H, W, C]
                t = rearrange(t, 'c f h w -> f h w c')
            elif t.dim() == 4 and t.shape[-1] in [1, 3, 4]:
                # Already [F, H, W, C]
                pass
            else:
                raise ValueError(f"Unexpected tensor shape: {t.shape}")

            # Normalize to [0, 1] if in [-1, 1]
            if t.min() < 0:
                t = (t + 1) / 2

            # Clamp and convert
            t = t.clamp(0, 1)
            return (t.cpu().numpy() * 255).astype(np.uint8)

        target_image_np = None
        if self.target_image is not None:
            target_image_np = process(self.target_image)

        return process(self.reference), process(self.target), process(self.prediction), target_image_np


class OmniTransferVisualizer:
    """Visualizer for OmniTransfer training with W&B integration.

    Creates comparison visualizations showing reference, target, and prediction
    side-by-side for training monitoring and debugging.
    """

    def __init__(
        self,
        log_to_wandb: bool = True,
        log_interval: int = 100,
        num_frames_to_log: int = 8,
        save_local: bool = False,
        local_save_dir: Path | None = None,
    ):
        """Initialize the visualizer.

        Args:
            log_to_wandb: Whether to log to W&B
            log_interval: Steps between logging visualizations
            num_frames_to_log: Number of frames to include in visualizations
            save_local: Whether to save visualizations locally
            local_save_dir: Directory for local saves
        """
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.log_interval = log_interval
        self.num_frames_to_log = num_frames_to_log
        self.save_local = save_local
        self.local_save_dir = local_save_dir

        if self.save_local and self.local_save_dir:
            self.local_save_dir.mkdir(parents=True, exist_ok=True)

        if self.log_to_wandb and not WANDB_AVAILABLE:
            logger.warning("W&B requested but not installed. Install with: pip install wandb")

    def create_frame_grid(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        frame_indices: list[int] | None = None,
    ) -> np.ndarray:
        """Create a grid of frames showing ref/target/pred comparison.

        Layout:
        ```
        Frame 0    Frame N/4   Frame N/2   Frame 3N/4  Frame N-1
        [ref]      [ref]       [ref]       [ref]       [ref]
        [tgt]      [tgt]       [tgt]       [tgt]       [tgt]
        [pred]     [pred]      [pred]      [pred]      [pred]
        ```

        Args:
            reference: Reference frames [F, H, W, C]
            target: Target frames [F, H, W, C]
            prediction: Prediction frames [F, H, W, C]
            frame_indices: Specific frame indices to use

        Returns:
            Grid image as numpy array [H_grid, W_grid, C]
        """
        num_frames = reference.shape[0]

        if frame_indices is None:
            # Select evenly spaced frames
            n = min(self.num_frames_to_log, num_frames)
            frame_indices = np.linspace(0, num_frames - 1, n, dtype=int).tolist()

        # Extract selected frames
        ref_frames = [reference[i] for i in frame_indices]
        tgt_frames = [target[i] for i in frame_indices]
        pred_frames = [prediction[i] for i in frame_indices]

        # Add labels to frames
        ref_labeled = [self._add_label(f, f"Ref t={i}") for i, f in zip(frame_indices, ref_frames)]
        tgt_labeled = [self._add_label(f, f"Target t={i}") for i, f in zip(frame_indices, tgt_frames)]
        pred_labeled = [self._add_label(f, f"Pred t={i}") for i, f in zip(frame_indices, pred_frames)]

        # Stack horizontally for each row
        ref_row = np.concatenate(ref_labeled, axis=1)
        tgt_row = np.concatenate(tgt_labeled, axis=1)
        pred_row = np.concatenate(pred_labeled, axis=1)

        # Stack vertically
        grid = np.concatenate([ref_row, tgt_row, pred_row], axis=0)

        return grid

    def create_single_frame_comparison(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        frame_idx: int = 0,
        target_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create a single frame comparison.

        For I2V mode (4 panels): Target Image | Reference Video | Ground Truth | Prediction
        For standard mode (3 panels): Reference | Target (GT) | Prediction

        Args:
            reference: Reference frames [F, H, W, C]
            target: Target frames [F, H, W, C]
            prediction: Prediction frames [F, H, W, C]
            frame_idx: Which frame to use
            target_image: Optional target image for I2V mode [F=1, H, W, C]

        Returns:
            Comparison image [H, W*N, C] where N=4 for I2V, N=3 for standard
        """
        if target_image is not None:
            # I2V mode: 4 panels
            # Target image is single frame, use frame 0
            img_frame = self._add_label(target_image[0], "Target Image")
            ref_frame = self._add_label(reference[frame_idx], "Ref Video")
            tgt_frame = self._add_label(target[frame_idx], "Ground Truth")
            pred_frame = self._add_label(prediction[frame_idx], "Prediction")
            return np.concatenate([img_frame, ref_frame, tgt_frame, pred_frame], axis=1)
        else:
            # Standard mode: 3 panels
            ref_frame = self._add_label(reference[frame_idx], "Reference")
            tgt_frame = self._add_label(target[frame_idx], "Target (GT)")
            pred_frame = self._add_label(prediction[frame_idx], "Prediction")
            return np.concatenate([ref_frame, tgt_frame, pred_frame], axis=1)

    def create_difference_map(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        frame_idx: int = 0,
        amplify: float = 3.0,
    ) -> np.ndarray:
        """Create a difference/error map between target and prediction.

        Args:
            target: Target frames [F, H, W, C]
            prediction: Prediction frames [F, H, W, C]
            frame_idx: Which frame to use
            amplify: Amplification factor for visibility

        Returns:
            Difference map image [H, W, C]
        """
        tgt = target[frame_idx].astype(np.float32)
        pred = prediction[frame_idx].astype(np.float32)

        # Compute absolute difference
        diff = np.abs(tgt - pred)

        # Amplify for visibility
        diff = np.clip(diff * amplify, 0, 255).astype(np.uint8)

        return self._add_label(diff, f"Error Map (x{amplify})")

    def _add_label(
        self,
        image: np.ndarray,
        label: str,
        font_scale: float = 0.5,
        thickness: int = 1,
    ) -> np.ndarray:
        """Add a text label to an image.

        Args:
            image: Input image [H, W, C]
            label: Text label to add
            font_scale: Font size scale
            thickness: Text thickness

        Returns:
            Labeled image
        """
        try:
            import cv2

            # Create copy to avoid modifying original
            img = image.copy()

            # Add background bar for label
            bar_height = 25
            img_with_bar = np.zeros((img.shape[0] + bar_height, img.shape[1], img.shape[2]), dtype=np.uint8)
            img_with_bar[bar_height:] = img
            img_with_bar[:bar_height] = 40  # Dark gray background

            # Add text
            cv2.putText(
                img_with_bar,
                label,
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            return img_with_bar

        except ImportError:
            # If cv2 not available, return image as-is
            return image

    def log_reconstruction(
        self,
        sample: ReconstructionSample,
        prefix: str = "train",
    ) -> dict[str, any]:
        """Log a reconstruction sample to W&B.

        Args:
            sample: ReconstructionSample containing ref/target/pred (and optional target_image for I2V)
            prefix: Prefix for W&B keys (e.g., "train", "val")

        Returns:
            Dictionary of logged metrics/images
        """
        ref_np, tgt_np, pred_np, target_image_np = sample.to_numpy()

        # Create visualizations (4-panel for I2V, 3-panel for standard)
        frame_grid = self.create_frame_grid(ref_np, tgt_np, pred_np)
        single_comparison = self.create_single_frame_comparison(
            ref_np, tgt_np, pred_np, frame_idx=0, target_image=target_image_np
        )
        mid_comparison = self.create_single_frame_comparison(
            ref_np, tgt_np, pred_np,
            frame_idx=min(ref_np.shape[0] // 2, tgt_np.shape[0] // 2),
            target_image=target_image_np
        )
        diff_map = self.create_difference_map(tgt_np, pred_np, frame_idx=0)

        # Compute metrics
        mse = np.mean((tgt_np.astype(np.float32) - pred_np.astype(np.float32)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / (mse + 1e-8))

        log_dict = {
            f"{prefix}/frame_grid": wandb.Image(frame_grid, caption=f"Step {sample.step}: {sample.prompt}"),
            f"{prefix}/comparison_first": wandb.Image(single_comparison, caption="First frame comparison"),
            f"{prefix}/comparison_mid": wandb.Image(mid_comparison, caption="Mid frame comparison"),
            f"{prefix}/error_map": wandb.Image(diff_map, caption="Prediction error"),
            f"{prefix}/mse": mse,
            f"{prefix}/psnr": psnr,
            f"{prefix}/task": sample.task.value,
        }

        if sample.loss is not None:
            log_dict[f"{prefix}/sample_loss"] = sample.loss

        # Log to W&B
        if self.log_to_wandb and wandb.run is not None:
            wandb.log(log_dict, step=sample.step)

        # Save locally if requested
        if self.save_local and self.local_save_dir:
            save_path = self.local_save_dir / f"step_{sample.step:06d}"
            save_path.mkdir(exist_ok=True)

            Image.fromarray(frame_grid).save(save_path / "frame_grid.png")
            Image.fromarray(single_comparison).save(save_path / "comparison_first.png")
            Image.fromarray(diff_map).save(save_path / "error_map.png")

        return log_dict

    def log_batch_reconstructions(
        self,
        references: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        tasks: list[OmniTransferTask],
        prompts: list[str],
        step: int,
        losses: list[float] | None = None,
        max_samples: int = 4,
        prefix: str = "train",
    ) -> dict[str, any]:
        """Log multiple reconstruction samples from a batch.

        Args:
            references: Batch of reference videos [B, C, F, H, W]
            targets: Batch of target videos [B, C, F, H, W]
            predictions: Batch of predictions [B, C, F, H, W]
            tasks: List of task types for each sample
            prompts: List of prompts for each sample
            step: Training step
            losses: Optional per-sample losses
            max_samples: Maximum samples to log
            prefix: W&B key prefix

        Returns:
            Dictionary of all logged items
        """
        batch_size = min(references.shape[0], max_samples)
        all_logs = {}

        for i in range(batch_size):
            sample = ReconstructionSample(
                reference=references[i],
                target=targets[i],
                prediction=predictions[i],
                task=tasks[i] if i < len(tasks) else tasks[0],
                prompt=prompts[i] if i < len(prompts) else prompts[0],
                step=step,
                loss=losses[i] if losses and i < len(losses) else None,
            )

            sample_logs = self.log_reconstruction(
                sample,
                prefix=f"{prefix}/sample_{i}",
            )
            all_logs.update(sample_logs)

        return all_logs

    def create_video_comparison(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
        fps: float = 25.0,
    ) -> np.ndarray:
        """Create a side-by-side video comparison.

        Args:
            reference: Reference frames [F, H, W, C]
            target: Target frames [F, H, W, C]
            prediction: Prediction frames [F, H, W, C]
            fps: Frame rate

        Returns:
            Combined video frames [F, H, W*3, C]
        """
        # Add labels to first frame only (or could do all)
        num_frames = min(reference.shape[0], target.shape[0], prediction.shape[0])

        combined_frames = []
        for i in range(num_frames):
            ref_frame = reference[i]
            tgt_frame = target[i]
            pred_frame = prediction[i]

            # Only add labels to first frame
            if i == 0:
                ref_frame = self._add_label(ref_frame, "Reference")
                tgt_frame = self._add_label(tgt_frame, "Target (GT)")
                pred_frame = self._add_label(pred_frame, "Prediction")

            combined = np.concatenate([ref_frame, tgt_frame, pred_frame], axis=1)
            combined_frames.append(combined)

        return np.stack(combined_frames, axis=0)

    def log_video_comparison(
        self,
        sample: ReconstructionSample,
        fps: float = 25.0,
        prefix: str = "train",
    ) -> dict[str, any]:
        """Log a video comparison to W&B.

        Args:
            sample: ReconstructionSample
            fps: Frame rate for video
            prefix: W&B key prefix

        Returns:
            Dictionary with logged video
        """
        ref_np, tgt_np, pred_np = sample.to_numpy()

        combined_video = self.create_video_comparison(ref_np, tgt_np, pred_np, fps)

        # Convert to format W&B expects [T, C, H, W]
        video_wandb = rearrange(combined_video, 't h w c -> t c h w')

        log_dict = {
            f"{prefix}/video_comparison": wandb.Video(
                video_wandb,
                fps=int(fps),
                caption=f"Step {sample.step}: {sample.prompt} ({sample.task.value})"
            ),
        }

        if self.log_to_wandb and wandb.run is not None:
            wandb.log(log_dict, step=sample.step)

        return log_dict


class OmniTransferWandBCallback:
    """Callback for W&B logging during OmniTransfer training.

    Integrates with the trainer to log:
    - Training metrics (loss, learning rate, etc.)
    - Reconstruction visualizations at specified intervals
    - Validation samples
    - Model checkpoints
    """

    def __init__(
        self,
        project: str = "ltx2-omnitransfer",
        entity: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
        log_interval: int = 100,
        reconstruction_interval: int = 500,
        num_frames_to_log: int = 8,
        max_samples_per_log: int = 4,
        save_local: bool = False,
        local_save_dir: Path | None = None,
    ):
        """Initialize W&B callback.

        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            config: Config dict to log
            tags: Tags for the run
            log_interval: Steps between metric logging
            reconstruction_interval: Steps between reconstruction logging
            num_frames_to_log: Frames per reconstruction
            max_samples_per_log: Max batch samples to log
            save_local: Whether to save locally too
            local_save_dir: Local save directory
        """
        self.project = project
        self.entity = entity
        self.config = config
        self.tags = tags or []
        self.log_interval = log_interval
        self.reconstruction_interval = reconstruction_interval
        self.max_samples_per_log = max_samples_per_log

        self.visualizer = OmniTransferVisualizer(
            log_to_wandb=True,
            log_interval=reconstruction_interval,
            num_frames_to_log=num_frames_to_log,
            save_local=save_local,
            local_save_dir=local_save_dir,
        )

        self._initialized = False

    def init_wandb(self, run_name: str | None = None):
        """Initialize W&B run.

        Args:
            run_name: Optional run name
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, skipping initialization")
            return

        if self._initialized:
            return

        wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
            name=run_name,
        )

        self._initialized = True
        logger.info(f"W&B initialized: {wandb.run.url}")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
    ):
        """Log training metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step
        """
        if not self._initialized or wandb.run is None:
            return

        wandb.log(metrics, step=step)

    def log_reconstructions(
        self,
        references: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        tasks: list[OmniTransferTask],
        prompts: list[str],
        step: int,
        losses: list[float] | None = None,
        prefix: str = "train",
    ):
        """Log reconstruction visualizations.

        Args:
            references: Batch of reference videos [B, C, F, H, W]
            targets: Batch of target videos [B, C, F, H, W]
            predictions: Batch of predictions [B, C, F, H, W]
            tasks: Task types for each sample
            prompts: Prompts for each sample
            step: Training step
            losses: Per-sample losses
            prefix: Key prefix
        """
        if not self._initialized or wandb.run is None:
            return

        self.visualizer.log_batch_reconstructions(
            references=references,
            targets=targets,
            predictions=predictions,
            tasks=tasks,
            prompts=prompts,
            step=step,
            losses=losses,
            max_samples=self.max_samples_per_log,
            prefix=prefix,
        )

    def should_log_metrics(self, step: int) -> bool:
        """Check if metrics should be logged at this step."""
        return step % self.log_interval == 0

    def should_log_reconstructions(self, step: int) -> bool:
        """Check if reconstructions should be logged at this step."""
        return step % self.reconstruction_interval == 0

    def finish(self):
        """Finish W&B run."""
        if self._initialized and wandb.run is not None:
            wandb.finish()
            self._initialized = False


def decode_latents_for_visualization(
    latents: torch.Tensor,
    vae_decoder: torch.nn.Module,
    chunk_size: int = 1,
) -> torch.Tensor:
    """Decode latents to pixel space for visualization.

    Args:
        latents: Latent tensors [B, C, F, H, W]
        vae_decoder: VAE decoder module
        chunk_size: Batch size for decoding (to manage memory)

    Returns:
        Decoded video tensors [B, C, F, H, W] in pixel space
    """
    batch_size = latents.shape[0]
    decoded = []

    # Get decoder device (from first parameter)
    decoder_device = next(vae_decoder.parameters()).device
    original_device = latents.device

    with torch.inference_mode():
        for i in range(0, batch_size, chunk_size):
            chunk = latents[i:i + chunk_size]
            # Move chunk to decoder's device and match dtype
            decoder_dtype = next(vae_decoder.parameters()).dtype
            chunk = chunk.to(device=decoder_device, dtype=decoder_dtype)
            # VideoDecoder uses forward(), not decode()
            # Call the module directly which invokes forward()
            decoded_chunk = vae_decoder(chunk)
            # Convert to float32 for visualization (PIL/torchvision need float32)
            decoded_chunk = decoded_chunk.to(device=original_device, dtype=torch.float32)
            decoded.append(decoded_chunk)

    return torch.cat(decoded, dim=0)
