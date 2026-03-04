#!/usr/bin/env python3
"""Sanity check VAE decoding for all OmniTransfer task modes.

This script verifies that VAE decoding works correctly for each task type
(effect, motion, camera, id, style) by:
1. Loading samples from each task type
2. Decoding latents to pixel space
3. Creating visualization grids
4. Saving debug images

Usage:
    python scripts/sanity_check_vae_modes.py \
        --data-root /media/2TB/omnitransfer_unified_5task \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --output-dir ./outputs/vae_sanity_check
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from einops import rearrange

from ltx_trainer import logger


def load_latent(path: Path) -> torch.Tensor:
    """Load a latent tensor from .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        # Handle different formats
        if "latent" in data:
            return data["latent"]
        elif "video_latent" in data:
            return data["video_latent"]
        else:
            # Return first tensor found
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    return v
    return data


def decode_latent_to_pixels(
    latent: torch.Tensor,
    vae_decoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Decode latent to pixel space.

    Args:
        latent: Latent tensor [C, F, H, W] or [B, C, F, H, W]
        vae_decoder: VAE decoder module
        device: Device to use

    Returns:
        Decoded pixels [B, C, F, H, W] in range [-1, 1]
    """
    # Ensure batch dimension
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)

    latent = latent.to(device, dtype=torch.bfloat16)

    with torch.inference_mode():
        decoded = vae_decoder(latent)

    return decoded


def latent_to_visualization(latent: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """Convert latent to visualization-ready numpy array.

    Args:
        latent: Latent tensor [C, F, H, W]
        normalize: Whether to normalize to [0, 255]

    Returns:
        Numpy array [F, H, W, 3] in uint8
    """
    # Take first 3 channels or average if more
    if latent.shape[0] > 3:
        # Average across channels to get grayscale, then repeat for RGB
        vis = latent.mean(dim=0, keepdim=True)  # [1, F, H, W]
        vis = vis.repeat(3, 1, 1, 1)  # [3, F, H, W]
    else:
        vis = latent[:3]  # [3, F, H, W]

    # Rearrange to [F, H, W, C]
    vis = rearrange(vis, 'c f h w -> f h w c')

    # Convert to float32 for numpy compatibility
    vis = vis.float()

    if normalize:
        # Normalize to [0, 1]
        vis = vis - vis.min()
        vis = vis / (vis.max() + 1e-8)

    # Convert to uint8
    vis = (vis.cpu().numpy() * 255).astype(np.uint8)

    return vis


def pixels_to_visualization(pixels: torch.Tensor) -> np.ndarray:
    """Convert decoded pixels to visualization-ready numpy array.

    Args:
        pixels: Decoded pixels [B, C, F, H, W] or [C, F, H, W] in range [-1, 1]

    Returns:
        Numpy array [F, H, W, 3] in uint8
    """
    if pixels.dim() == 5:
        pixels = pixels[0]  # Remove batch dim

    # pixels is [C, F, H, W], rearrange to [F, H, W, C]
    vis = rearrange(pixels, 'c f h w -> f h w c')

    # Convert to float32 for numpy compatibility
    vis = vis.float()

    # Convert from [-1, 1] to [0, 1]
    vis = (vis + 1) / 2
    vis = vis.clamp(0, 1)

    # Convert to uint8
    vis = (vis.cpu().numpy() * 255).astype(np.uint8)

    return vis


def create_comparison_grid(
    ref_vis: np.ndarray,
    tgt_vis: np.ndarray,
    task_name: str,
    frame_indices: list[int] | None = None,
) -> np.ndarray:
    """Create a comparison grid showing reference and target.

    Args:
        ref_vis: Reference visualization [F, H, W, C]
        tgt_vis: Target visualization [F, H, W, C]
        task_name: Name of the task
        frame_indices: Which frames to show

    Returns:
        Grid image [H_grid, W_grid, C]
    """
    num_frames = min(ref_vis.shape[0], tgt_vis.shape[0])

    if frame_indices is None:
        # Select 4 evenly spaced frames
        n = min(4, num_frames)
        frame_indices = np.linspace(0, num_frames - 1, n, dtype=int).tolist()

    rows = []

    # Reference row
    ref_frames = [ref_vis[i] for i in frame_indices]
    ref_row = np.concatenate(ref_frames, axis=1)

    # Target row
    tgt_frames = [tgt_vis[i] for i in frame_indices]
    tgt_row = np.concatenate(tgt_frames, axis=1)

    # Add labels
    try:
        import cv2

        # Add label bar
        bar_height = 30
        h, w = ref_row.shape[:2]

        ref_labeled = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
        ref_labeled[bar_height:] = ref_row
        ref_labeled[:bar_height] = 40
        cv2.putText(ref_labeled, f"{task_name.upper()} Reference", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        tgt_labeled = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
        tgt_labeled[bar_height:] = tgt_row
        tgt_labeled[:bar_height] = 40
        cv2.putText(tgt_labeled, f"{task_name.upper()} Target", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        grid = np.concatenate([ref_labeled, tgt_labeled], axis=0)
    except ImportError:
        grid = np.concatenate([ref_row, tgt_row], axis=0)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Sanity check VAE decoding for all task modes")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to preprocessed training data")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to LTX-2 model (for VAE decoding)")
    parser.add_argument("--output-dir", type=str, default="./outputs/vae_sanity_check",
                        help="Output directory for visualizations")
    parser.add_argument("--skip-vae", action="store_true",
                        help="Skip VAE decoding, only visualize latents")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {device}")

    # Find task types from metadata or directory structure
    task_types = ["effect", "motion", "camera", "id", "style"]

    # Check for metadata.json
    metadata_path = data_root / "metadata.json"
    sample_tasks = {}

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Map samples to tasks from "pairs" array
        pairs = metadata.get("pairs", [])
        for pair in pairs:
            sample_id = str(pair.get("id", ""))
            task = pair.get("task_type", "effect")
            if task not in sample_tasks:
                sample_tasks[task] = []
            sample_tasks[task].append(sample_id)

        logger.info(f"Found metadata with {len(pairs)} pairs, tasks: {list(sample_tasks.keys())}")
    else:
        # Check for task-specific directories or use numeric indices
        latents_dir = data_root / "latents"
        if latents_dir.exists():
            samples = sorted([p.stem for p in latents_dir.glob("*.pt")])
            # Assign samples to tasks cyclically for testing
            for i, sample_id in enumerate(samples[:30]):  # First 30 samples
                task = task_types[i % len(task_types)]
                if task not in sample_tasks:
                    sample_tasks[task] = []
                sample_tasks[task].append(sample_id)

            logger.info(f"Assigned {len(samples)} samples to tasks cyclically")

    # Load VAE decoder if needed
    vae_decoder = None
    if not args.skip_vae and args.model_path:
        logger.info("Loading VAE decoder...")
        try:
            from ltx_trainer.model_loader import load_video_vae_decoder
            vae_decoder = load_video_vae_decoder(args.model_path)
            vae_decoder = vae_decoder.to(device)
            vae_decoder.eval()
            logger.info("VAE decoder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load VAE decoder: {e}")
            logger.warning("Falling back to latent-space visualization")

    # Process each task type
    all_grids = []

    for task_name in task_types:
        if task_name not in sample_tasks or not sample_tasks[task_name]:
            logger.warning(f"No samples found for task: {task_name}")
            continue

        sample_id = sample_tasks[task_name][0]  # Use first sample
        logger.info(f"\nProcessing {task_name} task (sample: {sample_id})")

        # Load latents - try both naming conventions (0.pt and 000.pt)
        ref_path = data_root / "reference_latents" / f"{sample_id}.pt"
        if not ref_path.exists():
            ref_path = data_root / "reference_latents" / f"{int(sample_id):03d}.pt"

        tgt_path = data_root / "latents" / f"{sample_id}.pt"
        if not tgt_path.exists():
            tgt_path = data_root / "latents" / f"{int(sample_id):03d}.pt"

        if not ref_path.exists():
            logger.warning(f"Reference latent not found: {ref_path}")
            continue
        if not tgt_path.exists():
            logger.warning(f"Target latent not found: {tgt_path}")
            continue

        ref_latent = load_latent(ref_path)
        tgt_latent = load_latent(tgt_path)

        logger.info(f"  Reference latent shape: {ref_latent.shape}")
        logger.info(f"  Target latent shape: {tgt_latent.shape}")

        # Decode or visualize latents
        if vae_decoder is not None:
            try:
                logger.info("  Decoding reference...")
                ref_decoded = decode_latent_to_pixels(ref_latent, vae_decoder, device)
                ref_vis = pixels_to_visualization(ref_decoded)

                logger.info("  Decoding target...")
                tgt_decoded = decode_latent_to_pixels(tgt_latent, vae_decoder, device)
                tgt_vis = pixels_to_visualization(tgt_decoded)

                vis_type = "decoded"
            except Exception as e:
                logger.warning(f"  VAE decoding failed: {e}")
                ref_vis = latent_to_visualization(ref_latent)
                tgt_vis = latent_to_visualization(tgt_latent)
                vis_type = "latent"
        else:
            ref_vis = latent_to_visualization(ref_latent)
            tgt_vis = latent_to_visualization(tgt_latent)
            vis_type = "latent"

        logger.info(f"  Visualization type: {vis_type}")
        logger.info(f"  Reference vis shape: {ref_vis.shape}")
        logger.info(f"  Target vis shape: {tgt_vis.shape}")

        # Create comparison grid
        grid = create_comparison_grid(ref_vis, tgt_vis, task_name)
        all_grids.append((task_name, grid))

        # Save individual task grid
        task_output = output_dir / f"{task_name}_{vis_type}.png"
        Image.fromarray(grid).save(task_output)
        logger.info(f"  Saved: {task_output}")

    # Create combined multi-task grid
    if all_grids:
        logger.info("\nCreating combined multi-task grid...")

        # Stack all task grids vertically
        combined = np.concatenate([g for _, g in all_grids], axis=0)

        # Resize if too large
        max_height = 2000
        if combined.shape[0] > max_height:
            scale = max_height / combined.shape[0]
            new_h = max_height
            new_w = int(combined.shape[1] * scale)
            pil_img = Image.fromarray(combined)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            combined = np.array(pil_img)

        combined_path = output_dir / "all_tasks_combined.png"
        Image.fromarray(combined).save(combined_path)
        logger.info(f"Saved combined grid: {combined_path}")

        # Check file size
        file_size_kb = combined_path.stat().st_size / 1024
        logger.info(f"Combined grid size: {file_size_kb:.1f} KB")

    logger.info("\n=== Sanity Check Complete ===")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
