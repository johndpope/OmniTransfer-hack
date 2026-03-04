#!/usr/bin/env python3
# ruff: noqa: T201
"""Encode Grok isometric videos into I2V training dataset.

Processes 8 Grok-generated isometric 3D videos (784x1168, 145 frames @ 24fps):
  1. Extracts overlapping 25-frame clips (stride=8) → ~16 clips/video = ~128 total
  2. Center-crops + resizes to portrait 512x768 (preserves isometric structure)
  3. Encodes each clip with VAE → video latent [128, F_lat, H_lat, W_lat]
  4. Encodes first frame → reference image latent [128, 1, H_lat, W_lat]
  5. Generates diverse motion prompts from actions.txt
  6. Writes dataset in PrecomputedDataset format

Output structure:
    /media/2TB/isometric_i2v_training/
    ├── latents/              # Video clips [128, 4, 24, 16] (ground truth)
    ├── reference_latents/    # First frame [128, 1, 24, 16] (I2V condition)
    ├── metadata.json         # Clip provenance, prompts, video IDs
    └── (conditions_final/)   # Text embeddings - computed separately (~28GB)

Usage:
    cd packages/ltx-trainer

    # Encode (VAE only, ~8GB VRAM on cuda:1)
    uv run python scripts/encode_isometric_videos.py

    # Then compute text embeddings separately (~28GB VRAM)
    uv run python scripts/compute_final_embeddings.py \
        --dataset-dir /media/2TB/isometric_i2v_training \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma \
        --from-scratch
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder
from ltx_trainer.video_utils import read_video

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
GROK_VIDEO_DIR = Path("/media/12TB/isometric_3d/r2_native_dataset/new_grok_videos")
OUTPUT_DIR = Path("/media/2TB/isometric_i2v_training")
MODEL_PATH = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
ACTIONS_FILE = Path(
    "/home/johndpope/Documents/GitHub/PresidentialDilema-FastApi"
    "/grok-video-extractor-flutter/actions.txt"
)

# Portrait orientation for isometric (taller than wide)
TARGET_WIDTH = 768
TARGET_HEIGHT = 1152

# Clip extraction params
CLIP_FRAMES = 25       # Must satisfy frames % 8 == 1
CLIP_STRIDE = 8        # Overlap stride (frames between clip starts)
MAX_CLIPS_PER_VIDEO = 16  # Cap to avoid too many similar clips

# Base prompts for isometric animation
BASE_PROMPTS = [
    "Static camera, fixed isometric viewpoint. {action}. No camera movement.",
    "Isometric 3D view, camera stays completely still. {action}.",
    "Fixed isometric angle, no camera motion. {action}. Static camera.",
    "3D isometric scene with static camera. {action}.",
]

# Isometric-specific action templates when actions.txt not available
FALLBACK_ACTIONS = [
    "Two women conversing with animated hand gestures in a modern bar",
    "A woman speaks cheerfully, nodding and shifting her weight",
    "Characters talk expressively, one gesticulating while the other listens",
    "A woman laughs and waves her hand while her friend reacts",
    "Two people have an animated discussion with natural body language",
    "A woman leans forward speaking passionately, the other nods along",
    "Characters exchange words with expressive facial movements",
    "A woman raises her eyebrows and tilts her head while speaking",
]


def load_actions(actions_file: Path) -> list[str]:
    """Load character actions from actions.txt.

    Format is 'key: description' per line (comments start with #).
    We extract the description part for use in prompts.
    """
    if not actions_file.exists():
        logger.warning(f"Actions file not found: {actions_file}, using fallbacks")
        return FALLBACK_ACTIONS

    actions = []
    with open(actions_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: "key: description" — use the description part
            if ":" in line:
                desc = line.split(":", 1)[1].strip()
                if desc:
                    # Capitalize first letter for prompt readability
                    actions.append(desc[0].upper() + desc[1:] if len(desc) > 1 else desc.upper())
            else:
                actions.append(line)

    logger.info(f"Loaded {len(actions)} actions from {actions_file}")
    return actions if actions else FALLBACK_ACTIONS


def center_crop_portrait(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Center-crop video frames to target aspect ratio, then resize.

    Args:
        frames: [F, C, H, W] tensor in [0, 1]
        target_h: Target height
        target_w: Target width

    Returns:
        [F, C, target_h, target_w] tensor
    """
    _, _, h, w = frames.shape
    target_aspect = target_w / target_h
    source_aspect = w / h

    if abs(source_aspect - target_aspect) > 0.01:
        if source_aspect > target_aspect:
            # Source is wider → crop width
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            frames = frames[:, :, :, start_x : start_x + new_w]
        else:
            # Source is taller → crop height
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            frames = frames[:, :, start_y : start_y + new_h, :]

    return F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)


def extract_clips(
    frames: torch.Tensor,
    clip_frames: int = CLIP_FRAMES,
    stride: int = CLIP_STRIDE,
    max_clips: int = MAX_CLIPS_PER_VIDEO,
) -> list[torch.Tensor]:
    """Extract overlapping clips from a video.

    Args:
        frames: [F, C, H, W] tensor
        clip_frames: Frames per clip (must satisfy clip_frames % 8 == 1)
        stride: Frames between clip start positions
        max_clips: Maximum clips to extract

    Returns:
        List of [clip_frames, C, H, W] tensors
    """
    total_frames = frames.shape[0]
    clips = []

    for start in range(0, total_frames - clip_frames + 1, stride):
        clip = frames[start : start + clip_frames]
        clips.append(clip)
        if len(clips) >= max_clips:
            break

    return clips


def encode_clip(
    vae_encoder: torch.nn.Module,
    clip: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a video clip to VAE latent.

    Args:
        clip: [F, C, H, W] tensor in [0, 1]

    Returns:
        Latent tensor [128, F_lat, H_lat, W_lat]
    """
    # [F, C, H, W] → [1, C, F, H, W], normalize to [-1, 1]
    batch = rearrange(clip, "f c h w -> 1 c f h w") * 2.0 - 1.0
    batch = batch.to(device, dtype=dtype)

    with torch.inference_mode():
        latent = vae_encoder(batch)

    return latent.squeeze(0).cpu()  # [128, F_lat, H_lat, W_lat]


def encode_first_frame(
    vae_encoder: torch.nn.Module,
    clip: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode the first frame of a clip as single-frame latent.

    Returns:
        Latent tensor [128, 1, H_lat, W_lat]
    """
    # clip shape: [F, C, H, W]. Extract first frame → [C, H, W]
    # Then build [1, C, 1, H, W] for VAE (B=1, C=3, F=1, H, W)
    frame = clip[0]  # [C, H, W]
    batch = frame.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0  # [1, C, 1, H, W]
    batch = batch.to(device, dtype=dtype)

    with torch.inference_mode():
        latent = vae_encoder(batch)

    return latent.squeeze(0).cpu()  # [128, 1, H_lat, W_lat]


def generate_prompts(
    num_clips: int,
    actions: list[str],
    seed: int = 42,
) -> list[str]:
    """Generate diverse prompts for each clip.

    Combines base isometric prompt templates with character actions.
    """
    rng = random.Random(seed)
    prompts = []

    for _ in range(num_clips):
        template = rng.choice(BASE_PROMPTS)
        action = rng.choice(actions)
        prompt = template.format(action=action)
        prompts.append(prompt)

    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode Grok isometric videos into I2V training dataset"
    )
    parser.add_argument(
        "--video-dir", type=Path, default=GROK_VIDEO_DIR,
        help="Directory with Grok MP4 videos + .txt captions",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output dataset directory",
    )
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH,
        help="LTX-2 model checkpoint for VAE encoder",
    )
    parser.add_argument(
        "--actions-file", type=Path, default=ACTIONS_FILE,
        help="Path to actions.txt with character action descriptions",
    )
    parser.add_argument(
        "--target-width", type=int, default=TARGET_WIDTH,
        help="Target width (must be divisible by 32)",
    )
    parser.add_argument(
        "--target-height", type=int, default=TARGET_HEIGHT,
        help="Target height (must be divisible by 32)",
    )
    parser.add_argument(
        "--clip-frames", type=int, default=CLIP_FRAMES,
        help="Frames per clip (must satisfy frames %% 8 == 1)",
    )
    parser.add_argument(
        "--clip-stride", type=int, default=CLIP_STRIDE,
        help="Stride between clip start positions",
    )
    parser.add_argument(
        "--max-clips-per-video", type=int, default=MAX_CLIPS_PER_VIDEO,
        help="Maximum clips to extract per video",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1",
        help="Device for VAE encoder (default: cuda:1 = RTX PRO 4000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    args = parser.parse_args()

    # Validate
    if args.target_width % 32 != 0 or args.target_height % 32 != 0:
        raise ValueError(f"Dimensions must be divisible by 32: {args.target_width}x{args.target_height}")
    if args.clip_frames % 8 != 1:
        raise ValueError(f"clip_frames must satisfy frames %% 8 == 1, got {args.clip_frames}")

    # ── Step 1: Discover videos ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Discovering Grok videos")
    print("=" * 60)

    video_files = sorted(args.video_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in {args.video_dir}")

    print(f"Found {len(video_files)} videos in {args.video_dir}")
    for v in video_files:
        print(f"  {v.name} ({v.stat().st_size / 1e6:.1f} MB)")

    # ── Step 2: Load actions for prompt generation ───────────────────────
    actions = load_actions(args.actions_file)
    print(f"\nLoaded {len(actions)} action descriptions")

    # ── Step 3: Extract clips from all videos ────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Loading videos + extracting clips")
    print("=" * 60)

    all_clips: list[dict] = []  # {clip: Tensor, video_id: str, start_frame: int}

    for video_path in tqdm(video_files, desc="Loading videos"):
        video_id = video_path.stem

        # Read all frames
        frames, fps = read_video(video_path, max_frames=None)
        print(f"  {video_id}: {frames.shape[0]} frames @ {fps}fps, {frames.shape[2]}x{frames.shape[3]}")

        # Center-crop + resize to target portrait dimensions
        frames = center_crop_portrait(frames, args.target_height, args.target_width)

        # Extract clips
        clips = extract_clips(
            frames,
            clip_frames=args.clip_frames,
            stride=args.clip_stride,
            max_clips=args.max_clips_per_video,
        )

        for i, clip in enumerate(clips):
            start_frame = i * args.clip_stride
            all_clips.append({
                "clip": clip,
                "video_id": video_id,
                "start_frame": start_frame,
                "video_path": str(video_path),
            })

        print(f"    → {len(clips)} clips extracted")

    print(f"\nTotal: {len(all_clips)} clips from {len(video_files)} videos")

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        print(f"  {len(all_clips)} video latents in latents/")
        print(f"  {len(all_clips)} reference latents in reference_latents/")
        print(f"  Target: {args.target_width}x{args.target_height} portrait")
        print(f"  Clip: {args.clip_frames} frames, stride {args.clip_stride}")
        return

    # ── Step 4: Encode with VAE ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Encoding clips with VAE")
    print("=" * 60)

    # Create output directories
    latents_dir = args.output_dir / "latents"
    ref_dir = args.output_dir / "reference_latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE encoder
    print(f"Loading VAE encoder on {args.device}...")
    vae_encoder = load_video_vae_encoder(args.model_path, dtype=torch.bfloat16)
    vae_encoder = vae_encoder.to(args.device)
    vae_encoder.eval()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(args.device) / 1e9
        print(f"  VAE VRAM: {alloc:.1f}GB")

    # Generate prompts
    prompts = generate_prompts(len(all_clips), actions, seed=args.seed)

    metadata_entries = []

    for idx, clip_info in enumerate(tqdm(all_clips, desc="Encoding")):
        clip = clip_info["clip"]

        # Check if already encoded (zero-padded to match compute_final_embeddings.py)
        latent_path = latents_dir / f"{idx:03d}.pt"
        ref_path = ref_dir / f"{idx:03d}.pt"

        if latent_path.exists() and ref_path.exists():
            # Load existing to get shape for metadata
            metadata_entries.append({
                "id": idx,
                "video_id": clip_info["video_id"],
                "start_frame": clip_info["start_frame"],
                "caption": prompts[idx],
                "cached": True,
            })
            continue

        try:
            # Encode full clip → video latent
            video_latent = encode_clip(vae_encoder, clip, args.device, torch.bfloat16)

            # Encode first frame → reference latent
            ref_latent = encode_first_frame(vae_encoder, clip, args.device, torch.bfloat16)

            # Save video latent
            torch.save({
                "latents": video_latent,  # [128, F_lat, H_lat, W_lat]
                "num_frames": torch.tensor([video_latent.shape[1]]),  # Latent temporal dim, NOT raw frame count
                "height": torch.tensor([video_latent.shape[2]]),
                "width": torch.tensor([video_latent.shape[3]]),
            }, latent_path)

            # Save reference (first frame) latent
            torch.save({
                "latents": ref_latent,  # [128, 1, H_lat, W_lat]
                "num_frames": torch.tensor([1]),
                "height": torch.tensor([ref_latent.shape[2]]),
                "width": torch.tensor([ref_latent.shape[3]]),
            }, ref_path)

            metadata_entries.append({
                "id": idx,
                "video_id": clip_info["video_id"],
                "start_frame": clip_info["start_frame"],
                "caption": prompts[idx],
                "video_latent_shape": list(video_latent.shape),
                "ref_latent_shape": list(ref_latent.shape),
            })

            if idx == 0:
                print(f"\n  First clip shapes:")
                print(f"    Video latent: {video_latent.shape}")
                print(f"    Ref latent:   {ref_latent.shape}")

        except Exception as e:
            logger.error(f"Failed to encode clip {idx}: {e}")
            continue

    # Cleanup VAE
    del vae_encoder
    torch.cuda.empty_cache()
    gc.collect()

    # ── Step 5: Write metadata ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Writing metadata")
    print("=" * 60)

    metadata = {
        "task_type": "motion_transfer",
        "description": (
            "Isometric 3D I2V training dataset from Grok-generated videos. "
            "reference_latents/ = first frame (I2V condition), "
            "latents/ = full video clip (ground truth). "
            "Diverse action prompts from actions.txt."
        ),
        "source": "grok_isometric_videos",
        "num_samples": len(metadata_entries),
        "clip_frames": args.clip_frames,
        "clip_stride": args.clip_stride,
        "resolution": f"{args.target_width}x{args.target_height}",
        "num_source_videos": len(video_files),
        "pairs": metadata_entries,
    }

    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nDataset written to {args.output_dir}")
    print(f"  latents/:           {len(metadata_entries)} clips")
    print(f"  reference_latents/: {len(metadata_entries)} first-frames")
    print(f"  metadata.json:      {len(metadata_entries)} entries")
    print(f"\nNext step: compute text embeddings (separate process, ~28GB VRAM):")
    print(f"  uv run python scripts/compute_final_embeddings.py \\")
    print(f"    --dataset-dir {args.output_dir} \\")
    print(f"    --model-path {args.model_path} \\")
    print(f"    --text-encoder-path /media/2TB/ltx-models/gemma \\")
    print(f"    --from-scratch")


if __name__ == "__main__":
    main()
