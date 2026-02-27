#!/usr/bin/env python3
# ruff: noqa: T201
"""Extract every frame from isometric videos as individual T2I training samples.

Reads the 67 isometric videos from isometric_scd_v2, extracts ALL frames (up to 25
per video), resizes each to 480×704, VAE-encodes as 1-frame latent, and hardlinks
the parent video's conditions_final embedding for each frame.

This produces ~1,675 single-frame samples (67 videos × 25 frames) covering the
full isometric style distribution — 25× more than the video-level T2V dataset.

Data flow:
    For each video (67 total):
        For each frame (25 per video):
            frame → resize 480×704 → VAE encode → latent [128, 1, 22, 15]
            conditions_final → hardlink from parent video's embedding

Output structure:
    /media/2TB/omnitransfer/data/isometric_t2i/
    ├── latents/              # 1-frame latents [128, 1, H_lat, W_lat]
    ├── conditions_final/     # Hardlinked text embeddings [1024, 3840]
    └── metadata.json

Usage:
    cd ltx-trainer
    python scripts/extract_frames_for_t2i.py
    python scripts/extract_frames_for_t2i.py --dry-run     # preview only
    python scripts/extract_frames_for_t2i.py --skip-vae    # hardlink conditions only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

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
SOURCE_METADATA = Path("/media/2TB/omnitransfer/data/isometric_scd_v2/metadata.json")
SOURCE_CONDITIONS = Path("/media/2TB/omnitransfer/data/isometric_scd_v2/conditions_final")
OUTPUT_DIR = Path("/media/2TB/omnitransfer/data/isometric_t2i")
MODEL_PATH = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")

TARGET_WIDTH = 480
TARGET_HEIGHT = 704
MAX_FRAMES_PER_VIDEO = 25  # Match training trim count

VAE_DEVICE = "cuda:1"  # RTX PRO 4000 (24GB) — VAE encoder (~8GB)


def resize_frame(frame: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize a single frame [C, H, W] to target dimensions."""
    return F.interpolate(
        frame.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
    ).squeeze(0)


def encode_single_frame(
    vae_encoder: torch.nn.Module,
    frame: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a single frame to VAE latent.

    Args:
        frame: [C, H, W] tensor in [0, 1]
    Returns:
        Latent tensor [128, 1, H_lat, W_lat]
    """
    # [C, H, W] → [1, C, 1, H, W], normalize to [-1, 1]
    batch = frame.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0  # [1, C, 1, H, W]
    batch = batch.to(device, dtype=dtype)

    with torch.inference_mode():
        latent = vae_encoder(batch)

    return latent.squeeze(0).cpu()  # [128, 1, H_lat, W_lat]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract every frame from isometric videos for T2I training"
    )
    parser.add_argument("--source-metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--source-conditions", type=Path, default=SOURCE_CONDITIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_VIDEO)
    parser.add_argument("--vae-device", type=str, default=VAE_DEVICE)
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE encoding, only do hardlinks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── Load source metadata ──────────────────────────────────────────────
    print("=" * 60)
    print("Loading source metadata")
    print("=" * 60)

    with open(args.source_metadata) as f:
        source_meta = json.load(f)

    pairs = source_meta["pairs"]
    print(f"Source dataset: {len(pairs)} videos")
    print(f"Max frames per video: {args.max_frames}")
    print(f"Expected output: ~{len(pairs) * args.max_frames} single-frame samples")

    # Latent math for 1 frame
    lat_h = args.target_height // 32
    lat_w = args.target_width // 32
    print(f"Resolution: {args.target_width}x{args.target_height} → latent {lat_w}x{lat_h}")
    print(f"Per-frame latent shape: [128, 1, {lat_h}, {lat_w}]")

    if args.dry_run:
        # Count actual frames per video
        total_frames = 0
        for pair in pairs[:5]:
            mp4 = Path(pair["mp4_path"])
            if mp4.exists():
                frames, _ = read_video(mp4, max_frames=args.max_frames)
                n = min(frames.shape[0], args.max_frames)
                print(f"  {pair['uuid'][:20]}...: {n} frames")
                total_frames += n

        avg = total_frames / min(5, len(pairs))
        est_total = int(avg * len(pairs))
        print(f"\n[DRY RUN] Estimated ~{est_total} total frames from {len(pairs)} videos")
        disk_gb = est_total * 128 * lat_h * lat_w * 2 / 1e9  # bf16 = 2 bytes
        print(f"  Latent disk: ~{disk_gb:.1f}GB")
        print(f"  Conditions: hardlinks → 0 extra disk")
        return

    # Create output dirs
    latents_dir = args.output_dir / "latents"
    conditions_dir = args.output_dir / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: VAE encode frames ────────────────────────────────────────
    if not args.skip_vae:
        print("\n" + "=" * 60)
        print("Phase 1: VAE encoding individual frames")
        print("=" * 60)

        print(f"Loading VAE encoder on {args.vae_device}...")
        vae_encoder = load_video_vae_encoder(args.model_path, dtype=torch.bfloat16)
        vae_encoder = vae_encoder.to(args.vae_device)
        vae_encoder.eval()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(args.vae_device) / 1e9
            print(f"  VAE VRAM: {alloc:.1f}GB")

        global_idx = 0
        total_encoded = 0
        total_skipped = 0
        total_failed = 0

        for video_idx, pair in enumerate(tqdm(pairs, desc="Videos")):
            mp4_path = Path(pair["mp4_path"])

            if not mp4_path.exists():
                logger.warning(f"Video not found: {mp4_path}")
                total_failed += 1
                # Still increment global_idx for the expected frames
                global_idx += args.max_frames
                continue

            try:
                frames, fps = read_video(mp4_path, max_frames=args.max_frames)
                n_frames = min(frames.shape[0], args.max_frames)
                frames = frames[:n_frames]

                for frame_idx in range(n_frames):
                    latent_path = latents_dir / f"{global_idx:06d}.pt"

                    if latent_path.exists():
                        total_skipped += 1
                        global_idx += 1
                        continue

                    # Resize frame to target dims
                    frame = resize_frame(frames[frame_idx], args.target_height, args.target_width)

                    # Encode single frame
                    latent = encode_single_frame(vae_encoder, frame, args.vae_device, torch.bfloat16)

                    # Save in PrecomputedDataset format
                    torch.save(
                        {
                            "latents": latent,  # [128, 1, H_lat, W_lat]
                            "num_frames": torch.tensor([1]),
                            "height": torch.tensor([latent.shape[2]]),
                            "width": torch.tensor([latent.shape[3]]),
                            "fps": torch.tensor([24.0]),
                        },
                        latent_path,
                    )

                    total_encoded += 1
                    global_idx += 1

                    if total_encoded == 1:
                        print(f"\n  First latent shape: {latent.shape}")

            except Exception as e:
                logger.error(f"Failed to process video {video_idx} ({pair['uuid']}): {e}")
                total_failed += 1
                # Increment past the expected frames for this video
                global_idx += args.max_frames
                continue

        # Cleanup VAE
        del vae_encoder
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n  Encoded: {total_encoded}, Skipped (cached): {total_skipped}, Failed videos: {total_failed}")
    else:
        print("\n[SKIP] VAE encoding (--skip-vae)")
        # Count existing latents to know the global index range
        global_idx = len(list(latents_dir.glob("*.pt")))

    # ── Phase 2: Hardlink conditions_final ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Hardlinking conditions_final embeddings")
    print("=" * 60)

    global_idx = 0
    linked = 0
    skipped = 0

    for video_idx, pair in enumerate(pairs):
        source_cond = args.source_conditions / f"{video_idx:06d}.pt"

        if not source_cond.exists():
            logger.warning(f"Source condition not found: {source_cond}")
            global_idx += args.max_frames
            continue

        # Read the video to count actual frames (must match Phase 1)
        mp4_path = Path(pair["mp4_path"])
        if mp4_path.exists():
            frames, _ = read_video(mp4_path, max_frames=args.max_frames)
            n_frames = min(frames.shape[0], args.max_frames)
            del frames  # Free memory
        else:
            n_frames = args.max_frames

        for frame_idx in range(n_frames):
            target_cond = conditions_dir / f"{global_idx:06d}.pt"

            if target_cond.exists():
                skipped += 1
                global_idx += 1
                continue

            # Only link if the corresponding latent exists
            latent_path = latents_dir / f"{global_idx:06d}.pt"
            if not latent_path.exists():
                global_idx += 1
                continue

            try:
                os.link(str(source_cond), str(target_cond))
                linked += 1
            except OSError:
                # Fallback: copy if hardlink fails (cross-device)
                import shutil
                shutil.copy2(str(source_cond), str(target_cond))
                linked += 1

            global_idx += 1

    print(f"  Linked: {linked}, Skipped (existing): {skipped}")

    # ── Write metadata ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Writing metadata")
    print("=" * 60)

    num_latents = len(list(latents_dir.glob("*.pt")))
    num_conditions = len(list(conditions_dir.glob("*.pt")))

    # Build per-frame metadata
    frame_entries = []
    global_idx = 0
    for video_idx, pair in enumerate(pairs):
        mp4_path = Path(pair["mp4_path"])
        if mp4_path.exists():
            frames, _ = read_video(mp4_path, max_frames=args.max_frames)
            n_frames = min(frames.shape[0], args.max_frames)
            del frames
        else:
            n_frames = args.max_frames

        for frame_idx in range(n_frames):
            if (latents_dir / f"{global_idx:06d}.pt").exists():
                frame_entries.append({
                    "id": global_idx,
                    "source_video_idx": video_idx,
                    "frame_idx": frame_idx,
                    "uuid": pair["uuid"],
                    "prompt": pair.get("prompt", ""),
                })
            global_idx += 1

    metadata = {
        "task_type": "isometric_t2i",
        "description": (
            f"T2I dataset: {num_latents} single-frame samples extracted from "
            f"{len(pairs)} isometric Grok videos (25 frames each). "
            f"Resolution: {args.target_width}x{args.target_height}, 1 frame per sample."
        ),
        "source_dataset": str(args.source_metadata),
        "num_samples": num_latents,
        "num_conditions": num_conditions,
        "num_source_videos": len(pairs),
        "frames_per_video": args.max_frames,
        "resolution": f"{args.target_width}x{args.target_height}",
        "num_frames": 1,
        "has_final_embeddings": True,
        "conditions_final_dir": "conditions_final",
        "frame_entries": frame_entries,
    }

    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nDataset written to {args.output_dir}")
    print(f"  latents/:           {num_latents} files")
    print(f"  conditions_final/:  {num_conditions} files")
    print(f"  metadata.json:      {len(frame_entries)} entries")
    print(f"  Source videos:      {len(pairs)}")
    print(f"  Frames/video:       ~{num_latents / max(len(pairs), 1):.1f} avg")

    if num_latents != num_conditions:
        print(f"\n  WARNING: Mismatch: {num_latents} latents vs {num_conditions} conditions!")
        print(f"  Re-run with --skip-vae to fill missing hardlinks")


if __name__ == "__main__":
    main()
