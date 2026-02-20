#!/usr/bin/env python3
# ruff: noqa: T201
"""Build cross-clip identity preservation pairs from existing isometric I2V dataset.

Takes the 128 clips (8 videos × 16 clips) encoded by encode_isometric_videos.py
and creates (reference, target) pairs where both clips come from the SAME video
but different temporal windows. This teaches the model to preserve identity
(scene, characters, style) across different temporal content.

Pairing strategy:
- For each video (16 clips), pair clips with sufficient temporal separation
  (≥ MIN_SEPARATION clips apart) to avoid near-duplicate pairs.
- Each clip appears as BOTH reference and target in different pairs.
- Text embeddings are reused from the source dataset.

Output structure:
  /media/2TB/isometric_identity_training/
  ├── latents/           # Target video clip latents
  ├── reference_latents/ # Reference video clip latents (same video, different time)
  ├── conditions_final/  # Text embeddings (copied from source)
  └── metadata.json      # Pair provenance

Usage:
  cd packages/ltx-trainer
  uv run python scripts/build_identity_pairs.py
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_DIR = Path("/media/2TB/isometric_i2v_training")
OUTPUT_DIR = Path("/media/2TB/isometric_identity_training")

# Minimum clip index separation to avoid near-duplicate pairs.
# With clip stride=8 and clip length=25, clips 4 positions apart
# (32 frames) have zero overlap.
MIN_SEPARATION = 4

# Maximum pairs per video to keep dataset balanced
MAX_PAIRS_PER_VIDEO = 24


def main() -> None:
    # Load source metadata
    meta_path = SOURCE_DIR / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    pairs = meta["pairs"]
    print(f"Source dataset: {len(pairs)} clips")

    # Group clips by video_id
    video_clips: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        video_clips[p["video_id"]].append(p)

    # Sort each video's clips by start_frame
    for vid in video_clips:
        video_clips[vid].sort(key=lambda c: c["start_frame"])

    print(f"Videos: {len(video_clips)}")
    for vid, clips in sorted(video_clips.items()):
        print(f"  {vid[:8]}...: {len(clips)} clips")

    # ─────────────────────────────────────────────────────────────────────────
    # Generate cross-clip pairs
    # ─────────────────────────────────────────────────────────────────────────
    identity_pairs = []

    for vid, clips in sorted(video_clips.items()):
        n = len(clips)
        vid_pairs = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Ensure sufficient temporal separation
                if abs(i - j) < MIN_SEPARATION:
                    continue
                vid_pairs.append((clips[i], clips[j]))

        # Cap per-video pairs
        if len(vid_pairs) > MAX_PAIRS_PER_VIDEO:
            # Sample evenly spaced pairs
            step = len(vid_pairs) / MAX_PAIRS_PER_VIDEO
            vid_pairs = [vid_pairs[int(k * step)] for k in range(MAX_PAIRS_PER_VIDEO)]

        identity_pairs.extend(vid_pairs)

    print(f"\nGenerated {len(identity_pairs)} identity pairs "
          f"({len(identity_pairs) // len(video_clips)} per video avg)")

    # ─────────────────────────────────────────────────────────────────────────
    # Create output dataset
    # ─────────────────────────────────────────────────────────────────────────
    latents_dir = OUTPUT_DIR / "latents"
    ref_dir = OUTPUT_DIR / "reference_latents"
    cond_dir = OUTPUT_DIR / "conditions_final"

    for d in [latents_dir, ref_dir, cond_dir]:
        d.mkdir(parents=True, exist_ok=True)

    output_meta = []
    src_latents = SOURCE_DIR / "latents"
    src_cond = SOURCE_DIR / "conditions_final"

    for idx, (ref_clip, tgt_clip) in enumerate(identity_pairs):
        ref_id = ref_clip["id"]
        tgt_id = tgt_clip["id"]

        # Source file paths (zero-padded)
        ref_latent_src = src_latents / f"{ref_id:03d}.pt"
        tgt_latent_src = src_latents / f"{tgt_id:03d}.pt"
        tgt_cond_src = src_cond / f"{tgt_id:03d}.pt"

        if not ref_latent_src.exists() or not tgt_latent_src.exists():
            print(f"  Skipping pair {idx}: missing source files")
            continue

        # Output paths
        out_name = f"{idx:03d}.pt"

        # Copy target latent → latents/
        shutil.copy2(tgt_latent_src, latents_dir / out_name)

        # Copy reference latent → reference_latents/
        shutil.copy2(ref_latent_src, ref_dir / out_name)

        # Copy target's text embedding → conditions_final/
        if tgt_cond_src.exists():
            shutil.copy2(tgt_cond_src, cond_dir / out_name)
        else:
            # Fallback: use ref's embedding
            ref_cond_src = src_cond / f"{ref_id:03d}.pt"
            shutil.copy2(ref_cond_src, cond_dir / out_name)

        output_meta.append({
            "id": idx,
            "video_id": ref_clip["video_id"],
            "ref_clip_id": ref_id,
            "ref_start_frame": ref_clip["start_frame"],
            "tgt_clip_id": tgt_id,
            "tgt_start_frame": tgt_clip["start_frame"],
            "caption": tgt_clip.get("caption", "An isometric 3D scene with animated characters"),
        })

    # Write metadata
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump({"pairs": output_meta}, f, indent=2)

    print(f"\nDataset written to {OUTPUT_DIR}")
    print(f"  latents/:           {len(output_meta)} target clips")
    print(f"  reference_latents/: {len(output_meta)} reference clips")
    print(f"  conditions_final/:  {len(output_meta)} text embeddings")
    print(f"  metadata.json:      {len(output_meta)} pairs")

    # Verify shapes match
    tgt = torch.load(latents_dir / "000.pt", weights_only=True)
    ref = torch.load(ref_dir / "000.pt", weights_only=True)
    print(f"\n  Target latent:    {tgt['latents'].shape}")
    print(f"  Reference latent: {ref['latents'].shape}")


if __name__ == "__main__":
    main()
