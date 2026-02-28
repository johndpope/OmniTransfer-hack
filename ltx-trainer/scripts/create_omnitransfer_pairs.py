#!/usr/bin/env python3
"""Create pairs.json for OmniTransfer training from generated synthetic videos.

This script takes generated synthetic videos and creates the pairing structure
needed for OmniTransfer training. It identifies reference videos (static/baseline)
and pairs them with target videos (variations with motion, style, etc.).

For identity preservation training:
- Reference: Static portrait videos (reference_* mode)
- Targets: All other videos with the same identity but different scenarios

Usage:
    python scripts/create_omnitransfer_pairs.py \
        --videos-dir /media/2TB/omnitransfer_synthetic_videos \
        --output-dir /media/2TB/omnitransfer_raw_dataset

This creates:
    output_dir/
        videos/
            ref_001.mp4 -> symlink or copy
            tgt_001.mp4 -> symlink or copy
            ...
        pairs.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path


# Prompt category mappings (from generate_videos_local.py)
PROMPT_CATEGORIES = {
    "reference": "pure_id",
    "head_turn": "pure_id",
    "casual": "motion_casual",
    "expressive": "motion_expressive",
    "cyberpunk": "style_cyberpunk",
    "ghibli": "style_ghibli",
    "aging": "id_aging",
}

# Captions for each video type
CAPTIONS = {
    "reference_1": "A 32-year-old East Asian woman with shoulder-length black hair with caramel highlights, wearing a white turtleneck and jeans, standing in soft window light, neutral expression with slight smile, photorealistic portrait",
    "reference_2": "A 32-year-old East Asian woman, medium close-up portrait, soft diffused lighting, looking at camera, serene expression",
    "reference_3": "A 32-year-old East Asian woman, three-quarter view portrait, natural daylight, subtle smile",
    "head_turn_1": "A 32-year-old East Asian woman slowly turning head from left to right, maintaining neutral expression with slight smile, consistent identity",
    "head_turn_2": "A 32-year-old East Asian woman, gentle head movement looking from camera to side and back, smooth motion",
    "casual_1": "A 32-year-old East Asian woman sitting at a cafe table, turning pages of notebook, tucking hair behind ear, sipping coffee",
    "casual_2": "A 32-year-old East Asian woman walking through a sunlit park path, gentle breeze moving hair, natural gait",
    "casual_3": "A 32-year-old East Asian woman sitting on a park bench, reading a book, occasionally looking up and smiling",
    "expressive_1": "A 32-year-old East Asian woman sitting on a sofa, transitioning from neutral to smiling warmly, then laughing softly",
    "expressive_2": "A 32-year-old East Asian woman close-up, transitioning from thoughtful to surprised delight, genuine smile",
    "cyberpunk_1": "The same East Asian woman in cyberpunk aesthetic, glossy black latex top, neon-lit Tokyo alley at night",
    "cyberpunk_2": "The same East Asian woman in futuristic silver bodysuit, holographic visor, spaceship corridor",
    "ghibli_1": "The same East Asian woman in Studio Ghibli animation style, cozy wooden room with forest view, cream linen dress",
    "ghibli_2": "The same East Asian woman in Ghibli style, walking through wildflower field, white sundress, straw hat",
    "aging_1": "The same East Asian woman aged to 50 years old, silver-gray hair, same facial structure, elegant navy blouse",
}


def find_videos(videos_dir: Path) -> dict[str, Path]:
    """Find all generated videos by category."""
    videos = {}
    for mp4_file in videos_dir.glob("*.mp4"):
        # Parse filename: mode_number_seed.mp4
        name = mp4_file.stem
        parts = name.rsplit("_", 1)  # Split off seed
        if len(parts) == 2:
            video_key = parts[0]  # e.g., "reference_1"
            videos[video_key] = mp4_file
    return videos


def create_pairs(videos: dict[str, Path]) -> list[dict]:
    """Create reference-target pairs for OmniTransfer training.

    Strategy for identity preservation:
    - Use reference_1 as the primary reference (static, neutral baseline)
    - Pair it with all other videos as targets
    """
    pairs = []

    # Primary reference video
    primary_ref = "reference_1"
    if primary_ref not in videos:
        # Fall back to any reference video
        ref_candidates = [k for k in videos.keys() if k.startswith("reference")]
        if not ref_candidates:
            raise ValueError("No reference videos found! Need at least one reference_* video.")
        primary_ref = ref_candidates[0]

    ref_path = videos[primary_ref]

    # Create pairs: reference -> each target
    for video_key, video_path in videos.items():
        if video_key == primary_ref:
            continue  # Skip self-pairing

        pair = {
            "reference": ref_path.name,
            "target": video_path.name,
            "caption": CAPTIONS.get(video_key, f"A video of the same person in a different scenario ({video_key})"),
            "task_type": "identity_preservation",
            "source_category": PROMPT_CATEGORIES.get(video_key.rsplit("_", 1)[0], "unknown"),
        }
        pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Create pairs.json for OmniTransfer training from generated videos"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing generated videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for dataset structure",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of symlinking",
    )
    args = parser.parse_args()

    # Find videos
    videos = find_videos(args.videos_dir)
    print(f"Found {len(videos)} videos:")
    for key, path in sorted(videos.items()):
        print(f"  {key}: {path.name}")

    if not videos:
        print("ERROR: No videos found! Make sure video generation completed.")
        return

    # Create output structure
    output_videos_dir = args.output_dir / "videos"
    output_videos_dir.mkdir(parents=True, exist_ok=True)

    # Copy or symlink videos
    for video_key, video_path in videos.items():
        dest = output_videos_dir / video_path.name
        if dest.exists():
            print(f"  Skipping existing: {dest.name}")
            continue

        if args.copy_videos:
            print(f"  Copying: {video_path.name}")
            shutil.copy2(video_path, dest)
        else:
            print(f"  Symlinking: {video_path.name}")
            dest.symlink_to(video_path.absolute())

    # Create pairs
    pairs = create_pairs(videos)
    print(f"\nCreated {len(pairs)} training pairs")

    # Save pairs.json
    pairs_file = args.output_dir / "pairs.json"
    with open(pairs_file, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved pairs to: {pairs_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset structure created!")
    print("=" * 60)
    print(f"\nDirectory: {args.output_dir}")
    print(f"Videos: {len(videos)}")
    print(f"Training pairs: {len(pairs)}")
    print("\nNext steps:")
    print("1. Run dataset preparation:")
    print(f"   python scripts/prepare_omnitransfer_dataset.py \\")
    print(f"       --input-dir {args.output_dir} \\")
    print(f"       --output-dir /media/2TB/omnitransfer_processed \\")
    print(f"       --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \\")
    print(f"       --text-encoder-path /media/2TB/ltx-models/gemma")
    print("\n2. Start training:")
    print("   python scripts/train.py configs/ltx2_omnitransfer_lora.yaml")


if __name__ == "__main__":
    main()
