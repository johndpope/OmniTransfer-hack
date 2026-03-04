#!/usr/bin/env python3
"""Prepare style transfer training data using OmniTransfer reference videos.

This script:
1. Downloads OmniTransfer style reference videos
2. Uses existing identity videos as content targets
3. Processes videos using existing process_videos.py
4. Creates training pairs with proper metadata

Usage:
    python scripts/prepare_style_transfer_data.py \
        --output-dir /path/to/output \
        --model-path /path/to/ltx-2-19b-dev.safetensors \
        --text-encoder-path /path/to/gemma \
        --content-videos-dir /path/to/identity/videos
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests
import torch
from tqdm import tqdm


# OmniTransfer video URLs
OMNITRANSFER_BASE_URL = "https://pangzecheung.github.io/OmniTransfer/assets/videos"

STYLE_CATEGORIES = {
    "Effect": {
        "videos": [f"ref{i}" for i in range(1, 7)],
        "descriptions": [
            "glowing particle fire effect",
            "liquid metallic chrome effect",
            "burning flames smoke effect",
            "ice crystal frost effect",
            "electric lightning effect",
            "holographic rainbow effect"
        ]
    }
}

# Target dimensions matching OmniTransfer videos
TARGET_WIDTH = 768
TARGET_HEIGHT = 432
TARGET_FRAMES = 73
TARGET_FPS = 25


def download_style_videos(output_dir: Path, categories: list[str] = None) -> list[tuple[Path, str]]:
    """Download OmniTransfer style reference videos."""
    if categories is None:
        categories = list(STYLE_CATEGORIES.keys())

    videos_dir = output_dir / "style_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for category in categories:
        if category not in STYLE_CATEGORIES:
            print(f"Warning: Unknown category {category}")
            continue

        cat_info = STYLE_CATEGORIES[category]

        for video_name, description in zip(cat_info["videos"], cat_info["descriptions"]):
            url = f"{OMNITRANSFER_BASE_URL}/{category}/{video_name}.mp4"
            output_path = videos_dir / f"{category}_{video_name}.mp4"

            if output_path.exists() and output_path.stat().st_size > 10000:
                print(f"  Already exists: {output_path.name}")
                downloaded.append((output_path, description))
                continue

            print(f"  Downloading {category}/{video_name}.mp4...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                downloaded.append((output_path, description))

            except Exception as e:
                print(f"    Error downloading: {e}")

    return downloaded


def find_content_videos(content_dir: Path, max_videos: int = 6) -> list[tuple[Path, str]]:
    """Find existing content videos to use as targets."""
    content_videos = []

    # Look for mp4 files
    for video_path in sorted(content_dir.glob("**/*.mp4"))[:max_videos]:
        # Use filename as description
        desc = video_path.stem.replace("_", " ")
        content_videos.append((video_path, desc))

    return content_videos


def preprocess_video(input_path: Path, output_path: Path) -> bool:
    """Preprocess video to target dimensions."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
               f"fps={TARGET_FPS}",
        "-frames:v", str(TARGET_FRAMES),
        "-c:v", "libx264", "-crf", "18",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path}: {e}")
        return False


def create_dataset_csv(
    style_videos: list[tuple[Path, str]],
    content_videos: list[tuple[Path, str]],
    output_dir: Path
) -> tuple[Path, Path, list[dict]]:
    """Create CSV files for process_videos.py and metadata."""

    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)

    # Preprocess all videos to target dimensions
    print("\nPreprocessing videos to target dimensions...")

    processed_styles = []
    for video_path, desc in tqdm(style_videos, desc="Styles"):
        out_path = preprocessed_dir / f"style_{video_path.stem}.mp4"
        if out_path.exists() or preprocess_video(video_path, out_path):
            processed_styles.append((out_path, desc))

    processed_content = []
    for video_path, desc in tqdm(content_videos, desc="Content"):
        out_path = preprocessed_dir / f"content_{video_path.stem}.mp4"
        if out_path.exists() or preprocess_video(video_path, out_path):
            processed_content.append((out_path, desc))

    # Create CSV for all videos
    all_videos_csv = output_dir / "all_videos.csv"
    with open(all_videos_csv, 'w') as f:
        f.write("media_path,caption\n")
        for video_path, desc in processed_styles:
            f.write(f"{video_path},{desc} style video\n")
        for video_path, desc in processed_content:
            f.write(f"{video_path},{desc} content video\n")

    # Create training pairs metadata
    pairs = []
    pair_idx = 0

    for style_path, style_desc in processed_styles:
        for content_path, content_desc in processed_content:
            pairs.append({
                "idx": pair_idx,
                "file_name": f"{pair_idx:03d}.pt",
                "reference_file_name": f"{pair_idx:03d}.pt",
                "media_path": f"{pair_idx:03d}.pt",
                "text": f"Apply {style_desc} to {content_desc}. Style transfer video.",
                "style_source": style_path.name,
                "content_source": content_path.name
            })
            pair_idx += 1

    return all_videos_csv, preprocessed_dir, pairs


def run_process_videos(csv_path: Path, output_dir: Path, model_path: str) -> bool:
    """Run the existing process_videos.py script."""
    cmd = [
        "uv", "run", "python", "scripts/process_videos.py",
        str(csv_path),
        "--resolution-buckets", f"{TARGET_WIDTH}x{TARGET_HEIGHT}x{TARGET_FRAMES}",
        "--output-dir", str(output_dir / "encoded_latents"),
        "--model-path", model_path
    ]

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running process_videos.py: {e}")
        return False


def organize_latents(
    encoded_dir: Path,
    pairs: list[dict],
    output_dir: Path
) -> None:
    """Organize encoded latents into proper training structure."""

    latents_dir = output_dir / "latents"
    ref_latents_dir = output_dir / "reference_latents"
    latents_dir.mkdir(exist_ok=True)
    ref_latents_dir.mkdir(exist_ok=True)

    # Find encoded latents
    encoded_files = list(encoded_dir.rglob("*.pt"))

    # Map names to latent files
    latent_map = {}
    for lat_file in encoded_files:
        latent_map[lat_file.stem] = lat_file

    print(f"\nFound {len(latent_map)} encoded latents")

    # Organize by pairs
    import shutil

    for pair in tqdm(pairs, desc="Organizing latents"):
        idx = pair["idx"]
        style_name = pair["style_source"].replace(".mp4", "").replace("style_", "")
        content_name = pair["content_source"].replace(".mp4", "").replace("content_", "")

        # Find style latent (reference)
        style_key = f"style_{style_name}"
        if style_key in latent_map:
            shutil.copy(latent_map[style_key], ref_latents_dir / f"{idx:03d}.pt")

        # Find content latent (target)
        content_key = f"content_{content_name}"
        if content_key in latent_map:
            shutil.copy(latent_map[content_key], latents_dir / f"{idx:03d}.pt")


def process_text_embeddings(
    pairs: list[dict],
    output_dir: Path,
    text_encoder_path: str
) -> None:
    """Process text embeddings for all training pairs."""
    conditions_dir = output_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(pairs)} text embeddings...")

    # Load text encoder
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    model = AutoModelForCausalLM.from_pretrained(
        text_encoder_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()

    MAX_LEN = 256
    HIDDEN_DIM = 3840

    for pair in tqdm(pairs, desc="Encoding text"):
        output_path = conditions_dir / pair["file_name"]

        if output_path.exists():
            continue

        caption = pair["text"]

        with torch.no_grad():
            inputs = tokenizer(
                caption,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to("cuda")

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].squeeze(0)
            attention_mask = inputs.attention_mask.squeeze(0)

            # Pad to MAX_LEN
            seq_len = hidden_states.shape[0]
            if seq_len < MAX_LEN:
                pad_len = MAX_LEN - seq_len
                hidden_states = torch.cat([
                    hidden_states,
                    torch.zeros(pad_len, HIDDEN_DIM, device=hidden_states.device, dtype=hidden_states.dtype)
                ], dim=0)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=0)

            # Save (both as bfloat16 for trainer compatibility)
            torch.save({
                "prompt_embeds": hidden_states.cpu().to(torch.bfloat16),
                "prompt_attention_mask": attention_mask.cpu().to(torch.bfloat16)
            }, output_path)

    # Clean up
    del model, tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Prepare style transfer training data")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for processed data")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to LTX-2 safetensors checkpoint")
    parser.add_argument("--text-encoder-path", type=str, required=True,
                        help="Path to Gemma text encoder directory")
    parser.add_argument("--content-videos-dir", type=str, required=True,
                        help="Directory containing content/target videos")
    parser.add_argument("--max-content-videos", type=int, default=6,
                        help="Maximum number of content videos to use")
    parser.add_argument("--style-categories", type=str, nargs="+", default=["Effect"],
                        help="Style categories to download (Effect)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dir = output_dir / "processed"
    final_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Style Transfer Training Data Preparation")
    print("=" * 60)
    print(f"Output: {final_dir}")
    print(f"Target dims: {TARGET_WIDTH}x{TARGET_HEIGHT}, {TARGET_FRAMES} frames")
    print(f"Content source: {args.content_videos_dir}")
    print()

    # Step 1: Download style reference videos
    print("Step 1: Downloading style reference videos...")
    style_videos = download_style_videos(output_dir, args.style_categories)
    print(f"  Downloaded {len(style_videos)} style videos")

    # Step 2: Find content videos
    print("\nStep 2: Finding content videos...")
    content_dir = Path(args.content_videos_dir)
    content_videos = find_content_videos(content_dir, args.max_content_videos)
    print(f"  Found {len(content_videos)} content videos")

    if not content_videos:
        print("ERROR: No content videos found!")
        sys.exit(1)

    # Step 3: Create CSV and preprocess
    print("\nStep 3: Preprocessing videos and creating metadata...")
    csv_path, preprocessed_dir, pairs = create_dataset_csv(
        style_videos, content_videos, output_dir
    )
    print(f"  Created {len(pairs)} training pairs")

    # Step 4: Encode videos to latents
    print("\nStep 4: Encoding videos to VAE latents...")
    if run_process_videos(csv_path, output_dir, args.model_path):
        print("  Encoding complete")
    else:
        print("  WARNING: Encoding may have failed")

    # Step 5: Organize latents
    print("\nStep 5: Organizing latents into training structure...")
    organize_latents(
        output_dir / "encoded_latents",
        pairs,
        final_dir
    )

    # Step 6: Process text embeddings
    print("\nStep 6: Processing text embeddings...")
    process_text_embeddings(pairs, final_dir, args.text_encoder_path)

    # Step 7: Save metadata
    print("\nStep 7: Saving metadata...")
    with open(final_dir / "metadata.json", 'w') as f:
        json.dump(pairs, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Training data saved to: {final_dir}")
    print(f"  - latents/: {len(list((final_dir / 'latents').glob('*.pt')))} files")
    print(f"  - reference_latents/: {len(list((final_dir / 'reference_latents').glob('*.pt')))} files")
    print(f"  - conditions/: {len(list((final_dir / 'conditions').glob('*.pt')))} files")
    print(f"  - metadata.json: {len(pairs)} pairs")
    print()
    print("Update your training config to use:")
    print(f"  preprocessed_data_root: \"{final_dir}\"")


if __name__ == "__main__":
    main()
