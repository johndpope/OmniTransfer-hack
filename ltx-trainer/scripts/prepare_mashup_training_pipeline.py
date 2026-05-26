#!/usr/bin/env python3
"""Pipeline: Build OmniTransfer training dataset from Facebook/Instagram mashup reels.

Pipeline:
  1. Scan all scene-split clips from facebook_reels/scenes/
  2. Extract clean character reference frames from reel clips
  3. Build cross-paired reference/target dataset
  4. Create metadata.json + text prompts
  5. Output ready for VAE encoding + OmniTransfer training

Usage:
    uv run python scripts/prepare_mashup_training_pipeline.py \
        --source-dir /path/to/facebook_reels/scenes \
        --output-dir /media/2TB/omnitransfer/data/mashup_style

Output structure:
    {output_dir}/
    ├── raw_clips/           # All scene clips (symlinked or copied)
    ├── character_refs/      # Extracted character stills
    ├── metadata.json        # Training pairs with task=style_transfer
    ├── dataset.csv          # For process_videos.py
    └── train_config.yaml    # OmniTransfer training config
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# ── Known mapping for reel IDs → crossover movies ─────────────────────────
REEL_MOVIE_MAP: dict[str, str] = {
    "1199971765284042": "Back to the Future x Career Opportunities",
    "1256976252680327": "Labyrinth x Career Opportunities",
    "1689417079061469": "Return of the Living Dead x Career Opportunities",
    "2057624615188633": "Back to the Future x Career Opportunities",
    "2125476178020928": "Pretty Woman x Career Opportunities",
    "2156378868455964": "Ghostbusters x Career Opportunities",
    "3428533713973292": "Halloween x Career Opportunities",
    "770908555771949": "Kingpin x Career Opportunities",
    "797363763429829": "National Lampoon's Vacation x Career Opportunities",
    "820428490726045": "Red Sonja x Career Opportunities",
    "961517603217527": "Weird Science x Career Opportunities",
    # Instagram reels (same mashup style, need exact movie IDs verified)
    "instagram_DVofVseCBwz": "Career Opportunities mashup",
    "instagram_DV4T00Tj8x8": "Career Opportunities mashup",
    "instagram_DXGh6OaEnBG": "Career Opportunities mashup",
    "instagram_DVwVGuUkmHi": "Career Opportunities mashup",
}

# Instagram reels prefix
INSTAGRAM_PREFIX = "instagram_"

CHARACTER_CROSSOVERS: dict[str, list[dict]] = {
    "Career Opportunities": [
        {"character": "Josie McClellan", "actor": "Jennifer Connelly", "keywords": "Jennifer Connelly 1991 Career Opportunities department store"},
    ],
    "Back to the Future": [
        {"character": "Marty McFly", "actor": "Michael J. Fox", "keywords": "Michael J Fox Marty McFly 1985 Back to the Future dinner table"},
        {"character": "George McFly", "actor": "Crispin Glover", "keywords": "Crispin Glover George McFly 1985 dinner"},
        {"character": "Lorraine Baines", "actor": "Lea Thompson", "keywords": "Lea Thompson Lorraine Baines 1985 Back to the Future"},
    ],
    "Ghostbusters": [
        {"character": "Peter Venkman", "actor": "Bill Murray", "keywords": "Bill Murray Ghostbusters 1984"},
    ],
    "Labyrinth": [
        {"character": "Jareth", "actor": "David Bowie", "keywords": "David Bowie Jareth Labyrinth 1986 goblin king"},
    ],
}

TRAINING_RESOLUTION = (768, 1152)  # width, height — portrait to match reels
TARGET_FRAMES = 25  # Must satisfy frames % 8 == 1
TARGET_FPS = 24

CHARACTER_REF_FRAME_COUNT = 3  # Number of reference frames to extract per scene


def parse_args():
    p = argparse.ArgumentParser(description="Build OmniTransfer training dataset from reel mashups")
    p.add_argument("--source-dir", type=Path, required=True,
                   help="Path to facebook_reels/scenes/ directory")
    p.add_argument("--instagram-dir", type=Path, default=None,
                   help="Path to Instagram reel files (facebook_reels/) for additional clips")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for processed training data")
    p.add_argument("--resolution", type=int, nargs=2, default=list(TRAINING_RESOLUTION),
                   help="Target resolution width height (default: 768 1152)")
    p.add_argument("--frames", type=int, default=TARGET_FRAMES,
                   help="Target frame count per clip (default: 25, must be frames % 8 == 1)")
    p.add_argument("--fps", type=float, default=TARGET_FPS, help="Target FPS")
    p.add_argument("--skip-transcode", action="store_true",
                   help="Skip ffmpeg transcoding (use existing clips as-is)")
    p.add_argument("--extract-refs-only", action="store_true",
                   help="Only extract character reference frames, skip full dataset build")
    return p.parse_args()


def extract_character_frames(
    scene_dir: Path,
    output_dir: Path,
    num_frames: int = 3,
) -> list[dict]:
    """Extract clean character frames from scene clips.

    Picks the middle frame of each scene as a representative character still.

    Returns list of {scene_index, clip_name, frame_path, character_info}
    """
    refs = []
    thumb_dir = scene_dir / "thumbnails"
    if not thumb_dir.exists():
        print(f"  ⚠ No thumbnails in {scene_dir.name}")
        return refs

    char_output = output_dir / "character_refs" / scene_dir.name
    char_output.mkdir(parents=True, exist_ok=True)

    for thumb_path in sorted(thumb_dir.glob("*.jpg")):
        clip_name = thumb_path.stem  # e.g., Scene-001
        scene_idx = int(clip_name.split("-")[-1])

        # Copy thumbnail as character reference
        ref_path = char_output / f"{clip_name}_char_ref.jpg"
        shutil.copy2(thumb_path, ref_path)

        refs.append({
            "scene_index": scene_idx,
            "clip_name": clip_name,
            "ref_path": str(ref_path),
            "source_reel": scene_dir.name,
        })

    return refs


def transcode_to_training_format(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    frames: int,
    fps: float,
) -> bool:
    """Transcode a clip to target resolution + frame count using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps}"
        ),
        "-frames:v", str(frames),
        "-c:v", "libx264", "-crf", "18",
        "-an",  # strip audio for training
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ⚠ ffmpeg error: {e.stderr.decode()[-200:]}")
        return False


def build_dataset(
    source_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    frames: int,
    fps: float,
    skip_transcode: bool = False,
) -> list[dict]:
    """Build the full training dataset structure.

    Returns list of training pair metadata.
    """
    clips_dir = output_dir / "raw_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    all_clips: list[dict] = []

    for reel_dir in sorted(source_dir.iterdir()):
        if not reel_dir.is_dir():
            continue

        movie_name = REEL_MOVIE_MAP.get(reel_dir.name, reel_dir.name)
        print(f"\n{'='*60}")
        print(f"Reel: {reel_dir.name} — {movie_name}")

        for scene_file in sorted(reel_dir.glob("*-Scene-*.mp4")):
            scene_idx = int(scene_file.stem.split("-")[-1])
            out_name = f"{reel_dir.name}_Scene-{scene_idx:03d}.mp4"
            out_path = clips_dir / out_name

            if skip_transcode:
                # Use source clip directly when skipping transcode
                clip_info = {"reel_id": reel_dir.name, "scene": scene_idx, "path": str(scene_file)}
                all_clips.append(clip_info)
                print(f"  (skipped) {scene_file.name}")
                continue

            print(f"  Transcoding {out_name}...", end=" ")
            success = transcode_to_training_format(
                scene_file, out_path, width, height, frames, fps
            )
            if success:
                clip_info = {"reel_id": reel_dir.name, "scene": scene_idx, "path": str(out_path)}
                all_clips.append(clip_info)
                print(f"✅ {out_path.stat().st_size / 1e6:.1f}MB")
            else:
                print("❌")

    return all_clips


def create_training_pairs(
    all_clips: list[dict],
    output_dir: Path,
) -> list[dict]:
    """Create cross-paired training pairs for OmniTransfer style transfer.

    Strategy: For each reel, cross-pair scenes so the model learns the
    reel's consistent aesthetic as the "style" to transfer.

    Each pair:
      - reference: Scene-A from reel (style source)
      - target:    Scene-B from SAME reel (content to stylize)
      - task:      style_transfer
    """
    pairs = []

    # Group clips by reel
    by_reel: dict[str, list[dict]] = {}
    for clip in all_clips:
        by_reel.setdefault(clip["reel_id"], []).append(clip)

    pair_idx = 0
    for reel_id, clips in by_reel.items():
        if len(clips) < 2:
            continue

        movie_name = REEL_MOVIE_MAP.get(reel_id, reel_id)

        for ref_clip in clips:
            for tgt_clip in clips:
                if ref_clip["scene"] == tgt_clip["scene"]:
                    continue  # Don't pair a scene with itself

                prompt = generate_prompt(reel_id, ref_clip["scene"], tgt_clip["scene"])

                pairs.append({
                    "idx": pair_idx,
                    "file_name": f"{pair_idx:06d}.pt",
                    "reference_file_name": f"{pair_idx:06d}.pt",
                    "media_path": f"{pair_idx:06d}.pt",
                    "reel_id": reel_id,
                    "movie": movie_name,
                    "ref_scene": ref_clip["scene"],
                    "tgt_scene": tgt_clip["scene"],
                    "text": prompt,
                    "task_type": "style_transfer",
                })
                pair_idx += 1

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\n📝 Saved {len(pairs)} training pairs to {metadata_path}")

    return pairs


def generate_prompt(reel_id: str, ref_scene: int, tgt_scene: int) -> str:
    """Generate a text prompt describing the training pair.

    Different prompts for odd/even scene indices since the pattern is:
    odd scenes → Jennifer Connelly (department store)
    even scenes → crossover movie characters (dinner table)
    """
    movie_name = REEL_MOVIE_MAP.get(reel_id, "mashup video")

    # Scene 1, 3, 5, 7 → typically Jennifer Connelly scenes
    if ref_scene % 2 == 1 and tgt_scene % 2 == 1:
        return (
            f"A cinematic scene featuring Jennifer Connelly in a dimly lit "
            f"department store with colorful toys and carnival rides, "
            f"styled as a {movie_name} mashup, 1990s film aesthetic"
        )
    elif ref_scene % 2 == 1 and tgt_scene % 2 == 0:
        return (
            f"A dinner table scene with characters from {movie_name}, "
            f"styled with the visual aesthetic of a Jennifer Connelly "
            f"department store scene, warm film grain, cinematic lighting"
        )
    elif ref_scene % 2 == 0 and tgt_scene % 2 == 1:
        return (
            f"Jennifer Connelly in a department store, riding a mechanical "
            f"horse, wearing a white top and shorts, cinematic lighting, "
            f"style transfer from {movie_name} dinner scene aesthetic"
        )
    else:
        return (
            f"A character from {movie_name} at a dinner table, "
            f"cinematic 1990s film style, warm tones, film grain, "
            f"nostalgic movie atmosphere"
        )


def create_training_config(output_dir: Path, pairs: list[dict]) -> None:
    """Generate the OmniTransfer training config YAML."""
    config = f"""# =============================================================================
# OmniTransfer Style Transfer Config — Mashup Dataset
# =============================================================================
# Auto-generated from {len(pairs)} training pairs
# Generated by prepare_mashup_training_pipeline.py
# =============================================================================

model:
  model_path: "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
  training_mode: lora

lora:
  rank: 32
  alpha: 32

training_strategy:
  name: omnitransfer
  task_type: style_transfer
  enable_tpb: true
  enable_rcl: true
  enable_tma: false

  # Stage 1: Train DiT (LoRA) + TPB + ConceptEmbedding
  training_stage: 1

  # Style loss is CRITICAL for style transfer
  style_loss_weight: 0.5
  use_decoded_pixels_for_style: true
  use_vgg_style_features: true

  # Dynamic Identity Anchoring
  enable_concept_embeddings: true
  concept_embedding_dim: 128
  concept_embedding_task_specific: true

  # Data directories
  reference_latents_dir: reference_latents
  first_frame_latents_dir: first_frame_latents
  i2v_mode: false

  # W&B visualization
  log_reconstructions: true
  reconstruction_log_interval: 200
  log_video_comparisons: false

preprocessed_data_root: "{output_dir}"

optimization:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  optimizer_type: muon
  scheduler_type: cosine
  max_train_steps: 5000
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: bf16
  quantization: int8-quanto
  load_text_encoder_in_8bit: false

validation:
  interval: null  # Skip validation during training
"""
    config_path = output_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        f.write(config)
    print(f"⚙️  Training config saved to {config_path}")


def create_dataset_csv(
    all_clips: list[dict],
    output_dir: Path,
) -> Path:
    """Create CSV for process_videos.py input."""
    csv_path = output_dir / "dataset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["media_path", "caption"])
        for clip in all_clips:
            writer.writerow([clip["path"], f"Scene {clip['scene']} from reel {clip['reel_id']}"])
    print(f"📄 Dataset CSV: {csv_path} ({len(all_clips)} entries)")
    return csv_path


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = args.resolution
    frames = args.frames
    if frames % 8 != 1:
        print(f"⚠️  Adjusting frames from {frames} to {frames + (8 - (frames - 1) % 8)} (must be %8==1)")
        frames = frames + (8 - (frames - 1) % 8)

    print("=" * 60)
    print("🎬 OmniTransfer Mashup Training Pipeline")
    print("=" * 60)
    print(f"Source: {args.source_dir}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {width}×{height}, {frames} frames @ {args.fps}fps")
    print()

    # ── Step 1: Extract character reference frames ────────────────────────
    print("📸 Step 1: Extracting character reference frames...")
    all_refs = []
    for reel_dir in sorted(args.source_dir.iterdir()):
        if not reel_dir.is_dir():
            continue
        movie_name = REEL_MOVIE_MAP.get(reel_dir.name, reel_dir.name)
        refs = extract_character_frames(reel_dir, output_dir)
        all_refs.extend(refs)
        if refs:
            print(f"  {reel_dir.name} ({movie_name}): {len(refs)} character refs")

    print(f"   Total: {len(all_refs)} character reference frames")
    char_ref_dir = output_dir / "character_refs"
    print(f"   Saved to: {char_ref_dir}/")

    if args.extract_refs_only:
        print("\n✅ Reference extraction complete (--extract-refs-only)")
        return

    # ── Step 2: Build dataset (transcode clips to training format) ────────
    print(f"\n🎞️  Step 2: Processing {len(list(args.source_dir.iterdir()))} reels...")
    all_clips = build_dataset(
        args.source_dir, output_dir, width, height, frames, args.fps, args.skip_transcode,
    )
    print(f"   Total clips processed: {len(all_clips)}")

    if not all_clips:
        print("❌ No clips processed! Check source directory.")
        sys.exit(1)

    # ── Step 3: Create training pairs ─────────────────────────────────────
    print("\n🔗 Step 3: Creating cross-paired training pairs...")
    pairs = create_training_pairs(all_clips, output_dir)

    # ── Step 4: Create dataset CSV for VAE encoding ───────────────────────
    print("\n📄 Step 4: Creating dataset CSV...")
    csv_path = create_dataset_csv(all_clips, output_dir)

    # ── Step 5: Generate training config ─────────────────────────────────
    print("\n⚙️  Step 5: Generating training config...")
    create_training_config(output_dir, pairs)

    # ── Step 6: Print next steps ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ PIPELINE READY!")
    print("=" * 60)
    print()
    print("Next steps:")
    print()
    print("  1. Encode scene clips to VAE latents:")
    print(f"     uv run python scripts/process_videos.py {csv_path} \\")
    print(f"       --resolution-buckets {width}x{height}x{frames} \\")
    print(f"       --output-dir {output_dir / 'encoded_latents'} \\")
    print(f"       --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
    print()
    print("  2. Compute text embeddings (Gemma only, ~28GB):")
    print(f"     uv run python scripts/compute_text_embeddings.py \\")
    print(f"       --output-dir {output_dir} \\")
    print(f"       --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \\")
    print(f"       --text-encoder-path /media/2TB/ltx-models/gemma-3-12b-it")
    print()
    print("  3. Organize latents into training structure:")
    print(f"     # Copy encoded latents to:")
    print(f"     #   {output_dir}/latents/           (target)")
    print(f"     #   {output_dir}/reference_latents/ (reference)")
    print(f"     #   {output_dir}/conditions/        (text embeddings)")
    print()
    print("  4. Train:")
    print(f"     uv run python scripts/train.py {output_dir / 'train_config.yaml'}")
    print()
    print(f"  Character reference frames: {char_ref_dir}/")
    print("  You can use these as additional identity references")
    print("  by encoding them with scripts/encode_single_image.py")
    print()


if __name__ == "__main__":
    main()
