#!/usr/bin/env python3
"""Generate diorama videos from images using LTX-2 Image-to-Video.

Uses the base LTX-2 model's I2V capability to animate diorama images into
short videos. No Grok API needed — everything runs locally on GPU.

The output videos are then encoded to multi-frame VAE latents and assembled
into a cross-paired training dataset alongside existing image latents.

Pipeline:
  Phase 1: Generate I2V videos from diorama images (LTX-2 inference)
  Phase 2: Encode videos to multi-frame VAE latents
  Phase 3: Create cross-paired video training dataset

Key optimization: Uses pre-computed text embeddings (CachedPromptEmbeddings)
from the image dataset, completely skipping the 28GB Gemma text encoder.

VRAM budget (Phase 1):
  - Transformer bf16: ~20GB (on GPU during denoising)
  - VAE encoder: ~8GB (moved on/off GPU by sampler)
  - VAE decoder: ~8GB (moved on/off GPU by sampler)
  - Peak: ~28GB (transformer + VAE decoder during final decode)

Usage:
    # Generate videos + encode + cross-pair
    cd ~/Documents/GitHub/ltx2-omnitransfer/packages/ltx-trainer
    uv run python scripts/generate_diorama_videos.py \
        --output-dir /media/2TB/grok_diorama_video_crosspair

    # Encode-only mode (if videos already generated externally)
    uv run python scripts/generate_diorama_videos.py \
        --encode-only \
        --video-dir /path/to/existing/videos \
        --output-dir /media/2TB/grok_diorama_video_crosspair
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add scripts dir to path for SCENE_DESCRIPTIONS import
sys.path.insert(0, str(Path(__file__).parent))

from ltx_trainer import logger
from ltx_trainer.model_loader import load_model, load_video_vae_encoder
from ltx_trainer.validation_sampler import (
    CachedPromptEmbeddings,
    GenerationConfig,
    ValidationSampler,
)

# Import scene descriptions from the existing Grok data generation script
from generate_grok_training_data import SCENE_DESCRIPTIONS

TARGET_WIDTH = 832
TARGET_HEIGHT = 448
NUM_FRAMES = 25  # 25 frames -> 4 latent frames after 8x temporal compression
INFERENCE_STEPS = 30  # Good quality for training data
GUIDANCE_SCALE = 1.0  # No CFG (we don't have negative prompt embeddings)
STG_SCALE = 1.0  # Spatiotemporal guidance for quality (doesn't need negative prompt)


def collect_diorama_images(
    movie_dir: Path,
    grok_dir: Path | None = None,
) -> dict[str, Path]:
    """Collect one best diorama image per scene for I2V generation.

    Prefers: original diorama > Grok-generated > flipped variants.
    Returns dict[scene_name -> image_path].
    """
    scene_images: dict[str, Path] = {}

    # 1. Original dioramas (highest quality — artist-created)
    if movie_dir.exists():
        for scene_dir in sorted(movie_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name
            if scene not in SCENE_DESCRIPTIONS:
                continue
            dioramas = sorted(scene_dir.glob("diorama.*"))
            if dioramas:
                scene_images[scene] = dioramas[0]

    # 2. Grok-generated images (fill gaps)
    if grok_dir and (grok_dir / "images").exists():
        for f in sorted((grok_dir / "images").iterdir()):
            if not f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            # Extract scene name (remove _grok_v0, _existing_diorama, etc.)
            stem = f.stem
            for suffix in ("_grok_v0", "_existing_diorama"):
                stem = stem.replace(suffix, "")
            if stem in SCENE_DESCRIPTIONS and stem not in scene_images:
                scene_images[stem] = f

    return scene_images


def load_image_for_i2v(image_path: Path) -> torch.Tensor:
    """Load and prepare image for I2V conditioning.

    Returns tensor [C, H, W] in [0, 1] range (as expected by ValidationSampler).
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target_aspect = TARGET_WIDTH / TARGET_HEIGHT
    source_aspect = w / h

    # Center-crop to target aspect ratio
    if abs(source_aspect - target_aspect) > 0.01:
        if source_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img.crop((start_x, 0, start_x + new_w, h))
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img.crop((0, start_y, w, start_y + new_h))

    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    return transforms.ToTensor()(img)  # [C, H, W] in [0, 1]


def load_scene_embeddings(
    image_dataset_dir: Path,
) -> dict[str, Path]:
    """Load pre-computed text embeddings from image dataset.

    Returns dict[scene_name -> embedding_file_path].
    Maps each scene to exactly one conditions_final/*.pt file.
    """
    meta_path = image_dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json at {image_dataset_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    cond_dir = image_dataset_dir / "conditions_final"
    scene_to_emb: dict[str, Path] = {}

    for pair in meta["pairs"]:
        scene = pair["tgt_scene"]
        if scene in scene_to_emb:
            continue
        emb_path = cond_dir / f"{pair['id']}.pt"
        # Follow symlinks to real file
        if emb_path.is_symlink():
            emb_path = emb_path.resolve()
        if emb_path.exists():
            scene_to_emb[scene] = emb_path

    return scene_to_emb


def read_video_frames(video_path: Path, max_frames: int = NUM_FRAMES) -> torch.Tensor:
    """Read video frames using PyAV.

    Returns tensor [C, F, H, W] in [-1, 1] ready for VAE encoding.
    """
    import av

    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            if len(frames) >= max_frames:
                break
            arr = frame.to_ndarray(format="rgb24")
            frames.append(arr)

    if not frames:
        raise ValueError(f"No frames in {video_path}")

    # Stack: [F, H, W, C] -> [F, C, H, W]
    video = torch.from_numpy(np.stack(frames)).float().div(255.0)
    video = video.permute(0, 3, 1, 2)  # [F, C, H, W]

    # Resize to target resolution
    if video.shape[2] != TARGET_HEIGHT or video.shape[3] != TARGET_WIDTH:
        video = torch.nn.functional.interpolate(
            video,
            size=(TARGET_HEIGHT, TARGET_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

    # Permute to [C, F, H, W] and normalize to [-1, 1]
    video = video.permute(1, 0, 2, 3)  # [C, F, H, W]
    video = video * 2.0 - 1.0
    return video


# ---------------------------------------------------------------------------
# Phase 1: Generate I2V videos using pre-computed embeddings (no text encoder)
# ---------------------------------------------------------------------------


def phase1_generate_videos(
    scene_images: dict[str, Path],
    scene_embeddings: dict[str, Path],
    video_output_dir: Path,
    model_path: Path,
    device: str = "cuda:0",
    max_videos: int | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate I2V videos from diorama images using LTX-2.

    Uses CachedPromptEmbeddings to skip the 28GB text encoder entirely.
    Returns dict[scene_name -> video_path].
    """
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to scenes that have both images and embeddings
    scenes_ready = {
        s: img
        for s, img in scene_images.items()
        if s in scene_embeddings
    }

    # Check what's already generated
    scenes_to_generate = {}
    for scene, img_path in scenes_ready.items():
        video_path = video_output_dir / f"{scene}.mp4"
        if not video_path.exists():
            scenes_to_generate[scene] = img_path

    if not scenes_to_generate:
        logger.info("All videos already generated, skipping Phase 1")
        return {s: video_output_dir / f"{s}.mp4" for s in scenes_ready}

    if max_videos:
        scenes_to_generate = dict(list(scenes_to_generate.items())[:max_videos])

    logger.info(
        f"Phase 1: Generating {len(scenes_to_generate)} I2V videos "
        f"(skipping text encoder — using cached embeddings)"
    )

    # Load model WITHOUT text encoder (saves 28GB VRAM)
    logger.info("Loading LTX-2 model (transformer + VAE, no text encoder)...")
    components = load_model(
        checkpoint_path=model_path,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=True,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=False,
        text_encoder_path=None,
    )

    sampler = ValidationSampler(
        transformer=components.transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=None,
        audio_decoder=None,
        vocoder=None,
    )

    generated = {}
    for i, (scene, img_path) in enumerate(
        tqdm(scenes_to_generate.items(), desc="Generating I2V")
    ):
        video_path = video_output_dir / f"{scene}.mp4"

        try:
            # Load pre-computed embedding for this scene
            emb_data = torch.load(
                scene_embeddings[scene], map_location="cpu", weights_only=True
            )
            cached = CachedPromptEmbeddings(
                video_context_positive=emb_data["video_prompt_embeds"].unsqueeze(0),
                audio_context_positive=emb_data["audio_prompt_embeds"].unsqueeze(0),
                video_context_negative=None,  # No CFG (guidance_scale=1.0)
                audio_context_negative=None,
            )

            condition_image = load_image_for_i2v(img_path)

            gen_config = GenerationConfig(
                prompt="",  # Ignored when cached_embeddings provided
                negative_prompt="",
                height=TARGET_HEIGHT,
                width=TARGET_WIDTH,
                num_frames=NUM_FRAMES,
                frame_rate=25.0,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                seed=seed + i,  # Different seed per video for variety
                condition_image=condition_image,
                cached_embeddings=cached,
                generate_audio=False,
                stg_scale=STG_SCALE,
                stg_blocks=[29],
                stg_mode="stg_v",
            )

            video_output, _ = sampler.generate(config=gen_config, device=device)

            # save_video expects [F, C, H, W] or [C, F, H, W] — check format
            from ltx_trainer.video_utils import save_video

            save_video(video_tensor=video_output, output_path=video_path, fps=25.0)
            generated[scene] = video_path
            logger.info(f"  [{i+1}/{len(scenes_to_generate)}] {scene}: saved")

        except Exception as e:
            logger.error(f"  [{i+1}/{len(scenes_to_generate)}] {scene}: FAILED - {e}")
            import traceback

            traceback.print_exc()
            continue

    # Cleanup
    del sampler, components
    torch.cuda.empty_cache()
    gc.collect()

    # Merge with already-existing videos
    all_videos = {}
    for scene in scenes_ready:
        vp = video_output_dir / f"{scene}.mp4"
        if vp.exists():
            all_videos[scene] = vp

    logger.info(f"Phase 1 complete: {len(all_videos)} total videos")
    return all_videos


# ---------------------------------------------------------------------------
# Phase 2: Encode videos to multi-frame VAE latents
# ---------------------------------------------------------------------------


def phase2_encode_videos(
    scene_videos: dict[str, Path],
    cache_dir: Path,
    model_path: Path,
    device: str = "cuda:1",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Path]:
    """Encode video files to multi-frame VAE latents.

    Returns dict[scene_name -> latent_path].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    needs_encoding = {}
    scene_latents = {}
    for scene, video_path in scene_videos.items():
        latent_path = cache_dir / f"{scene}.pt"
        scene_latents[scene] = latent_path
        if not latent_path.exists():
            needs_encoding[scene] = video_path

    if not needs_encoding:
        logger.info(f"All {len(scene_latents)} video latents cached")
        return scene_latents

    logger.info(f"Phase 2: Encoding {len(needs_encoding)} videos to latents on {device}...")

    vae_encoder = load_video_vae_encoder(model_path, dtype=dtype)
    vae_encoder = vae_encoder.to(device)
    vae_encoder.eval()

    for scene, video_path in tqdm(needs_encoding.items(), desc="Encoding videos"):
        try:
            # Read video frames: [C, F, H, W] in [-1, 1]
            video_tensor = read_video_frames(video_path, max_frames=NUM_FRAMES)

            # Ensure frame count satisfies F % 8 == 1
            num_frames = video_tensor.shape[1]
            valid_frames = ((num_frames - 1) // 8) * 8 + 1
            if valid_frames < 1:
                valid_frames = 1
            video_tensor = video_tensor[:, :valid_frames]

            # Add batch dim: [1, C, F, H, W]
            video_tensor = video_tensor.unsqueeze(0).to(device, dtype=dtype)

            with torch.inference_mode():
                latent = vae_encoder(video_tensor)

            # Save: latent shape [C, F_lat, H_lat, W_lat]
            latent_squeezed = latent.squeeze(0).cpu()
            latent_data = {
                "latents": latent_squeezed,
                "num_frames": torch.tensor([valid_frames]),
                "height": torch.tensor([TARGET_HEIGHT]),
                "width": torch.tensor([TARGET_WIDTH]),
            }
            torch.save(latent_data, cache_dir / f"{scene}.pt")
            logger.info(
                f"  {scene}: [{video_tensor.shape[1]} frames] -> latent {latent_squeezed.shape}"
            )

        except Exception as e:
            logger.error(f"  {scene}: FAILED - {e}")
            scene_latents.pop(scene, None)
            continue

    del vae_encoder
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Phase 2 complete: {len(scene_latents)} video latents")
    return scene_latents


# ---------------------------------------------------------------------------
# Phase 3: Create cross-paired video training dataset
# ---------------------------------------------------------------------------


def phase3_create_crosspairs(
    scene_latents: dict[str, Path],
    output_dir: Path,
    image_dataset_dir: Path | None = None,
    refs_per_target: int = 2,
    seed: int = 42,
) -> int:
    """Create cross-paired video training dataset.

    For each target scene, pairs it with `refs_per_target` different reference
    scenes (ref != target). Reuses text embeddings from the image dataset
    via symlinks.
    """
    rng = random.Random(seed)
    valid_scenes = sorted(
        s
        for s in scene_latents
        if s in SCENE_DESCRIPTIONS and scene_latents[s].exists()
    )
    logger.info(f"Phase 3: Creating video cross-pairs from {len(valid_scenes)} scenes")

    latents_dir = output_dir / "latents"
    ref_dir = output_dir / "reference_latents"
    cond_dir = output_dir / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)

    # Build scene->embedding mapping from image dataset
    scene_to_emb: dict[str, Path] = {}
    if image_dataset_dir and (image_dataset_dir / "conditions_final").exists():
        scene_to_emb = load_scene_embeddings(image_dataset_dir)
        logger.info(f"  Reusing {len(scene_to_emb)} embeddings from image dataset")

    pairs = []
    pair_idx = 0
    for tgt_scene in valid_scenes:
        ref_candidates = [s for s in valid_scenes if s != tgt_scene]
        if not ref_candidates:
            continue

        chosen_refs = rng.sample(
            ref_candidates, k=min(refs_per_target, len(ref_candidates))
        )

        for ref_scene in chosen_refs:
            tgt_latent_path = scene_latents[tgt_scene]
            ref_latent_path = scene_latents[ref_scene]

            if not tgt_latent_path.exists() or not ref_latent_path.exists():
                continue

            # Symlink latents (avoid copying large files)
            tgt_link = latents_dir / f"{pair_idx}.pt"
            ref_link = ref_dir / f"{pair_idx}.pt"

            # Remove stale links
            for link in (tgt_link, ref_link):
                if link.exists() or link.is_symlink():
                    link.unlink()

            os.symlink(tgt_latent_path.resolve(), tgt_link)
            os.symlink(ref_latent_path.resolve(), ref_link)

            # Symlink embedding from image dataset
            cond_link = cond_dir / f"{pair_idx}.pt"
            if cond_link.exists() or cond_link.is_symlink():
                cond_link.unlink()

            if tgt_scene in scene_to_emb:
                os.symlink(scene_to_emb[tgt_scene].resolve(), cond_link)
            else:
                logger.warning(f"  No embedding for scene '{tgt_scene}' — skipping pair")
                tgt_link.unlink()
                ref_link.unlink()
                continue

            pairs.append(
                {
                    "id": pair_idx,
                    "ref_scene": ref_scene,
                    "tgt_scene": tgt_scene,
                    "caption": SCENE_DESCRIPTIONS[tgt_scene],
                }
            )
            pair_idx += 1

    metadata = {
        "task_type": "style_transfer",
        "data_type": "video",
        "num_pairs": len(pairs),
        "num_scenes": len(valid_scenes),
        "frames_per_video": NUM_FRAMES,
        "resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        "pairs": pairs,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Phase 3 complete: {len(pairs)} video cross-pairs")
    return len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diorama videos with LTX-2 I2V and encode for training"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/media/2TB/movie_dioramas"),
        help="Original movie diorama images (scene_name/diorama.png)",
    )
    parser.add_argument(
        "--grok-dir",
        type=Path,
        default=Path("/media/2TB/grok_training_data"),
        help="Grok-generated images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/2TB/grok_diorama_video_crosspair"),
        help="Output directory for video cross-paired dataset",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"),
    )
    parser.add_argument(
        "--image-dataset-dir",
        type=Path,
        default=Path("/media/2TB/grok_diorama_crosspair"),
        help="Image dataset to reuse text embeddings from",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Pre-existing video directory (skip Phase 1)",
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Skip video generation, only encode existing videos",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae-device", type=str, default="cuda:1")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--refs-per-target", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=25,
        help="Frames per video (must satisfy F %% 8 == 1). Default: 25",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without doing it",
    )
    args = parser.parse_args()

    global NUM_FRAMES
    NUM_FRAMES = args.num_frames

    # Validate frame count
    if (NUM_FRAMES - 1) % 8 != 0:
        valid = ((NUM_FRAMES - 1) // 8) * 8 + 1
        logger.warning(
            f"Frame count {NUM_FRAMES} doesn't satisfy F%%8==1, adjusting to {valid}"
        )
        NUM_FRAMES = valid

    video_dir = args.video_dir or (args.output_dir / "i2v_videos")

    # Load scene embeddings from image dataset
    scene_embeddings = load_scene_embeddings(args.image_dataset_dir)
    logger.info(f"Loaded embeddings for {len(scene_embeddings)} scenes")

    if args.dry_run:
        scene_images = collect_diorama_images(args.image_dir, args.grok_dir)
        ready = {s for s in scene_images if s in scene_embeddings}
        logger.info(f"\n--- DRY RUN ---")
        logger.info(f"Diorama images found: {len(scene_images)}")
        logger.info(f"Embeddings available: {len(scene_embeddings)}")
        logger.info(f"Ready to generate: {len(ready)}")
        for s in sorted(ready):
            logger.info(f"  {s}: {scene_images[s].name}")
        logger.info(f"Missing embeddings: {sorted(set(scene_images) - ready)}")
        cross_pairs = len(ready) * (len(ready) - 1) * args.refs_per_target // len(ready)
        logger.info(f"Estimated cross-pairs: ~{cross_pairs}")
        return

    # Phase 1: Generate videos (or skip)
    if args.encode_only:
        if not args.video_dir:
            parser.error("--video-dir required with --encode-only")
        scene_videos: dict[str, Path] = {}
        for f in sorted(args.video_dir.glob("*.mp4")):
            scene = f.stem
            if scene in SCENE_DESCRIPTIONS:
                scene_videos[scene] = f
        logger.info(f"Encode-only: found {len(scene_videos)} videos")
    else:
        scene_images = collect_diorama_images(args.image_dir, args.grok_dir)
        logger.info(f"Collected {len(scene_images)} scenes with diorama images")

        scene_videos = phase1_generate_videos(
            scene_images,
            scene_embeddings,
            video_dir,
            args.model_path,
            device=args.device,
            max_videos=args.max_videos,
            seed=args.seed,
        )

    if not scene_videos:
        logger.error("No videos to process!")
        return

    # Phase 2: Encode videos to VAE latents
    cache_dir = args.output_dir / "video_latent_cache"
    scene_latents = phase2_encode_videos(
        scene_videos,
        cache_dir,
        args.model_path,
        device=args.vae_device,
    )

    # Phase 3: Create cross-paired dataset
    num_pairs = phase3_create_crosspairs(
        scene_latents,
        args.output_dir,
        image_dataset_dir=args.image_dataset_dir,
        refs_per_target=args.refs_per_target,
        seed=args.seed,
    )

    logger.info(f"\nDONE: {num_pairs} video cross-pairs at {args.output_dir}")
    logger.info(f"Next: train with video data using a config pointing to {args.output_dir}")


if __name__ == "__main__":
    main()
