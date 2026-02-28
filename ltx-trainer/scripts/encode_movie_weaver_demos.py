#!/usr/bin/env python3
"""Encode Movie Weaver demos into training latents.

This script processes the downloaded Movie Weaver demos with split references:
- split_refs/R1.png → reference_latents/R1/{sample}.pt
- split_refs/R2.png → reference_latents/R2/{sample}.pt
- videos/*.mp4 → latents/{sample}.pt (ground truth result)

For Movie Weaver multi-concept training:
- Multiple reference images per sample (R1, R2, R3, R4)
- Concept assignments track which refs belong to same concept
- Prompts contain anchored tokens [R1], [R2], etc.

Usage:
    python scripts/encode_movie_weaver_demos.py \
        --input-dir /media/2TB/movie_weaver_demos \
        --output-dir /media/2TB/movie_weaver_training \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder, load_text_encoder
from ltx_trainer.video_utils import read_video


# Target dimensions (must be divisible by 32)
TARGET_H = 512
TARGET_W = 512


def load_image_as_latent_input(path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load image and prepare for VAE encoding [1, C, 1, H, W]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Center crop to square if needed
    if w != h:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

    # Resize to target
    img = img.resize((target_w, target_h), Image.LANCZOS)

    # Convert to tensor [C, H, W] in [0, 1]
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # Format as [1, C, 1, H, W] for VAE (single frame)
    tensor = tensor.unsqueeze(0).unsqueeze(2)

    # Normalize to [-1, 1]
    tensor = tensor * 2.0 - 1.0

    return tensor


def load_video_frames(path: Path, max_frames: int, target_h: int, target_w: int) -> tuple[torch.Tensor, float]:
    """Load video and prepare for VAE encoding [1, C, F, H, W]."""
    frames, fps = read_video(path, max_frames=max_frames)  # [F, C, H, W]

    # Center crop to square
    _, _, h, w = frames.shape
    if w != h:
        min_dim = min(w, h)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        frames = frames[:, :, start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Resize to target
    frames = F.interpolate(frames, size=(target_h, target_w), mode="bilinear", align_corners=False)

    # Trim to valid frame count (k*8 + 1)
    valid_frames = (frames.shape[0] - 1) // 8 * 8 + 1
    valid_frames = min(valid_frames, max_frames)
    frames = frames[:valid_frames]

    # Convert to [B, C, F, H, W]
    frames = rearrange(frames, "f c h w -> 1 c f h w")

    # Normalize to [-1, 1]
    frames = frames * 2.0 - 1.0

    return frames, fps


def encode_with_vae(vae_encoder, tensor: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Encode tensor with VAE."""
    tensor = tensor.to(device, dtype=dtype)
    with torch.inference_mode():
        latent = vae_encoder(tensor)
    return latent.cpu()


def encode_text(text_encoder, prompt: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Encode text prompt."""
    with torch.inference_mode():
        embeds = text_encoder.encode_prompt(
            prompt,
            device=device,
            dtype=dtype,
        )
    return embeds.cpu()


def process_demo(
    demo_dir: Path,
    vae_encoder,
    text_encoder,
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
    max_frames: int,
    sample_offset: int,
) -> list[dict]:
    """Process a single Movie Weaver demo directory.

    Returns list of processed sample metadata.
    """
    metadata_path = demo_dir / "metadata.json"
    split_dir = demo_dir / "split_refs"
    videos_dir = demo_dir / "videos"

    if not all(p.exists() for p in [metadata_path, split_dir]):
        logger.warning(f"Skipping {demo_dir.name}: missing required files")
        return []

    metadata = json.loads(metadata_path.read_text())
    prompt = metadata["prompt"]
    refs = metadata["refs"]
    concept_assignments = metadata["concept_assignments"]

    logger.info(f"Processing: {demo_dir.name}")
    logger.info(f"  Refs: {refs}")
    logger.info(f"  Concepts: {concept_assignments}")

    # Get list of split metadata entries
    split_meta_path = split_dir / "split_metadata.json"
    if not split_meta_path.exists():
        logger.warning(f"  No split metadata found")
        return []

    split_meta = json.loads(split_meta_path.read_text())
    samples = []

    # Process each image set (multiple composite images -> multiple training samples)
    for idx, split_entry in enumerate(split_meta.get("split_images", [])):
        sample_idx = sample_offset + idx
        logger.info(f"  Sample {sample_idx}: {split_entry['original']}")

        # Encode each reference image
        ref_latents = {}
        for ref_name, ref_info in split_entry["references"].items():
            ref_path = Path(ref_info["path"])
            if not ref_path.exists():
                logger.warning(f"    Missing ref: {ref_path}")
                continue

            logger.info(f"    Encoding {ref_name}: {ref_path.name}")
            img_tensor = load_image_as_latent_input(ref_path, TARGET_H, TARGET_W)
            ref_latent = encode_with_vae(vae_encoder, img_tensor, device, dtype)
            ref_latents[ref_name] = ref_latent

            # Save individual reference latent
            ref_output_dir = output_dir / f"reference_latents_{ref_name}"
            ref_output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ref_latent, ref_output_dir / f"{sample_idx}.pt")

        # Also save combined reference for multi-concept training
        if ref_latents:
            combined_refs = {
                "latents": {k: v for k, v in ref_latents.items()},
                "concept_assignments": concept_assignments,
                "ref_descriptions": {k: refs.get(k, "") for k in ref_latents.keys()},
            }
            multi_ref_dir = output_dir / "multi_concept_refs"
            multi_ref_dir.mkdir(parents=True, exist_ok=True)
            torch.save(combined_refs, multi_ref_dir / f"{sample_idx}.pt")

        # Find corresponding result video
        orig_name = Path(split_entry["original"]).stem  # e.g., "facebody_1"
        video_candidates = list(videos_dir.glob(f"{orig_name}*.mp4")) if videos_dir.exists() else []

        gt_latent = None
        if video_candidates:
            video_path = video_candidates[0]
            logger.info(f"    Encoding GT video: {video_path.name}")
            try:
                video_tensor, fps = load_video_frames(video_path, max_frames, TARGET_H, TARGET_W)
                gt_latent = encode_with_vae(vae_encoder, video_tensor, device, dtype)

                # Save ground truth latent
                gt_dir = output_dir / "latents"
                gt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(gt_latent, gt_dir / f"{sample_idx}.pt")
            except Exception as e:
                logger.warning(f"    Failed to encode video: {e}")

        # Encode text prompt (if text encoder provided)
        text_embeds = None
        if text_encoder is not None:
            logger.info(f"    Encoding prompt...")
            text_embeds = encode_text(text_encoder, prompt, device, dtype)
            text_dir = output_dir / "text_embeddings"
            text_dir.mkdir(parents=True, exist_ok=True)
            torch.save(text_embeds, text_dir / f"{sample_idx}.pt")
        else:
            # Save prompt text for later encoding
            prompt_dir = output_dir / "prompts"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dir / f"{sample_idx}.txt").write_text(prompt)

        # Build sample metadata
        sample_meta = {
            "idx": sample_idx,
            "demo": demo_dir.name,
            "prompt": prompt,
            "refs": refs,
            "concept_assignments": concept_assignments,
            "has_gt_video": gt_latent is not None,
            "num_refs": len(ref_latents),
        }
        samples.append(sample_meta)

        # Memory cleanup
        del ref_latents, gt_latent
        if text_embeds is not None:
            del text_embeds
        gc.collect()
        torch.cuda.empty_cache()

    return samples


def main():
    parser = argparse.ArgumentParser(description="Encode Movie Weaver demos")
    parser.add_argument("--input-dir", type=Path, default=Path("/media/2TB/movie_weaver_demos"))
    parser.add_argument("--output-dir", type=Path, default=Path("/media/2TB/movie_weaver_training"))
    parser.add_argument("--model-path", type=Path, default=Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"))
    parser.add_argument("--text-encoder-path", type=Path, default=Path("/media/2TB/ltx-models/gemma"))
    parser.add_argument("--skip-text-encoding", action="store_true", help="Skip text encoding (use later)")
    parser.add_argument("--max-frames", type=int, default=33, help="Max frames (must be k*8+1)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    # Validate frame count
    if args.max_frames % 8 != 1:
        raise ValueError(f"max_frames must be k*8+1, got {args.max_frames}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    logger.info("=" * 60)
    logger.info("Movie Weaver Demo Encoder")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Max frames: {args.max_frames}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    logger.info("Loading VAE encoder...")
    vae_encoder = load_video_vae_encoder(args.model_path)
    vae_encoder = vae_encoder.to(args.device, dtype=dtype)
    vae_encoder.eval()

    text_encoder = None
    if not args.skip_text_encoding:
        logger.info("Loading text encoder...")
        text_encoder = load_text_encoder(args.model_path, args.text_encoder_path, device=args.device, dtype=dtype)
    else:
        logger.info("Skipping text encoder (--skip-text-encoding)")

    # Process all demos
    all_samples = []
    sample_offset = 0

    for demo_dir in sorted(args.input_dir.iterdir()):
        if demo_dir.is_dir() and (demo_dir / "metadata.json").exists():
            samples = process_demo(
                demo_dir, vae_encoder, text_encoder,
                args.output_dir, args.device, dtype,
                args.max_frames, sample_offset
            )
            all_samples.extend(samples)
            sample_offset += len(samples)

    # Save master manifest
    manifest = {
        "samples": all_samples,
        "total_samples": len(all_samples),
        "encoding_config": {
            "target_h": TARGET_H,
            "target_w": TARGET_W,
            "max_frames": args.max_frames,
            "dtype": args.dtype,
        },
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info("\n" + "=" * 60)
    logger.info("ENCODING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Manifest: {manifest_path}")

    # Print reference latent directories
    for ref_dir in sorted(args.output_dir.glob("reference_latents_*")):
        count = len(list(ref_dir.glob("*.pt")))
        logger.info(f"  {ref_dir.name}: {count} files")


if __name__ == "__main__":
    main()
