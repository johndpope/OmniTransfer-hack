#!/usr/bin/env python3
"""Prepare dataset for OmniTransfer training.

This script processes video pairs (reference and target) for OmniTransfer training:
1. Encodes reference videos to latents (stored in reference_latents/)
2. Encodes target videos to latents (stored in latents/)
3. Computes text embeddings from captions (stored in conditions/)

Dataset structure expected:
    input_dir/
        pairs.json           # JSON file mapping reference to target videos
        videos/
            ref_001.mp4
            tgt_001.mp4
            ...

Output structure:
    output_dir/
        reference_latents/
            ref_001.safetensors
            ...
        latents/
            tgt_001.safetensors
            ...
        conditions/
            tgt_001.safetensors
            ...

Quote: "We collected our own data sets from the Internet to support
spatio-temporal video transfer tasks." (Section 5.1, OmniTransfer paper)

Usage:
    python scripts/prepare_omnitransfer_dataset.py \\
        --input-dir /path/to/raw_dataset \\
        --output-dir /path/to/processed_dataset \\
        --model-path /path/to/ltx2_model.safetensors \\
        --text-encoder-path /path/to/gemma-3-12b-it
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import torch.nn.functional as F
from einops import rearrange

from ltx_trainer import logger
from ltx_trainer.model_loader import (
    load_video_vae_encoder,
    load_text_encoder,
)
from ltx_trainer.video_utils import read_video


def resize_video_for_vae(video: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
    """Resize video tensor to target dimensions for VAE encoding.

    Args:
        video: Video tensor [F, C, H, W] in range [0, 1]
        target_width: Target width (must be divisible by 32)
        target_height: Target height (must be divisible by 32)

    Returns:
        Resized video tensor [F, C, H, W]
    """
    # video is [F, C, H, W]
    f, c, h, w = video.shape

    # Compute resize dimensions preserving aspect ratio
    aspect_ratio = w / h
    target_aspect = target_width / target_height

    if aspect_ratio > target_aspect:
        # Wider - resize to target height, crop width
        resize_h = target_height
        resize_w = int(target_height * aspect_ratio)
    else:
        # Taller - resize to target width, crop height
        resize_w = target_width
        resize_h = int(target_width / aspect_ratio)

    # Resize
    video = F.interpolate(
        video, size=(resize_h, resize_w), mode="bilinear", align_corners=False
    )

    # Center crop
    h_start = (resize_h - target_height) // 2
    w_start = (resize_w - target_width) // 2
    video = video[:, :, h_start:h_start + target_height, w_start:w_start + target_width]

    return video


def prepare_video_for_vae(video: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Convert video from [F, C, H, W] to [B, C, F, H, W] format for VAE.

    Args:
        video: Video tensor [F, C, H, W] in range [0, 1]
        max_frames: Maximum frames (must satisfy frames % 8 == 1)

    Returns:
        Video tensor [B, C, F, H, W] in range [-1, 1]
    """
    # Trim to valid frame count (k*8 + 1)
    valid_frames = (video.shape[0] - 1) // 8 * 8 + 1
    valid_frames = min(valid_frames, max_frames)
    video = video[:valid_frames]

    # Convert from [F, C, H, W] to [B, C, F, H, W]
    video = rearrange(video, "f c h w -> 1 c f h w")

    # Normalize from [0, 1] to [-1, 1] for VAE
    video = video * 2.0 - 1.0

    return video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for OmniTransfer training"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing videos and pairs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for processed latents",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to LTX-2 model checkpoint",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=Path,
        required=True,
        help="Path to Gemma text encoder",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=960,
        help="Target video width (default: 960)",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=544,
        help="Target video height (default: 544)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=97,
        help="Maximum frames per video (default: 97, must satisfy frames %% 8 == 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type (default: bfloat16)",
    )
    return parser.parse_args()


def stage1_encode_videos(
    args,
    pairs: list[dict],
    ref_latents_dir: Path,
    tgt_latents_dir: Path,
    dtype: torch.dtype,
) -> None:
    """Stage 1: Encode all videos to latents using VAE encoder.

    Loads only the VAE encoder (~5GB) to fit in limited VRAM.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Encoding videos to latents")
    logger.info("=" * 60)

    # Load VAE encoder
    logger.info("Loading VAE encoder...")
    vae_encoder = load_video_vae_encoder(args.model_path, dtype=dtype)
    vae_encoder = vae_encoder.to(args.device)
    vae_encoder.eval()

    # Collect all unique videos to process
    videos_to_process = []
    for pair in pairs:
        ref_name = Path(pair["reference"]).stem
        tgt_name = Path(pair["target"]).stem
        ref_output = ref_latents_dir / f"{ref_name}.safetensors"
        tgt_output = tgt_latents_dir / f"{tgt_name}.safetensors"

        if not ref_output.exists():
            videos_to_process.append({
                "video_path": args.input_dir / "videos" / pair["reference"],
                "output_path": ref_output,
                "name": ref_name,
            })
        if not tgt_output.exists():
            videos_to_process.append({
                "video_path": args.input_dir / "videos" / pair["target"],
                "output_path": tgt_output,
                "name": tgt_name,
            })

    # Remove duplicates (same reference used in multiple pairs)
    seen = set()
    unique_videos = []
    for v in videos_to_process:
        if v["name"] not in seen:
            seen.add(v["name"])
            unique_videos.append(v)

    logger.info(f"Encoding {len(unique_videos)} unique videos to latents")

    for video_info in tqdm(unique_videos, desc="Encoding videos"):
        try:
            frames, _ = read_video(
                video_info["video_path"],
                max_frames=args.max_frames,
            )
            frames = resize_video_for_vae(
                frames,
                target_width=args.target_width,
                target_height=args.target_height,
            )
            frames = prepare_video_for_vae(frames, args.max_frames)
            frames = frames.to(args.device, dtype=dtype)

            with torch.inference_mode():
                latent = vae_encoder(frames)

            # Save latent
            torch.save(
                {
                    "latents": latent.cpu(),
                    "num_frames": torch.tensor([latent.shape[2]]),
                    "height": torch.tensor([latent.shape[3]]),
                    "width": torch.tensor([latent.shape[4]]),
                },
                video_info["output_path"],
            )

            # Clear CUDA cache after each video
            del frames, latent
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error encoding {video_info['name']}: {e}")
            continue

    # Unload VAE encoder
    del vae_encoder
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    logger.info("VAE encoder unloaded, VRAM freed")


def stage2_compute_embeddings(
    args,
    pairs: list[dict],
    conditions_dir: Path,
    dtype: torch.dtype,
) -> None:
    """Stage 2: Compute text embeddings for all captions.

    Loads only the text encoder (~27GB) after VAE is unloaded.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Computing text embeddings")
    logger.info("=" * 60)

    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoder = load_text_encoder(
        checkpoint_path=args.model_path,
        gemma_model_path=args.text_encoder_path,
        device=args.device,
        dtype=dtype,
    )
    text_encoder.eval()

    # Process each pair's caption
    for pair in tqdm(pairs, desc="Computing embeddings"):
        tgt_name = Path(pair["target"]).stem
        cond_output = conditions_dir / f"{tgt_name}.safetensors"

        if cond_output.exists():
            continue

        caption = pair.get("caption", "A video")

        try:
            with torch.inference_mode():
                embeddings = text_encoder(caption)

            # Save embeddings
            torch.save(
                {
                    "video_prompt_embeds": embeddings.video_encoding.cpu(),
                    "audio_prompt_embeds": embeddings.audio_encoding.cpu()
                    if embeddings.audio_encoding is not None
                    else None,
                    "prompt_attention_mask": embeddings.attention_mask.cpu(),
                },
                cond_output,
            )

        except Exception as e:
            logger.error(f"Error computing embeddings for {tgt_name}: {e}")
            continue

    # Unload text encoder
    del text_encoder
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    logger.info("Text encoder unloaded, VRAM freed")


def main():
    args = parse_args()

    # Validate frame count
    if args.max_frames % 8 != 1:
        raise ValueError(
            f"max_frames must satisfy frames %% 8 == 1 for LTX-2, "
            f"got {args.max_frames}. Valid values: 1, 9, 17, 25, ..., 97, ..."
        )

    # Setup dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Create output directories
    ref_latents_dir = args.output_dir / "reference_latents"
    tgt_latents_dir = args.output_dir / "latents"
    conditions_dir = args.output_dir / "conditions"

    ref_latents_dir.mkdir(parents=True, exist_ok=True)
    tgt_latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs configuration
    pairs_file = args.input_dir / "pairs.json"
    if not pairs_file.exists():
        raise FileNotFoundError(
            f"pairs.json not found in {args.input_dir}. "
            "Expected JSON file with structure: "
            '[{"reference": "ref_001.mp4", "target": "tgt_001.mp4", "caption": "..."}, ...]'
        )

    with open(pairs_file) as f:
        pairs = json.load(f)

    logger.info(f"Found {len(pairs)} video pairs to process")

    # STAGED PROCESSING to avoid OOM
    # Stage 1: VAE encoding (~5GB VRAM for encoder + ~10GB for video tensors)
    stage1_encode_videos(args, pairs, ref_latents_dir, tgt_latents_dir, dtype)

    # Stage 2: Text embedding (~27GB VRAM for Gemma)
    stage2_compute_embeddings(args, pairs, conditions_dir, dtype)

    logger.info(f"Dataset preparation complete. Output saved to {args.output_dir}")

    # Create dataset metadata
    metadata = {
        "num_pairs": len(pairs),
        "target_width": args.target_width,
        "target_height": args.target_height,
        "max_frames": args.max_frames,
        "pairs": [
            {
                "reference": Path(p["reference"]).stem,
                "target": Path(p["target"]).stem,
                "caption": p.get("caption", ""),
            }
            for p in pairs
        ],
    }

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved to metadata.json")


if __name__ == "__main__":
    main()
