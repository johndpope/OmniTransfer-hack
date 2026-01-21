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

from ltx_trainer import logger
from ltx_trainer.model_loader import (
    load_video_vae_encoder,
    load_text_encoder,
)
from ltx_trainer.video_utils import load_video_frames, resize_for_vae


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

    # Load models
    logger.info("Loading VAE encoder...")
    vae_encoder = load_video_vae_encoder(args.model_path, dtype=dtype)
    vae_encoder = vae_encoder.to(args.device)
    vae_encoder.eval()

    logger.info("Loading text encoder...")
    text_encoder = load_text_encoder(args.text_encoder_path, dtype=dtype)
    text_encoder = text_encoder.to(args.device)
    text_encoder.eval()

    # Process each pair
    for pair in tqdm(pairs, desc="Processing video pairs"):
        ref_video_path = args.input_dir / "videos" / pair["reference"]
        tgt_video_path = args.input_dir / "videos" / pair["target"]
        caption = pair.get("caption", "A video")

        # Get base names for output files
        ref_name = Path(pair["reference"]).stem
        tgt_name = Path(pair["target"]).stem

        # Check if already processed
        ref_output = ref_latents_dir / f"{ref_name}.safetensors"
        tgt_output = tgt_latents_dir / f"{tgt_name}.safetensors"
        cond_output = conditions_dir / f"{tgt_name}.safetensors"

        if ref_output.exists() and tgt_output.exists() and cond_output.exists():
            logger.debug(f"Skipping {tgt_name}, already processed")
            continue

        try:
            # Process reference video
            if not ref_output.exists():
                ref_frames = load_video_frames(
                    ref_video_path,
                    max_frames=args.max_frames,
                )
                ref_frames = resize_for_vae(
                    ref_frames,
                    target_width=args.target_width,
                    target_height=args.target_height,
                )
                ref_frames = ref_frames.to(args.device, dtype=dtype)

                with torch.inference_mode():
                    ref_latent = vae_encoder.encode(ref_frames.unsqueeze(0))

                # Save reference latent
                torch.save(
                    {
                        "latents": ref_latent.cpu(),
                        "num_frames": torch.tensor([ref_latent.shape[2]]),
                        "height": torch.tensor([ref_latent.shape[3]]),
                        "width": torch.tensor([ref_latent.shape[4]]),
                    },
                    ref_output,
                )

            # Process target video
            if not tgt_output.exists():
                tgt_frames = load_video_frames(
                    tgt_video_path,
                    max_frames=args.max_frames,
                )
                tgt_frames = resize_for_vae(
                    tgt_frames,
                    target_width=args.target_width,
                    target_height=args.target_height,
                )
                tgt_frames = tgt_frames.to(args.device, dtype=dtype)

                with torch.inference_mode():
                    tgt_latent = vae_encoder.encode(tgt_frames.unsqueeze(0))

                # Save target latent
                torch.save(
                    {
                        "latents": tgt_latent.cpu(),
                        "num_frames": torch.tensor([tgt_latent.shape[2]]),
                        "height": torch.tensor([tgt_latent.shape[3]]),
                        "width": torch.tensor([tgt_latent.shape[4]]),
                    },
                    tgt_output,
                )

            # Process caption
            if not cond_output.exists():
                with torch.inference_mode():
                    embeddings = text_encoder.encode(caption)

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
            logger.error(f"Error processing pair {tgt_name}: {e}")
            continue

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
