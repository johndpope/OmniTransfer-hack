#!/usr/bin/env python3
"""Compute text embeddings for OmniTransfer dataset.

This script ONLY computes text embeddings (no VAE loading) to avoid OOM issues.

Usage:
    python scripts/compute_text_embeddings.py \
        --output-dir /media/2TB/omnitransfer_effect_motion \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma
"""

import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder


def main():
    parser = argparse.ArgumentParser(description="Compute text embeddings only")
    parser.add_argument("--output-dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to LTX-2 model")
    parser.add_argument("--text-encoder-path", type=Path, required=True, help="Path to Gemma")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load-in-8bit", action="store_true", default=True, help="Load text encoder in 8-bit")
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # Find how many samples we have from latents directory
    lat_dir = args.output_dir / "latents"
    cond_dir = args.output_dir / "conditions"
    cond_dir.mkdir(parents=True, exist_ok=True)

    lat_files = sorted(lat_dir.glob("*.pt"))
    logger.info(f"Found {len(lat_files)} latent files")

    # Get task types from metadata
    import json
    metadata_file = args.output_dir / "metadata.json"
    task_types = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        for pair in metadata.get("pairs", []):
            task_types[pair["id"]] = pair.get("task_type", "effect")
        logger.info(f"Loaded task types from metadata: {task_types}")

    # Task-specific captions
    CAPTIONS = {
        "effect": "Apply visual effect from reference video to the target image. Video with artistic effect transfer.",
        "motion": "Animate the target image with motion from the reference video. Video with motion transfer.",
        "camera": "Apply camera movement from reference video to the static scene. Video with camera motion.",
    }

    # Load text encoder
    logger.info(f"Loading text encoder (8-bit={args.load_in_8bit})...")
    text_encoder = load_text_encoder(
        checkpoint_path=args.model_path,
        gemma_model_path=args.text_encoder_path,
        device=args.device,
        dtype=dtype,
        load_in_8bit=args.load_in_8bit,
    )
    text_encoder.eval()
    logger.info("Text encoder loaded")

    # Compute embeddings
    for lat_file in tqdm(lat_files, desc="Computing embeddings"):
        idx = int(lat_file.stem)
        cond_out = cond_dir / f"{idx:03d}.pt"

        if cond_out.exists():
            logger.info(f"Skipping {idx:03d} - already exists")
            continue

        # Get task type and caption
        task_type = task_types.get(idx, "effect")
        caption = CAPTIONS.get(task_type, "Video transfer task.")

        try:
            with torch.inference_mode():
                # Use _preprocess_text to get raw embeddings BEFORE connectors
                # The trainer applies connectors during training
                prompt_embeds, prompt_attention_mask = text_encoder._preprocess_text(
                    caption, padding_side="left"
                )

            torch.save({
                "prompt_embeds": prompt_embeds[0].cpu().contiguous(),
                "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
            }, cond_out)
            logger.info(f"Saved embeddings for {idx:03d}")

            # Clear cache after each
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error computing embeddings for {idx}: {e}")

    # Unload
    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()

    # Verify
    cond_files = sorted(cond_dir.glob("*.pt"))
    logger.info(f"Total condition files: {len(cond_files)}")
    logger.info(f"Missing: {[f.stem for f in lat_files if not (cond_dir / f.name).exists()]}")


if __name__ == "__main__":
    main()
