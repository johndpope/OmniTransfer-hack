#!/usr/bin/env python3
"""Compute final text embeddings for cross-paired diorama dataset.

Optimized: computes only unique captions (79 scenes), then copies to all
1110+ sample positions. This avoids redundant Gemma forward passes.

Usage:
    python scripts/compute_crosspair_embeddings.py \
        --dataset-dir /media/2TB/grok_diorama_crosspair \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute text embeddings for cross-paired dataset (deduped)"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--text-encoder-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-in-8bit", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Load metadata
    meta_path = args.dataset_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    pairs = metadata["pairs"]
    logger.info(f"Dataset has {len(pairs)} pairs")

    # Find unique captions
    unique_captions: dict[str, int] = {}  # caption → first pair index
    caption_to_pairs: dict[str, list[int]] = {}  # caption → list of pair indices
    for pair in pairs:
        cap = pair["caption"]
        idx = pair["id"]
        if cap not in unique_captions:
            unique_captions[cap] = idx
            caption_to_pairs[cap] = []
        caption_to_pairs[cap].append(idx)

    logger.info(f"Unique captions: {len(unique_captions)} (saves {len(pairs) - len(unique_captions)} redundant encodes)")

    # Output directory
    output_dir = args.dataset_dir / "conditions_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check what's already computed
    if not args.overwrite:
        existing = set(int(f.stem) for f in output_dir.glob("*.pt"))
        needed_ids = set(p["id"] for p in pairs)
        if needed_ids.issubset(existing):
            logger.info(f"All {len(needed_ids)} embeddings already exist. Use --overwrite to recompute.")
            return

    # Load text encoder
    logger.info(f"Loading text encoder (8-bit={args.load_in_8bit})...")
    text_encoder = load_text_encoder(
        checkpoint_path=args.model_path,
        gemma_model_path=args.text_encoder_path,
        device=args.device,
        dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit,
    )
    text_encoder.eval()
    logger.info("Text encoder loaded")

    # Phase 1: Compute unique embeddings
    logger.info(f"Computing {len(unique_captions)} unique embeddings...")
    unique_embeddings: dict[str, dict] = {}

    for caption in tqdm(unique_captions.keys(), desc="Encoding unique captions"):
        try:
            with torch.inference_mode():
                video_embeds, audio_embeds, attention_mask = text_encoder(caption)

            embedding_data = {
                "video_prompt_embeds": video_embeds[0].cpu().contiguous(),
                "audio_prompt_embeds": (
                    audio_embeds[0].cpu().contiguous()
                    if audio_embeds is not None
                    else video_embeds[0].cpu().contiguous()
                ),
                "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
                "is_final_embedding": True,
            }
            unique_embeddings[caption] = embedding_data
        except Exception as e:
            logger.error(f"Failed to encode: {caption[:50]}... - {e}")
            continue

    # Cleanup text encoder
    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"Computed {len(unique_embeddings)} unique embeddings, text encoder unloaded")

    # Phase 2: Copy to all pair positions
    logger.info(f"Writing embeddings to {len(pairs)} sample positions...")
    written = 0
    for caption, embedding_data in unique_embeddings.items():
        for pair_idx in caption_to_pairs[caption]:
            output_file = output_dir / f"{pair_idx}.pt"
            torch.save(embedding_data, output_file)
            written += 1

    logger.info(f"Wrote {written} embedding files to {output_dir}")

    # Verify
    sample = torch.load(output_dir / "0.pt", map_location="cpu", weights_only=True)
    logger.info("Sample output:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            logger.info(f"  {k}: {v.shape} ({v.dtype})")
        else:
            logger.info(f"  {k}: {v}")

    # Update metadata
    metadata["has_final_embeddings"] = True
    metadata["conditions_final_dir"] = "conditions_final"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Updated metadata.json")


if __name__ == "__main__":
    main()
