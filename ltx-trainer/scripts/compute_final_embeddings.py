#!/usr/bin/env python3
"""Compute FINAL text embeddings (post-connector) for training without loading models.

This script computes the final video/audio prompt embeddings that are normally
computed by the embedding connectors during training. By pre-computing these,
training can skip loading the text encoder entirely, saving ~28GB VRAM.

The script can either:
1. Convert existing raw embeddings (from compute_text_embeddings.py) to final form
2. Compute from scratch if no raw embeddings exist

Output format in conditions_final/*.pt:
    - video_prompt_embeds: [seq_len, 4096] - final video cross-attention context
    - audio_prompt_embeds: [seq_len, 4096] - final audio cross-attention context
    - prompt_attention_mask: [seq_len] - attention mask
    - is_final_embedding: True - marker to identify format

Usage:
    # Convert existing raw embeddings to final
    python scripts/compute_final_embeddings.py \
        --dataset-dir /media/2TB/omnitransfer_unified_5task \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma

    # Compute from scratch using metadata captions
    python scripts/compute_final_embeddings.py \
        --dataset-dir /media/2TB/omnitransfer_unified_5task \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma \
        --from-scratch
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute final embeddings (post-connector) for VRAM-free training"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing conditions/ and metadata.json",
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: dataset-dir/conditions_final)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Compute from scratch using metadata captions (ignore existing conditions/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        default=True,
        help="Load text encoder in 8-bit (default: True)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing final embeddings",
    )
    return parser.parse_args()


def convert_raw_to_final(
    text_encoder,
    raw_conditions_dir: Path,
    output_dir: Path,
    device: str,
    overwrite: bool = False,
) -> int:
    """Convert existing raw embeddings to final embeddings using connectors.

    Args:
        text_encoder: Loaded text encoder with connectors
        raw_conditions_dir: Directory with raw embeddings (prompt_embeds [1024, 3840])
        output_dir: Output directory for final embeddings
        device: Device to use
        overwrite: Whether to overwrite existing files

    Returns:
        Number of files processed
    """
    raw_files = sorted(raw_conditions_dir.glob("*.pt"))
    logger.info(f"Found {len(raw_files)} raw embedding files to convert")

    processed = 0
    for raw_file in tqdm(raw_files, desc="Converting raw → final"):
        output_file = output_dir / raw_file.name

        if output_file.exists() and not overwrite:
            logger.debug(f"Skipping {raw_file.name} - already exists")
            continue

        try:
            # Load raw embeddings
            raw_data = torch.load(raw_file, map_location="cpu", weights_only=True)

            # Check if already in final format
            if raw_data.get("is_final_embedding", False):
                logger.debug(f"Skipping {raw_file.name} - already final format")
                continue

            prompt_embeds = raw_data["prompt_embeds"]  # [1024, 3840]
            prompt_attention_mask = raw_data["prompt_attention_mask"]  # [1024]

            # Add batch dimension and move to device
            prompt_embeds = prompt_embeds.unsqueeze(0).to(device)  # [1, 1024, 3840]
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0).to(device)  # [1, 1024]

            # Run through connectors
            with torch.inference_mode():
                video_embeds, audio_embeds, attention_mask = text_encoder._run_connectors(
                    prompt_embeds, prompt_attention_mask
                )

            # Save final embeddings
            torch.save({
                "video_prompt_embeds": video_embeds[0].cpu().contiguous(),  # [seq_len, 4096]
                "audio_prompt_embeds": audio_embeds[0].cpu().contiguous(),  # [seq_len, 4096]
                "prompt_attention_mask": attention_mask[0].cpu().contiguous(),  # [seq_len]
                "is_final_embedding": True,  # Marker for trainer to identify format
            }, output_file)

            processed += 1

            # Clear cache periodically
            if processed % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error converting {raw_file.name}: {e}")
            continue

    return processed


def compute_from_scratch(
    text_encoder,
    metadata: dict,
    output_dir: Path,
    device: str,
    overwrite: bool = False,
) -> int:
    """Compute final embeddings from scratch using metadata captions.

    Args:
        text_encoder: Loaded text encoder (full, with Gemma model)
        metadata: Dataset metadata with pairs/captions
        output_dir: Output directory for final embeddings
        device: Device to use
        overwrite: Whether to overwrite existing files

    Returns:
        Number of files processed
    """
    pairs = metadata.get("pairs", [])
    logger.info(f"Computing final embeddings for {len(pairs)} samples from captions")

    processed = 0
    for pair in tqdm(pairs, desc="Computing final embeddings"):
        idx = pair["id"]
        output_file = output_dir / f"{idx:03d}.pt"

        if output_file.exists() and not overwrite:
            logger.debug(f"Skipping {idx:03d} - already exists")
            continue

        caption = pair.get("caption", "A video")

        try:
            with torch.inference_mode():
                # Full forward pass through text encoder (Gemma + connectors)
                video_embeds, audio_embeds, attention_mask = text_encoder(caption)

            # Save final embeddings
            torch.save({
                "video_prompt_embeds": video_embeds[0].cpu().contiguous(),
                "audio_prompt_embeds": audio_embeds[0].cpu().contiguous() if audio_embeds is not None else video_embeds[0].cpu().contiguous(),
                "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
                "is_final_embedding": True,
            }, output_file)

            processed += 1

            # Clear cache periodically
            if processed % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error computing embeddings for {idx}: {e}")
            continue

    return processed


def main():
    args = parse_args()

    # Setup dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Setup output directory
    output_dir = args.output_dir or (args.dataset_dir / "conditions_final")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing conditions
    raw_conditions_dir = args.dataset_dir / "conditions"
    has_raw_conditions = raw_conditions_dir.exists() and list(raw_conditions_dir.glob("*.pt"))

    # Load metadata
    metadata_file = args.dataset_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Load text encoder
    logger.info("Loading text encoder...")
    text_encoder = load_text_encoder(
        checkpoint_path=args.model_path,
        gemma_model_path=args.text_encoder_path,
        device=args.device,
        dtype=dtype,
        load_in_8bit=args.load_in_8bit,
    )
    text_encoder.eval()
    logger.info("Text encoder loaded")

    # Process embeddings
    if args.from_scratch or not has_raw_conditions:
        if not metadata.get("pairs"):
            raise ValueError(
                "No raw conditions found and metadata.json has no pairs. "
                "Either provide conditions/ directory or metadata with captions."
            )
        processed = compute_from_scratch(
            text_encoder, metadata, output_dir, args.device, args.overwrite
        )
    else:
        # For conversion, we only need connectors, can unload Gemma
        logger.info("Converting raw embeddings - unloading Gemma to save VRAM...")
        # Keep connectors but remove heavy model
        text_encoder.model = None
        text_encoder.tokenizer = None
        text_encoder.feature_extractor_linear = None
        gc.collect()
        torch.cuda.empty_cache()

        processed = convert_raw_to_final(
            text_encoder, raw_conditions_dir, output_dir, args.device, args.overwrite
        )

    # Cleanup
    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"✅ Processed {processed} embeddings → {output_dir}")

    # Verify output
    final_files = list(output_dir.glob("*.pt"))
    logger.info(f"Total final embedding files: {len(final_files)}")

    # Show sample
    if final_files:
        sample = torch.load(final_files[0], map_location="cpu", weights_only=True)
        logger.info("Sample output format:")
        for key, val in sample.items():
            if hasattr(val, "shape"):
                logger.info(f"  {key}: {val.shape} ({val.dtype})")
            else:
                logger.info(f"  {key}: {val}")

    # Update metadata to note final embeddings are available
    if metadata_file.exists():
        metadata["has_final_embeddings"] = True
        metadata["conditions_final_dir"] = "conditions_final"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Updated metadata.json with final embeddings info")


if __name__ == "__main__":
    main()
