#!/usr/bin/env python3
"""Compute text embedding for a single prompt using Gemma.

Loads the Gemma text encoder (optionally in 8-bit), computes the embedding
for one prompt, and saves in the format expected by PrecomputedDataset.

Usage:
    python scripts/compute_single_embedding.py \
        --prompt "isometric 3D view of this scene, photorealistic miniature diorama" \
        --output /path/to/conditions/0.pt \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma

Output format:
    {
        "prompt_embeds": Tensor[1024, 3840],       # bfloat16
        "prompt_attention_mask": Tensor[1024],      # int64
        "caption": str,
    }
"""

import argparse
import gc
from pathlib import Path

import torch

from ltx_trainer import logger
from ltx_trainer.model_loader import load_text_encoder


def compute_embedding(
    prompt: str,
    output_path: str | Path,
    model_path: str | Path,
    text_encoder_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = True,
) -> dict[str, torch.Tensor | str]:
    """Compute text embedding for a single prompt and save.

    Args:
        prompt: Text prompt to encode.
        output_path: Path to save the .pt file.
        model_path: Path to LTX-2 .safetensors checkpoint.
        text_encoder_path: Path to Gemma model directory.
        device: CUDA device string.
        dtype: Model dtype.
        load_in_8bit: Load text encoder in 8-bit mode.

    Returns:
        Dict with saved tensor data.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading text encoder (8-bit={load_in_8bit}) from {text_encoder_path}")
    text_encoder = load_text_encoder(
        checkpoint_path=model_path,
        gemma_model_path=text_encoder_path,
        device=device,
        dtype=dtype,
        load_in_8bit=load_in_8bit,
    )
    text_encoder.eval()

    logger.info(f"Encoding prompt: {prompt[:80]}...")
    with torch.inference_mode():
        prompt_embeds, prompt_attention_mask = text_encoder._preprocess_text(
            prompt, padding_side="left"
        )

    data = {
        "prompt_embeds": prompt_embeds[0].cpu().contiguous(),
        "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
        "caption": prompt,
    }

    torch.save(data, output_path)
    logger.info(f"Saved embedding {data['prompt_embeds'].shape} to {output_path}")

    # Cleanup
    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute text embedding for a single prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to encode")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt path")
    parser.add_argument("--model-path", type=Path, required=True, help="LTX-2 .safetensors")
    parser.add_argument("--text-encoder-path", type=Path, required=True, help="Gemma model directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load-in-8bit", action="store_true", default=True, help="Load in 8-bit")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit loading")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    compute_embedding(
        prompt=args.prompt,
        output_path=args.output,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        device=args.device,
        dtype=dtype_map[args.dtype],
        load_in_8bit=not args.no_8bit,
    )


if __name__ == "__main__":
    main()
