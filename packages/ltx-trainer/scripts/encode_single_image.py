#!/usr/bin/env python3
"""Encode a single image to VAE latent tensor for OmniTransfer training.

Loads the LTX-2 VAE encoder, processes one image, and saves the latent
in the format expected by PrecomputedDataset.

Usage:
    python scripts/encode_single_image.py \
        --image /path/to/image.jpg \
        --output /path/to/output.pt \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --target-height 448 --target-width 832

Output format:
    {
        "latents": Tensor[128, 1, H_lat, W_lat],  # bfloat16
        "num_frames": Tensor[1],
        "height": Tensor[H_lat],
        "width": Tensor[W_lat],
    }
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder


def load_and_prepare_image(
    image_path: Path,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Load image, center-crop to target aspect ratio, resize, and prepare for VAE.

    Args:
        image_path: Path to input image.
        target_h: Target height (must be divisible by 32).
        target_w: Target width (must be divisible by 32).

    Returns:
        Tensor of shape [1, C, 1, H, W] normalized to [-1, 1].
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target_aspect = target_w / target_h
    source_aspect = w / h

    # Center-crop to match target aspect ratio
    if abs(source_aspect - target_aspect) > 0.01:
        if source_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img.crop((start_x, 0, start_x + new_w, h))
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img.crop((0, start_y, w, start_y + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)

    # Convert to [1, C, 1, H, W] in [-1, 1]
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
    tensor = tensor * 2.0 - 1.0

    return tensor


def encode_image(
    image_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    target_h: int = 448,
    target_w: int = 832,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Encode a single image to VAE latent and save.

    Args:
        image_path: Path to input image.
        output_path: Path to save the .pt file.
        model_path: Path to LTX-2 .safetensors checkpoint.
        target_h: Target height (must be divisible by 32).
        target_w: Target width (must be divisible by 32).
        device: CUDA device string.
        dtype: Model dtype.

    Returns:
        Dict with saved tensor data.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading VAE encoder from {model_path}")
    vae_encoder = load_video_vae_encoder(model_path, dtype=dtype)
    vae_encoder = vae_encoder.to(device)
    vae_encoder.eval()

    logger.info(f"Encoding {image_path} at {target_w}x{target_h}")
    img_tensor = load_and_prepare_image(image_path, target_h, target_w)
    img_tensor = img_tensor.to(device, dtype=dtype)

    with torch.inference_mode():
        latent = vae_encoder(img_tensor)

    latent = latent.cpu()
    data = {
        "latents": latent.squeeze(0),  # [C, 1, H_lat, W_lat]
        "num_frames": torch.tensor([1]),
        "height": torch.tensor([latent.shape[3]]),
        "width": torch.tensor([latent.shape[4]]),
    }

    torch.save(data, output_path)
    logger.info(f"Saved latent {data['latents'].shape} to {output_path}")

    # Cleanup
    del vae_encoder
    torch.cuda.empty_cache()
    gc.collect()

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a single image to VAE latent")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt path")
    parser.add_argument("--model-path", type=Path, required=True, help="LTX-2 .safetensors")
    parser.add_argument("--target-height", type=int, default=448, help="Target height (div by 32)")
    parser.add_argument("--target-width", type=int, default=832, help="Target width (div by 32)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if args.target_height % 32 != 0 or args.target_width % 32 != 0:
        raise ValueError(f"Dimensions must be divisible by 32: {args.target_width}x{args.target_height}")

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    encode_image(
        image_path=args.image,
        output_path=args.output,
        model_path=args.model_path,
        target_h=args.target_height,
        target_w=args.target_width,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )


if __name__ == "__main__":
    main()
