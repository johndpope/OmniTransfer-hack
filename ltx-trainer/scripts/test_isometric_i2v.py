#!/usr/bin/env python3
# ruff: noqa: T201
"""Test LTX-2 Image-to-Video on isometric 3D images.

Two-phase approach to avoid OOM on 32GB GPU:
  Phase 1: Load text encoder alone (~28GB), compute embeddings, save, unload
  Phase 2: Load transformer + VAE (~28GB peak), run I2V with cached embeddings

Usage:
  cd packages/ltx-trainer
  uv run python scripts/test_isometric_i2v.py

  # Custom image + prompt:
  uv run python scripts/test_isometric_i2v.py \
    --image /path/to/image.jpg \
    --prompt "A woman dancing in an isometric room" \
    --output /media/2TB/i2v_test/custom.mp4
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from torchvision import transforms

from ltx_trainer.model_loader import load_model, load_text_encoder
from ltx_trainer.progress import StandaloneSamplingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.validation_sampler import (
    CachedPromptEmbeddings,
    GenerationConfig,
    ValidationSampler,
)
from ltx_trainer.video_utils import save_video

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
# bf16 checkpoint for text encoder (embeddings), FP8 for transformer (inference)
MODEL_PATH_BF16 = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
MODEL_PATH_FP8 = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev-fp8.safetensors"
TEXT_ENCODER_PATH = "/media/2TB/ltx-models/gemma/"
DEFAULT_IMAGE = "/media/12TB/isometric_3d/r2_native_dataset/images/harvested_0005_233409_3c28aa066a.jpg"
DEFAULT_PROMPT = (
    "Static camera, fixed isometric viewpoint, no camera movement. "
    "A woman speaking cheerfully with animated hand gestures and expressive body language. "
    "She shifts her weight, nods her head, and makes lively conversation. "
    "The camera does not move at all."
)
DEFAULT_NEGATIVE_PROMPT = (
    "camera movement, camera pan, camera zoom, dolly, tracking shot, "
    "camera shake, rotating camera, moving viewpoint, static people, frozen"
)
DEFAULT_OUTPUT = "/media/2TB/i2v_test/test_isometric_i2v.mp4"


def load_image(image_path: str) -> torch.Tensor:
    """Load an image and convert to tensor [C, H, W] in [0, 1]."""
    image = open_image_as_srgb(image_path)
    transform = transforms.ToTensor()
    return transform(image)


def free_vram() -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def phase1_compute_embeddings(
    prompt: str,
    negative_prompt: str,
    model_path: str,
    text_encoder_path: str,
    device: str = "cuda:0",
    cache_path: Path | None = None,
    load_in_8bit: bool = True,
) -> dict:
    """Phase 1: Load text encoder, compute embeddings, save, free VRAM.

    Returns dict with video/audio context tensors on CPU.
    """
    # Check for cached embeddings
    if cache_path and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print("=" * 60)
    print("Phase 1: Computing text embeddings")
    print("=" * 60)
    t0 = time.time()

    bit_str = "8-bit (~14GB)" if load_in_8bit else "bf16 (~28GB)"
    print(f"Loading text encoder to {device} ({bit_str})...")
    text_encoder = load_text_encoder(
        checkpoint_path=model_path,
        gemma_model_path=text_encoder_path,
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
    )

    print(f"  Loaded in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM: {alloc:.1f}GB allocated")

    # Compute embeddings
    print(f"Computing embeddings for prompt: {prompt[:80]}...")
    with torch.inference_mode():
        v_ctx_pos, a_ctx_pos, _ = text_encoder(prompt)
        v_ctx_neg, a_ctx_neg = None, None
        if negative_prompt:
            v_ctx_neg, a_ctx_neg, _ = text_encoder(negative_prompt)

    # Move to CPU
    embeddings = {
        "video_context_positive": v_ctx_pos.cpu(),
        "audio_context_positive": a_ctx_pos.cpu(),
        "video_context_negative": v_ctx_neg.cpu() if v_ctx_neg is not None else None,
        "audio_context_negative": a_ctx_neg.cpu() if a_ctx_neg is not None else None,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }

    print(f"  video_context_positive: {v_ctx_pos.shape}")
    print(f"  audio_context_positive: {a_ctx_pos.shape}")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"  Cached to {cache_path}")

    # Aggressively free text encoder VRAM
    print("Unloading text encoder...")
    del text_encoder
    free_vram()

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after cleanup: {alloc:.1f}GB")

    print(f"Phase 1 done in {time.time() - t0:.1f}s\n")
    return embeddings


def phase2_generate_video(
    embeddings: dict,
    image_path: str,
    output_path: str,
    model_path: str,
    lora_path: str | None = None,
    height: int = 448,
    width: int = 832,
    num_frames: int = 9,
    num_steps: int = 30,
    guidance_scale: float = 4.0,
    stg_scale: float = 1.0,
    seed: int = 42,
    device: str = "cuda:0",
) -> None:
    """Phase 2: Load transformer + VAE, run I2V with cached embeddings."""
    print("=" * 60)
    print("Phase 2: Generating I2V video")
    print("=" * 60)
    t0 = time.time()

    # Load conditioning image
    print(f"Loading image: {image_path}")
    condition_image = load_image(image_path)
    print(f"  Shape: {condition_image.shape}")

    # Load model to CPU, quantize block-by-block on GPU, then move to GPU.
    # bf16 transformer is ~28GB (won't fit on 32GB GPU).
    # After int8-quanto quantization: ~19GB (fits with VAE).
    print("Loading transformer + VAE to CPU...")
    components = load_model(
        checkpoint_path=model_path,
        device="cpu",
        dtype=torch.bfloat16,
        with_video_vae_encoder=True,  # Needed for I2V (encode first frame)
        with_video_vae_decoder=True,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=False,  # Skip — using cached embeddings
    )

    # Quantize transformer with int8-quanto (block-by-block on GPU)
    print("Quantizing transformer with int8-quanto (this takes ~20 min first time)...")
    t_quant = time.time()
    components.transformer = quantize_model(
        components.transformer, precision="int8-quanto", device=device,
    )
    print(f"  Quantization done in {time.time() - t_quant:.0f}s")

    # Move quantized transformer to GPU
    print(f"Moving quantized transformer to {device}...")
    components.transformer.to(device)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  Transformer VRAM: {alloc:.1f}GB")

    # Move VAE encoder + decoder to GPU
    print("Moving VAE to GPU...")
    if components.video_vae_encoder is not None:
        components.video_vae_encoder.to(device)
    if components.video_vae_decoder is not None:
        components.video_vae_decoder.to(device)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  Total VRAM: {alloc:.1f}GB")

    transformer = components.transformer

    # Apply LoRA if provided
    if lora_path:
        from inference import load_lora_weights

        transformer = load_lora_weights(transformer, lora_path)

    # Build CachedPromptEmbeddings
    cached = CachedPromptEmbeddings(
        video_context_positive=embeddings["video_context_positive"],
        audio_context_positive=embeddings["audio_context_positive"],
        video_context_negative=embeddings["video_context_negative"],
        audio_context_negative=embeddings["audio_context_negative"],
    )

    # Build generation config
    gen_config = GenerationConfig(
        prompt="",  # Ignored when cached_embeddings provided
        cached_embeddings=cached,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        condition_image=condition_image,
        generate_audio=False,
        stg_scale=stg_scale,
        stg_blocks=[29],
        stg_mode="stg_v",
    )

    print(f"\nI2V Generation:")
    print(f"  Image: {image_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames} @ 25fps")
    print(f"  Steps: {num_steps}, CFG: {guidance_scale}, STG: {stg_scale}")
    print(f"  Seed: {seed}")

    # Generate
    print("\nGenerating video...")
    with StandaloneSamplingProgress(num_steps=num_steps) as progress:
        sampler = ValidationSampler(
            transformer=transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=None,
            sampling_context=progress,
        )
        video, audio = sampler.generate(config=gen_config, device=device)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_tensor=video, output_path=out, fps=25.0)
    print(f"\nVideo saved to {output_path}")
    print(f"Phase 2 done in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LTX-2 I2V on isometric 3D images")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Input image path")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output video path")
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH_BF16,
        help="LTX-2 checkpoint (bf16, quantized at runtime with int8-quanto)",
    )
    parser.add_argument("--text-encoder-path", type=str, default=TEXT_ENCODER_PATH)
    parser.add_argument("--lora-path", type=str, default=None, help="Optional LoRA weights")
    parser.add_argument("--height", type=int, default=768, help="Portrait for isometric (divisible by 32)")
    parser.add_argument("--width", type=int, default=512, help="Portrait for isometric (divisible by 32)")
    parser.add_argument("--num-frames", type=int, default=25, help="Frames (must satisfy frames %% 8 == 1)")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--stg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--embedding-cache", type=str, default="/media/2TB/i2v_test/cached_embedding.pt",
        help="Path to cache/load pre-computed embeddings",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", default=True,
        help="Load text encoder in 8-bit (default: True, saves ~14GB VRAM)",
    )
    parser.add_argument(
        "--no-8bit", action="store_true",
        help="Load text encoder in full bf16 (requires ~28GB free VRAM)",
    )
    args = parser.parse_args()
    if args.no_8bit:
        args.load_in_8bit = False

    print("=" * 60)
    print("LTX-2 Isometric 3D Image-to-Video Test")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {args.output}")
    print()

    # Phase 1: Compute text embeddings (or load from cache)
    embeddings = phase1_compute_embeddings(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        device=args.device,
        cache_path=Path(args.embedding_cache),
        load_in_8bit=args.load_in_8bit,
    )

    # Phase 2: Generate I2V video with cached embeddings
    phase2_generate_video(
        embeddings=embeddings,
        image_path=args.image,
        output_path=args.output,
        model_path=args.model_path,
        lora_path=args.lora_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        stg_scale=args.stg_scale,
        seed=args.seed,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
