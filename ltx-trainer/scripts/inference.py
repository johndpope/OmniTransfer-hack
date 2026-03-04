#!/usr/bin/env python3
# ruff: noqa: T201
"""
CLI script for running LTX video/audio generation inference.
Usage:
    # Text-to-Video + Audio (default behavior)
    python scripts/inference.py --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat playing with a ball" --output output.mp4
    # Video only (skip audio)
    python scripts/inference.py --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat playing with a ball" --skip-audio --output output.mp4
    # Image-to-Video
    python scripts/inference.py --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat walking" --condition-image first_frame.png --output output.mp4
    # Video-to-Video (IC-LoRA style)
    python scripts/inference.py --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --prompt "A cat turning into a dog" --reference-video input.mp4 --output output.mp4
    # With LoRA weights
    python scripts/inference.py --checkpoint path/to/model.safetensors \
        --text-encoder-path path/to/gemma \
        --lora-path path/to/lora.safetensors \
        --prompt "A cat in my custom style" --output output.mp4
"""

import argparse
import re
from pathlib import Path

import torch
import torchaudio
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torchvision import transforms

from ltx_trainer.model_loader import load_model
from ltx_trainer.progress import StandaloneSamplingProgress
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video


def load_image(image_path: str) -> torch.Tensor:
    """Load an image and convert to tensor [C, H, W] in [0, 1]."""
    image = open_image_as_srgb(image_path)
    transform = transforms.ToTensor()
    return transform(image)


def extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Extract target module names from LoRA checkpoint keys.
    LoRA keys follow the pattern (after removing "diffusion_model." prefix):
    - transformer_blocks.0.attn1.to_k.lora_A.weight
    - transformer_blocks.0.ff.net.0.proj.lora_B.weight
    This extracts the full module path like "transformer_blocks.0.attn1.to_k".
    Using full paths is more robust than partial patterns.
    """
    target_modules = set()
    # Pattern to extract everything before .lora_A or .lora_B
    pattern = re.compile(r"(.+)\.lora_[AB]\.")

    for key in state_dict:
        match = pattern.match(key)
        if match:
            module_path = match.group(1)
            target_modules.add(module_path)

    return sorted(target_modules)


def load_lora_weights(transformer: torch.nn.Module, lora_path: str | Path) -> torch.nn.Module:
    """Load LoRA weights into the transformer model.
    The LoRA rank and target modules are automatically detected from the checkpoint.
    Alpha is set equal to rank (standard practice for inference).
    Args:
        transformer: The base transformer model
        lora_path: Path to the LoRA weights (.safetensors)
    Returns:
        The transformer model with LoRA weights applied
    """
    print(f"Loading LoRA weights from {lora_path}...")

    # Load the LoRA state dict
    state_dict = load_file(str(lora_path))

    # Remove "diffusion_model." prefix (ComfyUI-compatible format)
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

    # Extract target modules from the checkpoint
    target_modules = extract_lora_target_modules(state_dict)
    if not target_modules:
        raise ValueError(f"Could not extract target modules from LoRA checkpoint: {lora_path}")
    print(f"  Detected {len(target_modules)} target modules")

    # Auto-detect rank from the first lora_A weight shape
    lora_rank = None
    for key, value in state_dict.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
            break
    if lora_rank is None:
        raise ValueError("Could not auto-detect LoRA rank from weights")
    print(f"  LoRA rank: {lora_rank}")

    # Create LoRA config and wrap the model
    # Alpha = rank is standard for inference (maintains the trained scale)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
    )

    # Wrap the transformer with PEFT to add LoRA layers
    transformer = get_peft_model(transformer, lora_config)

    # Load the LoRA weights
    base_model = transformer.get_base_model()
    set_peft_model_state_dict(base_model, state_dict)

    print("✓ LoRA weights loaded successfully")
    return transformer


def main() -> None:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        description="LTX Video/Audio Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors",
        help="Path to model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default="/media/2TB/ltx-models/gemma",
        help="Path to Gemma text encoder directory (not needed with --cached-embedding)",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA weights (.safetensors)",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for generation (not needed with --cached-embedding)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=544,
        help="Video height (must be divisible by 32)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Video width (must be divisible by 32)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=97,
        help="Number of video frames (must be k*8 + 1)",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Video frame rate",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (CFG)",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=1.0,
        help="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. Default: 1.0",
    )
    parser.add_argument(
        "--stg-blocks",
        type=int,
        nargs="*",
        default=[29],
        help="Which transformer blocks to perturb for STG. Default: 29 (single block).",
    )
    parser.add_argument(
        "--stg-mode",
        type=str,
        default="stg_av",
        choices=["stg_av", "stg_v"],
        help="STG mode: 'stg_av' perturbs both audio and video, 'stg_v' perturbs video only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Conditioning arguments
    parser.add_argument(
        "--condition-image",
        type=str,
        default=None,
        help="Path to conditioning image for image-to-video generation",
    )
    parser.add_argument(
        "--reference-video",
        type=str,
        default=None,
        help="Path to reference video for video-to-video generation (IC-LoRA style)",
    )
    parser.add_argument(
        "--include-reference-in-output",
        action="store_true",
        help="Include reference video side-by-side with generated output (only for V2V)",
    )

    # Audio arguments
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio generation (by default, audio is generated alongside video)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video path (.mp4)",
    )
    parser.add_argument(
        "--audio-output",
        type=str,
        default=None,
        help="Output audio path (.wav, optional - if not provided, audio will be embedded in video)",
    )

    # Cached embedding (bypass text encoder)
    parser.add_argument(
        "--cached-embedding",
        type=str,
        default=None,
        help="Path to precomputed text embedding .pt file (skips loading Gemma text encoder)",
    )

    # Quantization
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8-quanto", "fp8-quanto"],
        help="Quantize transformer to reduce VRAM (required for 32GB GPUs)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Validate conditioning arguments
    if args.include_reference_in_output and args.reference_video is None:
        parser.error("--include-reference-in-output requires --reference-video")

    # Validate arguments
    generate_audio = not args.skip_audio
    use_cached = args.cached_embedding is not None

    print("=" * 80)
    print("LTX Video/Audio Generation")
    print("=" * 80)

    # Load cached embeddings if provided (bypasses text encoder entirely)
    cached_embeddings = None
    if use_cached:
        from ltx_trainer.validation_sampler import CachedPromptEmbeddings

        print(f"Loading cached embedding from {args.cached_embedding}...")
        emb = torch.load(args.cached_embedding, map_location="cpu", weights_only=True)
        video_embeds = emb["video_prompt_embeds"].unsqueeze(0).to(torch.bfloat16)
        audio_embeds = emb.get("audio_prompt_embeds", torch.zeros_like(emb["video_prompt_embeds"])).unsqueeze(0).to(torch.bfloat16)
        cached_embeddings = CachedPromptEmbeddings(
            video_context_positive=video_embeds,
            audio_context_positive=audio_embeds,
        )
        print(f"  Shape: {video_embeds.shape}")

    # Determine if we need VAE encoder (for image or video conditioning)
    need_vae_encoder = args.condition_image is not None or args.reference_video is not None

    components = load_model(
        checkpoint_path=args.checkpoint,
        device="cpu",  # Load to CPU first, sampler will move to device as needed
        dtype=torch.bfloat16,
        with_video_vae_encoder=need_vae_encoder,
        with_video_vae_decoder=True,
        with_audio_vae_decoder=generate_audio,
        with_vocoder=generate_audio,
        with_text_encoder=not use_cached,
        text_encoder_path=args.text_encoder_path if not use_cached else None,
    )

    # Quantize transformer if requested (must be done before .to(device) in sampler)
    transformer = components.transformer
    if args.quantization != "none":
        from ltx_trainer.quantization import quantize_model

        print(f"Quantizing transformer ({args.quantization})...")
        transformer = quantize_model(transformer, args.quantization, device=args.device)
        # Move to device after quantization (sampler's .to() will be a no-op)
        transformer = transformer.to(args.device)
        print(f"  GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    # Apply LoRA weights if provided
    if args.lora_path is not None:
        transformer = load_lora_weights(transformer, args.lora_path)

    # Load conditioning image if provided
    condition_image = None
    if args.condition_image:
        print(f"Loading conditioning image from {args.condition_image}...")
        condition_image = load_image(args.condition_image)

    # Load reference video if provided
    reference_video = None
    if args.reference_video:
        print(f"Loading reference video from {args.reference_video}...")
        reference_video, ref_fps = read_video(args.reference_video, max_frames=args.num_frames)
        print(f"  Loaded {reference_video.shape[0]} frames @ {ref_fps:.1f} fps")

    # Determine generation mode
    if args.reference_video is not None and args.condition_image is not None:
        mode = "Video-to-Video + Image Conditioning (V2V+I2V)"
    elif args.reference_video is not None:
        mode = "Video-to-Video (V2V)"
    elif args.condition_image is not None:
        mode = "Image-to-Video (I2V)"
    else:
        mode = "Text-to-Video (T2V)"

    print("\n" + "=" * 80)
    print("Generation Parameters")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames} @ {args.frame_rate} fps")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"CFG scale: {args.guidance_scale}")
    if args.stg_scale > 0:
        blocks_str = args.stg_blocks if args.stg_blocks else "all"
        print(f"STG scale: {args.stg_scale} (mode: {args.stg_mode}, blocks: {blocks_str})")
    else:
        print("STG: disabled")
    print(f"Seed: {args.seed}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    if condition_image is not None:
        print(f"Conditioning: Image ({args.condition_image})")
    if reference_video is not None:
        print(f"Reference: Video ({args.reference_video})")
        if args.include_reference_in_output:
            print("  → Will include reference side-by-side in output")
    if generate_audio:
        video_duration = args.num_frames / args.frame_rate
        print(f"Audio: Enabled (duration will match video: {video_duration:.2f}s)")
    print("=" * 80)

    print(f"\nGenerating {'video + audio' if generate_audio else 'video'}...")

    # Create generation config
    gen_config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        condition_image=condition_image,
        reference_video=reference_video,
        generate_audio=generate_audio,
        include_reference_in_output=args.include_reference_in_output,
        stg_scale=args.stg_scale,
        stg_blocks=args.stg_blocks,
        stg_mode=args.stg_mode,
        cached_embeddings=cached_embeddings,
    )

    # Generate with progress bar
    with StandaloneSamplingProgress(num_steps=args.num_inference_steps) as progress:
        # Create sampler with progress context
        sampler = ValidationSampler(
            transformer=transformer,
            vae_decoder=components.video_vae_decoder,
            vae_encoder=components.video_vae_encoder,
            text_encoder=components.text_encoder,
            audio_decoder=components.audio_vae_decoder if generate_audio else None,
            vocoder=components.vocoder if generate_audio else None,
            sampling_context=progress,
        )
        video, audio = sampler.generate(
            config=gen_config,
            device=args.device,
        )

    # Save video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get audio sample rate from vocoder if audio was generated
    audio_sample_rate = None
    if audio is not None and components.vocoder is not None:
        audio_sample_rate = components.vocoder.output_sample_rate

    # Save as image if single frame + image extension, otherwise video
    if args.num_frames == 1 and output_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        from torchvision.utils import save_image

        # video is [C, F, H, W] in [0, 1] — extract first frame as [C, H, W]
        frame = video[:, 0, :, :] if video.dim() == 4 else video
        save_image(frame, output_path)
        print(f"✓ Image saved to {args.output}")
    else:
        save_video(
            video_tensor=video,
            output_path=output_path,
            fps=args.frame_rate,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
        )
        print(f"✓ Video saved to {args.output}")

    # Save separate audio file if requested
    if audio is not None and args.audio_output is not None:
        audio_output_path = Path(args.audio_output)
        audio_output_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(audio_output_path),
            audio.cpu(),
            sample_rate=audio_sample_rate,
        )
        duration = audio.shape[1] / audio_sample_rate
        print(f"✓ Audio saved: {duration:.2f}s at {audio_sample_rate}Hz")

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
