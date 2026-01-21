#!/usr/bin/env python3
"""Generate synthetic videos in stages to fit in limited VRAM.

Stage 1: Compute text embeddings for all prompts (loads only Gemma ~12GB)
Stage 2: Generate videos one at a time (loads transformer ~20GB)

This approach allows generating videos on GPUs with 24-32GB VRAM by never
loading both the text encoder and transformer simultaneously.

Usage:
    # Full pipeline (both stages)
    python scripts/generate_videos_staged.py --mode all --num-videos 15

    # Stage 1 only: compute embeddings
    python scripts/generate_videos_staged.py --stage embeddings --mode all

    # Stage 2 only: generate videos (assumes embeddings exist)
    python scripts/generate_videos_staged.py --stage videos --mode all
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

# Import prompts from the main script
from generate_videos_local import PROMPT_MODES, NEGATIVE_PROMPT, find_ltx2_paths


@dataclass
class EmbeddingCache:
    """Cached text embeddings for a prompt."""
    prompt: str
    video_context_positive: torch.Tensor  # [seq_len, dim]
    audio_context_positive: torch.Tensor | None
    video_context_negative: torch.Tensor
    audio_context_negative: torch.Tensor | None


def cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_embeddings(
    prompts: list[dict],
    model_path: str,
    text_encoder_path: str,
    output_dir: Path,
    negative_prompt: str = NEGATIVE_PROMPT,
) -> dict[str, Path]:
    """Stage 1: Compute and cache text embeddings for all prompts."""
    print("\n" + "=" * 60)
    print("STAGE 1: Computing Text Embeddings")
    print("=" * 60)

    # Import here to avoid loading everything at startup
    sys.path.insert(0, "/home/johndpope/Documents/GitHub/LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "/home/johndpope/Documents/GitHub/LTX-2/packages/ltx-pipelines/src")
    from ltx_core.text_encoders.gemma import encode_text
    from ltx_pipelines.utils.model_ledger import ModelLedger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading text encoder from: {text_encoder_path}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")

    # Load text encoder using ModelLedger (proper API)
    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=model_path,
        gemma_root_path=text_encoder_path,
    )
    text_encoder = ledger.text_encoder()

    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after loading text encoder: {allocated:.1f} GB")

    # Create output directory
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embedding_paths = {}

    for i, prompt_data in enumerate(prompts):
        name = prompt_data["name"]
        prompt = prompt_data["prompt"]

        print(f"\n[{i+1}/{len(prompts)}] Encoding: {name}")

        # Encode prompt and negative prompt
        with torch.inference_mode():
            context_p, context_n = encode_text(
                text_encoder,
                prompts=[prompt, negative_prompt]
            )

        v_context_p, a_context_p = context_p
        v_context_n, a_context_n = context_n

        # Save embeddings
        embedding_path = embeddings_dir / f"{name}.pt"
        torch.save({
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "video_context_positive": v_context_p.cpu(),
            "audio_context_positive": a_context_p.cpu() if a_context_p is not None else None,
            "video_context_negative": v_context_n.cpu(),
            "audio_context_negative": a_context_n.cpu() if a_context_n is not None else None,
        }, embedding_path)

        embedding_paths[name] = embedding_path
        print(f"  Saved: {embedding_path}")

    # Cleanup text encoder
    del text_encoder
    cleanup_gpu()

    print(f"\nStage 1 complete. {len(embedding_paths)} embeddings saved to {embeddings_dir}")
    return embedding_paths


def generate_videos(
    prompts: list[dict],
    model_path: str,
    embeddings_dir: Path,
    output_dir: Path,
    width: int = 832,
    height: int = 448,
    num_frames: int = 65,
    base_seed: int = 42,
    num_inference_steps: int = 40,
    cfg_guidance_scale: float = 3.0,
    enable_fp8: bool = True,
) -> list[dict]:
    """Stage 2: Generate videos using cached embeddings."""
    print("\n" + "=" * 60)
    print("STAGE 2: Generating Videos")
    print("=" * 60)

    # Import here to avoid loading everything at startup
    sys.path.insert(0, "/home/johndpope/Documents/GitHub/LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "/home/johndpope/Documents/GitHub/LTX-2/packages/ltx-pipelines/src")

    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.helpers import (
        denoise_audio_video,
        euler_denoising_loop,
        guider_denoising_func,
    )
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_ledger import ModelLedger
    from ltx_pipelines.utils.types import PipelineComponents
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading transformer from: {model_path}")
    print(f"FP8 mode: {enable_fp8}")
    print(f"Device: {device}")

    # Load model components using ModelLedger (proper API)
    # Note: We skip video_encoder since we're doing T2V (starting from noise, not encoding an image)
    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=model_path,
        fp8transformer=enable_fp8,
    )
    transformer = ledger.transformer()
    video_decoder = ledger.video_decoder()
    audio_decoder = ledger.audio_decoder()
    vocoder = ledger.vocoder()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after loading models: {allocated:.1f} GB")

    # Pipeline components
    pipeline_components = PipelineComponents(dtype=dtype, device=device)

    results = []
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt_data in enumerate(prompts):
        name = prompt_data["name"]
        seed = base_seed + i

        print(f"\n[{i+1}/{len(prompts)}] Generating: {name}")

        # Load cached embeddings
        embedding_path = embeddings_dir / f"{name}.pt"
        if not embedding_path.exists():
            print(f"  ERROR: Embedding not found: {embedding_path}")
            results.append({"name": name, "success": False, "error": "Embedding not found"})
            continue

        embeddings = torch.load(embedding_path, weights_only=True)

        # Move embeddings to device
        v_context_p = embeddings["video_context_positive"].to(device=device, dtype=dtype)
        v_context_n = embeddings["video_context_negative"].to(device=device, dtype=dtype)
        a_context_p = embeddings["audio_context_positive"]
        a_context_n = embeddings["audio_context_negative"]
        if a_context_p is not None:
            a_context_p = a_context_p.to(device=device, dtype=dtype)
        if a_context_n is not None:
            a_context_n = a_context_n.to(device=device, dtype=dtype)

        start_time = time.time()

        try:
            with torch.inference_mode():
                # Setup generation
                generator = torch.Generator(device=device).manual_seed(seed)
                noiser = GaussianNoiser(generator=generator)
                stepper = EulerDiffusionStep()
                cfg_guider = CFGGuider(cfg_guidance_scale)

                sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
                    dtype=torch.float32, device=device
                )

                def denoising_loop(sigmas, video_state, audio_state, stepper):
                    return euler_denoising_loop(
                        sigmas=sigmas,
                        video_state=video_state,
                        audio_state=audio_state,
                        stepper=stepper,
                        denoise_fn=guider_denoising_func(
                            cfg_guider,
                            v_context_p,
                            v_context_n,
                            a_context_p,
                            a_context_n,
                            transformer=transformer,
                        ),
                    )

                output_shape = VideoPixelShape(
                    batch=1,
                    frames=num_frames,
                    width=width,
                    height=height,
                    fps=25.0,
                )

                video_state, audio_state = denoise_audio_video(
                    output_shape=output_shape,
                    conditionings=[],  # No image conditioning
                    noiser=noiser,
                    sigmas=sigmas,
                    stepper=stepper,
                    denoising_loop_fn=denoising_loop,
                    components=pipeline_components,
                    dtype=dtype,
                    device=device,
                )

                # Decode video
                decoded_video = vae_decode_video(video_state.latent, video_decoder)
                decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)

                # Save video
                output_path = videos_dir / f"{name}_{seed}.mp4"
                encode_video(
                    video=decoded_video,
                    fps=25.0,
                    audio=decoded_audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=str(output_path),
                    video_chunks_number=1,
                )

            duration = time.time() - start_time
            print(f"  ✓ Success: {output_path} ({duration:.1f}s)")
            results.append({
                "name": name,
                "success": True,
                "path": str(output_path),
                "duration": duration,
            })

        except Exception as e:
            duration = time.time() - start_time
            print(f"  ✗ Failed: {e}")
            results.append({
                "name": name,
                "success": False,
                "error": str(e),
                "duration": duration,
            })

        # Cleanup between videos
        cleanup_gpu()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic videos in stages for memory efficiency"
    )
    parser.add_argument(
        "--stage",
        choices=["both", "embeddings", "videos"],
        default="both",
        help="Which stage to run",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=list(PROMPT_MODES.keys()) + ["all"],
        default="reference",
        help="Prompt mode to use",
    )
    parser.add_argument(
        "--num-videos", "-n",
        type=int,
        default=3,
        help="Number of videos to generate",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="/media/2TB/omnitransfer_synthetic_videos",
        help="Output directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to LTX-2 model",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default=None,
        help="Path to Gemma text encoder",
    )
    parser.add_argument(
        "--width", type=int, default=832,
    )
    parser.add_argument(
        "--height", type=int, default=448,
    )
    parser.add_argument(
        "--frames", type=int, default=65,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="Disable FP8 mode (uses more memory)",
    )

    args = parser.parse_args()

    # Find model paths
    model_path = args.model_path
    text_encoder_path = args.text_encoder_path

    if not model_path or not text_encoder_path:
        auto_model, auto_text = find_ltx2_paths()
        model_path = model_path or auto_model
        text_encoder_path = text_encoder_path or auto_text

    if not model_path:
        print("ERROR: Could not find LTX-2 model")
        return
    if not text_encoder_path:
        print("ERROR: Could not find Gemma text encoder")
        return

    # Get prompts
    if args.mode == "all":
        prompts = []
        for mode_name, mode_data in PROMPT_MODES.items():
            for i, prompt in enumerate(mode_data["prompts"]):
                prompts.append({
                    "name": f"{mode_name}_{i+1}",
                    "category": mode_data["category"],
                    "prompt": prompt,
                })
    else:
        mode_data = PROMPT_MODES[args.mode]
        prompts = [
            {"name": f"{args.mode}_{i+1}", "category": mode_data["category"], "prompt": p}
            for i, p in enumerate(mode_data["prompts"])
        ]

    prompts = prompts[:args.num_videos]

    print(f"\nMode: {args.mode}")
    print(f"Prompts: {len(prompts)}")
    print(f"Model: {model_path}")
    print(f"Text encoder: {text_encoder_path}")
    print(f"Output: {args.output_dir}")

    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"

    # Stage 1: Compute embeddings
    if args.stage in ["both", "embeddings"]:
        compute_embeddings(
            prompts=prompts,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            output_dir=output_dir,
        )
        cleanup_gpu()

    # Stage 2: Generate videos
    if args.stage in ["both", "videos"]:
        results = generate_videos(
            prompts=prompts,
            model_path=model_path,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            base_seed=args.seed,
            enable_fp8=not args.no_fp8,
        )

        # Summary
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        # Save results
        with open(output_dir / "generation_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)


if __name__ == "__main__":
    main()
