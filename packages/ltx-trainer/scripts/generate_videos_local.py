#!/usr/bin/env python3
"""Generate synthetic videos locally using LTX-2 (no Temporal required).

This script generates identity-consistent videos for OmniTransfer training
by directly invoking the LTX-2 inference pipeline.

Requirements:
- LTX-2 model files (safetensors)
- Gemma text encoder
- CUDA GPU with sufficient VRAM (24GB+ recommended)

Usage:
    # Dry run - preview prompts
    python scripts/generate_videos_local.py --dry-run

    # Generate 5 videos with default settings
    python scripts/generate_videos_local.py --num-videos 5

    # Generate with custom model path
    python scripts/generate_videos_local.py \
        --model-path /path/to/ltx2.safetensors \
        --text-encoder-path /path/to/gemma \
        --num-videos 10

    # Generate I2V with reference image
    python scripts/generate_videos_local.py \
        --reference-image /path/to/ref.png \
        --num-videos 5

    # Use specific prompt mode
    python scripts/generate_videos_local.py --mode cyberpunk --num-videos 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# =============================================================================
# IDENTITY PROMPTS FOR TRAINING DATA
# =============================================================================

BASE_IDENTITY = """A 32-year-old East Asian woman with shoulder-length straight black hair with subtle caramel highlights, sharp almond-shaped eyes, small straight nose, full lips with natural pink tone, fair-to-light skin with a few light freckles across the bridge of the nose, slim athletic build, 165 cm tall"""

BASE_OUTFIT = """wearing a fitted white ribbed turtleneck sweater and high-waisted light-wash jeans, silver small hoop earrings"""

PROMPT_MODES = {
    "reference": {
        "name": "Static Reference",
        "category": "pure_id",
        "prompts": [
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, standing in soft natural window light, neutral-relaxed expression with very slight confident half-smile, consistent facial structure across every frame, 4K, photorealistic, studio quality",
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, medium close-up portrait, soft diffused lighting, looking directly at camera, serene expression, ultra-sharp focus on facial features",
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, three-quarter view portrait, natural daylight, subtle smile, professional headshot quality",
        ],
    },
    "head_turn": {
        "name": "Head Turn Motion",
        "category": "pure_id",
        "prompts": [
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, soft natural window light, slowly turning head from left to right over 4 seconds, maintaining neutral-relaxed expression with very slight confident half-smile, ultra-consistent identity in every frame",
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, studio lighting, gentle head movement looking from camera to the side and back, smooth motion, consistent appearance throughout",
        ],
    },
    "casual": {
        "name": "Casual Motion",
        "category": "motion_casual",
        "prompts": [
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, sitting at a wooden cafe table near a large window, soft morning light, slowly turning the pages of an open notebook, occasionally tucking hair behind her ear, sipping from a white ceramic coffee cup, relaxed posture, very slight confident half-smile, camera slowly zooming in over 8 seconds, ultra-consistent identity across every frame",
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, walking slowly through a sunlit park path, gentle breeze moving hair slightly, natural gait, peaceful expression, trees in soft focus background",
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, sitting on a park bench, reading a book, occasionally looking up and smiling, birds in background, golden hour lighting",
        ],
    },
    "expressive": {
        "name": "Expressive Motion",
        "category": "motion_expressive",
        "prompts": [
            f"The same 32-year-old East Asian woman with shoulder-length straight black hair with caramel highlights, same fitted white ribbed turtleneck and light jeans, sitting on a modern gray sofa in a minimalist living room, soft warm lamp light, looking directly at camera, starting with neutral expression, then slowly smiling warmly, eyes crinkling, then laughing softly for 3 seconds, covering mouth with hand, then returning to gentle smile while tilting head slightly, ultra-consistent face and clothing in every frame",
            f"{BASE_IDENTITY}, {BASE_OUTFIT}, close-up on face, transitioning from thoughtful expression to surprised delight, raising eyebrows, widening eyes, breaking into genuine smile, natural emotional progression",
        ],
    },
    "cyberpunk": {
        "name": "Cyberpunk Style",
        "category": "style_cyberpunk",
        "prompts": [
            f"The same 32-year-old East Asian woman, now in a futuristic cyberpunk aesthetic: white ribbed turtleneck replaced with glossy black latex high-neck top, reflective chrome choker, subtle holographic makeup on cheekbones, standing in a rainy neon-lit Tokyo alley at night, blue and pink neon reflections on wet skin and hair, but facial structure, eye shape, hair length and caramel highlights 100% identical to the reference, same confident half-smile",
            f"The same 32-year-old East Asian woman with identical facial features and hair, wearing futuristic silver bodysuit with glowing blue accents, holographic visor pushed up on forehead, standing in a sleek spaceship corridor with ambient blue lighting, same calm confident expression",
        ],
    },
    "ghibli": {
        "name": "Studio Ghibli Style",
        "category": "style_ghibli",
        "prompts": [
            f"The same 32-year-old East Asian woman rendered in Studio Ghibli animation style, soft watercolor textures, hand-drawn feel, sitting in a cozy wooden room with large windows showing a lush green forest, warm afternoon light, wearing a simple cream-colored linen dress, same facial proportions and gentle expression adapted to Ghibli aesthetic, wind gently moving hair",
            f"The same woman in Ghibli style, walking through a field of tall grass and wildflowers, billowing white sundress, straw hat in hand, clouds drifting in painted sky, same recognizable features in anime form",
        ],
    },
    "aging": {
        "name": "Age Progression",
        "category": "id_aging",
        "prompts": [
            f"The same East Asian woman now aged to approximately 50 years old, same facial bone structure and eye shape, hair now silver-gray but same length and slight wave, subtle smile lines around eyes, wearing elegant navy blouse, dignified expression, soft natural lighting, ultra-consistent identity despite age change",
            f"The same East Asian woman as a young woman around 20 years old, same distinctive features but with youthful skin, hair slightly longer and pure black without highlights, wearing casual university student clothes, bright optimistic expression",
        ],
    },
}

NEGATIVE_PROMPT = "worst quality, blurry, distorted face, extra limbs, bad anatomy, inconsistent identity, morphing face, changing appearance, deformed, ugly, bad proportions"


@dataclass
class GenerationResult:
    """Result of a video generation."""
    success: bool
    prompt_name: str
    video_path: str | None
    error: str | None
    duration_seconds: float
    prompt: str


def find_ltx2_paths() -> tuple[str | None, str | None]:
    """Try to find LTX-2 model paths automatically."""
    # Common locations to check
    model_locations = [
        "/media/2TB/models/ltx2",
        "/media/2tb/models/ltx2",
        "/mnt/2TB-backup/models/ltx2",
        os.path.expanduser("~/.cache/ltx2"),
        os.path.expanduser("~/models/ltx2"),
        "/opt/models/ltx2",
    ]

    model_path = None
    text_encoder_path = None

    for loc in model_locations:
        safetensors = os.path.join(loc, "ltx-2-19b-dev-fp8.safetensors")
        gemma = os.path.join(loc, "gemma-3-12b-it-qat-q4_0-unquantized")

        if os.path.exists(safetensors):
            model_path = safetensors
        if os.path.exists(gemma):
            text_encoder_path = gemma

        if model_path and text_encoder_path:
            break

    return model_path, text_encoder_path


def generate_video_subprocess(
    prompt: str,
    output_path: Path,
    model_path: str,
    text_encoder_path: str,
    reference_image: str | None = None,
    width: int = 832,
    height: int = 448,
    num_frames: int = 65,
    seed: int = 42,
    negative_prompt: str = NEGATIVE_PROMPT,
) -> GenerationResult:
    """Generate video using LTX-2 via subprocess."""
    start_time = time.time()

    # Find LTX-2 repo path
    ltx2_repo = os.getenv("LTX2_PATH", "/home/johndpope/Documents/GitHub/LTX-2")

    # Build command
    if reference_image:
        # Image-to-video pipeline
        pipeline = "ltx_pipelines.image_to_video"
        cmd = [
            sys.executable, "-m", pipeline,
            "--checkpoint-path", model_path,
            "--gemma-root", text_encoder_path,
            "--input-image", reference_image,
            "--prompt", prompt,
            "--negative-prompt", negative_prompt,
            "--width", str(width),
            "--height", str(height),
            "--num-frames", str(num_frames),
            "--seed", str(seed),
            "--output-path", str(output_path),
        ]
    else:
        # Text-to-video pipeline
        pipeline = "ltx_pipelines.text_to_video"
        cmd = [
            sys.executable, "-m", pipeline,
            "--checkpoint-path", model_path,
            "--gemma-root", text_encoder_path,
            "--prompt", prompt,
            "--negative-prompt", negative_prompt,
            "--width", str(width),
            "--height", str(height),
            "--num-frames", str(num_frames),
            "--seed", str(seed),
            "--output-path", str(output_path),
        ]

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ltx2_repo}/packages/ltx-pipelines/src:{ltx2_repo}/packages/ltx-core/src"

    try:
        print(f"  Running: {pipeline}")
        result = subprocess.run(
            cmd,
            cwd=ltx2_repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0 and output_path.exists():
            return GenerationResult(
                success=True,
                prompt_name=output_path.stem,
                video_path=str(output_path),
                error=None,
                duration_seconds=duration,
                prompt=prompt,
            )
        else:
            error_msg = result.stderr[:500] if result.stderr else f"Return code: {result.returncode}"
            return GenerationResult(
                success=False,
                prompt_name=output_path.stem,
                video_path=None,
                error=error_msg,
                duration_seconds=duration,
                prompt=prompt,
            )

    except subprocess.TimeoutExpired:
        return GenerationResult(
            success=False,
            prompt_name=output_path.stem,
            video_path=None,
            error="Timeout after 600 seconds",
            duration_seconds=600,
            prompt=prompt,
        )
    except Exception as e:
        return GenerationResult(
            success=False,
            prompt_name=output_path.stem,
            video_path=None,
            error=str(e),
            duration_seconds=time.time() - start_time,
            prompt=prompt,
        )


def run_generation(args):
    """Run video generation."""
    print("\n" + "=" * 70)
    print("OmniTransfer Local Video Generation")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Num videos: {args.num_videos}")
    print(f"Dry run: {args.dry_run}")
    print(f"Output dir: {args.output_dir}")
    print(f"Reference image: {args.reference_image or 'None (T2V mode)'}")
    print("=" * 70 + "\n")

    # Get prompts for selected mode
    if args.mode == "all":
        all_prompts = []
        for mode_name, mode_data in PROMPT_MODES.items():
            for i, prompt in enumerate(mode_data["prompts"]):
                all_prompts.append({
                    "name": f"{mode_name}_{i+1}",
                    "category": mode_data["category"],
                    "prompt": prompt,
                })
    else:
        mode_data = PROMPT_MODES.get(args.mode)
        if not mode_data:
            print(f"Unknown mode: {args.mode}")
            print(f"Available modes: {', '.join(PROMPT_MODES.keys())}, all")
            return
        all_prompts = [
            {"name": f"{args.mode}_{i+1}", "category": mode_data["category"], "prompt": p}
            for i, p in enumerate(mode_data["prompts"])
        ]

    # Limit to requested number
    prompts_to_use = all_prompts[:args.num_videos]

    if args.dry_run:
        print("DRY RUN MODE - Printing prompts only\n")
        for i, p in enumerate(prompts_to_use):
            print(f"\n{'─' * 60}")
            print(f"Prompt {i+1}: {p['name']}")
            print(f"Category: {p['category']}")
            print(f"{'─' * 60}")
            print(f"\n{p['prompt']}\n")
        return

    # Find model paths
    model_path = args.model_path
    text_encoder_path = args.text_encoder_path

    if not model_path or not text_encoder_path:
        auto_model, auto_text = find_ltx2_paths()
        model_path = model_path or auto_model
        text_encoder_path = text_encoder_path or auto_text

    if not model_path:
        print("ERROR: Could not find LTX-2 model. Specify --model-path")
        return
    if not text_encoder_path:
        print("ERROR: Could not find Gemma text encoder. Specify --text-encoder-path")
        return

    print(f"Model: {model_path}")
    print(f"Text encoder: {text_encoder_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate videos
    results = []

    for i, prompt_data in enumerate(prompts_to_use):
        print(f"\n[{i+1}/{len(prompts_to_use)}] Generating: {prompt_data['name']}")
        print(f"  Category: {prompt_data['category']}")

        output_path = output_dir / f"{prompt_data['name']}_{args.seed + i}.mp4"

        result = generate_video_subprocess(
            prompt=prompt_data["prompt"],
            output_path=output_path,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            reference_image=args.reference_image,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            seed=args.seed + i,
        )

        results.append(result)

        if result.success:
            print(f"  ✓ Success: {result.video_path}")
            print(f"  ✓ Duration: {result.duration_seconds:.1f}s")
        else:
            print(f"  ✗ Failed: {result.error}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_duration = sum(r.duration_seconds for r in successful) / len(successful)
        print(f"Average duration: {avg_duration:.1f}s")

    if failed:
        print("\nFailed generations:")
        for r in failed:
            print(f"  - {r.prompt_name}: {r.error}")

    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "model_path": model_path,
        "text_encoder_path": text_encoder_path,
        "results": [
            {
                "name": r.prompt_name,
                "success": r.success,
                "video_path": r.video_path,
                "error": r.error,
                "duration": r.duration_seconds,
                "prompt": r.prompt[:200] + "..." if len(r.prompt) > 200 else r.prompt,
            }
            for r in results
        ],
    }

    results_path = output_dir / "generation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    if successful:
        print(f"\nGenerated videos in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic videos locally for OmniTransfer training"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
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
        default="generated_videos",
        help="Output directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to LTX-2 safetensors model",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=str,
        default=None,
        help="Path to Gemma text encoder directory",
    )
    parser.add_argument(
        "--reference-image", "-r",
        type=str,
        default=None,
        help="Reference image for I2V mode",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=448,
        help="Video height",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=65,
        help="Number of frames (must be 8n+1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Print prompts without generating",
    )

    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
