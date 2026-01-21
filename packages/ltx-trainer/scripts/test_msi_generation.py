#!/usr/bin/env python3
"""Test script for video generation via Temporal workflows.

This script tests video generation using LTX-2 (T2V and I2V) and MSI server
for OmniTransfer training data generation.

Backends:
- ltx2: LTX-2 Text-to-Video via Temporal
- ltx2_i2v: LTX-2 Image-to-Video via Temporal (requires --reference-image)
- ltx2_cli: LTX-2 direct CLI (no Temporal)
- msi: MSI server WAN2.2 via Temporal (requires --reference-image)

Usage:
    # Dry run - test prompts without generation
    python scripts/test_msi_generation.py --dry-run

    # Test with LTX-2 T2V via Temporal
    python scripts/test_msi_generation.py --backend ltx2 --num-videos 2

    # Test with LTX-2 I2V via Temporal
    python scripts/test_msi_generation.py --backend ltx2_i2v --reference-image /path/to/ref.png --num-videos 2

    # Test with LTX-2 I2V using URL
    python scripts/test_msi_generation.py --backend ltx2_i2v --reference-image https://example.com/ref.png

    # Test with MSI server (when ready)
    python scripts/test_msi_generation.py --backend msi --reference-image https://example.com/ref.png
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project paths
sys.path.insert(0, "/home/johndpope/Documents/GitHub/PresidentialDilema-FastApi")
sys.path.insert(0, "/home/johndpope/Documents/GitHub/LTX-2")


# =============================================================================
# IDENTITY PROMPTS FOR TESTING
# =============================================================================

# Core identity for all tests
IDENTITY_CORE = """A 32-year-old East Asian woman with shoulder-length straight black hair with subtle caramel highlights, sharp almond-shaped eyes, small straight nose, full lips with natural pink tone, fair-to-light skin with a few light freckles across the bridge of the nose, slim athletic build, 165 cm tall, wearing a fitted white ribbed turtleneck sweater and high-waisted light-wash jeans, silver small hoop earrings"""

TEST_PROMPTS = [
    {
        "name": "static_reference",
        "type": "reference",
        "category": "pure_id",
        "prompt": f"""{IDENTITY_CORE}, standing in soft natural window light, neutral-relaxed expression with very slight confident half-smile, consistent facial structure across every frame, 4K, photorealistic, studio quality""",
        "difficulty": 1,
    },
    {
        "name": "head_turn",
        "type": "target",
        "category": "pure_id",
        "prompt": f"""{IDENTITY_CORE}, soft natural window light, slowly turning head from left to right over 4 seconds, maintaining neutral-relaxed expression with very slight confident half-smile, ultra-consistent identity in every frame""",
        "difficulty": 1,
    },
    {
        "name": "cafe_motion",
        "type": "target",
        "category": "motion_casual",
        "prompt": f"""{IDENTITY_CORE}, sitting at a wooden café table near a large window, soft morning light, slowly turning the pages of an open notebook, occasionally tucking hair behind her ear, sipping from a white ceramic coffee cup, relaxed posture, very slight confident half-smile, camera slowly zooming in over 8 seconds, ultra-consistent identity across every frame""",
        "difficulty": 1,
    },
    {
        "name": "emotional_laugh",
        "type": "target",
        "category": "motion_expressive",
        "prompt": f"""The same 32-year-old East Asian woman with shoulder-length straight black hair with caramel highlights, same fitted white ribbed turtleneck and light jeans, sitting on a modern gray sofa in a minimalist living room, soft warm lamp light, looking directly at camera, starting with neutral expression, then slowly smiling warmly, eyes crinkling, then laughing softly for 3 seconds, covering mouth with hand, then returning to gentle smile while tilting head slightly, ultra-consistent face and clothing in every frame""",
        "difficulty": 2,
    },
    {
        "name": "cyberpunk_style",
        "type": "target",
        "category": "style_cyberpunk",
        "prompt": f"""The same 32-year-old East Asian woman, now in a futuristic cyberpunk aesthetic: white ribbed turtleneck replaced with glossy black latex high-neck top, reflective chrome choker, subtle holographic makeup on cheekbones, standing in a rainy neon-lit Tokyo alley at night, blue and pink neon reflections on wet skin and hair, but facial structure, eye shape, hair length and caramel highlights 100% identical to the reference, same confident half-smile""",
        "difficulty": 3,
    },
]

NEGATIVE_PROMPT = "worst quality, blurry, distorted face, extra limbs, bad anatomy, inconsistent identity, morphing face, changing appearance"


@dataclass
class GenerationResult:
    """Result of a video generation."""
    success: bool
    prompt_name: str
    video_path: str | None
    error: str | None
    duration_seconds: float
    workflow_id: str | None


# =============================================================================
# TEMPORAL CLIENT
# =============================================================================

async def get_temporal_client():
    """Get Temporal client."""
    try:
        from temporalio.client import Client

        host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        namespace = os.getenv("TEMPORAL_NAMESPACE", "default")

        client = await Client.connect(host, namespace=namespace)
        print(f"✓ Connected to Temporal at {host}")
        return client
    except Exception as e:
        print(f"✗ Failed to connect to Temporal: {e}")
        return None


# =============================================================================
# VIDEO GENERATION
# =============================================================================

async def generate_video_ltx2_cli(
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    width: int = 832,
    height: int = 448,
    num_frames: int = 65,
    seed: int = 42,
) -> GenerationResult:
    """Generate video using LTX-2 CLI directly (subprocess)."""
    import subprocess
    import time

    start_time = time.time()

    ltx2_path = os.getenv("LTX2_PATH", "/home/johndpope/Documents/GitHub/LTX-2")
    model_path = os.getenv("LTX2_MODEL_PATH", "/media/2tb/models/ltx2")

    cmd = [
        "uv", "run", "python", "-m", "inference",
        "--model-path", f"{model_path}/ltx-2-19b-dev-fp8.safetensors",
        "--text-encoder-path", f"{model_path}/gemma-3-12b-it-qat-q4_0-unquantized",
        "--prompt", prompt,
        "--negative-prompt", negative_prompt,
        "--width", str(width),
        "--height", str(height),
        "--num-frames", str(num_frames),
        "--seed", str(seed),
        "--output-path", str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=ltx2_path,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            return GenerationResult(
                success=True,
                prompt_name=output_path.stem,
                video_path=str(output_path),
                error=None,
                duration_seconds=duration,
                workflow_id=None,
            )
        else:
            return GenerationResult(
                success=False,
                prompt_name=output_path.stem,
                video_path=None,
                error=result.stderr[:500],
                duration_seconds=duration,
                workflow_id=None,
            )

    except subprocess.TimeoutExpired:
        return GenerationResult(
            success=False,
            prompt_name=output_path.stem,
            video_path=None,
            error="Timeout after 600 seconds",
            duration_seconds=600,
            workflow_id=None,
        )
    except Exception as e:
        return GenerationResult(
            success=False,
            prompt_name=output_path.stem,
            video_path=None,
            error=str(e),
            duration_seconds=time.time() - start_time,
            workflow_id=None,
        )


async def generate_video_temporal_ltx2(
    client,
    prompt: str,
    negative_prompt: str,  # Not used by LTX2VideoInput but kept for API consistency
    output_name: str,
    width: int = 832,
    height: int = 448,
    num_frames: int = 65,
    seed: int = 42,
) -> GenerationResult:
    """Generate video using LTX-2 via Temporal workflow."""
    import time
    from app.temporal.ltx2_workflows import GenerateLTX2VideoWorkflow, LTX2VideoInput

    start_time = time.time()
    workflow_id = f"test-ltx2-{output_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        input_data = LTX2VideoInput(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            seed=seed,
            enhance_prompt=True,
            use_fp8=True,
            use_distilled=True,
            upload_to_supabase=False,  # Keep local
        )

        result = await client.execute_workflow(
            GenerateLTX2VideoWorkflow.run,
            input_data,
            id=workflow_id,
            task_queue="presidential-dilemma-gemini",
            execution_timeout=timedelta(minutes=30),
        )

        duration = time.time() - start_time

        if result.success:
            return GenerationResult(
                success=True,
                prompt_name=output_name,
                video_path=result.video_path,
                error=None,
                duration_seconds=duration,
                workflow_id=workflow_id,
            )
        else:
            return GenerationResult(
                success=False,
                prompt_name=output_name,
                video_path=None,
                error=result.error,
                duration_seconds=duration,
                workflow_id=workflow_id,
            )

    except Exception as e:
        return GenerationResult(
            success=False,
            prompt_name=output_name,
            video_path=None,
            error=str(e),
            duration_seconds=time.time() - start_time,
            workflow_id=workflow_id,
        )


async def generate_video_ltx2_i2v(
    client,
    prompt: str,
    negative_prompt: str,
    output_name: str,
    reference_image_url: str | None = None,
    reference_image_path: str | None = None,
    width: int = 832,
    height: int = 448,
    num_frames: int = 65,
    seed: int = 42,
) -> GenerationResult:
    """Generate video using LTX-2 I2V via Temporal workflow.

    LTX-2 supports both T2V and I2V in the same workflow.
    When reference image is provided, it uses I2V mode.
    """
    import time
    from app.temporal.ltx2_workflows import GenerateLTX2VideoWorkflow, LTX2VideoInput

    start_time = time.time()
    workflow_id = f"test-ltx2-i2v-{output_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Need either URL or path for I2V
    if reference_image_url is None and reference_image_path is None:
        return GenerationResult(
            success=False,
            prompt_name=output_name,
            video_path=None,
            error="I2V requires reference_image_url or reference_image_path",
            duration_seconds=0,
            workflow_id=None,
        )

    try:
        input_data = LTX2VideoInput(
            prompt=prompt,
            input_image_url=reference_image_url,
            input_image_path=reference_image_path,
            width=width,
            height=height,
            num_frames=num_frames,
            seed=seed,
            enhance_prompt=True,
            image_strength=1.0,
            use_fp8=True,
            use_distilled=True,
            upload_to_supabase=False,  # Local output only
        )

        result = await client.execute_workflow(
            GenerateLTX2VideoWorkflow.run,
            input_data,
            id=workflow_id,
            task_queue="presidential-dilemma-gemini",
            execution_timeout=timedelta(minutes=30),
        )

        duration = time.time() - start_time

        if result.success:
            return GenerationResult(
                success=True,
                prompt_name=output_name,
                video_path=result.video_path,
                error=None,
                duration_seconds=duration,
                workflow_id=workflow_id,
            )
        else:
            return GenerationResult(
                success=False,
                prompt_name=output_name,
                video_path=None,
                error=result.error,
                duration_seconds=duration,
                workflow_id=workflow_id,
            )

    except Exception as e:
        return GenerationResult(
            success=False,
            prompt_name=output_name,
            video_path=None,
            error=str(e),
            duration_seconds=time.time() - start_time,
            workflow_id=workflow_id,
        )


async def generate_video_msi(
    client,
    prompt: str,
    negative_prompt: str,
    output_name: str,
    reference_image_url: str | None = None,
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
) -> GenerationResult:
    """Generate video using MSI server (WAN2.2 via ComfyUI).

    Note: WAN2.2 requires a reference image for I2V.
    For T2V, we need to first generate an image.
    """
    import time

    start_time = time.time()
    workflow_id = f"test-msi-{output_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # WAN2.2 is I2V only - need reference image
    if reference_image_url is None:
        return GenerationResult(
            success=False,
            prompt_name=output_name,
            video_path=None,
            error="WAN2.2 requires reference_image_url (I2V only)",
            duration_seconds=0,
            workflow_id=None,
        )

    try:
        from app.temporal.wan2_workflows import GenerateWAN2VideoWorkflow, WAN2VideoInput

        input_data = WAN2VideoInput(
            shot_id=None,  # Direct generation, no shot
            prompt=prompt,
            reference_image_url=reference_image_url,
            width=width,
            height=height,
            num_frames=num_frames,
        )

        result = await client.execute_workflow(
            GenerateWAN2VideoWorkflow.run,
            input_data,
            id=workflow_id,
            task_queue="presidential-dilemma-gemini",
            execution_timeout=timedelta(minutes=30),
        )

        duration = time.time() - start_time

        return GenerationResult(
            success=True,
            prompt_name=output_name,
            video_path=result.video_path,
            error=None,
            duration_seconds=duration,
            workflow_id=workflow_id,
        )

    except Exception as e:
        return GenerationResult(
            success=False,
            prompt_name=output_name,
            video_path=None,
            error=str(e),
            duration_seconds=time.time() - start_time,
            workflow_id=workflow_id,
        )


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

async def run_test(args):
    """Run the video generation test."""

    print("\n" + "="*70)
    print("OmniTransfer MSI Generation Test")
    print("="*70)
    print(f"Backend: {args.backend}")
    print(f"Num videos: {args.num_videos}")
    print(f"Dry run: {args.dry_run}")
    print(f"Output dir: {args.output_dir}")
    print("="*70 + "\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select prompts to test
    test_prompts = TEST_PROMPTS[:args.num_videos]

    if args.dry_run:
        print("DRY RUN MODE - Printing prompts only\n")
        for i, p in enumerate(test_prompts):
            print(f"\n{'─'*60}")
            print(f"Prompt {i+1}: {p['name']}")
            print(f"Category: {p['category']}")
            print(f"Type: {p['type']}")
            print(f"Difficulty: {'★' * p['difficulty']}")
            print(f"{'─'*60}")
            print(f"\n{p['prompt']}\n")
        return

    # Get Temporal client
    client = None
    if args.backend in ["ltx2", "ltx2_i2v", "msi"]:
        client = await get_temporal_client()
        if client is None:
            print("Warning: Falling back to LTX-2 CLI mode")
            args.backend = "ltx2_cli"

    # Generate videos
    results = []

    for i, prompt_data in enumerate(test_prompts):
        print(f"\n[{i+1}/{len(test_prompts)}] Generating: {prompt_data['name']}")
        print(f"  Category: {prompt_data['category']}")
        print(f"  Type: {prompt_data['type']}")

        output_path = output_dir / f"{prompt_data['name']}.mp4"

        if args.backend == "ltx2_cli":
            result = await generate_video_ltx2_cli(
                prompt=prompt_data["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                output_path=output_path,
                width=args.width,
                height=args.height,
                num_frames=args.frames,
                seed=args.seed + i,
            )
        elif args.backend == "ltx2":
            result = await generate_video_temporal_ltx2(
                client=client,
                prompt=prompt_data["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                output_name=prompt_data["name"],
                width=args.width,
                height=args.height,
                num_frames=args.frames,
                seed=args.seed + i,
            )
        elif args.backend == "ltx2_i2v":
            # LTX-2 I2V mode
            if not args.reference_image:
                print("  ⚠ ltx2_i2v backend requires --reference-image")
                result = GenerationResult(
                    success=False,
                    prompt_name=prompt_data["name"],
                    video_path=None,
                    error="ltx2_i2v requires --reference-image",
                    duration_seconds=0,
                    workflow_id=None,
                )
            else:
                # Check if URL or path
                ref_url = args.reference_image if args.reference_image.startswith("http") else None
                ref_path = args.reference_image if not args.reference_image.startswith("http") else None
                result = await generate_video_ltx2_i2v(
                    client=client,
                    prompt=prompt_data["prompt"],
                    negative_prompt=NEGATIVE_PROMPT,
                    output_name=prompt_data["name"],
                    reference_image_url=ref_url,
                    reference_image_path=ref_path,
                    width=args.width,
                    height=args.height,
                    num_frames=args.frames,
                    seed=args.seed + i,
                )
        elif args.backend == "msi":
            # MSI/WAN2.2 requires reference image
            if not args.reference_image:
                print("  ⚠ MSI backend requires --reference-image (I2V only)")
                result = GenerationResult(
                    success=False,
                    prompt_name=prompt_data["name"],
                    video_path=None,
                    error="MSI requires --reference-image",
                    duration_seconds=0,
                    workflow_id=None,
                )
            else:
                result = await generate_video_msi(
                    client=client,
                    prompt=prompt_data["prompt"],
                    negative_prompt=NEGATIVE_PROMPT,
                    output_name=prompt_data["name"],
                    reference_image_url=args.reference_image,
                    width=args.width,
                    height=args.height,
                    num_frames=args.frames,
                )
        else:
            raise ValueError(f"Unknown backend: {args.backend}")

        results.append(result)

        if result.success:
            print(f"  ✓ Success: {result.video_path}")
            print(f"  ✓ Duration: {result.duration_seconds:.1f}s")
        else:
            print(f"  ✗ Failed: {result.error}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

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
        "backend": args.backend,
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": [
            {
                "name": r.prompt_name,
                "success": r.success,
                "video_path": r.video_path,
                "error": r.error,
                "duration": r.duration_seconds,
                "workflow_id": r.workflow_id,
            }
            for r in results
        ],
    }

    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test MSI server video generation for OmniTransfer"
    )

    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["ltx2", "ltx2_cli", "ltx2_i2v", "msi"],
        default="ltx2",
        help="Backend: ltx2 (T2V Temporal), ltx2_cli (direct), ltx2_i2v (I2V Temporal), msi (WAN2.2)",
    )
    parser.add_argument(
        "--reference-image", "-r",
        type=str,
        default=None,
        help="Reference image path or URL for I2V modes",
    )
    parser.add_argument(
        "--num-videos", "-n",
        type=int,
        default=2,
        help="Number of test videos to generate",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="test_generation",
        help="Output directory",
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
        help="Number of frames",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Print prompts without generating",
    )

    args = parser.parse_args()
    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
