#!/usr/bin/env python3
"""Generate OmniTransfer Training Dataset via Temporal Workflows.

This script generates synthetic video pairs for OmniTransfer training by:
1. Using LTX-2 (local) or WAN2.2 (MSI server via ComfyUI) for video generation
2. Creating reference/target video pairs for each training mode
3. Organizing outputs in OmniTransfer dataset format

Identity prompts are carefully designed for consistent person overfitting
across different transfer tasks (ID, style, motion).

Usage:
    # Generate all training modes
    python scripts/generate_omnitransfer_dataset.py --all --num-clips 32

    # Generate specific mode
    python scripts/generate_omnitransfer_dataset.py --mode pure_id --num-clips 16

    # Use MSI server (WAN2.2 via ComfyUI)
    python scripts/generate_omnitransfer_dataset.py --backend msi --mode motion

    # Dry run (print prompts only)
    python scripts/generate_omnitransfer_dataset.py --dry-run --all
"""

import argparse
import asyncio
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Add temporal path
sys.path.insert(0, "/home/johndpope/Documents/GitHub/PresidentialDilema-FastApi")

from app.temporal.client import get_temporal_client

# Temporal imports
try:
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    print("Warning: temporalio not installed. Using dry-run mode only.")


# =============================================================================
# IDENTITY PROMPTS - Carefully Crafted for Overfitting
# =============================================================================

# Base identity - core visual anchors for consistent person generation
BASE_IDENTITY = """A 32-year-old East Asian woman with shoulder-length straight black hair with subtle caramel highlights, sharp almond-shaped eyes, small straight nose, full lips with natural pink tone, fair-to-light skin with a few light freckles across the bridge of the nose, slim athletic build, 165 cm tall"""

BASE_OUTFIT = """wearing a fitted white ribbed turtleneck sweater and high-waisted light-wash jeans, silver small hoop earrings"""

BASE_LIGHTING = """soft natural window light"""

BASE_EXPRESSION = """neutral-relaxed expression with very slight confident half-smile"""


class TrainingMode(Enum):
    """OmniTransfer training modes with associated prompts."""

    # Appearance Transfer Modes (ID / Style dominant)
    PURE_ID = "pure_id"
    ID_AGING = "id_aging"
    ID_ACCESSORIES = "id_accessories"
    STYLE_PAINTERLY = "style_painterly"
    STYLE_CYBERPUNK = "style_cyberpunk"

    # Temporal / Motion Modes
    MOTION_CASUAL = "motion_casual"
    MOTION_EXPRESSIVE = "motion_expressive"
    MOTION_DANCE = "motion_dance"

    # Combined / Multi-task Modes (hardest)
    COMBINED_GHIBLI = "combined_ghibli"
    COMBINED_EFFECTS = "combined_effects"


@dataclass
class IdentityPrompt:
    """A carefully crafted prompt for consistent identity generation."""

    mode: TrainingMode
    name: str
    prompt: str
    negative_prompt: str = "worst quality, blurry, distorted face, extra limbs, bad anatomy, inconsistent identity"
    difficulty: int = 1  # 1-5 stars
    description: str = ""
    is_reference: bool = False  # True if this is the reference prompt for transfer
    motion_intensity: str = "low"  # low, medium, high


# =============================================================================
# PROMPT LIBRARY
# =============================================================================

IDENTITY_PROMPTS: dict[TrainingMode, list[IdentityPrompt]] = {

    # =========================================================================
    # 1. PURE ID TRANSFER - Most important single-person overfitting case
    # =========================================================================
    TrainingMode.PURE_ID: [
        IdentityPrompt(
            mode=TrainingMode.PURE_ID,
            name="pure_id_reference",
            is_reference=True,
            difficulty=1,
            description="Static reference - maximum facial/clothing anchors",
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, standing in {BASE_LIGHTING}, {BASE_EXPRESSION}, consistent facial structure across every frame, 4K, photorealistic, studio quality""",
        ),
        IdentityPrompt(
            mode=TrainingMode.PURE_ID,
            name="pure_id_slight_turn",
            difficulty=1,
            description="Slight head turn - identity preservation test",
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, {BASE_LIGHTING}, slowly turning head from left to right over 4 seconds, maintaining {BASE_EXPRESSION}, ultra-consistent identity in every frame""",
            motion_intensity="low",
        ),
        IdentityPrompt(
            mode=TrainingMode.PURE_ID,
            name="pure_id_blink",
            difficulty=1,
            description="Natural blink - micro-expression test",
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, {BASE_LIGHTING}, looking at camera with {BASE_EXPRESSION}, natural eye blinks every 2-3 seconds, perfectly consistent facial features""",
            motion_intensity="low",
        ),
        IdentityPrompt(
            mode=TrainingMode.PURE_ID,
            name="pure_id_different_angle",
            difficulty=2,
            description="Different camera angle - 3/4 view",
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, {BASE_LIGHTING}, 3/4 profile view, {BASE_EXPRESSION}, same consistent facial structure visible from this angle""",
            motion_intensity="low",
        ),
    ],

    # =========================================================================
    # 2. ID + AGING TEST
    # =========================================================================
    TrainingMode.ID_AGING: [
        IdentityPrompt(
            mode=TrainingMode.ID_AGING,
            name="id_aging_reference_32",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, {BASE_LIGHTING}, {BASE_EXPRESSION}, age 32, youthful appearance""",
        ),
        IdentityPrompt(
            mode=TrainingMode.ID_AGING,
            name="id_aging_42",
            difficulty=2,
            description="Same person 10 years older",
            prompt=f"""The same 42-year-old East Asian woman from before, now with very faint laugh lines at the outer corners of the eyes, same shoulder-length straight black hair with slightly more visible caramel highlights, still {BASE_OUTFIT}, {BASE_LIGHTING}, same {BASE_EXPRESSION}, same slim athletic build, identical bone structure and eye shape""",
        ),
        IdentityPrompt(
            mode=TrainingMode.ID_AGING,
            name="id_aging_52",
            difficulty=3,
            description="Same person 20 years older",
            prompt=f"""The same 52-year-old East Asian woman, now with visible crow's feet and subtle nasolabial folds, some grey streaks in her shoulder-length black hair with caramel highlights, same white ribbed turtleneck and light-wash jeans, same silver hoops, {BASE_LIGHTING}, same confident half-smile but with wisdom in the eyes, identical facial bone structure""",
        ),
    ],

    # =========================================================================
    # 3. ID + ACCESSORIES TEST
    # =========================================================================
    TrainingMode.ID_ACCESSORIES: [
        IdentityPrompt(
            mode=TrainingMode.ID_ACCESSORIES,
            name="id_accessories_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, {BASE_LIGHTING}, {BASE_EXPRESSION}, no makeup, no glasses""",
        ),
        IdentityPrompt(
            mode=TrainingMode.ID_ACCESSORIES,
            name="id_accessories_makeup_glasses",
            difficulty=2,
            description="Heavy makeup and glasses test",
            prompt=f"""The same 32-year-old East Asian woman, now wearing subtle matte red lipstick, light smoky eye shadow, thin black cat-eye eyeliner, and clear square-framed glasses, otherwise identical appearance, {BASE_OUTFIT}, {BASE_LIGHTING}, same slight confident half-smile, identical facial structure underneath the accessories""",
        ),
        IdentityPrompt(
            mode=TrainingMode.ID_ACCESSORIES,
            name="id_accessories_hat_scarf",
            difficulty=2,
            description="Hat and scarf test",
            prompt=f"""The same 32-year-old East Asian woman, now wearing a cream-colored wool beanie that covers part of her forehead, a matching cream knit scarf loosely around her neck, same white ribbed turtleneck visible underneath, same light-wash jeans, same silver hoops partially visible, {BASE_LIGHTING}, {BASE_EXPRESSION}, identical face shape and eye details""",
        ),
    ],

    # =========================================================================
    # 4. STYLE TRANSFER - PAINTERLY
    # =========================================================================
    TrainingMode.STYLE_PAINTERLY: [
        IdentityPrompt(
            mode=TrainingMode.STYLE_PAINTERLY,
            name="style_painterly_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, {BASE_LIGHTING}, {BASE_EXPRESSION}, photorealistic, 4K photography""",
        ),
        IdentityPrompt(
            mode=TrainingMode.STYLE_PAINTERLY,
            name="style_painterly_sargent",
            difficulty=2,
            description="John Singer Sargent oil painting style",
            prompt=f"""The same 32-year-old East Asian woman with shoulder-length straight black hair with caramel highlights, sharp almond eyes, fair skin with light freckles, slim athletic build, {BASE_OUTFIT}, in the distinctive oil-painting style of John Singer Sargent with rich impasto brushstrokes, dramatic chiaroscuro lighting, warm golden tones, soft focus background resembling an old European atelier, but identity and clothing exactly preserved""",
        ),
        IdentityPrompt(
            mode=TrainingMode.STYLE_PAINTERLY,
            name="style_painterly_watercolor",
            difficulty=2,
            description="Soft watercolor illustration style",
            prompt=f"""The same woman with identical facial features, rendered in delicate watercolor illustration style with soft washes of color, visible paper texture, slightly muted pastel tones, white ribbed turtleneck and light jeans preserved in the watercolor medium, gentle brushwork, but face shape, eye details, and hair highlights 100% consistent""",
        ),
    ],

    # =========================================================================
    # 5. STYLE TRANSFER - CYBERPUNK
    # =========================================================================
    TrainingMode.STYLE_CYBERPUNK: [
        IdentityPrompt(
            mode=TrainingMode.STYLE_CYBERPUNK,
            name="style_cyberpunk_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, {BASE_LIGHTING}, {BASE_EXPRESSION}, photorealistic daylight setting""",
        ),
        IdentityPrompt(
            mode=TrainingMode.STYLE_CYBERPUNK,
            name="style_cyberpunk_neon",
            difficulty=3,
            description="Cyberpunk neon aesthetic",
            prompt=f"""The same woman, now in a futuristic cyberpunk aesthetic: white ribbed turtleneck replaced with glossy black latex high-neck top, reflective chrome choker, subtle holographic makeup on cheekbones, standing in a rainy neon-lit Tokyo alley at night, blue and pink neon reflections on wet skin and hair, but facial structure, eye shape, hair length and caramel highlights 100% identical, same confident half-smile""",
        ),
        IdentityPrompt(
            mode=TrainingMode.STYLE_CYBERPUNK,
            name="style_cyberpunk_corporate",
            difficulty=3,
            description="Cyberpunk corporate style",
            prompt=f"""The same woman in a sleek cyberpunk corporate setting: wearing a tailored black blazer with subtle circuit-pattern embroidery, holographic ID badge, hair pulled back but same length and highlights visible, sitting at a glass desk with floating holographic displays, cool blue ambient lighting, face illuminated by screen glow, identical facial features and expression""",
        ),
    ],

    # =========================================================================
    # 6. MOTION - CASUAL (easiest temporal)
    # =========================================================================
    TrainingMode.MOTION_CASUAL: [
        IdentityPrompt(
            mode=TrainingMode.MOTION_CASUAL,
            name="motion_casual_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, sitting at a wooden café table near a large window, {BASE_LIGHTING}, holding a white ceramic coffee cup, {BASE_EXPRESSION}, static pose""",
        ),
        IdentityPrompt(
            mode=TrainingMode.MOTION_CASUAL,
            name="motion_casual_cafe",
            difficulty=1,
            description="Natural café gestures - easiest temporal lock",
            motion_intensity="low",
            prompt=f"""The same 32-year-old East Asian woman with shoulder-length straight black hair with caramel highlights, {BASE_OUTFIT}, sitting at a wooden café table near a large window, soft morning light, slowly turning the pages of an open notebook, occasionally tucking hair behind her ear, sipping from a white ceramic coffee cup, relaxed posture, very slight confident half-smile, camera slowly zooming in over 8 seconds, ultra-consistent identity across every frame""",
        ),
        IdentityPrompt(
            mode=TrainingMode.MOTION_CASUAL,
            name="motion_casual_phone",
            difficulty=1,
            description="Looking at phone - subtle motion",
            motion_intensity="low",
            prompt=f"""The same woman sitting at the café table, {BASE_LIGHTING}, looking down at smartphone in hands, occasionally scrolling with thumb, glancing up briefly then back to phone, same relaxed posture and outfit, identical face in every frame""",
        ),
    ],

    # =========================================================================
    # 7. MOTION - EXPRESSIVE (micro-expressions, lip-sync)
    # =========================================================================
    TrainingMode.MOTION_EXPRESSIVE: [
        IdentityPrompt(
            mode=TrainingMode.MOTION_EXPRESSIVE,
            name="motion_expressive_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, sitting on a modern gray sofa in a minimalist living room, soft warm lamp light, looking directly at camera, neutral expression""",
        ),
        IdentityPrompt(
            mode=TrainingMode.MOTION_EXPRESSIVE,
            name="motion_expressive_laugh",
            difficulty=2,
            description="Emotional progression - neutral to laugh",
            motion_intensity="medium",
            prompt=f"""The same woman, sitting on a modern gray sofa in a minimalist living room, soft warm lamp light, looking directly at camera, starting with neutral expression, then slowly smiling warmly, eyes crinkling, then laughing softly for 3 seconds, covering mouth with hand, then returning to gentle smile while tilting head slightly, ultra-consistent face and clothing in every frame""",
        ),
        IdentityPrompt(
            mode=TrainingMode.MOTION_EXPRESSIVE,
            name="motion_expressive_surprise",
            difficulty=2,
            description="Surprise reaction",
            motion_intensity="medium",
            prompt=f"""The same woman on the gray sofa, lamp light, looking at camera with neutral expression, then eyes widening in pleasant surprise, eyebrows raising, mouth opening slightly in an "oh!", then breaking into a delighted smile, same outfit and identical facial structure throughout""",
        ),
    ],

    # =========================================================================
    # 8. MOTION - DANCE (strong temporal test)
    # =========================================================================
    TrainingMode.MOTION_DANCE: [
        IdentityPrompt(
            mode=TrainingMode.MOTION_DANCE,
            name="motion_dance_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, wearing white ribbed turtleneck tucked into high-waisted light jeans, silver hoops, standing in an empty dance studio with floor-to-ceiling mirrors, soft even lighting, neutral standing pose""",
        ),
        IdentityPrompt(
            mode=TrainingMode.MOTION_DANCE,
            name="motion_dance_contemporary",
            difficulty=3,
            description="Contemporary dance - full body temporal test",
            motion_intensity="high",
            prompt=f"""The same 32-year-old East Asian woman with identical facial features and hair, wearing white ribbed turtleneck tucked into high-waisted light jeans, silver hoops, performing a smooth slow-tempo contemporary dance routine in an empty dance studio with floor-to-ceiling mirrors, soft even lighting, fluid arm waves, gentle hip sways, occasional hair flip, very consistent identity and outfit across the entire 12-second clip""",
        ),
        IdentityPrompt(
            mode=TrainingMode.MOTION_DANCE,
            name="motion_dance_spin",
            difficulty=3,
            description="Graceful spin - extreme motion test",
            motion_intensity="high",
            prompt=f"""The same woman in the dance studio, starting from standing pose, then executing a slow graceful spin with arms extended, hair flowing outward, then returning to face camera with the same confident half-smile, identical facial features visible through the entire rotation""",
        ),
    ],

    # =========================================================================
    # 9. COMBINED - GHIBLI (ID + Style + Motion)
    # =========================================================================
    TrainingMode.COMBINED_GHIBLI: [
        IdentityPrompt(
            mode=TrainingMode.COMBINED_GHIBLI,
            name="combined_ghibli_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, standing on a grassy hill at golden hour, wind gently blowing hair, photorealistic style""",
        ),
        IdentityPrompt(
            mode=TrainingMode.COMBINED_GHIBLI,
            name="combined_ghibli_animated",
            difficulty=4,
            description="Ghibli style with motion - hardest multi-modal",
            motion_intensity="medium",
            prompt=f"""The same woman, in a hand-drawn Studio Ghibli anime style with soft watercolor backgrounds, wearing her signature white ribbed turtleneck and light jeans, standing on a grassy hill at golden hour, wind gently blowing her hair, slowly raising arms as if embracing the sky, then spinning once gracefully, ultra-consistent facial features, eye shape, freckles, and clothing silhouette even in stylized animation""",
        ),
    ],

    # =========================================================================
    # 10. COMBINED - EFFECTS (ID + Effect + Camera) - Hardest
    # =========================================================================
    TrainingMode.COMBINED_EFFECTS: [
        IdentityPrompt(
            mode=TrainingMode.COMBINED_EFFECTS,
            name="combined_effects_reference",
            is_reference=True,
            difficulty=1,
            prompt=f"""{BASE_IDENTITY}, {BASE_OUTFIT}, sitting at a wooden café table near a large window, soft morning light, static camera, no effects""",
        ),
        IdentityPrompt(
            mode=TrainingMode.COMBINED_EFFECTS,
            name="combined_effects_bokeh_dolly",
            difficulty=5,
            description="Particles + dolly zoom - toughest temporal test",
            motion_intensity="medium",
            prompt=f"""The same woman sitting at the café table, soft morning light, slowly turning notebook pages and sipping coffee, but with beautiful golden particle bokeh drifting across the frame like fireflies, subtle lens flare on the window, very slow cinematic dolly zoom-in over 10 seconds, perfect identity preservation in every frame despite particles and camera movement""",
        ),
        IdentityPrompt(
            mode=TrainingMode.COMBINED_EFFECTS,
            name="combined_effects_rain",
            difficulty=5,
            description="Rain + reflections + motion",
            motion_intensity="medium",
            prompt=f"""The same woman now standing by a floor-to-ceiling window, watching rain fall outside, soft reflections of raindrops on her face, occasional lightning flash illuminating the scene, she slowly places hand on glass, then turns to look at camera with the same confident half-smile, identical facial features throughout despite complex lighting changes""",
        ),
    ],
}


# =============================================================================
# TEMPORAL WORKFLOW INPUTS
# =============================================================================

@dataclass
class VideoGenerationRequest:
    """Request for generating a single video."""
    prompt: str
    negative_prompt: str
    width: int = 832
    height: int = 448  # Landscape by default
    num_frames: int = 65  # ~4s at 16fps
    seed: int | None = None
    mode: TrainingMode = TrainingMode.PURE_ID
    prompt_name: str = ""
    output_dir: str = ""
    is_reference: bool = False


@dataclass
class DatasetGenerationConfig:
    """Configuration for dataset generation."""
    modes: list[TrainingMode]
    num_clips_per_prompt: int = 16
    output_dir: Path = Path("omnitransfer_dataset")
    backend: str = "ltx2"  # "ltx2" or "msi"
    seed_base: int = 42
    width: int = 832
    height: int = 448
    num_frames: int = 65
    dry_run: bool = False
    msi_url: str = "https://media-msi.covershot.app"


# =============================================================================
# VIDEO GENERATION FUNCTIONS
# =============================================================================

async def generate_video_ltx2(
    client: "Client",
    request: VideoGenerationRequest,
    task_queue: str = "presidential-dilemma-gemini",
) -> dict[str, Any]:
    """Generate video using local LTX-2 via Temporal workflow."""
    from app.temporal.ltx2_workflows import GenerateLTX2VideoWorkflow, LTX2VideoInput

    input_data = LTX2VideoInput(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_frames=request.num_frames,
        seed=request.seed,
    )

    workflow_id = f"omnitransfer-ltx2-{request.mode.value}-{request.prompt_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    result = await client.execute_workflow(
        GenerateLTX2VideoWorkflow.run,
        input_data,
        id=workflow_id,
        task_queue=task_queue,
        execution_timeout=timedelta(minutes=30),
    )

    return {
        "video_path": result.video_path,
        "video_url": result.video_url,
        "workflow_id": workflow_id,
    }


async def generate_video_msi(
    client: "Client",
    request: VideoGenerationRequest,
    msi_url: str,
    task_queue: str = "presidential-dilemma-gemini",
) -> dict[str, Any]:
    """Generate video using MSI server (WAN2.2 via ComfyUI) via Temporal workflow.

    Note: WAN2.2 requires a reference image for I2V. For T2V, we'd need to
    first generate an image then use I2V.
    """
    # For now, we'll use LTX-2 for T2V since WAN2.2 is I2V only
    # TODO: Add image generation step for pure T2V via MSI

    from app.temporal.wan2_workflows import GenerateWAN2VideoWorkflow, WAN2VideoInput

    # WAN2 requires a reference image - this would need to be generated first
    # For dry-run, we'll just return a placeholder
    print(f"  [MSI/WAN2.2] Would generate via {msi_url}")
    print(f"  Note: WAN2.2 requires reference image for I2V")

    return {
        "video_path": f"{request.output_dir}/placeholder.mp4",
        "video_url": None,
        "workflow_id": "msi-placeholder",
    }


def generate_video_dry_run(request: VideoGenerationRequest) -> dict[str, Any]:
    """Dry run - just print the prompt."""
    print(f"\n{'='*60}")
    print(f"Mode: {request.mode.value}")
    print(f"Name: {request.prompt_name}")
    print(f"Reference: {request.is_reference}")
    print(f"Dimensions: {request.width}x{request.height}, {request.num_frames} frames")
    print(f"Seed: {request.seed}")
    print(f"Output: {request.output_dir}")
    print(f"\nPrompt:\n{request.prompt[:200]}...")
    print(f"{'='*60}\n")

    return {
        "video_path": f"{request.output_dir}/dry_run.mp4",
        "video_url": None,
        "workflow_id": "dry-run",
    }


# =============================================================================
# DATASET GENERATION
# =============================================================================

async def generate_dataset(config: DatasetGenerationConfig):
    """Generate complete OmniTransfer training dataset."""

    print(f"\n{'#'*70}")
    print(f"# OmniTransfer Dataset Generation")
    print(f"# Backend: {config.backend}")
    print(f"# Modes: {[m.value for m in config.modes]}")
    print(f"# Clips per prompt: {config.num_clips_per_prompt}")
    print(f"# Output: {config.output_dir}")
    print(f"# Dry run: {config.dry_run}")
    print(f"{'#'*70}\n")

    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "reference_videos").mkdir(exist_ok=True)
    (config.output_dir / "target_videos").mkdir(exist_ok=True)
    (config.output_dir / "metadata").mkdir(exist_ok=True)

    # Get Temporal client if not dry run
    client = None
    if not config.dry_run and TEMPORAL_AVAILABLE:
        try:
            client = await get_temporal_client()
            print(f"Connected to Temporal server")
        except Exception as e:
            print(f"Warning: Could not connect to Temporal: {e}")
            print("Falling back to dry-run mode")
            config.dry_run = True

    # Track all generated pairs
    all_pairs = []
    total_videos = 0
    failed_videos = 0

    for mode in config.modes:
        prompts = IDENTITY_PROMPTS.get(mode, [])
        if not prompts:
            print(f"Warning: No prompts defined for mode {mode.value}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing mode: {mode.value}")
        print(f"Prompts: {len(prompts)}")
        print(f"{'='*60}")

        # Find reference prompt
        ref_prompts = [p for p in prompts if p.is_reference]
        target_prompts = [p for p in prompts if not p.is_reference]

        if not ref_prompts:
            print(f"Warning: No reference prompt for mode {mode.value}")
            continue

        ref_prompt = ref_prompts[0]

        # Generate reference videos
        print(f"\nGenerating {config.num_clips_per_prompt} reference clips...")
        ref_videos = []

        for clip_idx in range(config.num_clips_per_prompt):
            seed = config.seed_base + clip_idx

            request = VideoGenerationRequest(
                prompt=ref_prompt.prompt,
                negative_prompt=ref_prompt.negative_prompt,
                width=config.width,
                height=config.height,
                num_frames=config.num_frames,
                seed=seed,
                mode=mode,
                prompt_name=f"{ref_prompt.name}_{clip_idx:03d}",
                output_dir=str(config.output_dir / "reference_videos" / mode.value),
                is_reference=True,
            )

            try:
                if config.dry_run:
                    result = generate_video_dry_run(request)
                elif config.backend == "msi":
                    result = await generate_video_msi(client, request, config.msi_url)
                else:
                    result = await generate_video_ltx2(client, request)

                ref_videos.append({
                    "path": result["video_path"],
                    "seed": seed,
                    "clip_idx": clip_idx,
                })
                total_videos += 1

            except Exception as e:
                print(f"  Error generating reference clip {clip_idx}: {e}")
                failed_videos += 1

        # Generate target videos for each non-reference prompt
        for target_prompt in target_prompts:
            print(f"\nGenerating {config.num_clips_per_prompt} clips for: {target_prompt.name}")

            for clip_idx in range(config.num_clips_per_prompt):
                seed = config.seed_base + clip_idx

                request = VideoGenerationRequest(
                    prompt=target_prompt.prompt,
                    negative_prompt=target_prompt.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_frames=config.num_frames,
                    seed=seed,
                    mode=mode,
                    prompt_name=f"{target_prompt.name}_{clip_idx:03d}",
                    output_dir=str(config.output_dir / "target_videos" / mode.value),
                    is_reference=False,
                )

                try:
                    if config.dry_run:
                        result = generate_video_dry_run(request)
                    elif config.backend == "msi":
                        result = await generate_video_msi(client, request, config.msi_url)
                    else:
                        result = await generate_video_ltx2(client, request)

                    # Create pair entry
                    if clip_idx < len(ref_videos):
                        pair = {
                            "reference": ref_videos[clip_idx]["path"],
                            "target": result["video_path"],
                            "mode": mode.value,
                            "task_type": "temporal" if "motion" in mode.value or "combined" in mode.value else "appearance",
                            "ref_prompt": ref_prompt.prompt,
                            "tgt_prompt": target_prompt.prompt,
                            "seed": seed,
                            "difficulty": target_prompt.difficulty,
                            "motion_intensity": target_prompt.motion_intensity,
                        }
                        all_pairs.append(pair)

                    total_videos += 1

                except Exception as e:
                    print(f"  Error generating target clip {clip_idx}: {e}")
                    failed_videos += 1

    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "modes": [m.value for m in config.modes],
            "num_clips_per_prompt": config.num_clips_per_prompt,
            "backend": config.backend,
            "width": config.width,
            "height": config.height,
            "num_frames": config.num_frames,
            "seed_base": config.seed_base,
        },
        "statistics": {
            "total_videos": total_videos,
            "failed_videos": failed_videos,
            "total_pairs": len(all_pairs),
        },
        "pairs": all_pairs,
    }

    metadata_path = config.output_dir / "metadata" / "pairs.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'#'*70}")
    print(f"# Generation Complete")
    print(f"# Total videos: {total_videos}")
    print(f"# Failed: {failed_videos}")
    print(f"# Pairs created: {len(all_pairs)}")
    print(f"# Metadata saved to: {metadata_path}")
    print(f"{'#'*70}\n")

    return metadata


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate OmniTransfer training dataset via Temporal workflows"
    )

    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=[m.value for m in TrainingMode],
        help="Specific training mode to generate",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate all training modes",
    )
    parser.add_argument(
        "--appearance-only",
        action="store_true",
        help="Generate only appearance transfer modes",
    )
    parser.add_argument(
        "--motion-only",
        action="store_true",
        help="Generate only motion transfer modes",
    )

    # Generation parameters
    parser.add_argument(
        "--num-clips", "-n",
        type=int,
        default=16,
        help="Number of clips per prompt (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for generation (default: 42)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=448,
        help="Video height (default: 448)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=65,
        help="Number of frames (default: 65, ~4s at 16fps)",
    )

    # Backend
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["ltx2", "msi"],
        default="ltx2",
        help="Video generation backend (default: ltx2)",
    )
    parser.add_argument(
        "--msi-url",
        type=str,
        default="https://media-msi.covershot.app",
        help="MSI server URL for WAN2.2 generation",
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("omnitransfer_dataset"),
        help="Output directory (default: omnitransfer_dataset)",
    )

    # Dry run
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Print prompts without generating videos",
    )

    # List prompts
    parser.add_argument(
        "--list-prompts", "-l",
        action="store_true",
        help="List all available prompts and exit",
    )

    return parser.parse_args()


def list_all_prompts():
    """Print all available prompts."""
    print("\n" + "="*80)
    print("OMNITRANSFER IDENTITY PROMPTS")
    print("="*80)

    for mode, prompts in IDENTITY_PROMPTS.items():
        print(f"\n{'─'*80}")
        print(f"MODE: {mode.value}")
        print(f"{'─'*80}")

        for prompt in prompts:
            ref_marker = "📌 [REFERENCE]" if prompt.is_reference else ""
            diff_stars = "★" * prompt.difficulty + "☆" * (5 - prompt.difficulty)

            print(f"\n  {prompt.name} {ref_marker}")
            print(f"  Difficulty: {diff_stars}")
            print(f"  Motion: {prompt.motion_intensity}")
            if prompt.description:
                print(f"  Description: {prompt.description}")
            print(f"  Prompt: {prompt.prompt[:100]}...")

    print("\n" + "="*80 + "\n")


def main():
    args = parse_args()

    if args.list_prompts:
        list_all_prompts()
        return

    # Determine modes to generate
    modes = []
    if args.all:
        modes = list(TrainingMode)
    elif args.appearance_only:
        modes = [
            TrainingMode.PURE_ID,
            TrainingMode.ID_AGING,
            TrainingMode.ID_ACCESSORIES,
            TrainingMode.STYLE_PAINTERLY,
            TrainingMode.STYLE_CYBERPUNK,
        ]
    elif args.motion_only:
        modes = [
            TrainingMode.MOTION_CASUAL,
            TrainingMode.MOTION_EXPRESSIVE,
            TrainingMode.MOTION_DANCE,
        ]
    elif args.mode:
        modes = [TrainingMode(args.mode)]
    else:
        print("Error: Must specify --mode, --all, --appearance-only, or --motion-only")
        return

    config = DatasetGenerationConfig(
        modes=modes,
        num_clips_per_prompt=args.num_clips,
        output_dir=args.output_dir,
        backend=args.backend,
        seed_base=args.seed,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        dry_run=args.dry_run,
        msi_url=args.msi_url,
    )

    asyncio.run(generate_dataset(config))


if __name__ == "__main__":
    from datetime import timedelta
    main()
