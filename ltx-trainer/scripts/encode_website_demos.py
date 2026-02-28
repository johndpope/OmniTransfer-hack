#!/usr/bin/env python3
"""Encode OmniTransfer website demos into training latents.

This script processes the downloaded website demos:
- ref{N}.jpg → target_image_latents/{N}.pt (static image to animate)
- ref{N}.mp4 → reference_latents/{N}.pt (effect/motion source video)
- result{N}.mp4 → latents/{N}.pt (ground truth - OmniTransfer output)

SMART ASPECT RATIO HANDLING:
- Detects aspect ratio of each result video (ground truth)
- Encodes at appropriate dimensions (landscape 832x448 or portrait 448x832)
- Supports mixed aspect ratios in same dataset (batch_size=1)

Usage:
    python scripts/encode_website_demos.py \
        --input-dir /media/2TB/omnitransfer_website_demos \
        --output-dir /media/2TB/omnitransfer_effect_motion \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma
"""

import argparse
import gc
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder, load_text_encoder
from ltx_trainer.video_utils import read_video


# Supported resolutions (must be divisible by 32)
LANDSCAPE_DIMS = (448, 832)  # (H, W) - 1.86 aspect ratio
PORTRAIT_DIMS = (832, 448)   # (H, W) - 0.54 aspect ratio

# ============================================================================
# Per-sample captions from OmniTransfer website demos
# These match the actual text prompts used in the paper/website
# ============================================================================
SAMPLE_CAPTIONS = {
    # Effect Video Transfer demos
    "Effect/ref1": "A person with their eyes closed, transitioning from a calm to an expressive state with dramatic lighting effects.",
    "Effect/ref2": "A person's face with artistic visual effects, dramatic lighting and color transformations.",
    "Effect/ref3": "A person with expressive facial movements, artistic visual effects applied.",
    "Effect/ref4": "A person's portrait with stylized visual effects and dynamic lighting.",
    "Effect/ref5": "A person with artistic color grading and visual effects transformation.",
    "Effect/ref6": "A person with creative visual effects and dramatic artistic styling.",

    # Motion Video Transfer demos (pose-free animation)
    "Motion/ref1": "A man with a rugged beard, wearing a leather jacket, riding a vintage motorcycle along a desert highway.",
    "Motion/ref2": "A curly-haired person in a white short-sleeved shirt stands in front of a pink background. At first, the eyes are closed and the expression looks a bit sleepy. Then the mouth slowly opens wide as if yawning, and the hand raises to cover the mouth. During the yawn, the body posture remains basically stable.",
    "Motion/ref3": "A person performing dynamic body movements with expressive gestures.",
    "Motion/ref4": "A person walking with natural body motion and arm movements.",
    "Motion/ref5": "A person with animated facial expressions and head movements.",
    "Motion/ref6": "A person performing smooth body motion with natural movements.",

    # ID Video Transfer demos (identity preservation)
    "ID/ref1": "a blonde-haired woman in a black top, gently touches and bends down to the flowers picturesque background of more greenery and white flowers",
    "ID/ref2": "a man with a rugged beard, wearing a leather jacket, riding a vintage motorcycle along a desert highway.",
    "ID/ref3": "Sitting on a comfortable beige upholstered sofa in a room with a gray-blue background wall, a figure wearing a green plaid shirt has white round table in front of them, holding a piece of paper in one hand and supporting their head with the other, showing a slightly distressed expression, the tables are adorned with notebooks, pens, and small black objects, the figure makes slight movements throughout the scene, occasionally flipping through the paper and fidgeting with items on the table, with a green potted plant standing quietly beside the sofa.",
    "ID/ref4": "A person in a light blue denim jacket and white pants sits on a beach chair at sunset, reading a book and occasionally looking up to watch the sun dip below the horizon, with the sky turning shades of orange, and the sound of waves crashing on the shore.",
    "ID/ref5": "At a desk, a figure sits with an open book in front of them, surrounded by notebooks, colored pencils, and a pencil holder filled with vibrant colors. The figure's hands move gently as they read, shifting the pencils or flipping through pages. In the background, a figure wearing an apron moves casually through the kitchen, completing the warm, homey scene with a quiet energy.",
    "ID/ref6": "A person wearing a light brown leather jacket and dark jeans sat on a park bench, playing guitar and singing a soft folk song. There are two boxes next to me with some coins, and some people stop to listen.",

    # Style Video Transfer demos
    "Style/ref1": "A curly-haired person in a white short-sleeved shirt stands in front of a pink background. At first, the eyes are closed and the expression looks a bit sleepy. Then the mouth slowly opens wide as if yawning, and the hand raises to cover the mouth. During the yawn, the body posture remains basically stable.",
    "Style/ref2": "The picture shows a metallic arched passage. The setting-sun shines through the grids on the side of the passage, creating a warm-yellow halo. Several pedestrians are walking inside the passage, and their figures are slightly blurred due to the light and movement. The ground of the passage is flat, and there is a railing on one side. As the picture progresses, a person wearing red clothes and carrying a backpack enters the passage. The overall atmosphere is tranquil and a bit warm.",
    "Style/ref3": "A woman sits on the stairs, wearing a white patterned-top and dark pants. The staircase handrail is faintly visible in the background. There is a lit candle behind, giving off a soft light. At first, the woman looks sad with her eyes slightly closed. Then she slowly raises her hand to touch her face and hold her head, seemingly immersed in painful emotions. The overall atmosphere is somewhat depressing.",
    "Style/ref4": "Indoors, a black Mercedes-Benz sedan is parked on a metal platform. Behind the car are several motorcycles in colors like red and blue. A man wearing a black top and jeans, holding a piece of paper. Then the trunk of the car slowly opens. The man stands still and looks away. The whole scene seems to be a vehicle display or inspection.",
    "Style/ref5": "The video shows a person watering a row of plants on a balcony, tilting the watering can slowly over each pot.",
    "Style/ref6": "Reading book under the tree",
}

# Fallback task-specific captions
TASK_CAPTIONS = {
    "effect": "Apply visual effect from reference video to the target subject. Video with artistic effect transfer.",
    "motion": "Animate the target image with motion from the reference video. Video with natural motion transfer.",
    "camera": "Apply camera movement from reference video to the static scene. Video with camera motion.",
    "id": "Transfer the identity from reference to the target scene while preserving appearance. Video with identity preservation.",
    "style": "Apply the artistic style from reference video to the target content. Video with style transfer.",
}


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Get video dimensions using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    w, h = map(int, result.stdout.strip().split(","))
    return w, h


def get_target_dimensions(video_path: Path) -> tuple[int, int]:
    """Determine target encoding dimensions based on video aspect ratio."""
    w, h = get_video_dimensions(video_path)
    aspect = w / h

    if aspect > 1.0:  # Landscape
        return LANDSCAPE_DIMS
    else:  # Portrait or square
        return PORTRAIT_DIMS


def resize_for_vae(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize tensor to target dimensions (must be divisible by 32)."""
    if tensor.dim() == 4:  # [F, C, H, W]
        return F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    elif tensor.dim() == 3:  # [C, H, W]
        return F.interpolate(tensor.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
    else:
        raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")


def load_image(path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load image and convert to tensor [1, C, 1, H, W] for VAE.

    Uses center cropping to preserve aspect ratio.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    target_aspect = target_w / target_h
    source_aspect = w / h

    if abs(source_aspect - target_aspect) > 0.01:
        # Aspect ratios differ - center crop first
        if source_aspect > target_aspect:
            # Source is wider - crop width
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img.crop((start_x, 0, start_x + new_w, h))
        else:
            # Source is taller - crop height
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img.crop((0, start_y, w, start_y + new_h))

    # Now resize to exact target dimensions
    img = img.resize((target_w, target_h), Image.LANCZOS)

    # Convert to tensor [C, H, W] in [0, 1]
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # Convert to [B, C, F, H, W] format with single frame
    tensor = tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]

    # Normalize to [-1, 1] for VAE
    tensor = tensor * 2.0 - 1.0

    return tensor


def center_crop_and_resize(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Center crop to match target aspect ratio, then resize.

    This preserves the proper proportions of objects instead of stretching.
    """
    _, _, h, w = tensor.shape  # [F, C, H, W]
    target_aspect = target_w / target_h
    source_aspect = w / h

    if abs(source_aspect - target_aspect) < 0.01:
        # Aspect ratios are close enough, just resize
        return F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

    if source_aspect > target_aspect:
        # Source is wider - crop width (center crop horizontally)
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        tensor = tensor[:, :, :, start_x:start_x + new_w]
    else:
        # Source is taller - crop height (center crop vertically)
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        tensor = tensor[:, :, start_y:start_y + new_h, :]

    # Now resize to exact target dimensions
    return F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)


def load_video_frames(path: Path, max_frames: int, target_h: int, target_w: int) -> torch.Tensor:
    """Load video and convert to tensor [1, C, F, H, W] for VAE."""
    frames, fps = read_video(path, max_frames=max_frames)  # [F, C, H, W]

    # Center crop and resize to target dimensions (preserves proportions)
    frames = center_crop_and_resize(frames, target_h, target_w)

    # Trim to valid frame count (k*8 + 1)
    valid_frames = (frames.shape[0] - 1) // 8 * 8 + 1
    valid_frames = min(valid_frames, max_frames)
    frames = frames[:valid_frames]

    # Convert to [B, C, F, H, W]
    frames = rearrange(frames, "f c h w -> 1 c f h w")

    # Normalize to [-1, 1] for VAE
    frames = frames * 2.0 - 1.0

    return frames, fps


def encode_with_vae(vae_encoder, tensor: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Encode tensor with VAE."""
    tensor = tensor.to(device, dtype=dtype)
    with torch.inference_mode():
        latent = vae_encoder(tensor)
    return latent.cpu()


def main():
    parser = argparse.ArgumentParser(description="Encode OmniTransfer website demos")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with downloaded demos")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for latents")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to LTX-2 model")
    parser.add_argument("--text-encoder-path", type=Path, required=True, help="Path to Gemma")
    parser.add_argument("--auto-aspect", action="store_true", default=True,
                       help="Auto-detect aspect ratio from result video (default: True)")
    parser.add_argument("--target-height", type=int, default=448, help="Target height if not auto (default: 448)")
    parser.add_argument("--target-width", type=int, default=832, help="Target width if not auto (default: 832)")
    parser.add_argument("--max-frames", type=int, default=65, help="Max frames (default: 65, must be k*8+1)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--tasks", type=str, nargs="+", default=["Effect", "Motion"], help="Tasks to process")
    args = parser.parse_args()

    # Validate frame count
    if args.max_frames % 8 != 1:
        raise ValueError(f"max_frames must be k*8+1, got {args.max_frames}")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # Create output directories
    target_img_dir = args.output_dir / "target_image_latents"
    ref_lat_dir = args.output_dir / "reference_latents"
    gt_lat_dir = args.output_dir / "latents"
    cond_dir = args.output_dir / "conditions"

    for d in [target_img_dir, ref_lat_dir, gt_lat_dir, cond_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all triplets with per-sample aspect ratio detection
    triplets = []
    aspect_stats = {"landscape": 0, "portrait": 0}

    for task in args.tasks:
        task_dir = args.input_dir / task
        if not task_dir.exists():
            logger.warning(f"Task directory not found: {task_dir}")
            continue

        for i in range(1, 20):  # Check up to 20 samples per task
            ref_img = task_dir / f"ref{i}.jpg"
            ref_vid = task_dir / f"ref{i}.mp4"
            result_vid = task_dir / f"result{i}.mp4"

            if ref_img.exists() and ref_vid.exists() and result_vid.exists():
                # Detect aspect ratio from result video (ground truth)
                if args.auto_aspect:
                    target_h, target_w = get_target_dimensions(result_vid)
                    orig_w, orig_h = get_video_dimensions(result_vid)
                    is_landscape = target_w > target_h
                    aspect_stats["landscape" if is_landscape else "portrait"] += 1
                else:
                    target_h, target_w = args.target_height, args.target_width
                    orig_w, orig_h = target_w, target_h
                    is_landscape = target_w > target_h

                # Get per-sample caption or fallback to task caption
                caption_key = f"{task}/ref{i}"
                caption = SAMPLE_CAPTIONS.get(caption_key, TASK_CAPTIONS.get(task.lower(), "Video transfer task."))

                triplets.append({
                    "idx": len(triplets),
                    "task": task.lower(),
                    "ref_image": ref_img,
                    "ref_video": ref_vid,
                    "result_video": result_vid,
                    "target_h": target_h,
                    "target_w": target_w,
                    "original_dims": (orig_w, orig_h),
                    "is_landscape": is_landscape,
                    "caption_key": caption_key,
                    "caption": caption,
                })
                orient = "LANDSCAPE" if is_landscape else "PORTRAIT"
                logger.info(f"Found triplet: {task}/ref{i} - {orig_w}x{orig_h} → {target_w}x{target_h} ({orient})")

    logger.info(f"Found {len(triplets)} complete triplets")
    logger.info(f"Aspect ratio distribution: {aspect_stats}")

    if not triplets:
        logger.error("No triplets found!")
        return

    # ========================================================================
    # Stage 1: Encode all media with VAE
    # ========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: Encoding with VAE")
    logger.info("=" * 60)

    vae_encoder = load_video_vae_encoder(args.model_path, dtype=dtype)
    vae_encoder = vae_encoder.to(args.device)
    vae_encoder.eval()

    for triplet in tqdm(triplets, desc="Encoding"):
        idx = triplet["idx"]
        target_h = triplet["target_h"]
        target_w = triplet["target_w"]
        orient = "LANDSCAPE" if triplet["is_landscape"] else "PORTRAIT"

        # 1. Encode reference IMAGE (target to animate) - use SAME dims as result video
        img_out = target_img_dir / f"{idx:03d}.pt"
        if not img_out.exists():
            try:
                img_tensor = load_image(triplet["ref_image"], target_h, target_w)
                img_latent = encode_with_vae(vae_encoder, img_tensor, args.device, dtype)
                torch.save({
                    "latents": img_latent.squeeze(0),  # [C, 1, H, W]
                    "num_frames": torch.tensor([1]),
                    "height": torch.tensor([img_latent.shape[3]]),
                    "width": torch.tensor([img_latent.shape[4]]),
                    "task_type": triplet["task"],
                    "orientation": orient,
                }, img_out)
            except Exception as e:
                logger.error(f"Error encoding image {triplet['ref_image']}: {e}")
                continue

        # 2. Encode reference VIDEO (effect/motion source) - use SAME dims as result video
        ref_out = ref_lat_dir / f"{idx:03d}.pt"
        if not ref_out.exists():
            try:
                ref_tensor, fps = load_video_frames(triplet["ref_video"], args.max_frames, target_h, target_w)
                ref_latent = encode_with_vae(vae_encoder, ref_tensor, args.device, dtype)
                torch.save({
                    "latents": ref_latent.squeeze(0),  # [C, F, H, W]
                    "num_frames": torch.tensor([ref_latent.shape[2]]),
                    "height": torch.tensor([ref_latent.shape[3]]),
                    "width": torch.tensor([ref_latent.shape[4]]),
                    "fps": torch.tensor([fps]),
                    "task_type": triplet["task"],
                    "orientation": orient,
                }, ref_out)
            except Exception as e:
                logger.error(f"Error encoding ref video {triplet['ref_video']}: {e}")
                continue

        # 3. Encode RESULT video (ground truth)
        gt_out = gt_lat_dir / f"{idx:03d}.pt"
        if not gt_out.exists():
            try:
                gt_tensor, fps = load_video_frames(triplet["result_video"], args.max_frames, target_h, target_w)
                gt_latent = encode_with_vae(vae_encoder, gt_tensor, args.device, dtype)
                torch.save({
                    "latents": gt_latent.squeeze(0),  # [C, F, H, W]
                    "num_frames": torch.tensor([gt_latent.shape[2]]),
                    "height": torch.tensor([gt_latent.shape[3]]),
                    "width": torch.tensor([gt_latent.shape[4]]),
                    "fps": torch.tensor([fps]),
                    "task_type": triplet["task"],
                    "orientation": orient,
                }, gt_out)
            except Exception as e:
                logger.error(f"Error encoding result video {triplet['result_video']}: {e}")
                continue

        # Clear cache
        torch.cuda.empty_cache()

    # Unload VAE
    del vae_encoder
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("VAE encoder unloaded")

    # ========================================================================
    # Stage 2: Compute text embeddings
    # ========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Computing text embeddings")
    logger.info("=" * 60)

    text_encoder = load_text_encoder(
        checkpoint_path=args.model_path,
        gemma_model_path=args.text_encoder_path,
        device=args.device,
        dtype=dtype,
    )
    text_encoder.eval()

    for triplet in tqdm(triplets, desc="Computing embeddings"):
        idx = triplet["idx"]
        cond_out = cond_dir / f"{idx:03d}.pt"

        if cond_out.exists():
            continue

        # Use per-sample caption (already set in triplet)
        caption = triplet["caption"]
        logger.debug(f"Sample {idx} ({triplet['caption_key']}): {caption[:60]}...")

        try:
            with torch.inference_mode():
                # Use _preprocess_text for raw embeddings (required by trainer)
                # Returns (projected_features, attention_mask) tuple with batch dim
                prompt_embeds, attention_mask = text_encoder._preprocess_text(caption, padding_side="left")

            # Save WITHOUT batch dimension (trainer expects [seq, hidden] not [1, seq, hidden])
            torch.save({
                "prompt_embeds": prompt_embeds[0].cpu().contiguous(),
                "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
                "caption": caption,  # Store caption for reference
            }, cond_out)
        except Exception as e:
            logger.error(f"Error computing embeddings for {idx}: {e}")

    # Unload text encoder
    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Text encoder unloaded")

    # ========================================================================
    # Save metadata
    # ========================================================================
    metadata = {
        "total_pairs": len(triplets),
        "tasks": list(set(t["task"] for t in triplets)),
        "task_counts": {},
        "aspect_ratio_distribution": aspect_stats,
        "config": {
            "num_frames": args.max_frames,
            "training_type": "website_demos",
            "auto_aspect_ratio": args.auto_aspect,
            "supported_dimensions": {
                "landscape": f"{LANDSCAPE_DIMS[0]}x{LANDSCAPE_DIMS[1]}",
                "portrait": f"{PORTRAIT_DIMS[0]}x{PORTRAIT_DIMS[1]}",
            },
            "source": "https://pangzecheung.github.io/OmniTransfer/",
            "note": "Ground truth from OmniTransfer website demos - MIXED aspect ratios",
        },
        "pairs": [],
    }

    for task in metadata["tasks"]:
        metadata["task_counts"][task] = sum(1 for t in triplets if t["task"] == task)

    for triplet in triplets:
        metadata["pairs"].append({
            "id": triplet["idx"],
            "task_type": triplet["task"],
            "target_image": triplet["ref_image"].name,
            "reference_video": triplet["ref_video"].name,
            "ground_truth_video": triplet["result_video"].name,
            "is_website_demo": True,
            "dimensions": f"{triplet['target_h']}x{triplet['target_w']}",
            "orientation": "landscape" if triplet["is_landscape"] else "portrait",
            "original_dims": f"{triplet['original_dims'][0]}x{triplet['original_dims'][1]}",
            "caption": triplet["caption"],
        })

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ========================================================================
    # Verify dataset
    # ========================================================================
    logger.info("=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    # Check that reference and ground truth are DIFFERENT
    for triplet in triplets[:3]:  # Check first 3
        idx = triplet["idx"]
        ref = torch.load(ref_lat_dir / f"{idx:03d}.pt", weights_only=False)["latents"]
        gt = torch.load(gt_lat_dir / f"{idx:03d}.pt", weights_only=False)["latents"]
        diff = (ref - gt).abs().mean().item()
        logger.info(f"Triplet {idx}: ref vs ground_truth diff = {diff:.4f} {'✓ GOOD' if diff > 0.1 else '✗ BAD'}")

    logger.info(f"\nDataset created at: {args.output_dir}")
    logger.info(f"Total triplets: {len(triplets)}")
    logger.info(f"Tasks: {metadata['task_counts']}")


if __name__ == "__main__":
    main()
