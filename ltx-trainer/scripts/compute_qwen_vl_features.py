#!/usr/bin/env python3
"""Compute Qwen VL features for TMA (Task-adaptive Multimodal Alignment).

This script pre-computes Qwen VL hidden states for each video in the dataset,
enabling TMA training without loading the large MLLM during training (~14GB saved).

The script extracts visual and semantic features from reference and target videos
using Qwen2-VL, Qwen2.5-VL, or Qwen3-VL models.

Output format in qwen_vl_features/*.pt:
    - qwen_features: Tensor[seq_len, hidden_dim] - MLLM hidden states
    - task_type: str - Transfer task type (motion_transfer, style_transfer, etc.)
    - caption: str - Video caption/prompt
    - num_ref_frames: int - Number of reference frames used
    - model_name: str - Qwen VL model used

Usage:
    # Compute features using Qwen2.5-VL-7B
    python scripts/compute_qwen_vl_features.py \
        --dataset-dir /media/2TB/omnitransfer_unified_5task \
        --model-path Qwen/Qwen2.5-VL-7B-Instruct \
        --num-ref-frames 4

    # With custom output directory
    python scripts/compute_qwen_vl_features.py \
        --dataset-dir /media/2TB/omnitransfer_unified_5task \
        --model-path /path/to/local/qwen2.5-vl-7b \
        --output-dir /path/to/output/qwen_vl_features
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.omnitransfer.qwen_vl_integration import (
    QWEN_HIDDEN_DIMS,
    detect_qwen_model_type,
)

# System prompt for TMA feature extraction
SYSTEM_PROMPT = """You are an expert video analysis assistant. Analyze the reference and target content to understand:
1. The visual characteristics and identity features in the reference
2. The content and context of the target
3. How the transfer should be performed based on the task type

Provide a detailed understanding that can guide video generation."""

# Task-specific prompts for different transfer types
TASK_PROMPTS = {
    "motion_transfer": (
        "Task: Motion Transfer\n"
        "Analyze the motion patterns in the reference video and understand how they should "
        "be applied to animate the target image while preserving its visual identity."
    ),
    "pose_reenactment": (
        "Task: Pose Reenactment\n"
        "Extract the pose sequence from the reference and understand how to apply it "
        "to the target subject while maintaining their identity."
    ),
    "style_transfer": (
        "Task: Style Transfer\n"
        "Identify the visual style characteristics (colors, textures, artistic effects) "
        "in the reference and understand how to apply them to the target content."
    ),
    "identity_preservation": (
        "Task: Identity Preservation\n"
        "Focus on the identity features of the subject in the reference (face, appearance, "
        "distinguishing characteristics) and understand how to maintain these in new contexts."
    ),
    "action_customization": (
        "Task: Action Customization\n"
        "Analyze the specific action being performed in the reference and understand "
        "how to transfer it to the target subject."
    ),
    "scene_composition": (
        "Task: Scene Composition\n"
        "Extract scene elements and composition from the reference and understand "
        "how to integrate the target subject into this context."
    ),
    "movie_weaver": (
        "Task: Multi-Concept Personalization\n"
        "Identify the multiple concepts/identities in the reference images and understand "
        "how to maintain separation while generating coherent video content."
    ),
    # Short name aliases
    "motion": "Task: Motion Transfer\nAnalyze motion patterns for transfer.",
    "camera": "Task: Camera Movement\nExtract camera motion for transfer.",
    "effect": "Task: Effect Transfer\nIdentify visual effects for transfer.",
    "id": "Task: Identity Preservation\nFocus on identity features.",
    "style": "Task: Style Transfer\nExtract style characteristics.",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute Qwen VL features for TMA training"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing latents/ and metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen VL model path (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: dataset-dir/qwen_vl_features)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Directory containing source MP4 videos (default: dataset-dir/videos)",
    )
    parser.add_argument(
        "--num-ref-frames",
        type=int,
        default=4,
        help="Number of reference frames to extract per video",
    )
    parser.add_argument(
        "--target-frame",
        type=str,
        default="middle",
        choices=["first", "middle", "last"],
        help="Which frame to use as target (first/middle/last)",
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
        help="Load model in 8-bit for memory efficiency",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit for maximum memory efficiency",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (1 recommended for VRAM efficiency)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=28 * 28 * 576,
        help="Maximum pixels for image processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing features",
    )
    parser.add_argument(
        "--use-flash-attention",
        action="store_true",
        default=False,
        help="Use Flash Attention 2 (requires flash_attn package)",
    )
    return parser.parse_args()


def extract_frames_from_video(
    video_path: Path,
    num_ref_frames: int = 4,
    target_frame: str = "middle",
) -> tuple[list[Image.Image], Image.Image]:
    """Extract reference frames and target frame from a video.

    Args:
        video_path: Path to the video file
        num_ref_frames: Number of reference frames to extract (uniformly sampled)
        target_frame: Which frame to use as target ("first", "middle", "last")

    Returns:
        Tuple of (reference_frames, target_frame)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required: pip install opencv-python")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        raise ValueError(f"Video has insufficient frames: {total_frames}")

    # Calculate frame indices for reference (uniform sampling)
    ref_indices = [int(i * (total_frames - 1) / (num_ref_frames - 1)) for i in range(num_ref_frames)]

    # Calculate target frame index
    if target_frame == "first":
        target_idx = 0
    elif target_frame == "last":
        target_idx = total_frames - 1
    else:  # middle
        target_idx = total_frames // 2

    # Extract frames
    ref_frames = []
    target_frame_img = None
    all_indices = sorted(set(ref_indices + [target_idx]))

    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {idx} from {video_path}")
            continue

        # Convert BGR to RGB and to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        if idx in ref_indices:
            ref_frames.append(pil_frame)
        if idx == target_idx:
            target_frame_img = pil_frame

    cap.release()

    if len(ref_frames) == 0:
        raise ValueError(f"No reference frames extracted from {video_path}")
    if target_frame_img is None:
        target_frame_img = ref_frames[len(ref_frames) // 2]  # Fallback to middle ref

    return ref_frames, target_frame_img


def load_qwen_model(args):
    """Load Qwen VL model and processor."""
    try:
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError:
        raise ImportError("transformers required: pip install transformers>=4.43")

    model_type, hidden_dim = detect_qwen_model_type(args.model_path)
    logger.info(f"Loading Qwen VL model: {args.model_path} (type: {model_type}, hidden_dim: {hidden_dim})")

    # Setup dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Common loading kwargs
    load_kwargs = {
        "dtype": dtype,  # Updated from torch_dtype
        "device_map": "auto" if args.load_in_8bit or args.load_in_4bit else args.device,
    }

    if args.use_flash_attention:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    # Use BitsAndBytesConfig for quantization (modern API)
    if args.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )

    # Load model based on type
    if model_type == "qwen3vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_path, **load_kwargs
        )
    elif model_type == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, **load_kwargs
        )
    elif model_type == "qwen2.5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, **load_kwargs
        )
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, **load_kwargs
        )

    model.eval()
    model.config.use_cache = False

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path)

    logger.info(f"Model loaded: {model.__class__.__name__}")
    return model, processor, hidden_dim


def build_conversation(
    ref_frames: list[Image.Image],
    target_frame: Image.Image,
    caption: str,
    task_type: str,
) -> list[dict[str, Any]]:
    """Build conversation format for Qwen VL.

    Args:
        ref_frames: List of reference frame images
        target_frame: Target frame image
        caption: Video caption
        task_type: Transfer task type

    Returns:
        Conversation messages for Qwen VL
    """
    # Get task-specific prompt
    task_prompt = TASK_PROMPTS.get(task_type, TASK_PROMPTS.get("motion_transfer"))

    # Build content with images
    content = []

    # Add reference frames
    for i, _ in enumerate(ref_frames):
        content.append({"type": "image"})
        if i == 0:
            content.append({"type": "text", "text": f"Reference frame {i+1} (of {len(ref_frames)}):"})

    # Add target frame
    content.append({"type": "image"})
    content.append({"type": "text", "text": "Target frame:"})

    # Add task prompt and caption
    content.append({
        "type": "text",
        "text": f"\n{task_prompt}\n\nCaption: {caption}\n\nAnalyze the visual content and provide understanding for video generation."
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    return messages


@torch.inference_mode()
def extract_features(
    model,
    processor,
    ref_frames: list[Image.Image],
    target_frame: Image.Image,
    caption: str,
    task_type: str,
    device: str,
) -> torch.Tensor:
    """Extract MLLM hidden states for TMA.

    Args:
        model: Qwen VL model
        processor: Qwen VL processor
        ref_frames: Reference frame images
        target_frame: Target frame image
        caption: Video caption
        task_type: Transfer task type
        device: Device to use

    Returns:
        Hidden states tensor [seq_len, hidden_dim]
    """
    # Build conversation
    messages = build_conversation(ref_frames, target_frame, caption, task_type)

    # Combine all images
    all_images = ref_frames + [target_frame]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs
    inputs = processor(
        text=[text],
        images=all_images,
        padding=True,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Forward pass with hidden states
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

    # Get last hidden state [1, seq_len, hidden_dim]
    hidden_states = outputs.hidden_states[-1]

    # Return without batch dimension [seq_len, hidden_dim]
    return hidden_states[0].cpu()


def find_video_path(
    videos_dir: Path,
    pair: dict,
    idx: int,
) -> Path | None:
    """Find video path using various strategies.

    Handles both flat directory structure and task-organized structure.
    """
    # Task type to directory mapping (for website demos structure)
    TASK_DIR_MAP = {
        "effect": "Effect",
        "motion": "Motion",
        "camera": "Camera",
        "id": "ID",
        "style": "Style",
        "style_transfer": "Style",
        "motion_transfer": "Motion",
        "identity_preservation": "ID",
    }

    video_name = pair.get("video", pair.get("reference_video", f"{idx:03d}.mp4"))
    task_type = pair.get("task_type", "")

    # Strategy 1: Try direct path
    video_path = videos_dir / video_name
    if video_path.exists():
        return video_path

    # Strategy 2: Try task-organized directory (website demos structure)
    if task_type:
        task_dir = TASK_DIR_MAP.get(task_type.lower(), task_type.capitalize())
        video_path = videos_dir / task_dir / video_name
        if video_path.exists():
            return video_path

    # Strategy 3: Try indexed filename
    for ext in [".mp4", ".webm", ".avi", ".mov"]:
        video_path = videos_dir / f"{idx:03d}{ext}"
        if video_path.exists():
            return video_path

    # Strategy 4: Search in subdirectories
    for subdir in videos_dir.iterdir():
        if subdir.is_dir():
            video_path = subdir / video_name
            if video_path.exists():
                return video_path

    return None


def main():
    args = parse_args()

    # Setup directories
    output_dir = args.output_dir or (args.dataset_dir / "qwen_vl_features")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = args.videos_dir or (args.dataset_dir / "videos")

    # Load metadata
    metadata_file = args.dataset_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found in {args.dataset_dir}")

    with open(metadata_file) as f:
        metadata = json.load(f)

    pairs = metadata.get("pairs", [])
    if not pairs:
        raise ValueError("No pairs found in metadata.json")

    logger.info(f"Found {len(pairs)} samples to process")
    logger.info(f"Videos directory: {videos_dir}")

    # Load model
    model, processor, hidden_dim = load_qwen_model(args)
    logger.info(f"Qwen VL model loaded with hidden_dim={hidden_dim}")

    # Process each sample
    processed = 0
    skipped = 0
    errors = 0

    for pair in tqdm(pairs, desc="Computing Qwen VL features"):
        idx = pair["id"]
        output_file = output_dir / f"{idx:03d}.pt"

        if output_file.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            # Get video path using smart path finding
            video_path = find_video_path(videos_dir, pair, idx)

            if video_path is None:
                video_name = pair.get("video", pair.get("reference_video", f"{idx:03d}.mp4"))
                logger.warning(f"Video not found for sample {idx}: {video_name}")
                errors += 1
                continue

            # Extract frames
            ref_frames, target_frame = extract_frames_from_video(
                video_path,
                num_ref_frames=args.num_ref_frames,
                target_frame=args.target_frame,
            )

            # Get caption and task type
            caption = pair.get("caption", "A video")
            task_type = pair.get("task_type", metadata.get("task_type", "motion_transfer"))

            # Extract features
            features = extract_features(
                model, processor,
                ref_frames, target_frame,
                caption, task_type,
                args.device,
            )

            # Save features
            torch.save({
                "qwen_features": features.contiguous(),  # [seq_len, hidden_dim]
                "task_type": task_type,
                "caption": caption,
                "num_ref_frames": len(ref_frames),
                "model_name": args.model_path,
                "hidden_dim": hidden_dim,
            }, output_file)

            processed += 1

            # Clear cache periodically
            if processed % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            errors += 1
            continue

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Qwen VL Feature Extraction Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped (existing): {skipped}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Output: {output_dir}")

    # Verify output
    output_files = list(output_dir.glob("*.pt"))
    logger.info(f"Total feature files: {len(output_files)}")

    if output_files:
        sample = torch.load(output_files[0], map_location="cpu", weights_only=True)
        logger.info("\nSample output format:")
        for key, val in sample.items():
            if hasattr(val, "shape"):
                logger.info(f"  {key}: {val.shape} ({val.dtype})")
            else:
                logger.info(f"  {key}: {val}")

    # Update metadata
    metadata["has_qwen_vl_features"] = True
    metadata["qwen_vl_features_dir"] = "qwen_vl_features"
    metadata["qwen_vl_model"] = args.model_path
    metadata["qwen_vl_hidden_dim"] = hidden_dim
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("\nUpdated metadata.json with Qwen VL features info")


if __name__ == "__main__":
    main()
