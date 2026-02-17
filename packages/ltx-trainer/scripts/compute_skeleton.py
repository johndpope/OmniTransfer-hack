#!/usr/bin/env python3
"""Compute SMPL/MANO skeleton pseudo-GT from videos for 3DiMo geometric supervision.

This script extracts body pose (SMPL) and hand joints (MANO) from driving videos
using 4DHumans + HaMeR, saving the results as .pt files for training supervision.

Output format per sample (matching IMTalker skeleton_dataset.py format):
    body_pose: [T, 23, 3, 3]       - SMPL rotation matrices (no global orient)
    hand_joints_3d: [T, 21, 3]     - MANO 3D hand joints (right hand)
    body_joints_3d: [T, 44, 3]     - Full 3D body joints
    body_confidence: [T, 44]        - Per-joint confidence scores

Usage:
    python scripts/compute_skeleton.py \
        --video-dir /path/to/raw_videos \
        --output-dir /path/to/dataset/skeleton \
        --batch-size 16

Requirements:
    pip install 4dhuman hamer  # Or install from source
    # See: https://github.com/shubham-goel/4D-Humans
    # See: https://github.com/geopavlakos/hamer
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from ltx_trainer import logger


def extract_skeleton_4dhumans(
    video_path: Path,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, torch.Tensor] | None:
    """Extract SMPL body pose from video using 4DHumans.

    Args:
        video_path: Path to input video
        device: Device for inference
        batch_size: Batch size for frame processing

    Returns:
        Dictionary with body_pose, body_joints_3d, body_confidence, or None if failed
    """
    try:
        from hmr2.models import download_models, load_hmr2
        from hmr2.utils import recursive_to
        from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
    except ImportError:
        logger.error(
            "4DHumans (hmr2) not installed. Install from: "
            "https://github.com/shubham-goel/4D-Humans"
        )
        return None

    try:
        import cv2
    except ImportError:
        logger.error("OpenCV required: pip install opencv-python")
        return None

    # Load video frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        logger.warning(f"No frames extracted from {video_path}")
        return None

    logger.info(f"Extracted {len(frames)} frames from {video_path.name}")

    # Load HMR2 model
    model, model_cfg = load_hmr2()
    model = model.to(device)
    model.eval()

    all_body_pose = []
    all_body_joints = []
    all_confidence = []

    # Process frames in batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]

        # Create dataset for batch
        dataset = ViTDetDataset(model_cfg, batch_frames)

        with torch.inference_mode():
            for j in range(len(batch_frames)):
                sample = dataset[j]
                batch_input = recursive_to({k: v.unsqueeze(0) for k, v in sample.items()}, device)

                output = model(batch_input)

                # Extract SMPL body pose (excluding global orient)
                # body_pose: [1, 23, 3, 3] rotation matrices
                if "body_pose" in output:
                    body_pose = output["body_pose"][0].cpu()  # [23, 3, 3]
                    all_body_pose.append(body_pose)

                # Extract 3D body joints
                if "joints" in output:
                    joints_3d = output["joints"][0].cpu()  # [44, 3]
                    all_body_joints.append(joints_3d)

                # Extract confidence (from detection score)
                confidence = torch.ones(44)  # Default full confidence
                if "scores" in sample:
                    confidence = confidence * sample["scores"].item()
                all_confidence.append(confidence)

    if not all_body_pose:
        logger.warning(f"No body poses extracted from {video_path}")
        return None

    result = {
        "body_pose": torch.stack(all_body_pose),           # [T, 23, 3, 3]
        "body_joints_3d": torch.stack(all_body_joints) if all_body_joints else None,  # [T, 44, 3]
        "body_confidence": torch.stack(all_confidence),     # [T, 44]
    }

    return result


def extract_hand_joints_hamer(
    video_path: Path,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor | None:
    """Extract MANO hand joints from video using HaMeR.

    Args:
        video_path: Path to input video
        device: Device for inference
        batch_size: Batch size for frame processing

    Returns:
        hand_joints_3d: [T, 21, 3] or None if failed
    """
    try:
        from hamer.models import download_models, load_hamer
        from hamer.utils import recursive_to
    except ImportError:
        logger.warning(
            "HaMeR not installed. Hand joints will be skipped. "
            "Install from: https://github.com/geopavlakos/hamer"
        )
        return None

    try:
        import cv2
    except ImportError:
        return None

    # Load video frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    # Load HaMeR model
    model, model_cfg = load_hamer()
    model = model.to(device)
    model.eval()

    all_hand_joints = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]

        with torch.inference_mode():
            for frame in batch_frames:
                # HaMeR expects single frame input
                # Process and extract hand joints
                try:
                    output = model.predict(frame)
                    if "hand_joints_3d" in output and len(output["hand_joints_3d"]) > 0:
                        joints = output["hand_joints_3d"][0].cpu()  # [21, 3]
                        all_hand_joints.append(joints)
                    else:
                        # No hand detected - use zeros
                        all_hand_joints.append(torch.zeros(21, 3))
                except Exception:
                    all_hand_joints.append(torch.zeros(21, 3))

    if not all_hand_joints:
        return None

    return torch.stack(all_hand_joints)  # [T, 21, 3]


def process_video(
    video_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 16,
    skip_hands: bool = False,
) -> bool:
    """Process a single video and save skeleton .pt file.

    Args:
        video_path: Input video path
        output_path: Output .pt file path
        device: Device for inference
        batch_size: Batch size
        skip_hands: Skip hand joint extraction

    Returns:
        True if successful
    """
    logger.info(f"Processing: {video_path.name}")

    # Extract body pose with 4DHumans
    body_data = extract_skeleton_4dhumans(video_path, device, batch_size)
    if body_data is None:
        logger.warning(f"Skipping {video_path.name}: body extraction failed")
        return False

    # Extract hand joints with HaMeR (optional)
    hand_joints = None
    if not skip_hands:
        hand_joints = extract_hand_joints_hamer(video_path, device, batch_size)

    # Combine into output format
    result = {
        "body_pose": body_data["body_pose"],             # [T, 23, 3, 3]
        "hand_joints_3d": hand_joints,                    # [T, 21, 3] or None
        "body_joints_3d": body_data.get("body_joints_3d"),  # [T, 44, 3] or None
        "body_confidence": body_data.get("body_confidence"),  # [T, 44] or None
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    logger.info(
        f"Saved skeleton: {output_path.name} "
        f"(body_pose={result['body_pose'].shape}, "
        f"hands={'yes' if hand_joints is not None else 'no'})"
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract SMPL/MANO skeleton pseudo-GT from videos for 3DiMo training"
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing raw driving videos"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for skeleton .pt files (e.g., dataset/skeleton/)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for frame processing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--skip-hands", action="store_true",
        help="Skip hand joint extraction (faster, no HaMeR needed)"
    )
    parser.add_argument(
        "--video-extensions", type=str, nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        help="Video file extensions to process"
    )
    parser.add_argument(
        "--match-latents", type=str, default=None,
        help="If provided, only process videos matching filenames in this latents/ directory "
        "(ensures skeleton files match dataset sample indices)"
    )

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find video files
    video_files = []
    for ext in args.video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    video_files = sorted(video_files)

    if not video_files:
        logger.error(f"No video files found in {video_dir}")
        sys.exit(1)

    # If matching to latents directory, filter and rename
    if args.match_latents:
        latents_dir = Path(args.match_latents)
        latent_files = sorted(latents_dir.glob("*.pt"))
        if len(latent_files) != len(video_files):
            logger.warning(
                f"Latent count ({len(latent_files)}) != video count ({len(video_files)}). "
                f"Processing min({len(latent_files)}, {len(video_files)}) files."
            )
        video_files = video_files[:len(latent_files)]

    logger.info(f"Found {len(video_files)} videos to process")

    success_count = 0
    for i, video_path in enumerate(video_files):
        # Use numeric naming to match dataset convention (0.pt, 1.pt, ...)
        output_path = output_dir / f"{i}.pt"

        if output_path.exists():
            logger.info(f"Skipping {video_path.name}: output exists")
            success_count += 1
            continue

        try:
            if process_video(
                video_path, output_path, device,
                batch_size=args.batch_size,
                skip_hands=args.skip_hands,
            ):
                success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(video_files)} ({success_count} successful)")

    logger.info(f"Done: {success_count}/{len(video_files)} videos processed successfully")
    logger.info(f"Skeleton files saved to: {output_dir}")


if __name__ == "__main__":
    main()
