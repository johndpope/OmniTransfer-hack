#!/usr/bin/env python3
"""Fix OmniTransfer dataset for self-reconstruction training.

The current dataset has identical latents in reference_latents/ and latents/,
which is useless (trains identity mapping).

This script sets up SELF-RECONSTRUCTION training:
- Each video becomes its own training sample
- reference_latents = the video (motion source)
- target_image_latents = first frame (image to animate)
- latents = SAME video (ground truth = reconstruct yourself)

For cross-subject I2V, we DON'T have ground truth, so self-reconstruction
is the practical approach. The model learns to animate a first frame
to recreate the original video.

Usage:
    python scripts/fix_omnitransfer_dataset.py \
        --dataset-dir /media/2TB/omnitransfer_effect_motion \
        --mode verify

    python scripts/fix_omnitransfer_dataset.py \
        --dataset-dir /media/2TB/omnitransfer_effect_motion \
        --mode fix
"""

import argparse
import json
import shutil
import torch
from pathlib import Path


def verify_dataset(dataset_dir: Path) -> dict:
    """Verify dataset and report issues."""
    ref_dir = dataset_dir / "reference_latents"
    lat_dir = dataset_dir / "latents"
    img_dir = dataset_dir / "target_image_latents"

    issues = {
        "identical_pairs": [],
        "missing_files": [],
        "shape_mismatches": [],
    }
    stats = {
        "total_pairs": 0,
        "ref_count": 0,
        "lat_count": 0,
        "img_count": 0,
    }

    ref_files = sorted(ref_dir.glob("*.pt"))
    lat_files = sorted(lat_dir.glob("*.pt"))
    img_files = sorted(img_dir.glob("*.pt")) if img_dir.exists() else []

    stats["ref_count"] = len(ref_files)
    stats["lat_count"] = len(lat_files)
    stats["img_count"] = len(img_files)
    stats["total_pairs"] = len(ref_files)

    # Check for identical pairs
    for ref_file in ref_files:
        lat_file = lat_dir / ref_file.name
        if lat_file.exists():
            ref_data = torch.load(ref_file, weights_only=False)
            lat_data = torch.load(lat_file, weights_only=False)

            ref_lat = ref_data["latents"]
            lat_lat = lat_data["latents"]

            diff = (ref_lat - lat_lat).abs().mean().item()
            if diff < 0.001:  # Essentially identical
                issues["identical_pairs"].append({
                    "file": ref_file.name,
                    "diff": diff,
                })
        else:
            issues["missing_files"].append(f"latents/{ref_file.name}")

    return {"issues": issues, "stats": stats}


def fix_for_self_reconstruction(dataset_dir: Path, backup: bool = True):
    """Fix dataset for self-reconstruction training.

    Strategy: Each video becomes its own training sample.
    - reference_latents stays as-is (video to get motion from)
    - latents gets a COPY of reference_latents (ground truth = same video)
    - target_image_latents gets first frame extracted from reference video

    This way:
    - Model input: first frame + reference video (self)
    - Model output: reconstruct the reference video
    """
    ref_dir = dataset_dir / "reference_latents"
    lat_dir = dataset_dir / "latents"
    img_dir = dataset_dir / "target_image_latents"

    # Backup existing
    if backup:
        backup_dir = dataset_dir / "backup_latents"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(lat_dir, backup_dir)
        print(f"Backed up latents/ to backup_latents/")

    ref_files = sorted(ref_dir.glob("*.pt"))

    fixed_count = 0
    for ref_file in ref_files:
        lat_file = lat_dir / ref_file.name
        img_file = img_dir / ref_file.name

        # Load reference latent
        ref_data = torch.load(ref_file, weights_only=False)
        ref_lat = ref_data["latents"]  # [C, F, H, W]

        # Copy reference to latents (ground truth = reconstruct reference)
        # This is already the case if they're identical, but let's be explicit
        lat_data = {
            "latents": ref_lat.clone(),
            "num_frames": ref_data.get("num_frames", torch.tensor([ref_lat.shape[1]])),
            "height": ref_data.get("height", torch.tensor([ref_lat.shape[2]])),
            "width": ref_data.get("width", torch.tensor([ref_lat.shape[3]])),
            "fps": ref_data.get("fps", torch.tensor([25.0])),
            "task_type": ref_data.get("task_type", "self_reconstruction"),
        }
        torch.save(lat_data, lat_file)

        # Extract first frame for target_image_latents
        first_frame = ref_lat[:, 0:1, :, :]  # [C, 1, H, W]
        img_data = {
            "latents": first_frame,
            "num_frames": torch.tensor([1]),
            "height": ref_data.get("height", torch.tensor([ref_lat.shape[2]])),
            "width": ref_data.get("width", torch.tensor([ref_lat.shape[3]])),
            "fps": ref_data.get("fps", torch.tensor([25.0])),
            "task_type": "target_image",
        }
        img_dir.mkdir(parents=True, exist_ok=True)
        torch.save(img_data, img_file)

        fixed_count += 1

    print(f"Fixed {fixed_count} samples for self-reconstruction training")
    return fixed_count


def update_metadata_for_self_reconstruction(dataset_dir: Path):
    """Update metadata.json for self-reconstruction training."""
    metadata_file = dataset_dir / "metadata.json"

    if not metadata_file.exists():
        print("No metadata.json found, skipping metadata update")
        return

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Update config
    if "config" in metadata:
        metadata["config"]["training_type"] = "self_reconstruction"
        metadata["config"]["note"] = "Self-reconstruction: video reconstructs itself from first frame"

    # Update pairs to reflect self-reconstruction
    if "pairs" in metadata:
        for pair in metadata["pairs"]:
            pair["task_type"] = "self_reconstruction"
            pair["is_self_reconstruction"] = True
            pair["description"] = f"Self-reconstruction: {pair.get('reference_video', 'video')} -> reconstruct from first frame"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated metadata.json for self-reconstruction training")


def main():
    parser = argparse.ArgumentParser(description="Fix OmniTransfer dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory to fix",
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "fix"],
        default="verify",
        help="Mode: verify (check issues) or fix (apply fixes)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup existing latents before fixing",
    )
    args = parser.parse_args()

    if args.mode == "verify":
        print(f"Verifying dataset: {args.dataset_dir}")
        print("=" * 60)
        result = verify_dataset(args.dataset_dir)

        print(f"\nStats:")
        for key, value in result["stats"].items():
            print(f"  {key}: {value}")

        print(f"\nIssues:")
        issues = result["issues"]

        identical = issues["identical_pairs"]
        if identical:
            print(f"  CRITICAL: {len(identical)} pairs have IDENTICAL reference and target latents!")
            print(f"  This trains identity mapping (useless). First 5:")
            for item in identical[:5]:
                print(f"    - {item['file']}: diff={item['diff']:.6f}")
        else:
            print("  No identical pairs found (good!)")

        if issues["missing_files"]:
            print(f"  Missing files: {len(issues['missing_files'])}")
            for f in issues["missing_files"][:5]:
                print(f"    - {f}")

    elif args.mode == "fix":
        print(f"Fixing dataset for self-reconstruction: {args.dataset_dir}")
        print("=" * 60)

        # First verify
        result = verify_dataset(args.dataset_dir)
        identical = result["issues"]["identical_pairs"]

        if not identical:
            print("No identical pairs found - dataset may already be correct!")
            return

        print(f"Found {len(identical)} identical pairs, fixing...")

        # Fix latents
        fix_for_self_reconstruction(args.dataset_dir, backup=not args.no_backup)

        # Update metadata
        update_metadata_for_self_reconstruction(args.dataset_dir)

        print("\nDone! Dataset is now set up for self-reconstruction training.")
        print("The model will learn to reconstruct videos from their first frame.")


if __name__ == "__main__":
    main()
