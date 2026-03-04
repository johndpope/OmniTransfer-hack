#!/usr/bin/env python3
"""Live ingest watcher: poll a folder for new images, VAE-encode, and add to dataset.

Runs alongside training. Drops new latents + default conditions into the dataset
directory. The trainer (with `live_ingest_enabled: true`) picks them up at epoch
boundaries via `PrecomputedDataset.rescan()`.

Architecture:
    - VAE encoder loaded on cuda:1 (~8GB)  ← fits alongside training's int8 transformer on cuda:0
    - Default text embedding cloned from an existing conditions_final/*.pt
    - Atomic writes: .tmp → os.rename() prevents partial reads
    - PID lockfile prevents duplicate instances

Usage:
    python scripts/live_ingest.py \
        --watch-dir ~/isometric_inbox/ \
        --dataset-dir /media/2TB/omnitransfer/data/isometric_t2i_all \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --device cuda:1 --poll-interval 5
"""

import argparse
import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
LOCKFILE_NAME = ".live_ingest.lock"
MANIFEST_NAME = "live_ingest_manifest.json"


def acquire_lockfile(dataset_dir: Path) -> Path:
    """Create a PID lockfile. Raises if another instance is running."""
    lockfile = dataset_dir / LOCKFILE_NAME
    if lockfile.exists():
        try:
            old_pid = int(lockfile.read_text().strip())
            # Check if process is still alive
            os.kill(old_pid, 0)
            raise RuntimeError(
                f"Another live_ingest instance is running (PID {old_pid}). "
                f"Remove {lockfile} if this is stale."
            )
        except (ProcessLookupError, ValueError):
            logger.warning(f"Stale lockfile from PID, removing: {lockfile}")
            lockfile.unlink()

    lockfile.write_text(str(os.getpid()))
    return lockfile


def release_lockfile(lockfile: Path) -> None:
    """Remove lockfile on exit."""
    try:
        if lockfile.exists():
            lockfile.unlink()
    except OSError:
        pass


def load_manifest(dataset_dir: Path) -> dict:
    """Load the manifest of already-processed files."""
    manifest_path = dataset_dir / MANIFEST_NAME
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"processed_files": [], "next_index": 0}


def save_manifest(dataset_dir: Path, manifest: dict) -> None:
    """Save manifest atomically."""
    manifest_path = dataset_dir / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.rename(tmp_path, manifest_path)


def find_next_index(latents_dir: Path, manifest: dict) -> int:
    """Find the next available index by checking both manifest and existing files."""
    manifest_idx = manifest.get("next_index", 0)
    # Also check existing files in case manifest is out of sync
    existing = list(latents_dir.glob("*.pt"))
    if existing:
        max_existing = max(int(f.stem) for f in existing if f.stem.isdigit())
        return max(manifest_idx, max_existing + 1)
    return manifest_idx


def load_default_condition(dataset_dir: Path) -> dict[str, torch.Tensor] | None:
    """Load a default text embedding from the first conditions_final file."""
    conditions_dir = dataset_dir / "conditions_final"
    if not conditions_dir.exists():
        # Also try "conditions"
        conditions_dir = dataset_dir / "conditions"

    if not conditions_dir.exists():
        logger.warning(f"No conditions directory found in {dataset_dir}")
        return None

    pt_files = sorted(conditions_dir.glob("*.pt"))
    if not pt_files:
        logger.warning(f"No .pt files in {conditions_dir}")
        return None

    condition = torch.load(pt_files[0], map_location="cpu", weights_only=True)
    logger.info(f"Loaded default condition from {pt_files[0]}")
    return condition


def load_and_prepare_image(image_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load image, center-crop to target aspect ratio, resize, prepare for VAE.

    Returns:
        Tensor [1, C, 1, H, W] normalized to [-1, 1].
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target_aspect = target_w / target_h
    source_aspect = w / h

    # Center-crop to match target aspect ratio
    if abs(source_aspect - target_aspect) > 0.01:
        if source_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            img = img.crop((start_x, 0, start_x + new_w, h))
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            img = img.crop((0, start_y, w, start_y + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)

    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
    tensor = tensor * 2.0 - 1.0
    return tensor


def encode_and_save(
    image_path: Path,
    latent_path: Path,
    condition_path: Path,
    vae_encoder: torch.nn.Module,
    default_condition: dict[str, torch.Tensor],
    target_h: int,
    target_w: int,
    device: str,
    dtype: torch.dtype,
) -> bool:
    """Encode one image and atomically write latent + condition files.

    Returns True on success, False on failure.
    """
    try:
        img_tensor = load_and_prepare_image(image_path, target_h, target_w)
        img_tensor = img_tensor.to(device, dtype=dtype)

        with torch.inference_mode():
            latent = vae_encoder(img_tensor)

        latent = latent.cpu()
        data = {
            "latents": latent.squeeze(0),  # [C, 1, H_lat, W_lat]
            "num_frames": torch.tensor([1]),
            "height": torch.tensor([latent.shape[3]]),
            "width": torch.tensor([latent.shape[4]]),
        }

        # Atomic write: .tmp → rename
        latent_tmp = latent_path.with_suffix(".tmp")
        torch.save(data, latent_tmp)
        os.rename(latent_tmp, latent_path)

        condition_tmp = condition_path.with_suffix(".tmp")
        torch.save(default_condition, condition_tmp)
        os.rename(condition_tmp, condition_path)

        logger.info(
            f"Ingested {image_path.name} → {latent_path.name} "
            f"(latent shape: {data['latents'].shape})"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to encode {image_path}: {e}")
        # Clean up partial files
        for p in [latent_path, latent_path.with_suffix(".tmp"),
                  condition_path, condition_path.with_suffix(".tmp")]:
            if p.exists():
                p.unlink()
        return False


def watch_loop(
    watch_dir: Path,
    dataset_dir: Path,
    vae_encoder: torch.nn.Module,
    default_condition: dict[str, torch.Tensor],
    target_h: int,
    target_w: int,
    device: str,
    dtype: torch.dtype,
    poll_interval: float,
    conditions_subdir: str,
) -> None:
    """Main poll loop: check for new images, encode, write to dataset."""
    latents_dir = dataset_dir / "latents"
    conditions_dir = dataset_dir / conditions_subdir
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(dataset_dir)
    processed_set = set(manifest.get("processed_files", []))
    next_idx = find_next_index(latents_dir, manifest)

    logger.info(
        f"Watching {watch_dir} → {dataset_dir} "
        f"(poll every {poll_interval}s, next index: {next_idx:06d})"
    )

    try:
        while True:
            # Find new image files
            new_files = []
            for ext in IMAGE_EXTENSIONS:
                for f in watch_dir.glob(f"*{ext}"):
                    if f.name not in processed_set and not f.name.startswith("."):
                        new_files.append(f)
                # Also check uppercase extensions
                for f in watch_dir.glob(f"*{ext.upper()}"):
                    if f.name not in processed_set and not f.name.startswith("."):
                        new_files.append(f)

            # Deduplicate (glob patterns may overlap)
            new_files = sorted(set(new_files), key=lambda p: p.name)

            if new_files:
                logger.info(f"Found {len(new_files)} new image(s)")

            for image_path in new_files:
                idx_str = f"{next_idx:06d}"
                latent_path = latents_dir / f"{idx_str}.pt"
                condition_path = conditions_dir / f"{idx_str}.pt"

                success = encode_and_save(
                    image_path=image_path,
                    latent_path=latent_path,
                    condition_path=condition_path,
                    vae_encoder=vae_encoder,
                    default_condition=default_condition,
                    target_h=target_h,
                    target_w=target_w,
                    device=device,
                    dtype=dtype,
                )

                if success:
                    processed_set.add(image_path.name)
                    next_idx += 1
                    manifest["processed_files"] = sorted(processed_set)
                    manifest["next_index"] = next_idx
                    save_manifest(dataset_dir, manifest)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Shutting down live ingest watcher")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch folder for new images, VAE-encode, add to training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Watch inbox, encode to existing dataset
    python scripts/live_ingest.py \\
        --watch-dir ~/isometric_inbox/ \\
        --dataset-dir /media/2TB/omnitransfer/data/isometric_t2i_all \\
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors

    # Custom resolution on secondary GPU
    python scripts/live_ingest.py \\
        --watch-dir ~/inbox/ \\
        --dataset-dir /media/2TB/omnitransfer/data/custom \\
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \\
        --target-height 704 --target-width 480 \\
        --device cuda:1 --poll-interval 10
""",
    )
    parser.add_argument("--watch-dir", type=Path, required=True, help="Directory to watch for new images")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Training dataset root (must have latents/ and conditions_final/)")
    parser.add_argument("--model-path", type=Path, required=True, help="LTX-2 .safetensors checkpoint")
    parser.add_argument("--target-height", type=int, default=480, help="Target height, must be divisible by 32 (default: 480)")
    parser.add_argument("--target-width", type=int, default=704, help="Target width, must be divisible by 32 (default: 704)")
    parser.add_argument("--device", type=str, default="cuda:1", help="CUDA device for VAE encoder (default: cuda:1)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between folder checks (default: 5)")
    parser.add_argument("--conditions-subdir", type=str, default="conditions_final", help="Conditions subdirectory name (default: conditions_final)")

    args = parser.parse_args()

    # Validate
    if args.target_height % 32 != 0 or args.target_width % 32 != 0:
        raise ValueError(f"Dimensions must be divisible by 32: {args.target_width}x{args.target_height}")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    args.watch_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Acquire lockfile
    lockfile = acquire_lockfile(args.dataset_dir)
    atexit.register(release_lockfile, lockfile)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Load default condition
    default_condition = load_default_condition(args.dataset_dir)
    if default_condition is None:
        raise RuntimeError(
            f"No conditions found in {args.dataset_dir}. "
            "The dataset must have at least one existing condition file to use as a template."
        )

    # Load VAE encoder
    logger.info(f"Loading VAE encoder on {args.device} ({args.dtype})")
    vae_encoder = load_video_vae_encoder(args.model_path, dtype=dtype)
    vae_encoder = vae_encoder.to(args.device)
    vae_encoder.eval()
    logger.info("VAE encoder ready")

    # Run the watch loop
    watch_loop(
        watch_dir=args.watch_dir,
        dataset_dir=args.dataset_dir,
        vae_encoder=vae_encoder,
        default_condition=default_condition,
        target_h=args.target_height,
        target_w=args.target_width,
        device=args.device,
        dtype=dtype,
        poll_interval=args.poll_interval,
        conditions_subdir=args.conditions_subdir,
    )


if __name__ == "__main__":
    main()
