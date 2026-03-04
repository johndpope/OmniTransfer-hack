#!/usr/bin/env python3
"""
Preprocess scrya-downloads videos for SCD evolution training.

Encodes videos from scrya-downloads (Isometric 3D, PixelArt, PixelRig,
Realistic Photo, Untagged) into latent + text embedding format matching
the existing ditto_subset dataset for the evolution engine.

Two-phase pipeline (VAE and Gemma never loaded simultaneously):
  Phase 1: VAE encode videos → latents/  (cuda:1, ~8GB)
  Phase 2: Gemma text encode → conditions_final/  (cuda:0, ~28GB)

Usage:
    # Full pipeline (both phases):
    python scripts/preprocess_scrya_evolution.py

    # Phase 1 only (VAE encoding):
    python scripts/preprocess_scrya_evolution.py --skip-text

    # Phase 2 only (text encoding, after Phase 1):
    python scripts/preprocess_scrya_evolution.py --skip-vae

    # Then merge with ditto_subset for evolution:
    python scripts/preprocess_scrya_evolution.py --merge-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

# ── Constants ──
MODEL_PATH = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
GEMMA_PATH = "/media/2TB/ltx-models/gemma"
SCRYA_ROOT = Path("/home/johndpope/scrya-downloads")
OUTPUT_ROOT = Path("/media/2TB/omnitransfer/data/scrya_evolution")
MERGED_ROOT = Path("/media/2TB/omnitransfer/data/evolution_merged")
DITTO_ROOT = Path("/media/2TB/omnitransfer/data/ditto_subset")

# Match ditto_subset: 448×768, 25 frames → 4 latent frames
TARGET_H = 448
TARGET_W = 768
NUM_FRAMES = 25
FPS = 24.0

# Dual-GPU: VAE on cuda:1 (PRO 4000 24GB), Gemma on cuda:0 (RTX 5090 32GB)
VAE_DEVICE = "cuda:1"
TEXT_DEVICE = "cuda:0"

SUBDIRS = ["Isometric 3D", "PixelArt", "PixelRig", "Realistic Photo", "Untagged"]


def discover_videos() -> list[tuple[Path, str, str]]:
    """Find all MP4 files with matching text prompts.

    Returns list of (video_path, prompt_text, category).
    """
    results = []
    for subdir in SUBDIRS:
        dirpath = SCRYA_ROOT / subdir
        if not dirpath.exists():
            print(f"  Skipping missing dir: {dirpath}")
            continue

        mp4s = sorted(dirpath.glob("*.mp4"))
        found = 0
        for mp4 in mp4s:
            # Try _combined.txt first (better quality prompts), then plain .txt
            combined = mp4.with_name(mp4.stem + "_combined.txt")
            plain = mp4.with_suffix(".txt")

            if combined.exists():
                prompt = combined.read_text().strip()
            elif plain.exists():
                prompt = plain.read_text().strip()
            else:
                continue

            if not prompt:
                continue

            results.append((mp4, prompt, subdir))
            found += 1

        print(f"  {subdir}: {found} video+prompt pairs (of {len(mp4s)} mp4s)")

    return results


def load_video_frames(
    path: Path, num_frames: int, target_h: int, target_w: int
) -> torch.Tensor:
    """Load and resize video to [1, C, F, H, W] in [-1, 1].

    Uses center-crop to target aspect ratio before resize (handles varied
    source resolutions: landscape, portrait, square).
    """
    import torchvision.io as tvio

    video, _, info = tvio.read_video(str(path), pts_unit="sec", end_pts=num_frames / FPS + 0.5)
    # video: [T, H, W, C] uint8

    if video.shape[0] < num_frames:
        # Pad by repeating last frame
        pad = num_frames - video.shape[0]
        video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

    video = video[:num_frames]  # [F, H, W, C]

    target_aspect = target_w / target_h
    frames = []
    for i in range(video.shape[0]):
        img = Image.fromarray(video[i].numpy())
        w, h = img.size
        source_aspect = w / h

        # Center-crop to target aspect ratio
        if abs(source_aspect - target_aspect) > 0.01:
            if source_aspect > target_aspect:  # Too wide → crop width
                new_w = int(h * target_aspect)
                start_x = (w - new_w) // 2
                img = img.crop((start_x, 0, start_x + new_w, h))
            else:  # Too tall → crop height
                new_h = int(w / target_aspect)
                start_y = (h - new_h) // 2
                img = img.crop((0, start_y, w, start_y + new_h))

        # Resize to exact target
        img = img.resize((target_w, target_h), Image.LANCZOS)
        t = torch.from_numpy(np.array(img)).float() / 255.0
        frames.append(t)

    tensor = torch.stack(frames)  # [F, H, W, C]
    tensor = tensor.permute(3, 0, 1, 2)  # [C, F, H, W]
    tensor = tensor.unsqueeze(0)  # [1, C, F, H, W]
    tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    return tensor


def phase1_vae_encode(videos: list[tuple[Path, str, str]], output_dir: Path) -> dict[str, str]:
    """Phase 1: VAE encode videos → latents/.

    Returns dict mapping sample_id → prompt for Phase 2.
    """
    from ltx_trainer.model_loader import load_video_vae_encoder

    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 1: VAE Encoding ({len(videos)} videos) ===")
    print(f"  Target: {TARGET_H}×{TARGET_W}, {NUM_FRAMES} frames")
    print(f"  Device: {VAE_DEVICE}")

    vae_encoder = load_video_vae_encoder(MODEL_PATH, dtype=torch.bfloat16)
    vae_encoder = vae_encoder.to(VAE_DEVICE)
    vae_encoder.eval()

    encoded = 0
    skipped = 0
    failed = 0
    id_to_prompt: dict[str, str] = {}

    for idx, (video_path, prompt, category) in enumerate(videos):
        sample_id = f"{idx:06d}"
        latent_path = latent_dir / f"{sample_id}.pt"

        if latent_path.exists():
            skipped += 1
            id_to_prompt[sample_id] = prompt
            continue

        try:
            video_tensor = load_video_frames(video_path, NUM_FRAMES, TARGET_H, TARGET_W)
            video_tensor = video_tensor.to(VAE_DEVICE, dtype=torch.bfloat16)

            with torch.inference_mode():
                latent = vae_encoder(video_tensor)  # [1, 128, F_lat, H_lat, W_lat]

            latent_data = {
                "latents": latent.squeeze(0).cpu(),  # [128, F_lat, H_lat, W_lat]
                "num_frames": torch.tensor([latent.shape[2]]),
                "height": torch.tensor([latent.shape[3]]),
                "width": torch.tensor([latent.shape[4]]),
                "fps": torch.tensor([20.0]),
            }

            # Atomic write
            tmp = latent_path.with_suffix(".tmp")
            torch.save(latent_data, tmp)
            os.rename(tmp, latent_path)

            id_to_prompt[sample_id] = prompt
            encoded += 1

            if (idx + 1) % 20 == 0:
                print(f"  [{idx + 1}/{len(videos)}] Encoded {encoded}, skipped {skipped}, failed {failed}")

        except Exception as e:
            print(f"  FAILED {video_path.name}: {e}")
            failed += 1

    print(f"\n  Phase 1 complete: {encoded} encoded, {skipped} cached, {failed} failed")

    # Save prompt mapping for Phase 2
    mapping_path = output_dir / "prompt_mapping.json"
    mapping = {sid: {"prompt": p} for sid, p in id_to_prompt.items()}
    # Include source info
    for idx, (video_path, prompt, category) in enumerate(videos):
        sid = f"{idx:06d}"
        if sid in mapping:
            mapping[sid]["source"] = str(video_path)
            mapping[sid]["category"] = category
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    # Cleanup
    del vae_encoder
    gc.collect()
    torch.cuda.empty_cache()

    return id_to_prompt


def phase2_text_encode(output_dir: Path) -> int:
    """Phase 2: Gemma text encode prompts → conditions_final/.

    Loads prompt mapping from Phase 1 output.
    """
    from ltx_trainer.model_loader import load_text_encoder

    cond_dir = output_dir / "conditions_final"
    cond_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "prompt_mapping.json"
    if not mapping_path.exists():
        print("ERROR: prompt_mapping.json not found. Run Phase 1 first.")
        return 0

    with open(mapping_path) as f:
        mapping = json.load(f)

    latent_dir = output_dir / "latents"
    # Only encode prompts for samples that have latents
    to_encode = []
    for sid, info in mapping.items():
        if (latent_dir / f"{sid}.pt").exists():
            cond_path = cond_dir / f"{sid}.pt"
            if not cond_path.exists():
                to_encode.append((sid, info["prompt"]))

    if not to_encode:
        print("  All conditions already cached. Nothing to do.")
        return len(mapping)

    print(f"\n=== Phase 2: Text Encoding ({len(to_encode)} prompts) ===")
    print(f"  Device: {TEXT_DEVICE}")

    text_encoder = load_text_encoder(
        checkpoint_path=MODEL_PATH,
        gemma_model_path=GEMMA_PATH,
        device=TEXT_DEVICE,
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )
    text_encoder.eval()

    encoded = 0
    for idx, (sid, prompt) in enumerate(to_encode):
        cond_path = cond_dir / f"{sid}.pt"
        try:
            with torch.inference_mode():
                video_embeds, audio_embeds, attention_mask = text_encoder(prompt)

            emb_data = {
                "video_prompt_embeds": video_embeds.squeeze(0).cpu().contiguous(),
                "audio_prompt_embeds": (
                    audio_embeds.squeeze(0).cpu().contiguous()
                    if audio_embeds is not None
                    else video_embeds.squeeze(0).cpu().contiguous()
                ),
                "prompt_attention_mask": attention_mask.squeeze(0).cpu().contiguous(),
                "is_final_embedding": True,
            }

            tmp = cond_path.with_suffix(".tmp")
            torch.save(emb_data, tmp)
            os.rename(tmp, cond_path)
            encoded += 1

            if (idx + 1) % 20 == 0:
                print(f"  [{idx + 1}/{len(to_encode)}] Encoded {encoded}")

        except Exception as e:
            print(f"  FAILED {sid} ({prompt[:50]}...): {e}")

    print(f"\n  Phase 2 complete: {encoded} encoded")

    # Cleanup
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    return encoded


def create_merged_dataset(output_dir: Path) -> int:
    """Create merged dataset by symlinking ditto_subset + scrya_evolution.

    The evolution engine reads from a single data_root. This creates a
    unified directory with sequential sample IDs.
    """
    merged_latent_dir = MERGED_ROOT / "latents"
    merged_cond_dir = MERGED_ROOT / "conditions_final"
    merged_latent_dir.mkdir(parents=True, exist_ok=True)
    merged_cond_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing symlinks
    for f in merged_latent_dir.glob("*.pt"):
        if f.is_symlink():
            f.unlink()
    for f in merged_cond_dir.glob("*.pt"):
        if f.is_symlink():
            f.unlink()

    count = 0

    # Phase A: Symlink ditto_subset samples
    ditto_latents = sorted((DITTO_ROOT / "latents").glob("*.pt"))
    ditto_conds = sorted((DITTO_ROOT / "conditions_final").glob("*.pt"))
    ditto_ids = {p.stem for p in ditto_latents} & {p.stem for p in ditto_conds}

    print(f"\n=== Merging Datasets ===")
    print(f"  Ditto subset: {len(ditto_ids)} valid pairs")

    for old_id in sorted(ditto_ids):
        new_id = f"{count:06d}"
        src_lat = DITTO_ROOT / "latents" / f"{old_id}.pt"
        src_cond = DITTO_ROOT / "conditions_final" / f"{old_id}.pt"
        os.symlink(src_lat, merged_latent_dir / f"{new_id}.pt")
        os.symlink(src_cond, merged_cond_dir / f"{new_id}.pt")
        count += 1

    # Phase B: Symlink scrya_evolution samples
    scrya_latents = sorted((output_dir / "latents").glob("*.pt"))
    scrya_conds = sorted((output_dir / "conditions_final").glob("*.pt"))
    scrya_ids = {p.stem for p in scrya_latents} & {p.stem for p in scrya_conds}

    print(f"  Scrya evolution: {len(scrya_ids)} valid pairs")

    for old_id in sorted(scrya_ids):
        new_id = f"{count:06d}"
        src_lat = output_dir / "latents" / f"{old_id}.pt"
        src_cond = output_dir / "conditions_final" / f"{old_id}.pt"
        os.symlink(src_lat, merged_latent_dir / f"{new_id}.pt")
        os.symlink(src_cond, merged_cond_dir / f"{new_id}.pt")
        count += 1

    print(f"  MERGED TOTAL: {count} samples")
    print(f"  Output: {MERGED_ROOT}")

    # Save metadata
    meta = {
        "num_samples": count,
        "sources": {
            "ditto_subset": {"count": len(ditto_ids), "path": str(DITTO_ROOT)},
            "scrya_evolution": {"count": len(scrya_ids), "path": str(output_dir)},
        },
        "resolution": f"{TARGET_H}x{TARGET_W}",
        "num_frames": NUM_FRAMES,
    }
    with open(MERGED_ROOT / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return count


def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess scrya-downloads videos for SCD evolution"
    )
    p.add_argument("--skip-vae", action="store_true", help="Skip Phase 1 (VAE encoding)")
    p.add_argument("--skip-text", action="store_true", help="Skip Phase 2 (text encoding)")
    p.add_argument("--merge-only", action="store_true", help="Only create merged dataset")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    if args.merge_only:
        create_merged_dataset(output_dir)
        return

    # Discover videos
    print("Discovering videos in scrya-downloads...")
    videos = discover_videos()
    print(f"\nTotal: {len(videos)} video+prompt pairs")

    if not videos:
        print("No videos found!")
        return

    # Phase 1: VAE encode
    if not args.skip_vae:
        phase1_vae_encode(videos, output_dir)
    else:
        print("\nSkipping Phase 1 (VAE encoding)")

    # Phase 2: Text encode
    if not args.skip_text:
        phase2_text_encode(output_dir)
    else:
        print("\nSkipping Phase 2 (text encoding)")

    # Save metadata
    meta = {
        "num_samples": len(videos),
        "resolution": f"{TARGET_H}x{TARGET_W}",
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "source": str(SCRYA_ROOT),
        "categories": SUBDIRS,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Create merged dataset
    print("\n" + "=" * 60)
    create_merged_dataset(output_dir)

    print("\n" + "=" * 60)
    print("DONE! Evolution data ready at:")
    print(f"  Scrya only:    {output_dir}")
    print(f"  Merged (ditto+scrya): {MERGED_ROOT}")


if __name__ == "__main__":
    main()
