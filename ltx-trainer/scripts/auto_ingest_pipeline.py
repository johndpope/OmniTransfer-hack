#!/usr/bin/env python3
"""Auto-ingest pipeline: watch folder → caption → embed → VAE encode → add to dataset.

Runs entirely on cuda:1 (RTX PRO 4000, 24GB) alongside training on cuda:0.
Models are loaded sequentially since they don't fit simultaneously:

  Phase 1: Qwen2.5-VL-7B (8-bit, ~14GB) → caption images
  Phase 2: Gemma text encoder (8-bit, ~16GB) → compute final embeddings
  Phase 3: VAE encoder (~8GB) → encode latents

New images are added atomically to the dataset directory. The trainer
picks them up at epoch boundaries via PrecomputedDataset.rescan().

Usage:
    # Process new images from scrya-sync downloads
    python scripts/auto_ingest_pipeline.py \
        --watch-dir "/home/johndpope/scrya-downloads/Isometric 3D" \
        --dataset-dir /media/2TB/omnitransfer/data/isometric_t2i_scrya \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma

    # One-shot mode (process once, don't poll)
    python scripts/auto_ingest_pipeline.py \
        --watch-dir "/home/johndpope/scrya-downloads/Isometric 3D" \
        --dataset-dir /media/2TB/omnitransfer/data/isometric_t2i_scrya \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma \
        --one-shot

    # Use existing Scrya prompts instead of Qwen captioning
    python scripts/auto_ingest_pipeline.py \
        --watch-dir "/home/johndpope/scrya-downloads/Isometric 3D" \
        --dataset-dir /media/2TB/omnitransfer/data/isometric_t2i_scrya \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma \
        --use-scrya-prompts
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LOCKFILE_NAME = ".auto_ingest.lock"
MANIFEST_NAME = "auto_ingest_manifest.json"


# ─────────────────────────────────────────────────────────────────────────────
# Lockfile / manifest (reused from live_ingest.py)
# ─────────────────────────────────────────────────────────────────────────────


def acquire_lockfile(dataset_dir: Path) -> Path:
    lockfile = dataset_dir / LOCKFILE_NAME
    if lockfile.exists():
        try:
            old_pid = int(lockfile.read_text().strip())
            os.kill(old_pid, 0)
            raise RuntimeError(
                f"Another auto_ingest instance is running (PID {old_pid}). "
                f"Remove {lockfile} if this is stale."
            )
        except (ProcessLookupError, ValueError):
            print(f"  Stale lockfile from dead PID, removing: {lockfile}")
            lockfile.unlink()
    lockfile.write_text(str(os.getpid()))
    return lockfile


def release_lockfile(lockfile: Path) -> None:
    try:
        if lockfile.exists():
            lockfile.unlink()
    except OSError:
        pass


def load_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / MANIFEST_NAME
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"processed_files": [], "next_index": 0}


def save_manifest(dataset_dir: Path, manifest: dict) -> None:
    manifest_path = dataset_dir / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.rename(tmp_path, manifest_path)


def find_next_index(latents_dir: Path, manifest: dict) -> int:
    manifest_idx = manifest.get("next_index", 0)
    existing = list(latents_dir.glob("*.pt"))
    if existing:
        max_existing = max(int(f.stem) for f in existing if f.stem.isdigit())
        return max(manifest_idx, max_existing + 1)
    return manifest_idx


# ─────────────────────────────────────────────────────────────────────────────
# Image discovery
# ─────────────────────────────────────────────────────────────────────────────


def discover_new_images(
    watch_dir: Path,
    processed_set: set[str],
) -> list[Path]:
    """Find all new image files in watch_dir (recursive), excluding already processed."""
    new_files = []
    for ext in IMAGE_EXTENSIONS:
        for f in watch_dir.rglob(f"*{ext}"):
            # Skip thumbnails
            if "_thumb" in f.stem:
                continue
            # Use relative path as key (handles subdirs)
            rel_key = str(f.relative_to(watch_dir))
            if rel_key not in processed_set and not f.name.startswith("."):
                new_files.append(f)
    return sorted(set(new_files), key=lambda p: str(p))


def find_prompt_for_image(image_path: Path) -> str | None:
    """Find the Scrya prompt for an image.

    Checks:
    1. {stem}.txt alongside the image (per-image prompt)
    2. prompt.txt in the parent folder (shared prompt for variation groups)
    """
    # Per-image prompt
    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text().strip()

    # Shared prompt in parent folder (variation group)
    shared_prompt = image_path.parent / "prompt.txt"
    if shared_prompt.exists():
        return shared_prompt.read_text().strip()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Captioning with Qwen2.5-VL
# ─────────────────────────────────────────────────────────────────────────────


def caption_images_with_qwen(
    image_paths: list[Path],
    model_path: str,
    device: str,
) -> dict[str, str]:
    """Caption images with Qwen2.5-VL-7B (8-bit). Returns {path_str: caption}."""
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    print(f"\n{'='*60}")
    print(f"  Phase 1: Captioning {len(image_paths)} images with Qwen2.5-VL")
    print(f"{'='*60}")

    t_start = time.time()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    mem_gb = torch.cuda.memory_allocated(1) / 1e9
    print(f"  Qwen2.5-VL loaded ({mem_gb:.1f} GB)")

    captions: dict[str, str] = {}
    for img_path in tqdm(image_paths, desc="Captioning", unit="img"):
        try:
            image = Image.open(img_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": (
                                "Describe what is shown in this image in one sentence. "
                                "Focus on the subject, setting, and notable details. "
                                "Be concise and factual. Do not mention the art style, "
                                "camera angle, or rendering technique."
                            ),
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            caption = processor.decode(generated, skip_special_tokens=True).strip()

            # Clean common Qwen prefixes
            for prefix in ["The image shows ", "This image shows ", "In this image, "]:
                if caption.startswith(prefix):
                    caption = caption[len(prefix):]
                    caption = caption[0].upper() + caption[1:] if caption else caption
                    break

            # Format as training prompt
            caption = f"Isometric 3D view, static camera. {caption.rstrip('.')}. No camera movement."
            captions[str(img_path)] = caption

        except Exception as e:
            print(f"  Failed to caption {img_path.name}: {e}")

    elapsed = time.time() - t_start
    print(f"  Captioned {len(captions)}/{len(image_paths)} images in {elapsed:.0f}s")

    # Unload Qwen
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("  Qwen2.5-VL unloaded")

    return captions


def use_scrya_prompts(
    image_paths: list[Path],
    watch_dir: Path,
) -> dict[str, str]:
    """Use existing Scrya prompts from .txt files. Returns {path_str: caption}."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Using existing Scrya prompts for {len(image_paths)} images")
    print(f"{'='*60}")

    captions: dict[str, str] = {}
    missing = 0
    for img_path in image_paths:
        prompt = find_prompt_for_image(img_path)
        if prompt:
            # Format as training prompt with static camera prefix
            if not prompt.lower().startswith(("isometric", "static", "a 3d")):
                prompt = f"Isometric 3D view, static camera. {prompt.rstrip('.')}. No camera movement."
            captions[str(img_path)] = prompt
        else:
            missing += 1

    print(f"  Found prompts for {len(captions)}/{len(image_paths)} images ({missing} missing)")
    return captions


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Compute Gemma final embeddings
# ─────────────────────────────────────────────────────────────────────────────


def compute_embeddings(
    captions: dict[str, str],
    model_path: str | Path,
    text_encoder_path: str | Path,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute Gemma final embeddings for each caption. Returns {path_str: embedding_dict}."""
    from ltx_trainer.model_loader import load_text_encoder

    print(f"\n{'='*60}")
    print(f"  Phase 2: Computing Gemma embeddings for {len(captions)} captions")
    print(f"{'='*60}")

    t_start = time.time()

    text_encoder = load_text_encoder(
        str(model_path), str(text_encoder_path),
        device=device, dtype=torch.bfloat16,
        load_in_8bit=True,
    )
    text_encoder.eval()

    mem_gb = torch.cuda.memory_allocated(1) / 1e9
    print(f"  Gemma loaded ({mem_gb:.1f} GB)")

    embeddings: dict[str, dict[str, torch.Tensor]] = {}
    for path_str, caption in tqdm(captions.items(), desc="Embedding", unit="emb"):
        try:
            with torch.inference_mode():
                video_embeds, audio_embeds, attention_mask = text_encoder(caption)

            embeddings[path_str] = {
                "video_prompt_embeds": video_embeds.squeeze(0).cpu().contiguous(),
                "audio_prompt_embeds": (audio_embeds.squeeze(0).cpu().contiguous()
                                        if audio_embeds is not None
                                        else video_embeds.squeeze(0).cpu().contiguous()),
                "prompt_attention_mask": attention_mask.squeeze(0).cpu().contiguous(),
                "is_final_embedding": True,
            }
        except Exception as e:
            print(f"  Failed to embed: {e}")

    elapsed = time.time() - t_start
    print(f"  Computed {len(embeddings)} embeddings in {elapsed:.0f}s")

    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print("  Gemma unloaded")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: VAE encode + write to dataset
# ─────────────────────────────────────────────────────────────────────────────


def load_and_prepare_image(image_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load image, center-crop to target aspect ratio, resize, prepare for VAE.

    Returns:
        Tensor [1, C, 1, H, W] normalized to [-1, 1].
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    target_aspect = target_w / target_h
    source_aspect = w / h

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


def encode_and_write(
    image_paths: list[str],
    embeddings: dict[str, dict[str, torch.Tensor]],
    captions: dict[str, str],
    model_path: str | Path,
    dataset_dir: Path,
    target_h: int,
    target_w: int,
    device: str,
    watch_dir: Path,
    manifest: dict,
    processed_set: set[str],
) -> int:
    """VAE-encode images and write latents + conditions atomically. Returns count of new samples."""
    from ltx_trainer.model_loader import load_video_vae_encoder

    print(f"\n{'='*60}")
    print(f"  Phase 3: VAE encoding {len(image_paths)} images → dataset")
    print(f"{'='*60}")

    t_start = time.time()

    latents_dir = dataset_dir / "latents"
    conditions_dir = dataset_dir / "conditions_final"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    next_idx = find_next_index(latents_dir, manifest)

    vae_encoder = load_video_vae_encoder(str(model_path), dtype=torch.bfloat16)
    vae_encoder = vae_encoder.to(device)
    vae_encoder.eval()

    mem_gb = torch.cuda.memory_allocated(1) / 1e9
    print(f"  VAE encoder loaded ({mem_gb:.1f} GB)")

    count = 0
    for path_str in tqdm(image_paths, desc="Encoding", unit="img"):
        if path_str not in embeddings:
            continue

        img_path = Path(path_str)
        rel_key = str(img_path.relative_to(watch_dir))
        idx_str = f"{next_idx:06d}"
        latent_path = latents_dir / f"{idx_str}.pt"
        condition_path = conditions_dir / f"{idx_str}.pt"

        try:
            img_tensor = load_and_prepare_image(img_path, target_h, target_w)
            img_tensor = img_tensor.to(device, dtype=torch.bfloat16)

            with torch.inference_mode():
                latent = vae_encoder(img_tensor)

            latent = latent.cpu()
            latent_data = {
                "latents": latent.squeeze(0),  # [C, 1, H_lat, W_lat]
                "num_frames": torch.tensor([1]),
                "height": torch.tensor([latent.shape[3]]),
                "width": torch.tensor([latent.shape[4]]),
            }

            # Atomic write: .tmp → rename
            latent_tmp = latent_path.with_suffix(".tmp")
            torch.save(latent_data, latent_tmp)
            os.rename(latent_tmp, latent_path)

            condition_tmp = condition_path.with_suffix(".tmp")
            torch.save(embeddings[path_str], condition_tmp)
            os.rename(condition_tmp, condition_path)

            # Track success
            processed_set.add(rel_key)
            next_idx += 1
            count += 1

            manifest["processed_files"] = sorted(processed_set)
            manifest["next_index"] = next_idx
            save_manifest(dataset_dir, manifest)

        except Exception as e:
            print(f"  Failed to encode {img_path.name}: {e}")
            for p in [latent_path, latent_path.with_suffix(".tmp"),
                      condition_path, condition_path.with_suffix(".tmp")]:
                if p.exists():
                    p.unlink()

    elapsed = time.time() - t_start
    print(f"  Encoded {count} images in {elapsed:.0f}s")

    del vae_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print("  VAE encoder unloaded")

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def process_batch(
    new_images: list[Path],
    args: argparse.Namespace,
    watch_dir: Path,
    dataset_dir: Path,
    manifest: dict,
    processed_set: set[str],
) -> int:
    """Process a batch of new images through the full pipeline. Returns count ingested."""
    print(f"\n{'#'*60}")
    print(f"  Processing batch of {len(new_images)} new images")
    print(f"{'#'*60}")

    t_batch = time.time()

    # Phase 1: Caption
    if args.use_scrya_prompts:
        captions = use_scrya_prompts(new_images, watch_dir)
    else:
        captions = caption_images_with_qwen(
            new_images,
            model_path=args.qwen_model,
            device=args.device,
        )

    if not captions:
        print("  No captions generated, skipping batch")
        return 0

    # Phase 2: Gemma embeddings
    embeddings = compute_embeddings(
        captions,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        device=args.device,
    )

    if not embeddings:
        print("  No embeddings computed, skipping batch")
        return 0

    # Phase 3: VAE encode + write
    count = encode_and_write(
        image_paths=list(captions.keys()),
        embeddings=embeddings,
        captions=captions,
        model_path=args.model_path,
        dataset_dir=dataset_dir,
        target_h=args.target_height,
        target_w=args.target_width,
        device=args.device,
        watch_dir=watch_dir,
        manifest=manifest,
        processed_set=processed_set,
    )

    elapsed = time.time() - t_batch
    print(f"\n  Batch complete: {count} images ingested in {elapsed / 60:.1f} min")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-ingest pipeline: caption → embed → VAE encode → add to dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--watch-dir", type=Path, required=True,
                        help="Directory to watch for new images (recursive)")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Training dataset root (latents/ + conditions_final/)")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="LTX-2 .safetensors checkpoint")
    parser.add_argument("--text-encoder-path", type=Path, required=True,
                        help="Gemma text encoder directory")

    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Qwen2.5-VL model path or HF repo")
    parser.add_argument("--use-scrya-prompts", action="store_true",
                        help="Use existing .txt prompts from Scrya instead of Qwen captioning")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="CUDA device for all models (default: cuda:1)")

    parser.add_argument("--target-height", type=int, default=768,
                        help="Target height, must be divisible by 32")
    parser.add_argument("--target-width", type=int, default=1152,
                        help="Target width, must be divisible by 32")

    parser.add_argument("--batch-size", type=int, default=50,
                        help="Process images in batches of this size (limits model load/unload cycles)")
    parser.add_argument("--poll-interval", type=float, default=60.0,
                        help="Seconds between folder checks in watch mode")
    parser.add_argument("--one-shot", action="store_true",
                        help="Process once and exit (don't poll)")

    args = parser.parse_args()

    # Validate
    if args.target_height % 32 != 0 or args.target_width % 32 != 0:
        raise ValueError(f"Dimensions must be divisible by 32: {args.target_width}x{args.target_height}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.text_encoder_path.exists():
        raise FileNotFoundError(f"Text encoder not found: {args.text_encoder_path}")
    if not args.watch_dir.exists():
        raise FileNotFoundError(f"Watch directory not found: {args.watch_dir}")

    watch_dir = args.watch_dir
    dataset_dir = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Lockfile
    lockfile = acquire_lockfile(dataset_dir)
    atexit.register(release_lockfile, lockfile)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    manifest = load_manifest(dataset_dir)
    processed_set = set(manifest.get("processed_files", []))

    print(f"Auto-ingest pipeline")
    print(f"  Watch:     {watch_dir}")
    print(f"  Dataset:   {dataset_dir}")
    print(f"  Device:    {args.device}")
    print(f"  Resolution:{args.target_width}x{args.target_height}")
    print(f"  Captioning:{'Scrya prompts' if args.use_scrya_prompts else 'Qwen2.5-VL'}")
    print(f"  Mode:      {'one-shot' if args.one_shot else f'poll every {args.poll_interval}s'}")
    print(f"  Already processed: {len(processed_set)}")

    try:
        while True:
            new_images = discover_new_images(watch_dir, processed_set)

            if new_images:
                print(f"\nFound {len(new_images)} new image(s)")

                # Process in batches to limit memory per model-load cycle
                for i in range(0, len(new_images), args.batch_size):
                    batch = new_images[i : i + args.batch_size]
                    count = process_batch(
                        batch, args, watch_dir, dataset_dir,
                        manifest, processed_set,
                    )
                    if count > 0:
                        # Reload manifest in case it was updated
                        manifest = load_manifest(dataset_dir)
                        processed_set = set(manifest.get("processed_files", []))
            else:
                if args.one_shot:
                    print("No new images found.")
                    break

            if args.one_shot:
                break

            print(f"  Sleeping {args.poll_interval}s...")
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nShutting down auto-ingest pipeline")


if __name__ == "__main__":
    main()
