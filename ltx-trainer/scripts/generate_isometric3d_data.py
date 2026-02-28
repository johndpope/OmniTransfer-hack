#!/usr/bin/env python3
"""Generate photorealistic isometric 3D training data using Grok (xAI) API.

Transforms movie scene screenshots into photorealistic isometric 3D renderings.
This is NOT diorama/miniature — it's full-scale architecture viewed from an
elevated isometric camera angle (~30°), like architectural visualization.

Pipeline:
  1. For each movie scene screenshot, call Grok image edit API
  2. Save isometric 3D renderings
  3. Create horizontal flip augmentations (ADA)
  4. Encode to VAE latents
  5. Compute text embeddings
  6. Create cross-paired dataset for OmniTransfer training

Usage:
  export GROK_API_KEY=xai-...
  cd ~/Documents/GitHub/ltx2-omnitransfer/packages/ltx-trainer
  uv run python scripts/generate_isometric3d_data.py --dry-run
  uv run python scripts/generate_isometric3d_data.py

Cost: ~$1.58 for 79 scenes (1 variant × $0.02/image)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from generate_grok_training_data import SCENE_DESCRIPTIONS, NO_FLIP_SCENES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = "https://api.x.ai/v1"
PRICE_PER_IMAGE = 0.02

# The isometric 3D prompt — key distinction from diorama
ISOMETRIC_PROMPT = (
    "Transform this movie scene into a photorealistic isometric 3D rendering. "
    "Viewed from an elevated 30-degree angle looking down at the scene. "
    "Full-scale realistic environment with accurate lighting, materials, and proportions. "
    "Clean isometric perspective like an architectural visualization or video game render. "
    "NOT a miniature, NOT a diorama, NOT tilt-shift — this is full-scale and photorealistic."
)


def get_headers() -> dict:
    key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("Set GROK_API_KEY or XAI_API_KEY")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def image_to_base64(path: Path, max_size: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def download_url(url: str, save_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_isometric_image(
    scene_img: Path,
    scene_name: str,
    output_dir: Path,
) -> Path | None:
    """Call Grok image edit to transform movie screenshot → isometric 3D."""
    save_path = output_dir / f"{scene_name}_iso3d.jpg"
    if save_path.exists():
        logger.info(f"  {scene_name}: already exists, skipping")
        return save_path

    b64_uri = image_to_base64(scene_img)
    headers = get_headers()
    payload = {
        "model": "grok-imagine-image",
        "prompt": ISOMETRIC_PROMPT,
        "n": 1,
        "image": {"url": b64_uri, "type": "image_url"},
    }

    try:
        resp = requests.post(
            f"{API_BASE}/images/edits", headers=headers, json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"  {scene_name}: API error — {e}")
        return None

    images = data.get("data", [])
    if not images:
        logger.error(f"  {scene_name}: no images returned")
        return None

    img_data = images[0]
    if "b64_json" in img_data:
        raw = base64.b64decode(img_data["b64_json"])
        save_path.write_bytes(raw)
        return save_path
    elif "url" in img_data:
        if download_url(img_data["url"], save_path):
            return save_path

    return None


def flip_image(image_path: Path, output_dir: Path) -> Path | None:
    try:
        img = Image.open(image_path).convert("RGB")
        flipped = ImageOps.mirror(img)
        flip_path = output_dir / f"{image_path.stem}_flip{image_path.suffix}"
        flipped.save(flip_path, quality=90)
        return flip_path
    except Exception as e:
        logger.error(f"Flip failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────
def discover_scenes(movie_dir: Path) -> list[dict]:
    """Find all movie scenes with screenshot images."""
    scenes = []
    for d in sorted(movie_dir.iterdir()):
        if not d.is_dir() or d.name not in SCENE_DESCRIPTIONS:
            continue
        scene_img = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = d / f"scene{ext}"
            if candidate.exists():
                scene_img = candidate
                break
        if scene_img:
            scenes.append({"name": d.name, "scene_img": scene_img})
    return scenes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate photorealistic isometric 3D images from movie scenes via Grok API"
    )
    parser.add_argument("--movie-dir", type=Path, default=Path("/media/2TB/movie_dioramas"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/media/2TB/grok_isometric3d")
    )
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-flip", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Delay between API calls (s)")
    args = parser.parse_args()

    scenes = discover_scenes(args.movie_dir)
    logger.info(f"Found {len(scenes)} movie scenes with screenshots")

    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    img_dir = args.output_dir / "images"
    flip_dir = args.output_dir / "images_flipped"
    img_dir.mkdir(parents=True, exist_ok=True)
    flip_dir.mkdir(parents=True, exist_ok=True)

    est_cost = len(scenes) * PRICE_PER_IMAGE
    logger.info(f"Estimated cost: ${est_cost:.2f} for {len(scenes)} scenes")

    if est_cost > args.budget:
        logger.error(f"Estimated cost ${est_cost:.2f} exceeds budget ${args.budget:.2f}")
        return

    if args.dry_run:
        logger.info("\n--- DRY RUN ---")
        for s in scenes:
            logger.info(f"  Would generate: {s['name']}_iso3d.jpg from {s['scene_img'].name}")
        logger.info(f"Total: {len(scenes)} images × $0.02 = ${est_cost:.2f}")
        logger.info(f"With flip: ~{len(scenes) * 2 - len(NO_FLIP_SCENES)} total images")
        return

    # ─── Generate isometric 3D images ───
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating photorealistic isometric 3D images...")
    logger.info(f"{'='*60}")

    spent = 0.0
    generated = []
    failed = []

    for i, scene in enumerate(scenes):
        if spent + PRICE_PER_IMAGE > args.budget:
            logger.warning(f"Budget limit reached at scene {i}")
            break

        logger.info(f"[{i+1}/{len(scenes)}] {scene['name']}...")
        result = generate_isometric_image(scene["scene_img"], scene["name"], img_dir)

        if result:
            generated.append({"scene": scene["name"], "path": str(result)})
            # Only charge if we actually called the API (not if cached)
            if not (img_dir / f"{scene['name']}_iso3d.jpg").stat().st_size == 0:
                spent += PRICE_PER_IMAGE
        else:
            failed.append(scene["name"])

        time.sleep(args.rate_limit)

    logger.info(f"\nGenerated: {len(generated)}, Failed: {len(failed)}")
    if failed:
        logger.info(f"Failed scenes: {failed}")
    logger.info(f"Spent: ~${spent:.2f}")

    # ─── Flip augmentation ───
    flipped_images = []
    if not args.skip_flip:
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating horizontal flip augmentations...")
        logger.info(f"{'='*60}")

        for item in generated:
            scene_name = item["scene"]
            if scene_name in NO_FLIP_SCENES:
                logger.info(f"  Skipping flip for {scene_name}")
                continue
            src = Path(item["path"])
            flipped = flip_image(src, flip_dir)
            if flipped:
                flipped_images.append({"scene": scene_name, "path": str(flipped)})

        logger.info(f"Created {len(flipped_images)} flipped images")

    # ─── Write manifest ───
    all_images = generated + [
        {**f, "is_flipped": True} for f in flipped_images
    ]
    for item in generated:
        item["is_flipped"] = False

    manifest = {
        "style": "isometric_3d",
        "prompt": ISOMETRIC_PROMPT,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cost_spent": spent,
        "num_original": len(generated),
        "num_flipped": len(flipped_images),
        "num_total": len(all_images),
        "failed_scenes": failed,
        "scene_descriptions": SCENE_DESCRIPTIONS,
        "images": all_images,
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"\nManifest: {manifest_path}")

    # ─── Summary ───
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE")
    logger.info(f"{'='*60}")
    logger.info(f"Original images: {len(generated)}")
    logger.info(f"Flipped images:  {len(flipped_images)}")
    logger.info(f"Total:           {len(all_images)}")
    logger.info(f"Output:          {args.output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review images in {img_dir}/")
    logger.info(f"  2. Encode to latents + create cross-pairs:")
    logger.info(f"     uv run python scripts/encode_grok_dataset.py \\")
    logger.info(f"       --input-dir {args.output_dir} \\")
    logger.info(f"       --movie-dir {args.movie_dir} \\")
    logger.info(f"       --output-dir /media/2TB/grok_isometric3d_crosspair")


if __name__ == "__main__":
    main()
