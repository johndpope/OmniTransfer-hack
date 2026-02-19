#!/usr/bin/env python3
"""
Generate OmniTransfer style-transfer training data using Grok (xAI) API.

Pipeline:
  1. Image Edit: movie screenshot → diorama variants (n=4 per scene)
  2. Image-to-Video: animate best dioramas to short videos
  3. Neutral captions: generate content-only descriptions (no style leakage)
  4. ADA: horizontal flip augmentation on all outputs

Budget: hard cap at $5.00

Usage:
  export GROK_API_KEY=xai-...
  python scripts/generate_grok_training_data.py \
      --movie-dir /media/2TB/movie_dioramas \
      --output-dir /media/2TB/grok_training_data \
      --budget 5.0
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pricing constants (as of 2026-02)
# ─────────────────────────────────────────────────────────────────────────────
PRICE_IMAGE_EDIT = 0.02       # per image (grok-imagine-image)
PRICE_IMAGE_GEN = 0.02        # per image (grok-imagine-image)
PRICE_VIDEO_PER_SEC = 0.05    # per second (grok-imagine-video)
VIDEO_DURATION = 5             # seconds per video

# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = "https://api.x.ai/v1"


def get_headers() -> dict:
    key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("Set GROK_API_KEY or XAI_API_KEY environment variable")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def image_to_base64(path: str | Path, max_size: int = 1024) -> str:
    """Load image, resize if too large, return base64 data URI."""
    img = Image.open(path).convert("RGB")
    # Resize to save bandwidth and stay within API limits
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    # Encode as JPEG for smaller payloads
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def download_url(url: str, save_path: Path) -> bool:
    """Download a URL to a local file. Returns True on success."""
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.error(f"Download failed {url}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Image Edit — movie screenshot → diorama variants
# ─────────────────────────────────────────────────────────────────────────────
def generate_diorama_images(
    scene_path: Path,
    scene_name: str,
    output_dir: Path,
    n_variants: int = 4,
) -> list[Path]:
    """Call Grok image edit API to transform a movie screenshot into diorama variants."""
    b64_uri = image_to_base64(scene_path)

    prompt = (
        "Transform this movie scene into an isometric 3D miniature diorama. "
        "Photorealistic tiny detailed model with tilt-shift depth of field. "
        "Keep the scene composition but render as a tabletop miniature."
    )

    headers = get_headers()
    # Try image generation with reference image (edit endpoint)
    payload = {
        "model": "grok-imagine-image",
        "prompt": prompt,
        "n": n_variants,
        "image": {
            "url": b64_uri,
            "type": "image_url",
        },
    }

    logger.info(f"  Generating {n_variants} diorama variants for {scene_name}...")
    try:
        resp = requests.post(f"{API_BASE}/images/edits", headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        # If edit doesn't support n>1, fall back to single requests
        if resp.status_code == 422 or resp.status_code == 400:
            logger.warning(f"  Edit API may not support n={n_variants}, trying n=1 loop")
            return _generate_diorama_images_single(b64_uri, prompt, scene_name, output_dir, n_variants)
        logger.error(f"  API error for {scene_name}: {e} — {resp.text}")
        return []
    except Exception as e:
        logger.error(f"  Request failed for {scene_name}: {e}")
        return []

    # Save results
    saved = []
    images = data.get("data", [])
    for i, img_data in enumerate(images):
        fname = f"{scene_name}_grok_v{i}.jpg"
        save_path = output_dir / fname

        if "b64_json" in img_data:
            raw = base64.b64decode(img_data["b64_json"])
            save_path.write_bytes(raw)
            saved.append(save_path)
        elif "url" in img_data:
            if download_url(img_data["url"], save_path):
                saved.append(save_path)

    logger.info(f"  Saved {len(saved)} images for {scene_name}")
    return saved


def _generate_diorama_images_single(
    b64_uri: str,
    prompt: str,
    scene_name: str,
    output_dir: Path,
    n_variants: int,
) -> list[Path]:
    """Fallback: generate one image at a time if n>1 not supported."""
    headers = get_headers()
    saved = []
    for i in range(n_variants):
        payload = {
            "model": "grok-imagine-image",
            "prompt": prompt,
            "n": 1,
            "image": {
                "url": b64_uri,
                "type": "image_url",
            },
        }
        try:
            resp = requests.post(f"{API_BASE}/images/edits", headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            img_data = data["data"][0]

            fname = f"{scene_name}_grok_v{i}.jpg"
            save_path = output_dir / fname

            if "b64_json" in img_data:
                raw = base64.b64decode(img_data["b64_json"])
                save_path.write_bytes(raw)
                saved.append(save_path)
            elif "url" in img_data:
                if download_url(img_data["url"], save_path):
                    saved.append(save_path)

            time.sleep(0.3)  # Rate limit courtesy
        except Exception as e:
            logger.error(f"  Variant {i} failed for {scene_name}: {e}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Image-to-Video — animate diorama images
# ─────────────────────────────────────────────────────────────────────────────
def generate_diorama_video(
    image_path: Path,
    scene_name: str,
    output_dir: Path,
    duration: int = VIDEO_DURATION,
) -> Path | None:
    """Call Grok image-to-video API to animate a diorama image."""
    b64_uri = image_to_base64(image_path, max_size=720)

    prompt = (
        "Slow gentle camera movement around this miniature diorama. "
        "Subtle parallax and depth-of-field shift. Tilt-shift photography style."
    )

    headers = get_headers()
    payload = {
        "model": "grok-imagine-video",
        "prompt": prompt,
        "image_url": b64_uri,
        "duration": duration,
        "aspect_ratio": "16:9",
        "resolution": "480p",
    }

    logger.info(f"  Generating {duration}s video for {scene_name}...")
    try:
        resp = requests.post(f"{API_BASE}/videos/generations", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        request_id = data.get("request_id")
        if not request_id:
            logger.error(f"  No request_id in response: {data}")
            return None
    except Exception as e:
        logger.error(f"  Video request failed for {scene_name}: {e}")
        return None

    # Poll for completion
    # NOTE: Grok video API response format:
    #   - When ready: {"video": {"url": "...", "duration": N}, "model": "..."}
    #   - When pending: {"status": "pending"} or similar
    #   - No "status" field when complete — check for "video" key directly
    poll_url = f"{API_BASE}/videos/{request_id}"
    poll_headers = {"Authorization": get_headers()["Authorization"]}

    for attempt in range(120):  # max 10 minutes
        time.sleep(5)
        try:
            result = requests.get(poll_url, headers=poll_headers, timeout=30).json()

            # Check if video is ready (response contains "video" with "url")
            if "video" in result and isinstance(result["video"], dict):
                video_url = result["video"].get("url")
                if video_url:
                    save_path = output_dir / f"{scene_name}_video.mp4"
                    if download_url(video_url, save_path):
                        logger.info(f"  Video saved: {save_path}")
                        return save_path
                    else:
                        logger.error(f"  Video download failed for {scene_name}")
                        return None

            # Check explicit status field (if present)
            status = result.get("status", "")
            if status in ("expired", "failed", "error"):
                logger.error(f"  Video {status} for {scene_name}: {result}")
                return None

            # Check for error in response
            if "error" in result:
                logger.error(f"  Video error for {scene_name}: {result['error']}")
                return None

            if attempt % 6 == 0:
                logger.info(f"  Still processing {scene_name} (attempt {attempt}, status={status or 'generating'})...")

        except Exception as e:
            logger.warning(f"  Poll error: {e}")

    logger.error(f"  Video timed out for {scene_name}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Neutral content descriptions (no style leakage!)
# ─────────────────────────────────────────────────────────────────────────────
# Handcrafted neutral descriptions for key movie scenes
# These describe CONTENT only — no "isometric", "diorama", "3D", "miniature"
SCENE_DESCRIPTIONS = {
    "2001_space_odyssey": "a futuristic white room with glowing floor panels and a monolith",
    "alien_nostromo": "a dark industrial spaceship corridor with pipes and warning lights",
    "aliens_hive": "a cavernous alien nest with organic walls and dim emergency lighting",
    "american_beauty_roses": "a suburban house exterior with red rose petals scattered everywhere",
    "apocalypse_now_river": "a jungle river scene with a military patrol boat at dusk",
    "avatar_pandora": "a bioluminescent alien forest with glowing plants at night",
    "back_to_future_mall": "a parking lot at night with a DeLorean car and lightning",
    "batman_89_gotham": "a dark gothic city street with cathedral architecture",
    "blade_runner_2049_city": "a foggy dystopian cityscape with towering brutalist buildings",
    "blade_runner_rooftop": "a rainy rooftop at night with neon signs and flying vehicles",
    "breakfast_club_library": "a high school library with long tables and bookshelves",
    "casablanca_piano": "a 1940s nightclub interior with a grand piano and bar",
    "chinatown_office": "a vintage private detective office with blinds and desk lamp",
    "clockwork_orange_bar": "a surreal white bar with mannequin furniture and milk dispensers",
    "dark_knight_interrogation": "a stark concrete interrogation room with a single table",
    "die_hard_nakatomi": "a modern office building lobby with marble floors and Christmas decorations",
    "django_plantation": "a southern plantation mansion with white columns and cotton fields",
    "drive_highway": "a nighttime highway scene with city lights and a muscle car",
    "dune_desert": "a vast desert landscape with sand dunes and a distant spice harvester",
    "empire_strikes_back_hoth": "a frozen ice cave base with military equipment and snow",
    "et_bicycle": "a suburban neighborhood at night with a bicycle silhouetted against the moon",
    "eternal_sunshine_beach": "a frozen beach in winter with crumbling buildings",
    "exorcist_stairs": "steep stone stairs between row houses at night with a streetlamp",
    "fargo_snow": "a snowy midwest highway with a car accident scene",
    "fifth_element_taxi": "a futuristic city with flying taxis between towering buildings",
    "fight_club_basement": "a dingy basement with bare bulbs and concrete pillars",
    "forrest_gump_bench": "a park bench on a tree-lined sidewalk in a small town",
    "full_metal_jacket_barracks": "a military barracks with rows of bunk beds",
    "ghostbusters_library": "a grand old library with tall bookshelves and chandeliers",
    "gladiator_arena": "a Roman colosseum arena with sand floor and tiered seating",
    "godfather_office": "a dimly lit study with a wooden desk, lamp, and venetian blinds",
    "goodfellas_bar": "a 1970s Italian-American bar with wood paneling and neon beer signs",
    "good_will_hunting_bench": "a park bench overlooking a pond in autumn",
    "gravity_space": "an astronaut floating in orbit with Earth in the background",
    "heat_shootout": "a downtown city street with cars and buildings during a confrontation",
    "inception_hotel": "a hotel corridor that bends and rotates impossibly",
    "indiana_jones_boulder": "an ancient temple interior with stone walls and booby traps",
    "inglourious_basterds_tavern": "a 1940s basement tavern with wooden tables and dim lighting",
    "interstellar_wormhole": "a spacecraft approaching a glowing spherical wormhole in space",
    "jaws_boat": "a small fishing boat on the open ocean at sunset",
    "jurassic_park_trex": "a tropical park road at night with overturned vehicles in rain",
    "kill_bill_showdown": "a Japanese garden courtyard covered in snow",
    "leon_apartment": "a sparse New York apartment with a potted plant by the window",
    "lord_of_rings_fellowship": "a serene elven valley with waterfalls and ornate architecture",
    "lotr_two_towers_helms": "a massive stone fortress on a cliff during a thunderstorm",
    "mad_max_warrig": "a post-apocalyptic desert highway with armored vehicles",
    "matrix_lobby": "a grand government building lobby with marble columns and metal detectors",
    "memento_polaroid": "a motel room covered in polaroid photos and handwritten notes",
    "no_country_desert": "a desolate Texas desert gas station at dusk",
    "one_flew_cuckoo": "a 1960s psychiatric ward dayroom with linoleum floors",
    "parasite_basement": "a cramped underground bunker hidden beneath a modern house",
    "pirates_caribbean_ship": "a wooden pirate ship on rough seas at sunset",
    "planet_apes_statue": "a beach with the Statue of Liberty half-buried in sand",
    "point_break_surf": "an ocean wave with a surfer at golden hour",
    "psycho_shower": "a vintage bathroom with a shower curtain and tiled walls",
    "pulp_fiction_diner": "a retro 1950s-themed diner with red booths and checkered floor",
    "rain_man_highway": "a long straight highway through flat farmland",
    "revenant_forest": "a snowy wilderness forest with a frozen river",
    "robocop_detroit": "a gritty urban Detroit street with abandoned buildings",
    "rocky_steps": "wide stone steps leading up to a neoclassical museum building",
    "saving_private_ryan_beach": "a stormy beach during a military landing with obstacles",
    "scarface_mansion": "an opulent art deco mansion interior with a grand staircase",
    "schindlers_list_factory": "a World War II era factory floor with heavy machinery",
    "se7en_apartment": "a dark cramped apartment with peeling wallpaper and bare bulbs",
    "shawshank_rain": "a man standing in the rain in a prison courtyard at night",
    "shining_hallway": "a long hotel corridor with geometric carpet and wall sconces",
    "silence_lambs_cell": "an underground prison cell with glass walls and dim lighting",
    "social_network_office": "a modern tech startup office with whiteboards and laptops",
    "star_wars_cantina": "an alien cantina bar with diverse patrons and curved architecture",
    "t2_junkyard": "an industrial junkyard with molten metal and heavy machinery",
    "taxi_driver_mirror": "a New York City taxi interior at night with neon reflections",
    "terminator2_highway": "a Los Angeles highway with a truck and motorcycle chase",
    "the_birds_playground": "a seaside town playground with a jungle gym and schoolhouse",
    "there_will_be_blood_oil": "an early 1900s oil derrick in a dusty desert landscape",
    "thing_station": "an isolated Antarctic research station in a blizzard",
    "titanic_bow": "the bow of a large ocean liner at sunset on calm seas",
    "total_recall_mars": "a red Martian landscape with domed habitation structures",
    "truman_show_dome": "a perfect suburban neighborhood with an impossibly blue sky",
    "wall_e_wasteland": "a post-apocalyptic cityscape buried in compressed garbage towers",
    "whiplash_stage": "a jazz club stage with drum kit, spotlights, and music stands",
    "wizard_of_oz_tornado": "a Kansas farmhouse with a dark tornado funnel approaching",
}


def get_scene_description(scene_name: str) -> str:
    """Get a neutral content description for a scene (no style info)."""
    return SCENE_DESCRIPTIONS.get(scene_name, f"a cinematic scene from a movie")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: ADA — Horizontal flip augmentation
# ─────────────────────────────────────────────────────────────────────────────
# Scenes where horizontal flip does NOT make sense (strong left-right asymmetry, text, etc.)
NO_FLIP_SCENES = {
    "memento_polaroid",      # handwritten text
    "social_network_office",  # text on whiteboards
    "taxi_driver_mirror",     # mirror reflection has specific orientation
    "forrest_gump_bench",     # iconic composition
    "rocky_steps",            # iconic left-to-right run
}


def create_flipped_variant(image_path: Path, output_dir: Path) -> Path | None:
    """Create a horizontally flipped copy of an image for ADA."""
    try:
        img = Image.open(image_path).convert("RGB")
        flipped = ImageOps.mirror(img)
        # Save with _flip suffix
        stem = image_path.stem
        suffix = image_path.suffix
        flip_path = output_dir / f"{stem}_flip{suffix}"
        flipped.save(flip_path, quality=90)
        return flip_path
    except Exception as e:
        logger.error(f"Flip failed for {image_path}: {e}")
        return None


def create_flipped_video(video_path: Path, output_dir: Path) -> Path | None:
    """Create a horizontally flipped copy of a video using ffmpeg."""
    try:
        import subprocess
        stem = video_path.stem
        flip_path = output_dir / f"{stem}_flip.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "hflip",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            str(flip_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return flip_path
    except Exception as e:
        logger.error(f"Video flip failed for {video_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cost tracker
# ─────────────────────────────────────────────────────────────────────────────
class CostTracker:
    def __init__(self, budget: float):
        self.budget = budget
        self.spent = 0.0
        self.breakdown: list[dict] = []

    def can_afford(self, cost: float) -> bool:
        return (self.spent + cost) <= self.budget

    def charge(self, cost: float, description: str):
        self.spent += cost
        self.breakdown.append({"cost": cost, "desc": description, "total": self.spent})
        logger.info(f"  💰 ${cost:.2f} — {description} (total: ${self.spent:.2f} / ${self.budget:.2f})")

    def remaining(self) -> float:
        return self.budget - self.spent

    def summary(self) -> str:
        lines = [f"Budget: ${self.budget:.2f}, Spent: ${self.spent:.2f}, Remaining: ${self.remaining():.2f}"]
        for item in self.breakdown:
            lines.append(f"  ${item['cost']:.2f} — {item['desc']}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def discover_scenes(movie_dir: Path) -> list[dict]:
    """Find all movie scene directories with scene images."""
    scenes = []
    for d in sorted(movie_dir.iterdir()):
        if not d.is_dir():
            continue
        # Find scene image (prefer .jpg)
        scene_img = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = d / f"scene{ext}"
            if candidate.exists():
                scene_img = candidate
                break
        if scene_img is None:
            continue

        # Find existing diorama images
        dioramas = sorted(d.glob("diorama*.png")) + sorted(d.glob("diorama*.jpg"))

        scenes.append({
            "name": d.name,
            "dir": d,
            "scene_img": scene_img,
            "existing_dioramas": dioramas,
            "description": get_scene_description(d.name),
        })

    return scenes


def main():
    parser = argparse.ArgumentParser(description="Generate Grok training data for OmniTransfer")
    parser.add_argument("--movie-dir", type=Path, default=Path("/media/2TB/movie_dioramas"))
    parser.add_argument("--output-dir", type=Path, default=Path("/media/2TB/grok_training_data"))
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--n-image-variants", type=int, default=4, help="Variants per scene for image edit")
    parser.add_argument("--n-image-scenes", type=int, default=25, help="Max scenes for image generation")
    parser.add_argument("--n-video-scenes", type=int, default=10, help="Max scenes for video generation")
    parser.add_argument("--video-duration", type=int, default=5, help="Video duration in seconds")
    parser.add_argument("--skip-images", action="store_true", help="Skip image generation phase")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video generation phase")
    parser.add_argument("--skip-flip", action="store_true", help="Skip ADA flip augmentation")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without calling API")
    args = parser.parse_args()

    cost = CostTracker(args.budget)

    # Setup output dirs
    img_dir = args.output_dir / "images"
    vid_dir = args.output_dir / "videos"
    flip_img_dir = args.output_dir / "images_flipped"
    flip_vid_dir = args.output_dir / "videos_flipped"
    for d in [img_dir, vid_dir, flip_img_dir, flip_vid_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Discover scenes
    scenes = discover_scenes(args.movie_dir)
    logger.info(f"Found {len(scenes)} movie scenes")

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1: Generate diorama image variants
    # ─────────────────────────────────────────────────────────────────────
    all_images: list[dict] = []  # {scene_name, path, is_flipped, source}

    if not args.skip_images:
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: Image Edit (movie → diorama) — up to {args.n_image_scenes} scenes")
        logger.info(f"{'='*60}")

        # Pick diverse subset of scenes
        image_scenes = scenes[:args.n_image_scenes]
        random.seed(42)
        random.shuffle(scenes)
        image_scenes = scenes[:args.n_image_scenes]

        for scene in image_scenes:
            # Cost check: n variants at $0.02 each
            est_cost = args.n_image_variants * PRICE_IMAGE_EDIT
            if not cost.can_afford(est_cost):
                logger.warning(f"Budget exhausted at {scene['name']}. Stopping image generation.")
                break

            if args.dry_run:
                logger.info(f"  [DRY RUN] Would generate {args.n_image_variants} dioramas for {scene['name']}")
                cost.charge(est_cost, f"{args.n_image_variants}x diorama images: {scene['name']}")
                continue

            saved = generate_diorama_images(
                scene_path=scene["scene_img"],
                scene_name=scene["name"],
                output_dir=img_dir,
                n_variants=args.n_image_variants,
            )

            actual_cost = len(saved) * PRICE_IMAGE_EDIT
            if saved:
                cost.charge(actual_cost, f"{len(saved)}x diorama images: {scene['name']}")

            for p in saved:
                all_images.append({
                    "scene_name": scene["name"],
                    "path": str(p),
                    "is_flipped": False,
                    "source": "grok_edit",
                    "description": scene["description"],
                })

            time.sleep(0.5)  # Rate limit

    # Also include existing diorama images from movie_dioramas/
    logger.info(f"\nAdding existing diorama images from {args.movie_dir}...")
    for scene in scenes:
        for diorama_path in scene["existing_dioramas"]:
            # Copy to output dir for consistency
            dest = img_dir / f"{scene['name']}_existing_{diorama_path.stem}.jpg"
            if not dest.exists():
                try:
                    img = Image.open(diorama_path).convert("RGB")
                    img.save(dest, quality=90)
                except Exception as e:
                    logger.warning(f"Could not copy {diorama_path}: {e}")
                    continue

            all_images.append({
                "scene_name": scene["name"],
                "path": str(dest),
                "is_flipped": False,
                "source": "existing",
                "description": scene["description"],
            })

    logger.info(f"Total images before flip: {len(all_images)}")

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2: Generate diorama videos
    # ─────────────────────────────────────────────────────────────────────
    all_videos: list[dict] = []

    if not args.skip_videos:
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2: Image-to-Video — up to {args.n_video_scenes} dioramas")
        logger.info(f"{'='*60}")

        # Pick scenes that have good diorama images to animate
        # Prefer existing dioramas (known good quality)
        video_candidates = [s for s in scenes if s["existing_dioramas"]]
        random.shuffle(video_candidates)
        video_scenes = video_candidates[:args.n_video_scenes]

        for scene in video_scenes:
            est_cost = args.video_duration * PRICE_VIDEO_PER_SEC
            if not cost.can_afford(est_cost):
                logger.warning(f"Budget exhausted at {scene['name']}. Stopping video generation.")
                break

            # Use first existing diorama image as source
            source_img = scene["existing_dioramas"][0]

            if args.dry_run:
                logger.info(f"  [DRY RUN] Would generate {args.video_duration}s video for {scene['name']}")
                cost.charge(est_cost, f"{args.video_duration}s video: {scene['name']}")
                continue

            video_path = generate_diorama_video(
                image_path=source_img,
                scene_name=scene["name"],
                output_dir=vid_dir,
                duration=args.video_duration,
            )

            if video_path:
                cost.charge(est_cost, f"{args.video_duration}s video: {scene['name']}")
                all_videos.append({
                    "scene_name": scene["name"],
                    "path": str(video_path),
                    "is_flipped": False,
                    "source": "grok_video",
                    "description": scene["description"],
                    "duration": args.video_duration,
                })

    # Also include existing Grok videos
    grok_video_dir = Path("/media/12TB/isometric_3d/r2_native_dataset/new_grok_videos")
    if grok_video_dir.exists():
        logger.info(f"\nAdding existing Grok videos from {grok_video_dir}...")
        for mp4 in sorted(grok_video_dir.glob("*.mp4")):
            # Read associated prompt
            txt_file = mp4.with_suffix(".txt")
            desc = txt_file.read_text().strip()[:200] if txt_file.exists() else "an isometric scene"

            all_videos.append({
                "scene_name": mp4.stem[:20],
                "path": str(mp4),
                "is_flipped": False,
                "source": "existing_grok",
                "description": desc,
            })

    logger.info(f"Total videos before flip: {len(all_videos)}")

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3: ADA — Horizontal flip augmentation
    # ─────────────────────────────────────────────────────────────────────
    if not args.skip_flip:
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 3: ADA Horizontal Flip Augmentation")
        logger.info(f"{'='*60}")

        # Flip images
        new_images = []
        for item in all_images:
            scene = item["scene_name"]
            if scene in NO_FLIP_SCENES:
                logger.info(f"  Skipping flip for {scene} (asymmetric/text)")
                continue

            src = Path(item["path"])
            if not src.exists():
                continue

            if args.dry_run:
                new_images.append({**item, "is_flipped": True, "path": str(src).replace(".jpg", "_flip.jpg")})
                continue

            flipped = create_flipped_variant(src, flip_img_dir)
            if flipped:
                new_images.append({
                    "scene_name": scene,
                    "path": str(flipped),
                    "is_flipped": True,
                    "source": item["source"] + "_flip",
                    "description": item["description"],
                })

        all_images.extend(new_images)
        logger.info(f"  Added {len(new_images)} flipped images → total: {len(all_images)}")

        # Flip videos
        new_videos = []
        for item in all_videos:
            scene = item["scene_name"]
            if scene in NO_FLIP_SCENES:
                continue

            src = Path(item["path"])
            if not src.exists():
                continue

            if args.dry_run:
                new_videos.append({**item, "is_flipped": True})
                continue

            flipped = create_flipped_video(src, flip_vid_dir)
            if flipped:
                new_videos.append({
                    "scene_name": scene,
                    "path": str(flipped),
                    "is_flipped": True,
                    "source": item["source"] + "_flip",
                    "description": item["description"],
                    "duration": item.get("duration", 5),
                })

        all_videos.extend(new_videos)
        logger.info(f"  Added {len(new_videos)} flipped videos → total: {len(all_videos)}")

    # ─────────────────────────────────────────────────────────────────────
    # Phase 4: Write manifest
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 4: Writing manifest")
    logger.info(f"{'='*60}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cost_summary": {
            "budget": cost.budget,
            "spent": cost.spent,
            "remaining": cost.remaining(),
        },
        "counts": {
            "images_total": len(all_images),
            "images_original": len([i for i in all_images if not i["is_flipped"]]),
            "images_flipped": len([i for i in all_images if i["is_flipped"]]),
            "images_grok_new": len([i for i in all_images if i["source"] == "grok_edit"]),
            "images_existing": len([i for i in all_images if i["source"] == "existing"]),
            "videos_total": len(all_videos),
            "videos_original": len([v for v in all_videos if not v["is_flipped"]]),
            "videos_flipped": len([v for v in all_videos if v["is_flipped"]]),
            "videos_grok_new": len([v for v in all_videos if v["source"] == "grok_video"]),
            "videos_existing": len([v for v in all_videos if v["source"] == "existing_grok"]),
        },
        "scene_descriptions": {s["name"]: s["description"] for s in scenes},
        "images": all_images,
        "videos": all_videos,
        "cost_breakdown": cost.breakdown,
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Manifest written to {manifest_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Images:  {manifest['counts']['images_total']} total "
                f"({manifest['counts']['images_grok_new']} new + "
                f"{manifest['counts']['images_existing']} existing + "
                f"{manifest['counts']['images_flipped']} flipped)")
    logger.info(f"Videos:  {manifest['counts']['videos_total']} total "
                f"({manifest['counts']['videos_grok_new']} new + "
                f"{manifest['counts']['videos_existing']} existing + "
                f"{manifest['counts']['videos_flipped']} flipped)")
    logger.info(f"\n{cost.summary()}")
    logger.info(f"\nOutput: {args.output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review generated images/videos in {args.output_dir}")
    logger.info(f"  2. Encode to latents: python scripts/encode_grok_dataset.py --input-dir {args.output_dir}")
    logger.info(f"  3. Compute neutral text embeddings")
    logger.info(f"  4. Create cross-pair training dataset")
    logger.info(f"  5. Train with corrected OmniTransfer pipeline")


if __name__ == "__main__":
    main()
