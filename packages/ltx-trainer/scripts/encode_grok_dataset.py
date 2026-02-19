#!/usr/bin/env python3
"""Encode Grok-generated diorama images into a cross-paired OmniTransfer dataset.

This script:
  1. Collects ALL diorama images (original, Grok-generated, flipped)
  2. Encodes each to VAE latents [128, 1, 14, 26] at 832×448
  3. Creates cross-paired (ref, target) style-transfer training samples
  4. Writes metadata with neutral scene descriptions (no style leakage)

The output is a ready-to-train dataset where:
  - reference_latents/N.pt = diorama from scene_i (style source)
  - latents/N.pt = diorama from scene_j (ground truth, DIFFERENT scene)
  - conditions_final/N.pt = neutral text embedding for scene_j (computed separately)
  - metadata.json = cross-pair mapping + neutral captions

Usage:
    # Step 1: Encode images + create cross-pairs
    python scripts/encode_grok_dataset.py \
        --input-dir /media/2TB/grok_training_data \
        --movie-dir /media/2TB/movie_dioramas \
        --output-dir /media/2TB/grok_diorama_crosspair \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors

    # Step 2: Compute text embeddings (separate process, ~28GB VRAM)
    python scripts/compute_final_embeddings.py \
        --dataset-dir /media/2TB/grok_diorama_crosspair \
        --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \
        --text-encoder-path /media/2TB/ltx-models/gemma \
        --from-scratch
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_video_vae_encoder

# ─────────────────────────────────────────────────────────────────────────────
# Neutral scene descriptions (content-only, NO style words)
# These are copied from generate_grok_training_data.py to keep consistency
# ─────────────────────────────────────────────────────────────────────────────
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

TARGET_WIDTH = 832
TARGET_HEIGHT = 448


def load_and_prepare_image(
    image_path: Path,
    target_h: int = TARGET_HEIGHT,
    target_w: int = TARGET_WIDTH,
) -> torch.Tensor:
    """Load image, center-crop to target aspect ratio, resize, prepare for VAE.

    Returns tensor [1, C, 1, H, W] normalized to [-1, 1].
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


def extract_scene_name(filename: str) -> str:
    """Extract scene name from various file naming conventions.

    Examples:
        alien_nostromo_grok_v0.jpg → alien_nostromo
        alien_nostromo_existing_diorama.jpg → alien_nostromo
        alien_nostromo_existing_diorama_2.jpg → alien_nostromo
        alien_nostromo_existing_diorama_flip.jpg → alien_nostromo
        alien_nostromo_grok_v0_flip.jpg → alien_nostromo
    """
    stem = Path(filename).stem

    # Remove flip suffix first
    if stem.endswith("_flip"):
        stem = stem[: -len("_flip")]

    # Try to match known suffixes
    for suffix in [
        "_grok_v0", "_grok_v1", "_grok_v2", "_grok_v3",
        "_existing_diorama_2", "_existing_diorama",
    ]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    # Fallback: check if stem matches a known scene
    if stem in SCENE_DESCRIPTIONS:
        return stem

    # Last resort: try progressively shorter prefixes
    parts = stem.split("_")
    for i in range(len(parts), 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in SCENE_DESCRIPTIONS:
            return candidate

    return stem  # Unknown scene


def collect_diorama_images(
    grok_data_dir: Path,
    movie_dir: Path,
) -> dict[str, list[Path]]:
    """Collect all diorama images grouped by scene name.

    Sources:
        1. Original dioramas from movie_dir/*/diorama*.png
        2. Grok-generated images from grok_data_dir/images/*_grok_v*.jpg
        3. Existing diorama copies from grok_data_dir/images/*_existing_diorama*.jpg
        4. Flipped variants from grok_data_dir/images_flipped/*_flip.*

    Returns dict[scene_name → list of image paths].
    """
    scene_images: dict[str, list[Path]] = {}

    def add_image(scene: str, path: Path) -> None:
        if scene not in scene_images:
            scene_images[scene] = []
        if path not in scene_images[scene]:
            scene_images[scene].append(path)

    # 1. Original dioramas from movie_dir
    if movie_dir.exists():
        for scene_dir in sorted(movie_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene = scene_dir.name
            for f in scene_dir.glob("diorama*"):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    add_image(scene, f)

    # 2. Images from grok_data_dir (generated + existing copies)
    images_dir = grok_data_dir / "images"
    if images_dir.exists():
        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            scene = extract_scene_name(f.name)
            add_image(scene, f)

    # 3. Flipped variants
    flipped_dir = grok_data_dir / "images_flipped"
    if flipped_dir.exists():
        for f in sorted(flipped_dir.iterdir()):
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            scene = extract_scene_name(f.name)
            add_image(scene, f)

    return scene_images


def encode_all_images(
    scene_images: dict[str, list[Path]],
    cache_dir: Path,
    model_path: Path,
    device: str = "cuda:1",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, dict[Path, Path]]:
    """Encode all diorama images to VAE latents, with caching.

    Uses cuda:1 (RTX PRO 4000, 24GB) for the VAE encoder to keep
    cuda:0 (RTX 5090, 32GB) free for later training.

    Returns dict[scene → {image_path: latent_path}].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Count total images
    total = sum(len(imgs) for imgs in scene_images.values())

    # Check cache to see how many need encoding
    needs_encoding = []
    scene_latents: dict[str, dict[Path, Path]] = {}

    for scene, imgs in scene_images.items():
        scene_latents[scene] = {}
        for img_path in imgs:
            # Use hash of path for unique cache key
            cache_key = img_path.stem.replace(" ", "_")
            latent_path = cache_dir / f"{scene}__{cache_key}.pt"
            scene_latents[scene][img_path] = latent_path
            if not latent_path.exists():
                needs_encoding.append((img_path, latent_path))

    if not needs_encoding:
        logger.info(f"All {total} images already cached in {cache_dir}")
        return scene_latents

    logger.info(f"Encoding {len(needs_encoding)}/{total} images (rest cached)")

    # Load VAE encoder
    logger.info(f"Loading VAE encoder on {device}...")
    vae_encoder = load_video_vae_encoder(model_path, dtype=dtype)
    vae_encoder = vae_encoder.to(device)
    vae_encoder.eval()
    logger.info("VAE encoder loaded")

    # Encode
    for img_path, latent_path in tqdm(needs_encoding, desc="Encoding images"):
        try:
            img_tensor = load_and_prepare_image(img_path)
            img_tensor = img_tensor.to(device, dtype=dtype)

            with torch.inference_mode():
                latent = vae_encoder(img_tensor)

            # Save: [C, 1, H_lat, W_lat]
            data = {
                "latents": latent.squeeze(0).cpu(),  # [128, 1, 14, 26]
                "num_frames": torch.tensor([1]),
                "height": torch.tensor([latent.shape[3]]),
                "width": torch.tensor([latent.shape[4]]),
            }
            torch.save(data, latent_path)

        except Exception as e:
            logger.error(f"Failed to encode {img_path}: {e}")
            # Create entry anyway so we can skip it in pairing
            scene_latents[extract_scene_name(img_path.name)].pop(img_path, None)
            continue

    # Cleanup VAE
    del vae_encoder
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"Encoding complete. Cached in {cache_dir}")

    return scene_latents


def create_cross_pairs(
    scene_latents: dict[str, dict[Path, Path]],
    refs_per_target: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Create cross-paired (ref, target) training samples.

    For each target image from scene_j, select `refs_per_target` reference
    images from random OTHER scenes. This teaches the model to extract
    "isometric diorama" style from any reference and apply it to new content.

    Returns list of dicts with:
        - ref_scene, ref_image, ref_latent
        - tgt_scene, tgt_image, tgt_latent
        - caption (neutral content description of target scene)
    """
    rng = random.Random(seed)

    # Only use scenes that have images and known descriptions
    valid_scenes = [
        s for s in scene_latents
        if scene_latents[s] and s in SCENE_DESCRIPTIONS
    ]
    logger.info(f"Creating cross-pairs from {len(valid_scenes)} valid scenes")

    pairs = []
    for tgt_scene in valid_scenes:
        # Pool of reference scenes (everything except current)
        ref_scenes = [s for s in valid_scenes if s != tgt_scene]
        if not ref_scenes:
            continue

        for tgt_img, tgt_latent in scene_latents[tgt_scene].items():
            if not tgt_latent.exists():
                continue

            # Pick random ref scenes (with replacement if needed)
            chosen_refs = rng.choices(ref_scenes, k=min(refs_per_target, len(ref_scenes)))

            for ref_scene in chosen_refs:
                # Pick a random image from the ref scene
                ref_items = [
                    (img, lat) for img, lat in scene_latents[ref_scene].items()
                    if lat.exists()
                ]
                if not ref_items:
                    continue
                ref_img, ref_latent = rng.choice(ref_items)

                pairs.append({
                    "ref_scene": ref_scene,
                    "ref_image": str(ref_img),
                    "ref_latent": str(ref_latent),
                    "tgt_scene": tgt_scene,
                    "tgt_image": str(tgt_img),
                    "tgt_latent": str(tgt_latent),
                    "caption": SCENE_DESCRIPTIONS[tgt_scene],
                })

    rng.shuffle(pairs)
    logger.info(f"Created {len(pairs)} cross-paired training samples")
    return pairs


def write_training_dataset(
    pairs: list[dict],
    output_dir: Path,
) -> None:
    """Write the final dataset in PrecomputedDataset format.

    Creates:
        output_dir/latents/N.pt          - target latent (ground truth)
        output_dir/reference_latents/N.pt - reference latent (style source)
        output_dir/metadata.json          - pairs + captions for embedding computation
    """
    latents_dir = output_dir / "latents"
    ref_dir = output_dir / "reference_latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    metadata_pairs = []
    written = 0

    for i, pair in enumerate(tqdm(pairs, desc="Writing dataset")):
        try:
            # Copy latent files with sequential naming
            tgt_data = torch.load(pair["tgt_latent"], map_location="cpu", weights_only=True)
            ref_data = torch.load(pair["ref_latent"], map_location="cpu", weights_only=True)

            torch.save(tgt_data, latents_dir / f"{i}.pt")
            torch.save(ref_data, ref_dir / f"{i}.pt")

            metadata_pairs.append({
                "id": i,
                "ref_scene": pair["ref_scene"],
                "tgt_scene": pair["tgt_scene"],
                "caption": pair["caption"],
                "ref_image": pair["ref_image"],
                "tgt_image": pair["tgt_image"],
            })
            written += 1

        except Exception as e:
            logger.error(f"Failed to write pair {i}: {e}")
            continue

    # Write metadata
    metadata = {
        "task_type": "style_transfer",
        "num_pairs": written,
        "pairs": metadata_pairs,
        "description": (
            "Cross-paired diorama style transfer dataset. "
            "Reference provides isometric diorama style, target is ground truth. "
            "Captions are neutral content descriptions (no style words)."
        ),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Wrote {written} pairs to {output_dir}")
    logger.info(f"  latents/: {written} files")
    logger.info(f"  reference_latents/: {written} files")
    logger.info(f"  metadata.json: {written} pairs with captions")
    logger.info("")
    logger.info("Next: compute text embeddings (separate process, ~28GB):")
    logger.info(f"  python scripts/compute_final_embeddings.py \\")
    logger.info(f"    --dataset-dir {output_dir} \\")
    logger.info(f"    --model-path /media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors \\")
    logger.info(f"    --text-encoder-path /media/2TB/ltx-models/gemma \\")
    logger.info(f"    --from-scratch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode Grok diorama images into cross-paired OmniTransfer dataset"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Grok training data directory (with images/, images_flipped/)",
    )
    parser.add_argument(
        "--movie-dir", type=Path, default=Path("/media/2TB/movie_dioramas"),
        help="Original movie dioramas directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output dataset directory",
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"),
        help="LTX-2 model checkpoint for VAE encoder",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Latent cache directory (default: output-dir/latent_cache)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1",
        help="Device for VAE encoder (default: cuda:1 = RTX PRO 4000)",
    )
    parser.add_argument(
        "--refs-per-target", type=int, default=2,
        help="Number of reference images per target (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for cross-pair generation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just show statistics, don't encode or write",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or (args.output_dir / "latent_cache")

    # Step 1: Collect all diorama images
    logger.info("=" * 60)
    logger.info("Step 1: Collecting diorama images")
    logger.info("=" * 60)

    scene_images = collect_diorama_images(args.input_dir, args.movie_dir)

    total_images = sum(len(imgs) for imgs in scene_images.values())
    logger.info(f"Found {total_images} images across {len(scene_images)} scenes")

    # Show per-scene counts
    for scene in sorted(scene_images.keys()):
        count = len(scene_images[scene])
        desc = SCENE_DESCRIPTIONS.get(scene, "???")
        logger.info(f"  {scene:40s} {count:3d} images  ({desc[:50]})")

    if args.dry_run:
        # Estimate pairs
        valid = sum(1 for s in scene_images if s in SCENE_DESCRIPTIONS and scene_images[s])
        est_pairs = total_images * args.refs_per_target
        logger.info(f"\nDry run: {valid} valid scenes, ~{est_pairs} pairs estimated")
        return

    # Step 2: Encode all images to VAE latents
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Encoding images to VAE latents")
    logger.info("=" * 60)

    scene_latents = encode_all_images(
        scene_images,
        cache_dir=cache_dir,
        model_path=args.model_path,
        device=args.device,
    )

    # Step 3: Create cross-paired dataset
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: Creating cross-paired (ref, target) samples")
    logger.info("=" * 60)

    pairs = create_cross_pairs(
        scene_latents,
        refs_per_target=args.refs_per_target,
        seed=args.seed,
    )

    # Step 4: Write dataset
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: Writing training dataset")
    logger.info("=" * 60)

    write_training_dataset(pairs, args.output_dir)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Samples: {len(pairs)}")
    logger.info(f"Latent cache: {cache_dir}")


if __name__ == "__main__":
    main()
