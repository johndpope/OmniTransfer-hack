#!/usr/bin/env python3
"""Download Movie Weaver demo images and videos.

Downloads all reference images and result videos from the Movie Weaver
supplementary website for training data.

Movie Weaver (CVPR 2025): "Tuning-Free Multi-Concept Video Personalization
with Anchored Prompts"
https://jeff-liangf.github.io/projects/movieweaver/
"""

import os
import subprocess
from pathlib import Path

# Base URL for Movie Weaver supplementary materials
BASE_URL = "https://jeff-liangf.github.io/projects/movieweaver/supp/website"

# Output directory
OUTPUT_DIR = Path("/media/2TB/movie_weaver_demos")

# All prompts with reference structure
# Format: {
#   "name": identifier,
#   "prompt": full prompt text,
#   "refs": {"R1": "description", "R2": "description", ...},
#   "images": [list of image paths],
#   "videos": [list of video paths],
# }
MOVIE_WEAVER_DEMOS = [
    # =========================================================================
    # Figure 1 demos - Single person configurations
    # =========================================================================
    {
        "name": "fig1_face_only",
        "prompt": "The video shows a young man [R1] holding a dog in his arms. The man is standing on a sidewalk in front of a row of buildings. He is wearing a gray hoodie and has short brown hair. The dog is a small, fluffy, white and brown dog. The man is holding the dog close to his chest and looking at the camera. The background shows a row of buildings with large windows and a street with parked cars. The lighting suggests it is daytime.",
        "refs": {"R1": "man face"},
        "concept_assignments": [0],
        "images": ["videos/fig1/face_1.png", "videos/fig1/face_2.png"],
        "videos": ["videos/fig1/face_1.mp4", "videos/fig1/face_2.mp4"],
    },
    {
        "name": "fig1_face_body",
        "prompt": "The video shows a young man [R1] [R2] holding a dog in his arms. The man is standing on a sidewalk in front of a row of buildings. He is wearing a gray hoodie and has short brown hair. The dog is a small, fluffy, white and brown dog. The man is holding the dog close to his chest and looking at the camera.",
        "refs": {"R1": "man face", "R2": "man body"},
        "concept_assignments": [0, 0],  # Same person
        "images": ["videos/fig1/facebody_1.png", "videos/fig1/facebody_2.png"],
        "videos": ["videos/fig1/facebody_1.mp4", "videos/fig1/facebody_2.mp4"],
    },
    {
        "name": "fig1_face_body_animal",
        "prompt": "The video shows a young man [R1] [R2] holding a dog [R3] in his arms. The man is standing on a sidewalk in front of a row of buildings. He is wearing a gray hoodie and has short brown hair. The dog is a small, fluffy, white and brown dog.",
        "refs": {"R1": "man face", "R2": "man body", "R3": "dog"},
        "concept_assignments": [0, 0, 1],  # Person 0, Dog 1
        "images": ["videos/fig1/facebodyanimal_1.png", "videos/fig1/facebodyanimal_2.png"],
        "videos": ["videos/fig1/facebodyanimal_1.mp4", "videos/fig1/facebodyanimal_2.mp4"],
    },
    {
        "name": "fig1_two_face",
        "prompt": "A man [R1] and a woman [R2] are working in a data center with rows of server racks. They are both wearing professional attire and looking at a laptop screen together.",
        "refs": {"R1": "man face", "R2": "woman face"},
        "concept_assignments": [0, 1],  # Two different people
        "images": ["videos/fig1/twoface_1.png", "videos/fig1/twoface_2.png"],
        "videos": ["videos/fig1/twoface_1.mp4", "videos/fig1/twoface_2.mp4"],
    },
    {
        "name": "fig1_two_face_body",
        "prompt": "A man [R1] [R2] and a woman [R3] [R4] are working in a data center with rows of server racks. They are both wearing professional attire and looking at a laptop screen together.",
        "refs": {"R1": "man face", "R2": "man body", "R3": "woman face", "R4": "woman body"},
        "concept_assignments": [0, 0, 1, 1],  # Man=0, Woman=1
        "images": ["videos/fig1/twofacebody_1.png", "videos/fig1/twofacebody_2.png"],
        "videos": ["videos/fig1/twofacebody_1.mp4", "videos/fig1/twofacebody_2.mp4"],
    },

    # =========================================================================
    # Figure 5 demos - More two-person scenarios
    # =========================================================================
    {
        "name": "fig5_two_face_hardhat",
        "prompt": "A man [R1] and a man [R2] Hard Hat Walking, Talking, and Using Tablet in a warehouse. They are both wearing hard hats and high-visibility vests.",
        "refs": {"R1": "man1 face", "R2": "man2 face"},
        "concept_assignments": [0, 1],
        "images": [f"videos/fig5/twoface_{i}.png" for i in range(4)],
        "videos": [f"videos/fig5/twoface_{i}.mp4" for i in range(4)],
    },
    {
        "name": "fig5_face_body_animal_christmas",
        "prompt": "The video shows a man [R1] [R2] and a dog [R3] sitting at a table with a Christmas tree in the background. The man is wearing a red sweater and the dog is wearing a festive collar.",
        "refs": {"R1": "man face", "R2": "man body", "R3": "dog"},
        "concept_assignments": [0, 0, 1],
        "images": [f"videos/fig5/facebodyanimal_{i}.png" for i in range(4)],
        "videos": [f"videos/fig5/facebodyanimal_{i}.mp4" for i in range(4)],
    },
    {
        "name": "fig5_two_face_body_beach",
        "prompt": "a woman [R1] [R2] and a man [R3] [R4] eating salad after fitness workout on beach. Multiracial a woman and a man having a break on beach snacking on a vegan takeaway meal of green veggies laughing together. The video shows a woman and a man sitting on the beach, eating salads. They are both sitting on the sand, with their legs crossed and their feet pointed towards the camera. They are both looking at each other and smiling. The woman is holding a salad in her hand on the left, and the man is holding a salad in his hand on the right. The background is a beach with palm trees and a body of water. The sky is overcast. The camera is static.",
        "refs": {"R1": "woman face", "R2": "woman body", "R3": "man face", "R4": "man body"},
        "concept_assignments": [0, 0, 1, 1],
        "images": [f"videos/fig5/twofacebody_{i}.png" for i in range(4)],
        "videos": [f"videos/fig5/twofacebody_{i}.mp4" for i in range(4)],
    },

    # =========================================================================
    # Figure 6 demos - Comparison examples
    # =========================================================================
    {
        "name": "fig6_woman_dog_park",
        "prompt": "The video shows a woman [R1] [R2] sitting on a bench in a park, petting a dog [R3]. The woman is wearing a casual outfit and the dog is a golden retriever. The park has green trees in the background.",
        "refs": {"R1": "woman face", "R2": "woman body", "R3": "dog"},
        "concept_assignments": [0, 0, 1],
        "images": ["videos/fig6/movie_weaver_0.png"],
        "videos": ["videos/fig6/movie_weaver_0.mp4"],
    },
    {
        "name": "fig6_two_men_working",
        "prompt": "A man [R1] [R2] and a man [R3] [R4] working and taking notes together in table. They are both wearing business casual attire and appear to be collaborating on a project.",
        "refs": {"R1": "man1 face", "R2": "man1 body", "R3": "man2 face", "R4": "man2 body"},
        "concept_assignments": [0, 0, 1, 1],
        "images": ["videos/fig6/movie_weaver_1.png"],
        "videos": ["videos/fig6/movie_weaver_1.mp4"],
    },

    # =========================================================================
    # Figure 8 demos - Limitations / challenging cases
    # =========================================================================
    {
        "name": "fig8_basketball",
        "prompt": "A man [R1][R2] and another man [R3][R4] are playing basketball. They are on an outdoor basketball court with a hoop visible in the background.",
        "refs": {"R1": "man1 face", "R2": "man1 body", "R3": "man2 face", "R4": "man2 body"},
        "concept_assignments": [0, 0, 1, 1],
        "images": ["videos/fig8/limitation_1.png"],
        "videos": ["videos/fig8/limitation_1.mp4"],
    },
    {
        "name": "fig8_three_men_talking",
        "prompt": "A man [R1] in white T-shirt, a man [R2] in black leather jacket and a man [R3] in yellow shirt are talking. They appear to be having a casual conversation outdoors.",
        "refs": {"R1": "man1 face", "R2": "man2 face", "R3": "man3 face"},
        "concept_assignments": [0, 1, 2],  # Three different people
        "images": ["videos/fig8/limitation_2.png"],
        "videos": ["videos/fig8/limitation_2.mp4"],
    },

    # =========================================================================
    # Supplementary Figure 1 - Additional examples
    # =========================================================================
    {
        "name": "sup_fig1_motorcycle",
        "prompt": "A man wearing a black leather jacket and sunglasses [R1] is sitting on a motorcycle next to a man in a yellow T-shirt [R2]. They appear to be having a conversation on a sunny day.",
        "refs": {"R1": "man1 face", "R2": "man2 face"},
        "concept_assignments": [0, 1],
        "images": [f"videos/sup_fig1/set_{i}.png" for i in range(1, 5)],
        "videos": [f"videos/sup_fig1/set_{i}.mp4" for i in range(1, 5)],
    },

    # =========================================================================
    # Additional prompts from main page
    # =========================================================================
    {
        "name": "jogging_park",
        "prompt": "Young a man [R1] [R2] and a woman [R3] [R4] jogging in the park in slow motion. The video shows a man and woman jogging on a paved path. They are both jogging on a paved path that is surrounded by greenery. The man is holding a black whistle around his neck. The woman is holding her hands to her chest. The man is holding his hands to his sides. They are both jogging towards the camera. The camera is handheld.",
        "refs": {"R1": "man face", "R2": "man body", "R3": "woman face", "R4": "woman body"},
        "concept_assignments": [0, 0, 1, 1],
        "images": [],  # Need to find these
        "videos": [],
    },
]


def download_file(url: str, output_path: Path) -> bool:
    """Download a file using wget."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  [SKIP] Already exists: {output_path.name}")
        return True

    try:
        result = subprocess.run(
            ["wget", "-q", "-O", str(output_path), url],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            print(f"  [OK] Downloaded: {output_path.name}")
            return True
        else:
            print(f"  [FAIL] Failed: {url}")
            if output_path.exists():
                output_path.unlink()
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def download_demo(demo: dict, output_dir: Path) -> dict:
    """Download all files for a demo."""
    demo_dir = output_dir / demo["name"]
    demo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {demo['name']}")
    print(f"Prompt: {demo['prompt'][:80]}...")
    print(f"Refs: {demo['refs']}")
    print(f"Concept assignments: {demo['concept_assignments']}")

    # Save prompt
    prompt_file = demo_dir / "prompt.txt"
    prompt_file.write_text(demo["prompt"])

    # Save metadata
    import json
    meta_file = demo_dir / "metadata.json"
    meta_file.write_text(json.dumps({
        "name": demo["name"],
        "prompt": demo["prompt"],
        "refs": demo["refs"],
        "concept_assignments": demo["concept_assignments"],
    }, indent=2))

    # Download images
    downloaded_images = []
    for img_path in demo.get("images", []):
        url = f"{BASE_URL}/{img_path}"
        filename = img_path.split("/")[-1]
        output_path = demo_dir / "images" / filename
        if download_file(url, output_path):
            downloaded_images.append(str(output_path))

    # Download videos
    downloaded_videos = []
    for vid_path in demo.get("videos", []):
        url = f"{BASE_URL}/{vid_path}"
        filename = vid_path.split("/")[-1]
        output_path = demo_dir / "videos" / filename
        if download_file(url, output_path):
            downloaded_videos.append(str(output_path))

    return {
        "name": demo["name"],
        "dir": str(demo_dir),
        "images": downloaded_images,
        "videos": downloaded_videos,
    }


def main():
    print("=" * 60)
    print("Movie Weaver Demo Downloader")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for demo in MOVIE_WEAVER_DEMOS:
        result = download_demo(demo, OUTPUT_DIR)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    total_images = sum(len(r["images"]) for r in results)
    total_videos = sum(len(r["videos"]) for r in results)

    print(f"Total demos: {len(results)}")
    print(f"Total images: {total_images}")
    print(f"Total videos: {total_videos}")

    # Save master manifest
    import json
    manifest_file = OUTPUT_DIR / "manifest.json"
    manifest_file.write_text(json.dumps({
        "demos": MOVIE_WEAVER_DEMOS,
        "downloaded": results,
    }, indent=2))
    print(f"\nManifest saved to: {manifest_file}")


if __name__ == "__main__":
    main()
