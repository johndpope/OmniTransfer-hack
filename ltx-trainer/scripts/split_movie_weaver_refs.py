#!/usr/bin/env python3
"""Split Movie Weaver composite reference images into individual R1, R2, R3, R4 images.

Each composite image contains multiple references laid out horizontally.
This script splits them and saves individual reference images for training.
"""

import json
from pathlib import Path
from PIL import Image


def split_composite_image(
    image_path: Path,
    num_refs: int,
    ref_names: list[str],
    output_dir: Path,
) -> list[Path]:
    """Split a composite image into individual reference images.

    Args:
        image_path: Path to composite image
        num_refs: Number of references to split into
        ref_names: Names for each reference (e.g., ["R1", "R2"])
        output_dir: Directory to save split images

    Returns:
        List of paths to saved reference images
    """
    img = Image.open(image_path)
    width, height = img.size
    ref_width = width // num_refs

    saved_paths = []
    for i, ref_name in enumerate(ref_names):
        # Crop region for this reference
        left = i * ref_width
        right = (i + 1) * ref_width
        ref_img = img.crop((left, 0, right, height))

        # Save with descriptive name
        base_name = image_path.stem
        output_path = output_dir / f"{base_name}_{ref_name}.png"
        ref_img.save(output_path, "PNG")
        saved_paths.append(output_path)
        print(f"  Saved: {output_path.name} ({ref_width}x{height})")

    return saved_paths


def process_demo_directory(demo_dir: Path) -> dict:
    """Process all images in a Movie Weaver demo directory.

    Returns:
        Dictionary with split image paths and metadata
    """
    images_dir = demo_dir / "images"
    meta_path = demo_dir / "metadata.json"

    if not images_dir.exists() or not meta_path.exists():
        return {"error": "Missing images or metadata"}

    metadata = json.loads(meta_path.read_text())
    refs = metadata.get("refs", {})
    concept_assignments = metadata.get("concept_assignments", [])

    num_refs = len(refs)
    ref_names = list(refs.keys())  # ["R1", "R2", ...]

    print(f"\n{'='*60}")
    print(f"Processing: {demo_dir.name}")
    print(f"Refs: {refs}")
    print(f"Concept assignments: {concept_assignments}")

    # Create split output directory
    split_dir = demo_dir / "split_refs"
    split_dir.mkdir(exist_ok=True)

    results = {
        "demo": demo_dir.name,
        "refs": refs,
        "concept_assignments": concept_assignments,
        "split_images": [],
    }

    # Process each composite image
    for img_path in sorted(images_dir.glob("*.png")):
        print(f"\nSplitting: {img_path.name}")
        split_paths = split_composite_image(
            img_path, num_refs, ref_names, split_dir
        )

        # Build per-image reference mapping
        ref_mapping = {}
        for ref_name, split_path in zip(ref_names, split_paths):
            ref_mapping[ref_name] = {
                "path": str(split_path),
                "description": refs[ref_name],
                "concept": concept_assignments[ref_names.index(ref_name)] if concept_assignments else 0,
            }

        results["split_images"].append({
            "original": str(img_path),
            "references": ref_mapping,
        })

    # Save split metadata
    split_meta_path = split_dir / "split_metadata.json"
    split_meta_path.write_text(json.dumps(results, indent=2))
    print(f"\nSplit metadata saved: {split_meta_path}")

    return results


def main():
    """Split all Movie Weaver demo images."""
    base_dir = Path("/media/2TB/movie_weaver_demos")

    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        return

    print("=" * 60)
    print("Movie Weaver Reference Image Splitter")
    print("=" * 60)

    all_results = []

    # Process each demo directory
    for demo_dir in sorted(base_dir.iterdir()):
        if demo_dir.is_dir() and (demo_dir / "metadata.json").exists():
            result = process_demo_directory(demo_dir)
            all_results.append(result)

    # Save master manifest of split images
    manifest = {
        "base_dir": str(base_dir),
        "demos": all_results,
        "total_demos": len(all_results),
        "total_split_images": sum(
            len(r.get("split_images", [])) * len(r.get("refs", {}))
            for r in all_results
        ),
    }

    manifest_path = base_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total demos processed: {manifest['total_demos']}")
    print(f"Total individual refs: {manifest['total_split_images']}")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
