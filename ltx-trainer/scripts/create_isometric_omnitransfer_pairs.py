#!/usr/bin/env python3
"""Create cross-video OmniTransfer pairs from isometric I2V clips.

For OmniTransfer identity preservation, reference and target MUST come from
different source videos. This script reads the isometric I2V dataset (128 clips
from 8 videos) and creates cross-video pairings via symlinks.

Each clip gets paired with a clip from a DIFFERENT source video, so the model
must learn to transfer visual style/identity across different isometric scenes.

Usage:
    python scripts/create_isometric_omnitransfer_pairs.py

Output:
    /media/2TB/omnitransfer/data/isometric_omnitransfer/
    ├── latents/           # Target clips (symlinks to i2v latents)
    ├── reference_latents/ # Reference clips from different videos
    ├── conditions_final/  # Text embeddings (symlinks to i2v conditions)
    └── metadata.json      # Pairing information
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# Paths
SOURCE_DIR = Path("/media/2TB/omnitransfer/data/isometric_i2v")
OUTPUT_DIR = Path("/media/2TB/omnitransfer/data/isometric_omnitransfer")

# Seed for reproducibility
random.seed(42)


def main() -> None:
    # Load source metadata
    metadata = json.load(open(SOURCE_DIR / "metadata.json"))
    pairs = metadata["pairs"]

    # Group clips by source video
    video_clips: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        video_clips[pair["video_id"]].append(pair)

    video_ids = sorted(video_clips.keys())
    print(f"Found {len(pairs)} clips from {len(video_ids)} source videos")
    for vid in video_ids:
        print(f"  {vid[:8]}...: {len(video_clips[vid])} clips")

    # Create cross-video pairings
    # For each clip, pair it with a random clip from a different video
    output_pairs: list[dict] = []
    pair_id = 0

    for tgt_clip in pairs:
        tgt_vid = tgt_clip["video_id"]
        tgt_id = tgt_clip["id"]

        # Pick a random clip from a different source video
        other_vids = [v for v in video_ids if v != tgt_vid]
        ref_vid = random.choice(other_vids)
        ref_clip = random.choice(video_clips[ref_vid])

        output_pairs.append({
            "id": pair_id,
            "tgt_clip_id": tgt_id,
            "tgt_video_id": tgt_vid,
            "tgt_caption": tgt_clip.get("caption", ""),
            "ref_clip_id": ref_clip["id"],
            "ref_video_id": ref_vid,
            "ref_caption": ref_clip.get("caption", ""),
        })
        pair_id += 1

    # Verify all pairs are cross-video
    for p in output_pairs:
        assert p["tgt_video_id"] != p["ref_video_id"], (
            f"Pair {p['id']}: same video! tgt={p['tgt_video_id']}, ref={p['ref_video_id']}"
        )

    print(f"\nCreated {len(output_pairs)} cross-video pairs")

    # Create output directories
    for subdir in ["latents", "reference_latents", "conditions_final"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Detect filename format: check if 000.pt exists (zero-padded) or 0.pt (plain)
    if (SOURCE_DIR / "latents" / "000.pt").exists():
        fmt = lambda x: f"{x:03d}"  # noqa: E731
        print("Detected zero-padded filenames (000.pt, 001.pt, ...)")
    else:
        fmt = str
        print("Detected plain filenames (0.pt, 1.pt, ...)")

    # Create symlinks
    created = 0
    for p in output_pairs:
        pid = p["id"]

        # Target latent (what model should generate)
        tgt_src = SOURCE_DIR / "latents" / f"{fmt(p['tgt_clip_id'])}.pt"
        tgt_dst = OUTPUT_DIR / "latents" / f"{fmt(pid)}.pt"

        # Reference latent (from different video - identity/style source)
        # Use FIRST FRAME only (864 tokens) to fit in 32GB VRAM
        # Full clips (3456 tokens each) cause OOM: ref(3456)+tgt(3456)=6912 tokens
        ref_src = SOURCE_DIR / "reference_latents" / f"{fmt(p['ref_clip_id'])}.pt"
        ref_dst = OUTPUT_DIR / "reference_latents" / f"{fmt(pid)}.pt"

        # Text embedding (use target's caption)
        cond_src = SOURCE_DIR / "conditions_final" / f"{fmt(p['tgt_clip_id'])}.pt"
        cond_dst = OUTPUT_DIR / "conditions_final" / f"{fmt(pid)}.pt"

        # Create symlinks (remove existing)
        for src, dst in [(tgt_src, tgt_dst), (ref_src, ref_dst), (cond_src, cond_dst)]:
            if not src.exists():
                print(f"WARNING: Missing source file: {src}")
                continue
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            created += 1

    print(f"Created {created} symlinks in {OUTPUT_DIR}")

    # Save metadata
    out_meta = {
        "task_type": "identity_preservation",
        "description": (
            "Cross-video OmniTransfer pairs from isometric I2V clips. "
            "Reference and target always come from different source videos."
        ),
        "source": "isometric_i2v",
        "num_samples": len(output_pairs),
        "num_source_videos": len(video_ids),
        "clip_frames": metadata.get("clip_frames", 25),
        "resolution": metadata.get("resolution", "768x1152"),
        "pairs": output_pairs,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(out_meta, f, indent=2)

    print(f"Saved metadata to {OUTPUT_DIR / 'metadata.json'}")

    # Print sample pairings
    print("\nSample pairings (first 5):")
    for p in output_pairs[:5]:
        print(
            f"  #{p['id']}: tgt={p['tgt_video_id'][:8]}..clip{p['tgt_clip_id']} "
            f"<- ref={p['ref_video_id'][:8]}..clip{p['ref_clip_id']}"
        )


if __name__ == "__main__":
    main()
