#!/usr/bin/env python3
"""Extract clean front-facing character portraits from the reel scenes.

For each single-person scene, find the frame with the largest, most-centered
frontal face and crop a portrait (face + upper body / outfit). These become the
one-image references for the LTX-2.3 IC-LoRA-Ingredients pipeline.

Usage:
    python tools/extract_portraits.py \
        --clips-dir /media/2TB/omnitransfer/data/mashup_v2/clips \
        --gender-json /media/2TB/omnitransfer/data/mashup_v2/scene_gender.json \
        --out-dir /media/2TB/omnitransfer/data/mashup_v2/portraits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

SINGLE = {"one_woman", "one_man"}


def best_portrait(clip: Path, casc: cv2.CascadeClassifier, samples: int = 10):
    """Return (score, frame_idx, bbox, frame_bgr) of the best frontal face, or None."""
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    best = None
    for fi in range(0, n, max(1, n // samples)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in casc.detectMultiScale(gray, 1.1, 6, minSize=(80, 80)):
            cx = x + w / 2
            centered = 1 - abs(cx - bgr.shape[1] / 2) / (bgr.shape[1] / 2)
            score = w * h * max(centered, 0.1)
            if best is None or score > best[0]:
                best = (score, fi, (x, y, w, h), bgr.copy())
    cap.release()
    return best


def crop_portrait(bgr, box):
    x, y, w, h = box
    H, W = bgr.shape[:2]
    x0 = max(0, int(x - 0.6 * w)); x1 = min(W, int(x + 1.6 * w))
    y0 = max(0, int(y - 0.5 * h)); y1 = min(H, int(y + 3.2 * h))
    return bgr[y0:y1, x0:x1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--gender-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-face", type=int, default=90, help="min face box side (px)")
    args = ap.parse_args()

    gender = json.load(open(args.gender_json))
    casc = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [n for n, g in gender.items() if g in SINGLE]
    made = 0
    manifest = {}
    for name in sorted(scenes):
        clip = args.clips_dir / f"{name}.mp4"
        if not clip.exists():
            continue
        best = best_portrait(clip, casc)
        if best is None:
            continue
        score, fi, box, bgr = best
        if min(box[2], box[3]) < args.min_face:
            continue
        crop = crop_portrait(bgr, box)
        out = args.out_dir / f"{name}.png"
        cv2.imwrite(str(out), crop)
        manifest[name] = {"gender": gender[name], "frame": fi,
                          "face_px": int(min(box[2], box[3]))}
        made += 1

    (args.out_dir / "portraits.json").write_text(json.dumps(manifest, indent=2))
    from collections import Counter
    print(f"Extracted {made} portraits -> {args.out_dir}")
    print("by gender:", dict(Counter(m["gender"] for m in manifest.values())))


if __name__ == "__main__":
    main()
