#!/usr/bin/env python3
"""Classify the main-person gender of each scene clip with Qwen2.5-VL.

Writes {scene_name: "man"|"woman"|"both"|"none"} to scene_gender.json so the
dataset builder can pair references with gender-matched targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

CATEGORIES = ["one_woman", "one_man", "two_women", "two_men", "man_and_woman",
              "group", "none"]
Q = ("Categorize the people in this image. Reply with EXACTLY one of these labels "
     "and nothing else: 'one_woman' (a single woman), 'one_man' (a single man), "
     "'two_women' (two women), 'two_men' (two men), 'man_and_woman' (one man and "
     "one woman), 'group' (three or more people), 'none' (no people visible). "
     "Choose based on the clearly visible foreground people.")


def mid_frame(clip: Path, size: int = 448) -> Image.Image:
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return Image.new("RGB", (size, size))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.resize(rgb, (size, size)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model-path", type=str, default="/media/2TB/ltx-models/qwen2.5-vl-7b")
    args = ap.parse_args()

    clips = sorted((args.data_root / "clips").glob("*.mp4"))
    proc = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print(f"Classifying {len(clips)} scenes...")

    out: dict[str, str] = {}
    for i, clip in enumerate(clips):
        img = mid_frame(clip)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        ans = proc.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip().lower()
        ans_norm = ans.replace(" ", "_").replace("-", "_")
        label = next((c for c in CATEGORIES if c in ans_norm), "none")
        out[clip.stem] = label
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(clips)}")

    (args.data_root / "scene_gender.json").write_text(json.dumps(out, indent=2))
    from collections import Counter
    print("Distribution:", dict(Counter(out.values())))
    print(f"Wrote {args.data_root / 'scene_gender.json'}")


if __name__ == "__main__":
    main()
