#!/usr/bin/env python3
"""Generate a video with the LTX-2.3 IC-LoRA-Ingredients LoRA from reference image(s).

Thin, reproducible wrapper around castlehill's `ltx_pipelines.distilled`. Bakes in
the config that actually works on this machine:
  - gemma-root = the FULL multi-shard gemma (the fp4 gemma is text-only -> its
    vision tower can't init -> meta-tensor crash).
  - fp8-cast quantization, Ingredients LoRA, per-image frame-0 conditioning.

Usage:
    python tools/ingredients_generate.py \
        --image /path/to/portrait.png \
        --prompt "A man walks down a rainy neon street at night, cinematic" \
        --output /media/2TB/omnitransfer/inference/out.mp4
    # up to 4 --image for multi-ingredient composites
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CASTLEHILL = Path(os.getenv("LTX2_PATH", "/home/johndpope/Documents/GitHub/ltx2-castlehill"))
VENV_PY = CASTLEHILL / ".venv/bin/python"
MODELS = Path("/media/2TB/ltx-models")
DISTILLED = MODELS / "ltx2.3/ltx-2.3-22b-distilled.safetensors"
UPSCALER = MODELS / "ltx2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT = MODELS / "gemma"  # full multi-shard model (has the vision tower)
INGREDIENTS_LORA = MODELS / "LTX-2.3-22b-IC-LoRA-Ingredients/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True,
                    help="Reference/ingredient image (repeat for up to 4).")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--num-frames", type=int, default=97)  # % 8 == 1
    ap.add_argument("--frame-rate", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--image-strength", type=float, default=1.0)
    ap.add_argument("--quantization", default="fp8-cast")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PY), "-m", "ltx_pipelines.distilled",
        "--distilled-checkpoint-path", str(DISTILLED),
        "--spatial-upsampler-path", str(UPSCALER),
        "--gemma-root", str(GEMMA_ROOT),
        "--prompt", args.prompt,
        "--width", str(args.width), "--height", str(args.height),
        "--num-frames", str(args.num_frames), "--frame-rate", str(args.frame_rate),
        "--seed", str(args.seed),
        "--lora", str(INGREDIENTS_LORA), str(args.lora_strength),
    ]
    for img in args.image:
        cmd += ["--image", str(img), "0", str(args.image_strength)]
    cmd += ["--output-path", args.output, "--quantization", args.quantization]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CASTLEHILL}/packages/ltx-pipelines/src:{CASTLEHILL}/packages/ltx-core/src"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    print("Generating ->", args.output)
    r = subprocess.run(cmd, cwd=str(CASTLEHILL), env=env)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
