#!/usr/bin/env python3
"""Rebuild the facebook_reels mashup dataset for paper-aligned OmniTransfer training.

Fixes two paper-alignment issues in the original mashup_training data:
  1. Single-frame latents (F=1) -> re-encode MULTI-FRAME (49 frames, F=7 latent).
     OmniTransfer is a video-reference method; TPB Delta=(f,0,0) and RCL need
     temporal extent (arXiv:2601.14250v1, Sec 4.2/4.3).
  2. Prompts that leaked the target style -> NEUTRAL, content-only prompts.
     Style/identity must come from the reference video, not the text prompt.

This driver does the CPU-side orchestration only. GPU encoding is delegated to the
battle-tested `process_videos.py` (VAE) and `process_captions.py` (LTX-2.3 text
feature extractor -> video_prompt_embeds). Staged so the two big models never load
together.

Stages:
  prepare   : transcode scenes -> 768x1152x49, write scenes.csv + captions.csv +
              metadata_v2.json + neutral prompts + train_config.yaml.
              (then run process_videos.py and process_captions.py — see printed cmds)
  finalize  : symlink indexed latents/, reference_latents/, conditions/ from the
              per-scene latent pool + per-pair condition files.

Usage:
  python tools/rebuild_mashup_v2.py prepare  --scenes-dir facebook_reels/scenes \
      --output-dir /media/2TB/omnitransfer/data/mashup_v2
  # ...run the two printed GPU commands...
  python tools/rebuild_mashup_v2.py finalize --output-dir /media/2TB/omnitransfer/data/mashup_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2

# Reel -> crossover movie (content identity, NOT style — safe to put in prompt).
REEL_MOVIE_MAP: dict[str, str] = {
    "1199971765284042": "Back to the Future x Career Opportunities",
    "1256976252680327": "Labyrinth x Career Opportunities",
    "1689417079061469": "Return of the Living Dead x Career Opportunities",
    "2057624615188633": "Back to the Future x Career Opportunities",
    "2125476178020928": "Pretty Woman x Career Opportunities",
    "2156378868455964": "Ghostbusters x Career Opportunities",
    "3428533713973292": "Halloween x Career Opportunities",
    "770908555771949": "Kingpin x Career Opportunities",
    "797363763429829": "National Lampoon's Vacation x Career Opportunities",
    "820428490726045": "Red Sonja x Career Opportunities",
    "961517603217527": "Weird Science x Career Opportunities",
    "instagram_DVofVseCBwz": "a movie mashup",
    "instagram_DV4T00Tj8x8": "a movie mashup",
    "instagram_DXGh6OaEnBG": "a movie mashup",
    "instagram_DVwVGuUkmHi": "a movie mashup",
}

WIDTH, HEIGHT = 768, 1152          # portrait, matches vertical reels
FRAMES = 25                        # 25 % 8 == 1 -> F=4 latent frames. 49-frame (F=7)
                                   # OOMs a 22B model on 32GB (12096 RCL tokens); 25f
                                   # keeps full res + real temporal extent at ~6912 tokens.
FPS = 24
MODEL_PATH = "/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
TEXT_ENCODER_PATH = "/media/2TB/ltx-models/gemma"


def neutral_prompt(movie: str) -> str:
    """Content-only prompt. NO style words (no 'film grain', 'aesthetic', 'cinematic

    lighting', etc.) — style comes from the reference video per the paper.
    """
    return f"A scene from {movie}."


def scene_name(reel_id: str, scene_idx: int) -> str:
    return f"{reel_id}_Scene-{scene_idx:03d}"


def frame_count(path: Path) -> int:
    v = cv2.VideoCapture(str(path))
    n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    v.release()
    return n


def transcode(src: Path, dst: Path) -> bool:
    """Transcode a scene clip to WIDTHxHEIGHT, FRAMES frames, FPS (audio stripped)."""
    ffmpeg_bin = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
    cmd = [
        ffmpeg_bin, "-y", "-i", str(src),
        "-vf", (
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},fps={FPS}"
        ),
        "-frames:v", str(FRAMES),
        "-c:v", "libx264", "-crf", "18", "-an",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dst.exists()


def prepare(scenes_dir: Path, output_dir: Path, min_frames: int) -> None:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # 1. Enumerate + transcode eligible scenes (>= min_frames source frames).
    scenes: list[dict] = []           # {reel_id, scene, name, clip}
    skipped_short = 0
    for reel_dir in sorted(scenes_dir.iterdir()):
        if not reel_dir.is_dir():
            continue
        reel_id = reel_dir.name
        for scene_file in sorted(reel_dir.glob("*-Scene-*.mp4")):
            n = frame_count(scene_file)
            scene_idx = int(scene_file.stem.split("-")[-1])
            name = scene_name(reel_id, scene_idx)
            if n < min_frames:
                skipped_short += 1
                print(f"  skip (short, {n}f): {name}")
                continue
            dst = clips_dir / f"{name}.mp4"
            if not dst.exists():
                ok = transcode(scene_file, dst)
                print(f"  {'ok' if ok else 'FAIL'}: {name} ({n}f -> {FRAMES}f)")
                if not ok:
                    continue
            scenes.append({"reel_id": reel_id, "scene": scene_idx, "name": name,
                           "clip": str(dst)})
    print(f"\nTranscoded {len(scenes)} scenes ({skipped_short} skipped as too short)")

    # 2. scenes.csv for process_videos.py (VAE) — bare filenames, run from clips_dir.
    scenes_csv = output_dir / "scenes.csv"
    with open(scenes_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["media_path", "caption"])
        for s in scenes:
            w.writerow([f"clips/{s['name']}.mp4", "scene"])  # relative to scenes.csv dir

    # 3. Cross-pair scenes WITHIN each reel (ref != tgt).
    by_reel: dict[str, list[dict]] = {}
    for s in scenes:
        by_reel.setdefault(s["reel_id"], []).append(s)

    pairs: list[dict] = []
    idx = 0
    for reel_id, reel_scenes in by_reel.items():
        if len(reel_scenes) < 2:
            continue
        movie = REEL_MOVIE_MAP.get(reel_id, "a movie mashup")
        for ref in reel_scenes:
            for tgt in reel_scenes:
                if ref["scene"] == tgt["scene"]:
                    continue
                pairs.append({
                    "idx": idx,
                    "id": idx,
                    "file_name": f"{idx:06d}.pt",             # target latent
                    "reference_file_name": f"{idx:06d}.pt",   # reference latent (diff content)
                    "reel_id": reel_id,
                    "movie": movie,
                    "ref_scene": ref["scene"],
                    "tgt_scene": tgt["scene"],
                    "ref_name": ref["name"],
                    "tgt_name": tgt["name"],
                    "text": neutral_prompt(movie),
                    "task_type": "style_transfer",
                })
                idx += 1
    print(f"Built {len(pairs)} cross-pairs from {len(by_reel)} reels")

    with open(output_dir / "metadata_v2.json", "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)
    # process_captions/embeddings expect metadata.json with pairs
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)

    # 4. captions.csv for process_captions.py (one row per pair index).
    captions_csv = output_dir / "captions.csv"
    with open(captions_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["media_path", "caption"])
        for p in pairs:
            w.writerow([f"{p['idx']:06d}.mp4", p["text"]])  # -> conditions/{idx:06d}.pt

    # 5. train_config.yaml (paper-aligned + repo conventions: Muon, checkpoints, W&B).
    write_config(output_dir, len(pairs))

    # 6. Print the GPU commands to run next.
    print("\n" + "=" * 70)
    print("PREPARE DONE. Now run the two GPU stages (staged, models don't co-load):\n")
    print("# A) VAE-encode scenes -> multi-frame latent pool (GPU 0 is fine):")
    print(f"cd {clips_dir} && \\")
    print("  PYTHONPATH=... CUDA_VISIBLE_DEVICES=0 \\")
    print(f"  python {Path('ltx-trainer/scripts/process_videos.py').resolve()} \\")
    print(f"    {scenes_csv} --resolution-buckets {WIDTH}x{HEIGHT}x{FRAMES} \\")
    print(f"    --output-dir {output_dir / 'scene_latents'} --model-path {MODEL_PATH}\n")
    print("# B) Text-encode neutral prompts -> conditions (needs ~28GB — 5090):")
    print(f"  python {Path('ltx-trainer/scripts/process_captions.py').resolve()} \\")
    print(f"    {captions_csv} --output-dir {output_dir / 'conditions'} \\")
    print(f"    --model-path {MODEL_PATH} --text-encoder-path {TEXT_ENCODER_PATH}\n")
    print("Then: python tools/rebuild_mashup_v2.py finalize --output-dir", output_dir)


def finalize(output_dir: Path) -> None:
    """Symlink indexed latents/, reference_latents/ from the scene-latent pool."""
    with open(output_dir / "metadata_v2.json") as f:
        pairs = json.load(f)["pairs"]

    pool = output_dir / "scene_latents" / "clips"  # process_videos preserves clips/ prefix
    lat = output_dir / "latents"
    ref = output_dir / "reference_latents"
    for d in (lat, ref):
        d.mkdir(parents=True, exist_ok=True)

    missing = 0
    made = 0
    for p in pairs:
        tgt_src = pool / f"{p['tgt_name']}.pt"
        ref_src = pool / f"{p['ref_name']}.pt"
        if not tgt_src.exists() or not ref_src.exists():
            missing += 1
            continue
        tgt_link = lat / f"{p['idx']:06d}.pt"
        ref_link = ref / f"{p['idx']:06d}.pt"
        for link, src in ((tgt_link, tgt_src), (ref_link, ref_src)):
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src.resolve())
        made += 1

    cond = output_dir / "conditions"
    n_cond = len(list(cond.glob("*.pt"))) if cond.exists() else 0
    print(f"Linked {made} pairs (latents + reference_latents). {missing} missing scene latents.")
    print(f"conditions/: {n_cond} files.  latents/: {len(list(lat.glob('*.pt')))}")
    if made and n_cond >= made:
        print("READY to train:", output_dir / "train_config.yaml")
    else:
        print("WARN: conditions count < pairs — run process_captions.py (stage B) first.")


def write_config(output_dir: Path, n_pairs: int) -> None:
    exp_out = "/media/2TB/omnitransfer/output/mashup_v2"
    cfg = f"""# OmniTransfer style-transfer — mashup_v2 (paper-aligned: 49-frame video refs,
# neutral prompts). Auto-generated by tools/rebuild_mashup_v2.py from {n_pairs} pairs.
model:
  model_path: "{MODEL_PATH}"
  training_mode: lora

lora:
  rank: 32
  alpha: 32

training_strategy:
  name: omnitransfer
  task_type: style_transfer
  training_stage: 1
  enable_tpb: true
  enable_rcl: true
  enable_tma: false
  enable_concept_embeddings: true
  concept_embedding_dim: 128
  concept_embedding_task_specific: true
  i2v_mode: false
  reference_latents_dir: reference_latents
  # Pixel-VGG style loss disabled: it decodes latents through the VAE decoder,
  # which the trainer offloads to CPU between uses -> CPU conv hangs the step.
  # Not part of the OmniTransfer paper (loss is flow-matching MSE + TPB/RCL).
  style_loss_weight: 0.0
  use_decoded_pixels_for_style: false
  use_vgg_style_features: false
  # W&B reconstruction/video logging (multi-frame now -> visualize several frames)
  log_reconstructions: true
  reconstruction_log_interval: 100
  num_frames_to_visualize: 4
  log_video_comparisons: true
  video_log_interval: 500

data:
  preprocessed_data_root: "{output_dir}"
  final_embeddings_dir: conditions
  use_cached_final_embeddings: true
  num_dataloader_workers: 0

optimization:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-5        # AdamW LoRA. 1e-4 diverged to NaN (~step 400); 3e-5 + warmup stable.
  warmup_steps: 100            # stabilizes early training (bf16 + int8-quanto + batch 1)
  optimizer_type: adamw        # Muon's torch impl rejects the 3D ConceptEmbedding param
  scheduler_type: cosine
  scheduler_params:
    eta_min: 1.0e-4
  weight_decay: 0.01
  steps: 4000
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: bf16
  quantization: int8-quanto
  load_text_encoder_in_8bit: false

output_dir: "{exp_out}"

checkpoints:
  interval: 500
  keep_last_n: 4

validation:
  interval: null

wandb:
  enabled: true
  project: omnitransfer-mashup-v2
  log_validation_videos: false
"""
    (output_dir / "train_config.yaml").write_text(cfg)
    print(f"Wrote {output_dir / 'train_config.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--scenes-dir", type=Path, required=True)
    pp.add_argument("--output-dir", type=Path, required=True)
    pp.add_argument("--min-frames", type=int, default=FRAMES)
    fp = sub.add_parser("finalize")
    fp.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "prepare":
        prepare(args.scenes_dir, args.output_dir, args.min_frames)
    elif args.cmd == "finalize":
        finalize(args.output_dir)


if __name__ == "__main__":
    main()
