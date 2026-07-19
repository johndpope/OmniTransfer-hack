#!/usr/bin/env python3
"""Precompute Qwen-VL (MLLM) features for OmniTransfer TMA / MetaQuery (Stage 2/3).

The OmniTransfer strategy consumes a cached ``qwen_vl_features/`` directory when
TMA is enabled (arXiv:2601.14250, Sec 4.4): per training sample it loads the
MLLM's hidden states over the reference frames + target first frame + prompt, and
the ``MetaQueryBank`` aggregates them via cross-attention through a trainable
connector. This script produces those features.

Each output ``qwen_vl_features/{idx:06d}.pt`` matches PrecomputedDataset's expected
format::

    {
      "qwen_features": Tensor[seq_len, hidden_dim],   # MLLM last hidden state
      "task_type": str,
      "caption": str,
      "num_ref_frames": int,
      "model_name": str,
      "hidden_dim": int,
    }

Usage (real features — needs the Qwen-VL weights, ~16GB):
    python scripts/compute_qwen_vl_features.py \
        --data-root /media/2TB/omnitransfer/data/mashup_v2 \
        --model-path Qwen/Qwen2.5-VL-7B-Instruct

Usage (wiring validation — no MLLM, correctly-shaped random features):
    python scripts/compute_qwen_vl_features.py \
        --data-root /media/2TB/omnitransfer/data/mashup_v2 --dummy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch


def read_frames(clip: Path, num: int, size: int) -> torch.Tensor:
    """Read up to `num` evenly-spaced frames from a clip as [T, C, H, W] in [0,1]."""
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = [int(i * total / num) for i in range(num)] if num > 1 else [0]
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    cap.release()
    if not frames:
        frames = [torch.zeros(3, size, size)]
    return torch.stack(frames)  # [T, C, H, W]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Dataset root with metadata.json + clips/ (from rebuild_mashup_v2.py)")
    ap.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir-name", type=str, default="qwen_vl_features")
    ap.add_argument("--num-ref-frames", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dummy", action="store_true",
                    help="Skip the MLLM and write correctly-shaped random features "
                         "(for validating the Stage-2 training loop end-to-end).")
    ap.add_argument("--dummy-seq-len", type=int, default=64)
    ap.add_argument("--dummy-hidden-dim", type=int, default=3584)  # Qwen2.5-VL-7B
    args = ap.parse_args()

    meta = json.load(open(args.data_root / "metadata.json"))
    pairs = meta.get("pairs", meta if isinstance(meta, list) else [])
    out_dir = args.data_root / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = args.data_root / "clips"

    if args.dummy:
        print(f"[dummy] writing {len(pairs)} random feature files "
              f"([{args.dummy_seq_len}, {args.dummy_hidden_dim}]) to {out_dir}")
        for p in pairs:
            idx = int(p.get("id", p.get("idx")))
            torch.save({
                "qwen_features": torch.randn(args.dummy_seq_len, args.dummy_hidden_dim) * 0.02,
                "task_type": p.get("task_type", "style_transfer"),
                "caption": p.get("text", ""),
                "num_ref_frames": args.num_ref_frames,
                "model_name": "dummy",
                "hidden_dim": args.dummy_hidden_dim,
            }, out_dir / f"{idx:06d}.pt")
        print(f"[dummy] done: {len(list(out_dir.glob('*.pt')))} files")
        return

    # Real extraction — reuse the repo's Qwen-VL wrapper. Load the module directly
    # by file path to avoid ltx_trainer.omnitransfer.__init__ (which imports strategy
    # and hits a known circular import); qwen_vl_integration itself only imports
    # stdlib/torch/transformers at module level.
    import importlib.util

    qvi_path = Path(__file__).resolve().parent.parent / "src/ltx_trainer/omnitransfer/qwen_vl_integration.py"
    spec = importlib.util.spec_from_file_location("qwen_vl_integration", qvi_path)
    qvi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qvi)
    QwenVLConfig = qvi.QwenVLConfig
    QwenVLFeatureExtractor = qvi.QwenVLFeatureExtractor

    device = torch.device(args.device)
    extractor = QwenVLFeatureExtractor(
        QwenVLConfig(
            model_path=args.model_path,
            device=args.device,
            use_flash_attention=False,  # not installed; fall back to SDPA
        )
    ).to(device).eval()
    hidden_dim = extractor.hidden_dim
    print(f"Loaded {args.model_path} (hidden_dim={hidden_dim}); extracting {len(pairs)} samples")

    done = 0
    for p in pairs:
        idx = int(p.get("id", p.get("idx")))
        out_path = out_dir / f"{idx:06d}.pt"
        if out_path.exists():
            done += 1
            continue
        ref_clip = clips / f"{p['ref_name']}.mp4"
        tgt_clip = clips / f"{p['tgt_name']}.mp4"
        if not ref_clip.exists() or not tgt_clip.exists():
            continue
        ref = read_frames(ref_clip, args.num_ref_frames, args.image_size).unsqueeze(0).to(device)
        tgt = read_frames(tgt_clip, 1, args.image_size)[0].unsqueeze(0).to(device)
        with torch.inference_mode():
            feats = extractor.extract_features(ref, tgt, p.get("text", ""))  # [1, seq, hidden]
        torch.save({
            "qwen_features": feats[0].float().cpu().contiguous(),
            "task_type": p.get("task_type", "style_transfer"),
            "caption": p.get("text", ""),
            "num_ref_frames": args.num_ref_frames,
            "model_name": args.model_path,
            "hidden_dim": hidden_dim,
        }, out_path)
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(pairs)}")
    print(f"Done: {len(list(out_dir.glob('*.pt')))} feature files in {out_dir}")


if __name__ == "__main__":
    main()
