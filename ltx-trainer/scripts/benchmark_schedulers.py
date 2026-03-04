#!/usr/bin/env python3
# ruff: noqa: T201
"""Benchmark learned sigma schedulers vs default LTX2Scheduler.

Compares LTX2Scheduler, BézierFlow, and BSplineFlow at 4-step and 8-step
denoising on single-frame SCD decoding. Measures MSE vs 30-step teacher
reference, wall-clock time, and reports sigma values.

Usage:
    python scripts/benchmark_schedulers.py \
        --bezier-schedule /media/2TB/omnitransfer/output/bezierflow/schedule_muon_v2.pt \
        --bspline-schedule /media/2TB/omnitransfer/output/bsplineflow/schedule.pt \
        --test-embeddings /media/2TB/omnitransfer/data/ditto_subset/conditions_final/ \
        --num-test 5 \
        --output /media/2TB/omnitransfer/output/scheduler_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

# Set CUDA arch before importing quanto
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape


def load_scd_model(
    checkpoint: str,
    lora_path: str,
    quantization: str,
    encoder_layers: int,
    decoder_combine: str,
    device: str,
) -> tuple:
    """Load and return (scd_model, patchifier, latent_channels)."""
    from ltx_trainer.model_loader import load_transformer

    print("[1/4] Loading transformer on CPU...")
    transformer = load_transformer(checkpoint, device="cpu", dtype=torch.bfloat16)

    # Apply LoRA on CPU BEFORE quantization (PEFT can't wrap QLinear)
    if lora_path and Path(lora_path).exists():
        import re as _re

        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        print(f"  Applying LoRA: {Path(lora_path).name}")
        lora_sd = load_file(lora_path)
        lora_sd = {k.replace("diffusion_model.", "", 1): v for k, v in lora_sd.items()}
        # Normalize SCD paths
        normalized = {}
        for key, value in lora_sd.items():
            if key.startswith("encoder_blocks.") or key.startswith("decoder_blocks."):
                continue
            if key.startswith("base_model."):
                key = key[len("base_model."):]
            normalized[key] = value
        lora_sd = normalized
        target_modules = sorted(
            {m.group(1) for key in lora_sd if (m := _re.match(r"(.+)\.lora_[AB]\.", key))}
        )
        rank = next(t.shape[0] for k, t in lora_sd.items() if "lora_A" in k and t.ndim == 2)
        print(f"  LoRA rank={rank}, {len(target_modules)} targets")
        lora_config = LoraConfig(
            r=rank, lora_alpha=rank, target_modules=target_modules, lora_dropout=0.0
        )
        transformer = get_peft_model(transformer, lora_config)
        set_peft_model_state_dict(transformer.get_base_model(), lora_sd)
        transformer = transformer.get_base_model()

    if quantization != "none":
        from ltx_trainer.quantization import quantize_model

        print(f"  Quantizing ({quantization})...")
        transformer = quantize_model(transformer, quantization, device=device)

    transformer = transformer.to(device)

    from ltx_core.model.transformer.scd_model import LTXSCDModel

    scd_model = LTXSCDModel(
        base_model=transformer,
        encoder_layers=encoder_layers,
        decoder_input_combine=decoder_combine,
    )
    scd_model.eval()

    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_channels = scd_model.base_model.patchify_proj.in_features

    return scd_model, patchifier, latent_channels


def load_test_embeddings(
    embeddings_dir: Path,
    num_test: int,
    device: str,
    dtype: torch.dtype,
) -> list[dict[str, torch.Tensor]]:
    """Load diverse test embeddings, skipping first 100 (used in training)."""
    files = sorted(embeddings_dir.glob("*.pt"))
    # Skip first 100 files (may have been used in scheduler training)
    test_files = files[100:]
    if len(test_files) < num_test:
        test_files = files  # Fallback if not enough
    # Pick evenly spaced samples for diversity
    step = max(1, len(test_files) // num_test)
    selected = [test_files[i * step] for i in range(num_test)]

    samples = []
    for f in selected:
        cond = torch.load(f, map_location="cpu", weights_only=True)
        embeds = cond.get("video_prompt_embeds", cond.get("prompt_embeds"))
        mask = cond.get("prompt_attention_mask", None)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        samples.append({
            "prompt_embeds": embeds.to(device, dtype),
            "prompt_mask": mask.to(device) if mask is not None else None,
            "name": f.stem,
        })
    return samples


@torch.inference_mode()
def run_denoising(
    scd_model,
    patchifier,
    sigmas: torch.Tensor,
    z_init: torch.Tensor,
    enc_features: torch.Tensor,
    enc_features_null: torch.Tensor,
    positions: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor | None,
    null_embeds: torch.Tensor,
    null_mask: torch.Tensor | None,
    guidance_scale: float,
    tokens_per_frame: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    """Run Euler denoising and return (final_latent, wall_time_seconds)."""
    num_steps = len(sigmas) - 1
    use_cfg = guidance_scale > 1.0

    z = patchifier.patchify(z_init.clone())
    t_start = time.perf_counter()

    for step in range(num_steps):
        sigma = sigmas[step]
        sigma_next = sigmas[step + 1]

        dec_modality = Modality(
            enabled=True,
            latent=z,
            timesteps=torch.full((1, tokens_per_frame), sigma.item(), device=device, dtype=dtype),
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_mask,
        )
        velocity, _ = scd_model.forward_decoder(
            video=dec_modality,
            encoder_features=enc_features,
            audio=None,
            perturbations=None,
        )

        if use_cfg:
            uncond_mod = Modality(
                enabled=True,
                latent=z,
                timesteps=torch.full(
                    (1, tokens_per_frame), sigma.item(), device=device, dtype=dtype
                ),
                positions=positions,
                context=null_embeds,
                context_mask=null_mask,
            )
            v_uncond, _ = scd_model.forward_decoder(
                video=uncond_mod,
                encoder_features=enc_features_null,
                audio=None,
                perturbations=None,
            )
            velocity = v_uncond + guidance_scale * (velocity - v_uncond)

        dt = sigma_next - sigma
        z = (z.float() + velocity.float() * dt.float()).to(dtype)

    t_elapsed = time.perf_counter() - t_start
    return z, t_elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sigma schedulers for SCD inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bezier-schedule",
        default="/media/2TB/omnitransfer/output/bezierflow/schedule_muon_v2.pt",
        help="BézierFlow schedule .pt file",
    )
    parser.add_argument(
        "--bspline-schedule",
        default="/media/2TB/omnitransfer/output/bsplineflow/schedule.pt",
        help="BSplineFlow schedule .pt file",
    )
    parser.add_argument(
        "--test-embeddings",
        default="/media/2TB/omnitransfer/data/ditto_subset/conditions_final/",
        help="Directory of precomputed text embeddings",
    )
    parser.add_argument("--num-test", type=int, default=5, help="Number of test embeddings")
    parser.add_argument(
        "--output",
        default="/media/2TB/omnitransfer/output/scheduler_benchmark.json",
        help="Output JSON path for benchmark results",
    )
    # Model args
    parser.add_argument(
        "--checkpoint",
        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-distilled.safetensors",
        help="Base model checkpoint",
    )
    parser.add_argument(
        "--lora-path",
        default="/media/2TB/omnitransfer/output/scd_token_concat/checkpoints/lora_weights_step_01000.safetensors",
        help="SCD LoRA checkpoint",
    )
    parser.add_argument(
        "--quantization",
        default="int8-quanto",
        choices=["fp8-quanto", "int8-quanto", "none"],
    )
    parser.add_argument("--encoder-layers", type=int, default=32)
    parser.add_argument("--decoder-combine", default="token_concat")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = "cuda:0"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    latent_h = args.height // 32
    latent_w = args.width // 32
    tokens_per_frame = latent_h * latent_w
    latent_channels = 128
    CHUNK_LATENT = 4

    print()
    print("=" * 70)
    print("  Sigma Scheduler Benchmark: LTX2 vs BézierFlow vs BSplineFlow")
    print("=" * 70)
    print(f"  Resolution:  {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"  Tokens/frame: {tokens_per_frame}")
    print(f"  CFG scale:   {args.guidance_scale}")
    print(f"  Test samples: {args.num_test}")
    print()

    # ── Load model ──
    scd_model, patchifier, latent_channels = load_scd_model(
        args.checkpoint,
        args.lora_path,
        args.quantization,
        args.encoder_layers,
        args.decoder_combine,
        device,
    )

    scale_factors = SpatioTemporalScaleFactors.default()

    def get_positions_for_frame(frame_idx: int) -> torch.Tensor:
        all_coords = patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=frame_idx + 1,
                height=latent_h,
                width=latent_w,
                batch=1,
                channels=latent_channels,
            ),
            device=device,
        )
        px = get_pixel_coords(latent_coords=all_coords, scale_factors=scale_factors, causal_fix=True).to(dtype)
        px[:, 0, ...] = px[:, 0, ...] / args.fps
        start = frame_idx * tokens_per_frame
        end = start + tokens_per_frame
        return px[:, :, start:end, :]

    # ── Load test embeddings ──
    print(f"\n[2/4] Loading {args.num_test} test embeddings...")
    test_dir = Path(args.test_embeddings)
    samples = load_test_embeddings(test_dir, args.num_test, device, dtype)
    print(f"  Loaded: {[s['name'] for s in samples]}")

    null_embeds = torch.zeros_like(samples[0]["prompt_embeds"])
    null_mask = (
        torch.zeros_like(samples[0]["prompt_mask"])
        if samples[0]["prompt_mask"] is not None
        else None
    )

    # ── Build scheduler configs ──
    print("\n[3/4] Building scheduler configurations...")

    # Load learned schedulers
    scheduler_configs = {}

    # LTX2Scheduler (default) — uses shifted logit-normal schedule
    dummy_latent = torch.empty(1, 1, CHUNK_LATENT, latent_h, latent_w)
    ltx2_sched = LTX2Scheduler()
    for n_steps in [4, 8]:
        sigmas = ltx2_sched.execute(steps=n_steps, latent=dummy_latent).to(device=device, dtype=dtype)
        scheduler_configs[f"LTX2Scheduler_{n_steps}"] = {
            "sigmas": sigmas,
            "label": "LTX2Scheduler",
            "steps": n_steps,
            "sigma_values": sigmas.tolist(),
        }

    # BézierFlow
    bezier_path = Path(args.bezier_schedule)
    if bezier_path.exists():
        from ltx_trainer.bezierflow import BezierScheduler

        bezier = BezierScheduler.load(bezier_path, device="cpu")
        for n_steps in [4, 8]:
            sigmas = bezier.get_sigma_schedule(n_steps).to(device=device, dtype=dtype)
            scheduler_configs[f"BezierFlow_{n_steps}"] = {
                "sigmas": sigmas,
                "label": "BézierFlow",
                "steps": n_steps,
                "sigma_values": sigmas.tolist(),
            }
        del bezier
        print(f"  BézierFlow loaded: {bezier_path.name}")
    else:
        print(f"  WARNING: BézierFlow not found at {bezier_path}")

    # BSplineFlow
    bspline_path = Path(args.bspline_schedule)
    if bspline_path.exists():
        from ltx_trainer.bsplineflow import BSplineScheduler

        bspline = BSplineScheduler.load(bspline_path, device="cpu")
        for n_steps in [4, 8]:
            sigmas = bspline.get_sigma_schedule(n_steps).to(device=device, dtype=dtype)
            scheduler_configs[f"BSplineFlow_{n_steps}"] = {
                "sigmas": sigmas,
                "label": "BSplineFlow",
                "steps": n_steps,
                "sigma_values": sigmas.tolist(),
            }
        del bspline
        print(f"  BSplineFlow loaded: {bspline_path.name}")
    else:
        print(f"  WARNING: BSplineFlow not found at {bspline_path}")

    # Teacher schedule (30-step reference)
    teacher_sigmas = ltx2_sched.execute(steps=30, latent=dummy_latent).to(device=device, dtype=dtype)
    print(f"  Teacher: 30-step LTX2Scheduler σ=[{teacher_sigmas[0]:.4f}→{teacher_sigmas[-1]:.4f}]")
    print(f"  Configs: {list(scheduler_configs.keys())}")

    # ── Run benchmark ──
    print(f"\n[4/4] Running benchmark ({len(scheduler_configs)} configs × {args.num_test} samples)...")

    positions = get_positions_for_frame(0)
    use_cfg = args.guidance_scale > 1.0

    results = {}
    for config_name, config in scheduler_configs.items():
        results[config_name] = {
            "label": config["label"],
            "steps": config["steps"],
            "sigma_values": config["sigma_values"],
            "mse_per_sample": [],
            "time_per_sample": [],
        }

    from ltx_core.model.transformer.scd_model import KVCache

    # For each test sample: generate teacher reference, then benchmark each scheduler
    with torch.inference_mode():
      for si, sample in enumerate(samples):
        prompt_embeds = sample["prompt_embeds"]
        prompt_mask = sample["prompt_mask"]
        sample_name = sample["name"]
        print(f"\n  Sample {si + 1}/{args.num_test}: {sample_name}")

        # Use the SAME random noise for all configs on this sample
        torch.manual_seed(args.seed + si)
        z_init = torch.randn(1, latent_channels, 1, latent_h, latent_w, device=device, dtype=dtype)

        # ── Encode once (shared across all configs) ──
        enc_input = torch.zeros(1, latent_channels, 1, latent_h, latent_w, device=device, dtype=dtype)
        enc_patch = patchifier.patchify(enc_input)
        enc_modality = Modality(
            enabled=True,
            latent=enc_patch,
            timesteps=torch.zeros(1, tokens_per_frame, device=device, dtype=dtype),
            positions=get_positions_for_frame(0),
            context=prompt_embeds,
            context_mask=prompt_mask,
        )
        kv_cache = KVCache.empty()
        kv_cache.is_cache_step = True
        enc_out, _ = scd_model.forward_encoder(
            video=enc_modality,
            audio=None,
            perturbations=None,
            kv_cache=kv_cache,
            tokens_per_frame=tokens_per_frame,
        )
        enc_features = enc_out.x.detach().clone()
        del enc_out, kv_cache, enc_modality, enc_input, enc_patch

        if use_cfg:
            enc_patch_null = patchifier.patchify(
                torch.zeros(1, latent_channels, 1, latent_h, latent_w, device=device, dtype=dtype)
            )
            enc_null_mod = Modality(
                enabled=True,
                latent=enc_patch_null,
                timesteps=torch.zeros(1, tokens_per_frame, device=device, dtype=dtype),
                positions=get_positions_for_frame(0),
                context=null_embeds,
                context_mask=null_mask,
            )
            kv_null = KVCache.empty()
            kv_null.is_cache_step = True
            enc_out_null, _ = scd_model.forward_encoder(
                video=enc_null_mod,
                audio=None,
                perturbations=None,
                kv_cache=kv_null,
                tokens_per_frame=tokens_per_frame,
            )
            enc_features_null = enc_out_null.x.detach().clone()
            del enc_out_null, kv_null, enc_null_mod, enc_patch_null
        else:
            enc_features_null = enc_features

        torch.cuda.empty_cache()

        # ── Teacher reference (30 steps) ──
        print(f"    Teacher (30 steps)...", end=" ", flush=True)
        z_teacher, t_teacher = run_denoising(
            scd_model,
            patchifier,
            teacher_sigmas,
            z_init,
            enc_features,
            enc_features_null,
            positions,
            prompt_embeds,
            prompt_mask,
            null_embeds,
            null_mask,
            args.guidance_scale,
            tokens_per_frame,
            device,
            dtype,
        )
        print(f"{t_teacher:.2f}s")

        # ── Benchmark each scheduler config ──
        for config_name, config in scheduler_configs.items():
            z_result, t_elapsed = run_denoising(
                scd_model,
                patchifier,
                config["sigmas"],
                z_init,
                enc_features,
                enc_features_null,
                positions,
                prompt_embeds,
                prompt_mask,
                null_embeds,
                null_mask,
                args.guidance_scale,
                tokens_per_frame,
                device,
                dtype,
            )

            mse = torch.nn.functional.mse_loss(z_result.float(), z_teacher.float()).item()
            results[config_name]["mse_per_sample"].append(mse)
            results[config_name]["time_per_sample"].append(t_elapsed)

            print(
                f"    {config_name:25s} | MSE={mse:.6f} | {t_elapsed:.3f}s "
                f"({config['steps']} steps)"
            )
            del z_result

        # Cleanup between samples
        del z_teacher, z_init, enc_features, enc_features_null
        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate and report ──
    print("\n")
    print("=" * 90)
    print("  BENCHMARK RESULTS")
    print("=" * 90)
    header = f"{'Scheduler':15s} {'Steps':>5s} {'MSE vs Teacher':>14s} {'s/frame':>8s} {'σ values'}"
    print(f"  {header}")
    print(f"  {'-' * 85}")

    summary = []
    for config_name, r in results.items():
        avg_mse = sum(r["mse_per_sample"]) / len(r["mse_per_sample"])
        avg_time = sum(r["time_per_sample"]) / len(r["time_per_sample"])
        sigmas_str = "[" + ", ".join(f"{s:.3f}" for s in r["sigma_values"]) + "]"
        print(f"  {r['label']:15s} {r['steps']:5d} {avg_mse:14.6f} {avg_time:8.3f}s {sigmas_str}")
        summary.append({
            "config": config_name,
            "label": r["label"],
            "steps": r["steps"],
            "avg_mse": avg_mse,
            "avg_time_s": avg_time,
            "sigma_values": r["sigma_values"],
            "mse_per_sample": r["mse_per_sample"],
            "time_per_sample": r["time_per_sample"],
        })
    print(f"  {'-' * 85}")

    # ── Relative improvements ──
    ltx2_4 = next((s for s in summary if s["config"] == "LTX2Scheduler_4"), None)
    if ltx2_4:
        print("\n  Relative to LTX2Scheduler (4-step):")
        for s in summary:
            if s["steps"] == 4 and s["config"] != "LTX2Scheduler_4":
                ratio = s["avg_mse"] / ltx2_4["avg_mse"] if ltx2_4["avg_mse"] > 0 else float("inf")
                print(f"    {s['label']:15s}: MSE ratio = {ratio:.3f}x {'(better)' if ratio < 1 else '(worse)'}")

    ltx2_8 = next((s for s in summary if s["config"] == "LTX2Scheduler_8"), None)
    if ltx2_8:
        print("\n  Relative to LTX2Scheduler (8-step):")
        for s in summary:
            if s["steps"] == 8 and s["config"] != "LTX2Scheduler_8":
                ratio = s["avg_mse"] / ltx2_8["avg_mse"] if ltx2_8["avg_mse"] > 0 else float("inf")
                print(f"    {s['label']:15s}: MSE ratio = {ratio:.3f}x {'(better)' if ratio < 1 else '(worse)'}")

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "benchmark_config": {
            "resolution": f"{args.width}x{args.height}",
            "quantization": args.quantization,
            "guidance_scale": args.guidance_scale,
            "num_test_samples": args.num_test,
            "teacher_steps": 30,
            "seed": args.seed,
        },
        "results": summary,
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
