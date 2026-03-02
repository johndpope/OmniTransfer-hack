#!/usr/bin/env python3
"""
SCD Evolution — Gradient-free fine-tuning for autoregressive quality.

Optimizes SCD decoder LoRA weights using EggRoll-style evolution (hash-based
antithetic perturbation + ES gradient) to minimize AR rollout quality drift.

Prerequisite: A gradient-trained SCD LoRA checkpoint.

Usage:
    python scripts/evolve_scd.py \
        --lora-path /media/2TB/omnitransfer/output/scd_distilled_perframe/checkpoints/lora_weights_step_02000.safetensors \
        --data-root /media/2TB/omnitransfer/data/ditto_subset \
        --output-dir /media/2TB/omnitransfer/output/scd_evolution \
        --population-size 4 \
        --num-generations 200

    # With VAE decode + pixel metrics on second GPU:
    python scripts/evolve_scd.py \
        --lora-path /path/to/lora.safetensors \
        --data-root /media/2TB/omnitransfer/data/ditto_subset \
        --use-vae-decoder \
        --w-pixel-lpips 0.15 --w-pixel-ssim 0.1 \
        --eval-batch-size 2 \
        --output-dir /media/2TB/omnitransfer/output/scd_evolution

    # With YAML config:
    python scripts/evolve_scd.py --config configs/ltx2_scd_evolution.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Ensure CUDA arch is set for RTX 5090 before any CUDA imports
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SCD Evolution — gradient-free AR quality fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (overrides defaults, CLI overrides config file)
    p.add_argument("--config", type=str, help="YAML config file path")

    # Model
    p.add_argument(
        "--checkpoint",
        default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors",
        help="Base model checkpoint",
    )
    p.add_argument("--lora-path", type=str, required=False, help="SCD LoRA checkpoint to evolve")
    p.add_argument("--encoder-layers", type=int, default=32, help="SCD encoder layers")
    p.add_argument("--decoder-combine", default="add", choices=["add", "token_concat"])
    p.add_argument(
        "--quantization",
        default="int8-quanto",
        choices=["int8-quanto", "fp8-quanto", "none"],
    )
    p.add_argument("--distilled", action="store_true", help="Use distilled sigma schedule")

    # Data
    p.add_argument(
        "--data-root",
        default="/media/2TB/omnitransfer/data/ditto_subset",
        help="Precomputed dataset root",
    )
    p.add_argument("--conditions-dir", default="conditions_final")

    # Evolution
    p.add_argument("--population-size", type=int, default=4, help="Antithetic pairs per generation")
    p.add_argument("--num-generations", type=int, default=200)
    p.add_argument("--noise-scale", type=float, default=0.005, help="Initial perturbation scale")
    p.add_argument("--noise-decay", type=float, default=0.998, help="Noise scale decay per gen")
    p.add_argument("--noise-min", type=float, default=1e-6, help="Minimum noise scale")
    p.add_argument("--update-scale", type=float, default=0.002, help="ES learning rate")
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Samples per perturbation evaluation (higher = less noisy, slower)",
    )

    # AR Rollout
    p.add_argument("--ar-frames", type=int, default=4, help="Frames per AR evaluation")
    p.add_argument("--num-inference-steps", type=int, default=15, help="Denoising steps (30→15 for speed)")

    # Fitness weights
    p.add_argument("--w-fm-loss", type=float, default=0.5, help="Weight for FM velocity MSE")
    p.add_argument("--w-latent-recon", type=float, default=0.3, help="Weight for latent recon MSE")
    p.add_argument("--w-temporal-coherence", type=float, default=0.2, help="Weight for temporal coherence gap")
    p.add_argument("--w-pixel-lpips", type=float, default=0.0, help="Weight for pixel LPIPS (requires VAE)")
    p.add_argument("--w-pixel-ssim", type=float, default=0.0, help="Weight for pixel SSIM (requires VAE)")

    # Dual-GPU
    p.add_argument("--use-vae-decoder", action="store_true", help="Load VAE decoder on second GPU")
    p.add_argument("--vae-device", default="cuda:1", help="Device for VAE decoder")

    # Output
    p.add_argument(
        "--output-dir",
        default="/media/2TB/omnitransfer/output/scd_evolution",
    )
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=5)

    # W&B
    p.add_argument("--wandb-project", default="scd-evolution")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # Resumption
    p.add_argument("--resume", type=str, help="Checkpoint name to resume from (e.g. 'gen_0050')")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f) or {}

    # Build EvolutionConfig from YAML + CLI (CLI overrides YAML)
    from ltx_trainer.evolution.engine import EvolutionConfig, SCDEvolutionEngine

    # Flatten nested YAML sections
    flat = {}
    for section_key, section_val in yaml_config.items():
        if isinstance(section_val, dict):
            flat.update(section_val)
        else:
            flat[section_key] = section_val

    # Map YAML keys to EvolutionConfig fields
    yaml_to_field = {
        "checkpoint": "checkpoint",
        "lora_path": "lora_path",
        "encoder_layers": "encoder_layers",
        "decoder_combine": "decoder_combine",
        "quantization": "quantization",
        "distilled": "distilled",
        "root": "data_root",
        "data_root": "data_root",
        "conditions_dir": "conditions_dir",
        "population_size": "population_size",
        "num_generations": "num_generations",
        "noise_scale": "noise_scale",
        "noise_decay": "noise_decay",
        "noise_min": "noise_min",
        "update_scale": "update_scale",
        "eval_batch_size": "eval_batch_size",
        "ar_frames": "ar_frames",
        "num_inference_steps": "num_inference_steps",
        "w_fm_loss": "w_fm_loss",
        "w_latent_recon": "w_latent_recon",
        "w_temporal_coherence": "w_temporal_coherence",
        "w_pixel_lpips": "w_pixel_lpips",
        "w_pixel_ssim": "w_pixel_ssim",
        "use_vae_decoder": "use_vae_decoder",
        "vae_device": "vae_device",
        "dir": "output_dir",
        "output_dir": "output_dir",
        "checkpoint_every": "checkpoint_every",
        "log_every": "log_every",
        "enabled": "wandb_enabled",
        "wandb_enabled": "wandb_enabled",
        "project": "wandb_project",
        "wandb_project": "wandb_project",
        "seed": "seed",
    }

    config_kwargs = {}
    for yaml_key, field_name in yaml_to_field.items():
        if yaml_key in flat:
            config_kwargs[field_name] = flat[yaml_key]

    # CLI overrides (only set if explicitly provided)
    cli_overrides = {
        "checkpoint": args.checkpoint,
        "lora_path": args.lora_path,
        "encoder_layers": args.encoder_layers,
        "decoder_combine": args.decoder_combine,
        "quantization": args.quantization,
        "distilled": args.distilled,
        "data_root": args.data_root,
        "conditions_dir": args.conditions_dir,
        "population_size": args.population_size,
        "num_generations": args.num_generations,
        "noise_scale": args.noise_scale,
        "noise_decay": args.noise_decay,
        "noise_min": args.noise_min,
        "update_scale": args.update_scale,
        "eval_batch_size": args.eval_batch_size,
        "ar_frames": args.ar_frames,
        "num_inference_steps": args.num_inference_steps,
        "w_fm_loss": args.w_fm_loss,
        "w_latent_recon": args.w_latent_recon,
        "w_temporal_coherence": args.w_temporal_coherence,
        "w_pixel_lpips": args.w_pixel_lpips,
        "w_pixel_ssim": args.w_pixel_ssim,
        "use_vae_decoder": args.use_vae_decoder,
        "vae_device": args.vae_device,
        "output_dir": args.output_dir,
        "checkpoint_every": args.checkpoint_every,
        "log_every": args.log_every,
        "wandb_enabled": not args.no_wandb,
        "wandb_project": args.wandb_project,
        "seed": args.seed,
    }

    # Only override if CLI provided non-default values
    for key, val in cli_overrides.items():
        if val is not None:
            config_kwargs[key] = val

    config = EvolutionConfig(**config_kwargs)

    # Validate
    if config.lora_path is None:
        print("ERROR: --lora-path is required (SCD LoRA checkpoint to evolve)")
        sys.exit(1)

    if not Path(config.lora_path).exists():
        print(f"ERROR: LoRA checkpoint not found: {config.lora_path}")
        sys.exit(1)

    if not Path(config.data_root).exists():
        print(f"ERROR: Data root not found: {config.data_root}")
        sys.exit(1)

    # Print config summary
    print("=" * 60)
    print("SCD Evolution Configuration")
    print("=" * 60)
    print(f"  Checkpoint:      {config.checkpoint}")
    print(f"  LoRA:            {config.lora_path}")
    print(f"  Data:            {config.data_root}")
    print(f"  Output:          {config.output_dir}")
    print(f"  Quantization:    {config.quantization}")
    print(f"  Population:      {config.population_size} pairs")
    print(f"  Generations:     {config.num_generations}")
    print(f"  AR frames:       {config.ar_frames}")
    print(f"  Inference steps: {config.num_inference_steps}")
    print(f"  Eval batch size: {config.eval_batch_size}")
    print(f"  Noise:           {config.noise_scale} (decay={config.noise_decay})")
    print(f"  Update scale:    {config.update_scale}")
    print(f"  Fitness weights: fm={config.w_fm_loss}, recon={config.w_latent_recon}, "
          f"tcoh={config.w_temporal_coherence}")
    if config.use_vae_decoder:
        print(f"  VAE decoder:     {config.vae_device}")
        print(f"  Pixel weights:   lpips={config.w_pixel_lpips}, ssim={config.w_pixel_ssim}")
    print(f"  W&B:             {'enabled' if config.wandb_enabled else 'disabled'}")
    print("=" * 60)

    # Run evolution
    engine = SCDEvolutionEngine(config)
    engine.setup()

    # Resume from checkpoint if requested
    if args.resume:
        engine.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume} (gen={engine.state.generation})")

    engine.run()


if __name__ == "__main__":
    main()
