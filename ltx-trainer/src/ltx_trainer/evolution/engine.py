"""
SCD Evolution Engine — gradient-free fine-tuning for autoregressive quality.

Orchestrates the evolution loop:
  1. Load model (transformer → quantize → LoRA → SCD wrap)
  2. Load precomputed dataset (latents + conditions)
  3. For each generation:
     a. Generate antithetic perturbation pairs
     b. Evaluate fitness via AR rollout against GT
     c. Compute ES gradient and update decoder LoRA weights
     d. Anneal noise scale
  4. Save evolved checkpoint
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ltx_trainer import logger
from ltx_trainer.evolution.fitness import ARRolloutEvaluator, FitnessResult
from ltx_trainer.evolution.perturbation import (
    HashPerturbation,
    SelectiveLoRAPerturbation,
    generate_perturbation_seeds,
)


@dataclass
class EvolutionConfig:
    """Configuration for SCD evolution."""

    # Model
    checkpoint: str = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
    lora_path: str | None = None
    encoder_layers: int = 32
    decoder_combine: str = "add"
    quantization: str = "int8-quanto"
    distilled: bool = False

    # Data
    data_root: str = "/media/2TB/omnitransfer/data/ditto_subset"
    conditions_dir: str = "conditions_final"

    # Evolution
    population_size: int = 4  # Number of antithetic pairs
    num_generations: int = 200
    noise_scale: float = 0.005
    noise_decay: float = 0.998
    noise_min: float = 1e-6
    update_scale: float = 0.002
    eval_batch_size: int = 1  # Samples per perturbation evaluation (higher = less noisy)

    # AR Rollout
    ar_frames: int = 4
    num_inference_steps: int = 15

    # Fitness weights (latent-space)
    w_fm_loss: float = 0.5
    w_latent_recon: float = 0.3
    w_temporal_coherence: float = 0.2

    # Fitness weights (pixel-space, requires VAE on second GPU)
    w_pixel_lpips: float = 0.0
    w_pixel_ssim: float = 0.0

    # Dual-GPU: VAE decoder on cuda:1 for pixel metrics
    use_vae_decoder: bool = False
    vae_device: str = "cuda:1"

    # Output
    output_dir: str = "/media/2TB/omnitransfer/output/scd_evolution"
    checkpoint_every: int = 25
    log_every: int = 5

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "scd-evolution"

    # Misc
    seed: int = 42


@dataclass
class EvolutionState:
    """Mutable state tracked across generations."""

    generation: int = 0
    best_fitness: float = float("-inf")
    current_noise_scale: float = 0.005
    fitness_history: list[float] = field(default_factory=list)
    generations_without_improvement: int = 0

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "current_noise_scale": self.current_noise_scale,
            "fitness_history": self.fitness_history[-100:],
            "generations_without_improvement": self.generations_without_improvement,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvolutionState:
        return cls(
            generation=d["generation"],
            best_fitness=d["best_fitness"],
            current_noise_scale=d["current_noise_scale"],
            fitness_history=d.get("fitness_history", []),
            generations_without_improvement=d.get("generations_without_improvement", 0),
        )


class SCDEvolutionEngine:
    """Main evolution loop for SCD decoder LoRA fine-tuning."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.state = EvolutionState(current_noise_scale=config.noise_scale)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        # Lazily initialized
        self.scd_model: nn.Module | None = None
        self.perturbation_handler: SelectiveLoRAPerturbation | None = None
        self.evaluator: ARRolloutEvaluator | None = None
        self.dataset_samples: list[dict] | None = None
        self.wandb_run = None

    def setup(self) -> None:
        """Initialize model, dataset, perturbation handler, and evaluator."""
        self._load_model()
        self._load_dataset()
        self._init_perturbation()
        self._load_vae_decoder()
        self._init_evaluator()
        self._init_wandb()
        self._init_output_dir()

    def _load_model(self) -> None:
        """Load transformer → quantize → apply LoRA → wrap with SCD."""
        from ltx_core.model.transformer.scd_model import LTXSCDModel

        from ltx_trainer.model_loader import load_transformer

        logger.info(f"Loading transformer from {self.config.checkpoint}")
        transformer = load_transformer(self.config.checkpoint, device="cpu", dtype=self.dtype)

        # Quantize if requested
        if self.config.quantization != "none":
            from ltx_trainer.quantization import quantize_model

            logger.info(f"Quantizing with {self.config.quantization}")
            transformer = quantize_model(transformer, self.config.quantization, device="cuda:0")
        else:
            transformer = transformer.to(self.device)

        # Apply LoRA checkpoint
        if self.config.lora_path:
            logger.info(f"Loading LoRA from {self.config.lora_path}")
            self._apply_lora(transformer)

        # Wrap with SCD
        self.scd_model = LTXSCDModel(
            base_model=transformer,
            encoder_layers=self.config.encoder_layers,
            decoder_input_combine=self.config.decoder_combine,
        )
        self.scd_model.eval()

        # Get model dimensions
        self._latent_channels = transformer.patchify_proj.in_features  # 128

        logger.info(
            f"SCD model ready: {self.config.encoder_layers} encoder + "
            f"{48 - self.config.encoder_layers} decoder blocks"
        )

    def _load_vae_decoder(self) -> None:
        """Optionally load VAE decoder on secondary GPU for pixel-space metrics."""
        self.vae_decoder = None
        self.lpips_net = None

        if not self.config.use_vae_decoder:
            return

        from ltx_trainer.model_loader import load_video_vae_decoder

        vae_device = torch.device(self.config.vae_device)
        logger.info(f"Loading VAE decoder on {vae_device}")
        self.vae_decoder = load_video_vae_decoder(
            self.config.checkpoint, device=str(vae_device), dtype=self.dtype,
        )
        self.vae_decoder.eval()

        # Load LPIPS if weight > 0
        if self.config.w_pixel_lpips > 0:
            try:
                import lpips

                self.lpips_net = lpips.LPIPS(net="alex").to(vae_device)
                self.lpips_net.eval()
                logger.info("LPIPS network loaded (alex)")
            except ImportError:
                logger.warning("lpips package not installed, disabling LPIPS metric")
                self.config.w_pixel_lpips = 0.0

    def _apply_lora(self, transformer: nn.Module) -> None:
        """Apply a saved LoRA checkpoint to the transformer."""
        from peft import PeftModel, LoraConfig as PeftLoraConfig
        from safetensors.torch import load_file

        state_dict = load_file(self.config.lora_path)

        # Detect LoRA rank from state dict
        rank = None
        for key, tensor in state_dict.items():
            if "lora_A" in key:
                rank = tensor.shape[0]
                break

        if rank is None:
            raise ValueError(f"Could not detect LoRA rank from {self.config.lora_path}")

        logger.info(f"Detected LoRA rank={rank}")

        # Collect target modules from state dict
        target_modules = set()
        for key in state_dict.keys():
            # Extract module path between "base_model.model." and ".lora_A/B"
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p in ("lora_A", "lora_B"):
                    target_modules.add(parts[i - 1])
                    break

        lora_config = PeftLoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=list(target_modules),
            lora_dropout=0.0,
            bias="none",
        )

        # Wrap with PEFT
        transformer = PeftModel(transformer, lora_config)

        # Normalize state dict keys (SCD training may save with various prefixes)
        normalized = {}
        for key, val in state_dict.items():
            # Strip "base_model.model." prefix if present
            nk = key
            if nk.startswith("base_model.model."):
                nk = nk[len("base_model.model."):]
            # Re-add PEFT prefix
            if not nk.startswith("base_model.model."):
                nk = f"base_model.model.{nk}"
            normalized[nk] = val.to(self.device)

        # Load with strict=False to handle any extra keys
        missing, unexpected = transformer.load_state_dict(normalized, strict=False)
        if missing:
            logger.warning(f"Missing keys during LoRA load: {len(missing)}")
        if unexpected:
            logger.warning(f"Unexpected keys during LoRA load: {len(unexpected)}")

        # Merge and unload to get a flat model with LoRA baked in, then re-apply
        # Actually, for evolution we need the LoRA params separate and trainable.
        # Keep the PeftModel as-is — evolution will perturb lora_A/lora_B in-place.

    def _load_dataset(self) -> None:
        """Load precomputed latents + conditions from data_root."""
        data_root = Path(self.config.data_root)
        latents_dir = data_root / "latents"
        conditions_dir = data_root / self.config.conditions_dir

        if not latents_dir.exists():
            raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
        if not conditions_dir.exists():
            raise FileNotFoundError(f"Conditions directory not found: {conditions_dir}")

        # Collect matching pairs
        latent_files = sorted(latents_dir.glob("*.pt"))
        self.dataset_samples = []

        for lf in latent_files:
            cf = conditions_dir / lf.name
            if cf.exists():
                self.dataset_samples.append({
                    "latent_path": str(lf),
                    "condition_path": str(cf),
                })

        logger.info(f"Loaded {len(self.dataset_samples)} dataset samples from {data_root}")

        if len(self.dataset_samples) == 0:
            raise ValueError(f"No matching latent/condition pairs found in {data_root}")

        # Probe first sample for dimensions
        sample_latent = torch.load(self.dataset_samples[0]["latent_path"], weights_only=True)
        if isinstance(sample_latent, dict):
            sample_latent = sample_latent.get("latent", sample_latent.get("video_latent", next(iter(sample_latent.values()))))

        # Expected shape: [C, F, H, W] or [1, C, F, H, W]
        if sample_latent.dim() == 5:
            sample_latent = sample_latent[0]
        self._latent_shape = sample_latent.shape  # [C, F, H, W]
        logger.info(f"Latent shape: {self._latent_shape}")

    def _init_perturbation(self) -> None:
        """Initialize selective perturbation handler for decoder LoRA params."""
        self.perturbation_handler = SelectiveLoRAPerturbation(
            model=self.scd_model,
            encoder_layers=self.config.encoder_layers,
            device=str(self.device),
            dtype=self.dtype,
            use_gaussian=True,
        )

        if self.perturbation_handler.num_params == 0:
            raise RuntimeError(
                "No decoder LoRA parameters found! Make sure the LoRA checkpoint "
                "contains parameters for transformer_blocks >= encoder_layers."
            )

    def _init_evaluator(self) -> None:
        """Initialize AR rollout fitness evaluator."""
        from ltx_core.components.patchifiers import (
            SpatioTemporalScaleFactors,
            VideoLatentPatchifier,
            VideoLatentShape,
            get_pixel_coords,
        )

        C, F, H, W = self._latent_shape
        patchifier = VideoLatentPatchifier(patch_size=1)

        # Position computation helper (replicates scd_inference.py pattern)
        fps = 24.0  # Standard LTX-2 fps

        def get_positions_for_frame(
            frame_idx: int,
            target_device: torch.device | None = None,
        ) -> torch.Tensor:
            dev = target_device or self.device
            n_frames = frame_idx + 1
            coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    frames=n_frames, height=H, width=W, batch=1, channels=C,
                ),
                device=dev,
            )
            scale_factors = SpatioTemporalScaleFactors.default()
            px = get_pixel_coords(
                latent_coords=coords, scale_factors=scale_factors, causal_fix=True,
            ).to(self.dtype)
            px[:, 0, ...] = px[:, 0, ...] / fps

            tpf = H * W
            start = frame_idx * tpf
            end = start + tpf
            return px[:, :, start:end, :]

        self.evaluator = ARRolloutEvaluator(
            scd_model=self.scd_model,
            patchifier=patchifier,
            get_positions_for_frame_fn=get_positions_for_frame,
            tokens_per_frame=H * W,
            latent_h=H,
            latent_w=W,
            latent_channels=C,
            num_inference_steps=self.config.num_inference_steps,
            ar_frames=self.config.ar_frames,
            w_fm_loss=self.config.w_fm_loss,
            w_latent_recon=self.config.w_latent_recon,
            w_temporal_coherence=self.config.w_temporal_coherence,
            w_pixel_lpips=self.config.w_pixel_lpips,
            w_pixel_ssim=self.config.w_pixel_ssim,
            device=self.device,
            dtype=self.dtype,
            vae_decoder=self.vae_decoder,
            vae_device=self.config.vae_device if self.config.use_vae_decoder else None,
            lpips_net=self.lpips_net,
        )

    def _init_wandb(self) -> None:
        """Initialize W&B logging if enabled."""
        if not self.config.wandb_enabled:
            return

        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=f"scd-evo-{time.strftime('%m%d-%H%M')}",
                config={
                    "population_size": self.config.population_size,
                    "num_generations": self.config.num_generations,
                    "noise_scale": self.config.noise_scale,
                    "noise_decay": self.config.noise_decay,
                    "update_scale": self.config.update_scale,
                    "ar_frames": self.config.ar_frames,
                    "num_inference_steps": self.config.num_inference_steps,
                    "w_fm_loss": self.config.w_fm_loss,
                    "w_latent_recon": self.config.w_latent_recon,
                    "w_temporal_coherence": self.config.w_temporal_coherence,
                    "w_pixel_lpips": self.config.w_pixel_lpips,
                    "w_pixel_ssim": self.config.w_pixel_ssim,
                    "eval_batch_size": self.config.eval_batch_size,
                    "use_vae_decoder": self.config.use_vae_decoder,
                    "encoder_layers": self.config.encoder_layers,
                    "decoder_combine": self.config.decoder_combine,
                    "quantization": self.config.quantization,
                    "lora_path": self.config.lora_path,
                    "data_root": self.config.data_root,
                    "num_evolved_params": self.perturbation_handler.num_params,
                },
            )
            logger.info(f"W&B initialized: {self.wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed, disabling logging")
            self.config.wandb_enabled = False

    def _init_output_dir(self) -> None:
        """Create output directory structure."""
        out = Path(self.config.output_dir)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out}")

    def _load_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Load a single dataset sample to GPU."""
        sample = self.dataset_samples[idx]

        latent = torch.load(sample["latent_path"], weights_only=True)
        if isinstance(latent, dict):
            latent = latent.get("latent", latent.get("video_latent", next(iter(latent.values()))))
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)  # [C,F,H,W] -> [1,C,F,H,W]

        condition = torch.load(sample["condition_path"], weights_only=True)
        if isinstance(condition, dict):
            prompt_embeds = condition["prompt_embeds"]
            prompt_mask = condition.get("prompt_attention_mask", None)
        else:
            prompt_embeds = condition
            prompt_mask = None

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)  # [seq, dim] -> [1, seq, dim]
        if prompt_mask is not None and prompt_mask.dim() == 1:
            prompt_mask = prompt_mask.unsqueeze(0)

        return {
            "latent": latent.to(self.device, self.dtype),
            "prompt_embeds": prompt_embeds.to(self.device, self.dtype),
            "prompt_mask": prompt_mask.to(self.device) if prompt_mask is not None else None,
        }

    def _evaluate_baseline(self) -> FitnessResult:
        """Evaluate current (unperturbed) model fitness."""
        sample_idx = random.randint(0, len(self.dataset_samples) - 1)
        sample = self._load_sample(sample_idx)
        return self.evaluator.evaluate(
            gt_latents=sample["latent"],
            prompt_embeds=sample["prompt_embeds"],
            prompt_mask=sample["prompt_mask"],
            seed=self.config.seed,
        )

    def run_generation(self) -> dict:
        """Run one generation: evaluate antithetic pairs, update weights.

        Returns dict with generation stats.
        """
        gen = self.state.generation
        seeds = generate_perturbation_seeds(self.config.population_size)
        fitness_diffs: dict[int, float] = {}
        all_pos_fitness: list[float] = []
        all_neg_fitness: list[float] = []

        batch_size = self.config.eval_batch_size

        for seed_idx, seed in enumerate(seeds):
            # Load batch of samples for this perturbation evaluation
            sample_indices = [
                random.randint(0, len(self.dataset_samples) - 1)
                for _ in range(batch_size)
            ]
            samples = [self._load_sample(idx) for idx in sample_indices]

            eval_seed = self.config.seed + gen * 1000 + seed_idx

            # +ε evaluation
            self.perturbation_handler.apply_perturbation(
                HashPerturbation(seed, self.state.current_noise_scale, +1)
            )
            if batch_size == 1:
                fitness_pos = self.evaluator.evaluate(
                    gt_latents=samples[0]["latent"],
                    prompt_embeds=samples[0]["prompt_embeds"],
                    prompt_mask=samples[0].get("prompt_mask"),
                    seed=eval_seed,
                )
            else:
                fitness_pos = self.evaluator.evaluate_batch(
                    samples=samples, seed_base=eval_seed,
                )
            self.perturbation_handler.revert_to_original()

            # -ε evaluation
            self.perturbation_handler.apply_perturbation(
                HashPerturbation(seed, self.state.current_noise_scale, -1)
            )
            if batch_size == 1:
                fitness_neg = self.evaluator.evaluate(
                    gt_latents=samples[0]["latent"],
                    prompt_embeds=samples[0]["prompt_embeds"],
                    prompt_mask=samples[0].get("prompt_mask"),
                    seed=eval_seed,
                )
            else:
                fitness_neg = self.evaluator.evaluate_batch(
                    samples=samples, seed_base=eval_seed,
                )
            self.perturbation_handler.revert_to_original()

            fitness_diffs[seed] = fitness_pos.total - fitness_neg.total
            all_pos_fitness.append(fitness_pos.total)
            all_neg_fitness.append(fitness_neg.total)

            # Free GPU memory
            del samples
            torch.cuda.empty_cache()

        # ES gradient update
        num_updates = self.perturbation_handler.update_from_votes(
            seeds=seeds,
            fitness_diffs=fitness_diffs,
            update_scale=self.config.update_scale,
            noise_scale=self.state.current_noise_scale,
        )

        # Anneal noise
        self.state.current_noise_scale = max(
            self.state.current_noise_scale * self.config.noise_decay,
            self.config.noise_min,
        )

        # Track best fitness
        mean_fitness = sum(all_pos_fitness + all_neg_fitness) / len(all_pos_fitness + all_neg_fitness)
        self.state.fitness_history.append(mean_fitness)

        if mean_fitness > self.state.best_fitness:
            self.state.best_fitness = mean_fitness
            self.state.generations_without_improvement = 0
        else:
            self.state.generations_without_improvement += 1

        self.state.generation += 1

        return {
            "generation": gen,
            "mean_fitness": mean_fitness,
            "best_fitness": self.state.best_fitness,
            "mean_pos_fitness": sum(all_pos_fitness) / len(all_pos_fitness),
            "mean_neg_fitness": sum(all_neg_fitness) / len(all_neg_fitness),
            "mean_fitness_diff": sum(abs(d) for d in fitness_diffs.values()) / len(fitness_diffs),
            "num_updates": num_updates,
            "noise_scale": self.state.current_noise_scale,
            "gens_no_improve": self.state.generations_without_improvement,
        }

    def save_checkpoint(self, name: str) -> None:
        """Save evolved LoRA parameters and evolution state."""
        out = Path(self.config.output_dir) / "checkpoints"

        # Save evolved params (compatible with scd_inference.py LoRA loading)
        param_path = out / f"{name}_params.safetensors"
        evolved_state = self.perturbation_handler.state_dict()

        from safetensors.torch import save_file

        # Convert to float32 for compatibility
        save_dict = {k: v.float().cpu() for k, v in evolved_state.items()}
        save_file(save_dict, str(param_path))

        # Save evolution state
        state_path = out / f"{name}_state.json"
        with open(state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

        logger.info(f"Checkpoint saved: {param_path}")

    def save_full_lora(self, name: str) -> None:
        """Save a full LoRA checkpoint (all params, not just evolved ones).

        This produces a checkpoint directly loadable by scd_inference.py.
        """
        out = Path(self.config.output_dir) / "checkpoints"
        path = out / f"{name}.safetensors"

        from safetensors.torch import save_file

        # Collect all LoRA parameters from the model
        lora_state = {}
        for pname, param in self.scd_model.named_parameters():
            if "lora_A" in pname or "lora_B" in pname:
                lora_state[pname] = param.data.float().cpu()

        save_file(lora_state, str(path))
        logger.info(f"Full LoRA checkpoint saved: {path} ({len(lora_state)} tensors)")

    def load_checkpoint(self, name: str) -> None:
        """Load a previously saved evolution checkpoint."""
        out = Path(self.config.output_dir) / "checkpoints"

        param_path = out / f"{name}_params.safetensors"
        state_path = out / f"{name}_state.json"

        if param_path.exists():
            from safetensors.torch import load_file

            evolved_state = load_file(str(param_path))
            # Convert back to model dtype
            evolved_state = {k: v.to(self.dtype).to(self.device) for k, v in evolved_state.items()}
            self.perturbation_handler.load_state_dict(evolved_state)
            logger.info(f"Loaded evolved params from {param_path}")

        if state_path.exists():
            with open(state_path) as f:
                self.state = EvolutionState.from_dict(json.load(f))
            logger.info(f"Loaded evolution state: gen={self.state.generation}")

    def run(self) -> None:
        """Full evolution loop."""
        logger.info("=" * 60)
        logger.info("SCD Evolution — Starting")
        logger.info(f"  Population: {self.config.population_size} pairs")
        logger.info(f"  Generations: {self.config.num_generations}")
        logger.info(f"  AR frames: {self.config.ar_frames}")
        logger.info(f"  Inference steps: {self.config.num_inference_steps}")
        logger.info(f"  Evolved params: {self.perturbation_handler.num_params:,}")
        logger.info("=" * 60)

        # Evaluate baseline (unperturbed)
        baseline = self._evaluate_baseline()
        logger.info(f"Baseline fitness: {baseline}")

        if self.wandb_run:
            self.wandb_run.log(
                {
                    "baseline/total": baseline.total,
                    "baseline/fm_loss": baseline.fm_loss,
                    "baseline/latent_recon": baseline.latent_recon,
                    "baseline/temporal_coh": baseline.temporal_coh,
                },
                step=0,
            )

        start_gen = self.state.generation
        total_start = time.time()

        for gen_offset in range(self.config.num_generations - start_gen):
            gen = start_gen + gen_offset
            t0 = time.time()

            stats = self.run_generation()

            elapsed = time.time() - t0

            # Log
            if gen % self.config.log_every == 0 or gen == 0:
                logger.info(
                    f"Gen {stats['generation']:4d} | "
                    f"fitness={stats['mean_fitness']:.4f} (best={stats['best_fitness']:.4f}) | "
                    f"diff={stats['mean_fitness_diff']:.4f} | "
                    f"updates={stats['num_updates']} | "
                    f"noise={stats['noise_scale']:.5f} | "
                    f"{elapsed:.1f}s"
                )

            if self.wandb_run:
                self.wandb_run.log(
                    {
                        "evolution/mean_fitness": stats["mean_fitness"],
                        "evolution/best_fitness": stats["best_fitness"],
                        "evolution/mean_fitness_diff": stats["mean_fitness_diff"],
                        "evolution/num_updates": stats["num_updates"],
                        "evolution/noise_scale": stats["noise_scale"],
                        "evolution/gens_no_improve": stats["gens_no_improve"],
                        "evolution/time_per_gen": elapsed,
                    },
                    step=gen + 1,
                )

            # Checkpoint
            if (gen + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"gen_{gen + 1:04d}")
                self.save_full_lora(f"lora_evolved_gen_{gen + 1:04d}")

            # Early stopping (50 generations without improvement)
            if self.state.generations_without_improvement >= 50:
                logger.info(f"Early stopping: 50 generations without improvement at gen {gen}")
                break

        total_elapsed = time.time() - total_start
        logger.info(f"Evolution complete in {total_elapsed / 60:.1f} minutes")

        # Save final checkpoint
        self.save_checkpoint("final")
        self.save_full_lora("lora_evolved_final")

        # Final evaluation
        final = self._evaluate_baseline()
        logger.info(f"Final fitness: {final}")
        logger.info(f"Improvement: {final.total - baseline.total:.4f}")

        if self.wandb_run:
            self.wandb_run.log(
                {
                    "final/total": final.total,
                    "final/fm_loss": final.fm_loss,
                    "final/latent_recon": final.latent_recon,
                    "final/temporal_coh": final.temporal_coh,
                    "final/improvement": final.total - baseline.total,
                },
                step=self.state.generation,
            )
            self.wandb_run.finish()
