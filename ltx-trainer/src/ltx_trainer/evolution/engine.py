"""
SCD Evolution Engine — gradient-free fine-tuning for autoregressive quality.

Orchestrates the evolution loop:
  1. Load model (transformer → quantize → SCD wrap → inject manual LoRA)
  2. Load precomputed dataset (latents + conditions)
  3. For each generation:
     a. Generate antithetic perturbation pairs
     b. Evaluate fitness via AR rollout against GT
     c. Compute ES gradient and update decoder LoRA weights
     d. Anneal noise scale
  4. Save evolved checkpoint

NOTE: Uses manual LoRA injection (not PEFT) to avoid the PEFT+quanto circular
dependency. PEFT can't wrap QLinear (quantized); quanto quantizes LoRA's
nn.Linear adapters. Manual LoRA wraps quantized modules with bfloat16 params.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_trainer import logger
from ltx_trainer.evolution.fitness import ARRolloutEvaluator, FitnessNormalization, FitnessResult
from ltx_trainer.evolution.perturbation import (
    HashPerturbation,
    SelectiveLoRAPerturbation,
    generate_perturbation_seeds,
)


# ── Manual LoRA (PEFT-free, works with quantized modules) ──


class ManualLoRA(nn.Module):
    """Lightweight LoRA wrapper compatible with quantized (QLinear) base modules.

    Unlike PEFT, this doesn't replace the base module's type — it wraps it.
    The base module (possibly quantized) handles the main computation, and
    the LoRA params (always bfloat16) add a low-rank update on top.

    Parameter naming for perturbation handler compatibility:
      {parent_path}.lora_A  — nn.Parameter [rank, in_features]
      {parent_path}.lora_B  — nn.Parameter [out_features, rank]
    """

    def __init__(
        self,
        base_module: nn.Module,
        rank: int,
        alpha: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.base = base_module
        in_features = base_module.in_features
        out_features = base_module.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        self.scaling = (alpha or rank) / rank

        # Preserve attributes that downstream code may check
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base_out = self.base(x, *args, **kwargs)
        # LoRA path in bfloat16: (x @ A^T) @ B^T * scaling
        x_f = x.to(self.lora_A.dtype)
        lora_out = F.linear(F.linear(x_f, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


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
        """Load transformer → quantize → SCD wrap → inject manual LoRA.

        Order of operations (avoids PEFT+quanto circular dependency):
          1. Load bf16 transformer to CPU
          2. Read patchify_proj dimensions (before quantization changes types)
          3. Quantize with quanto (plain nn.Linear → QLinear)
          4. Move to GPU
          5. Wrap with LTXSCDModel (encoder/decoder split)
          6. Inject ManualLoRA on decoder attention layers
          7. Load LoRA weights from checkpoint

        This bypasses PEFT entirely. ManualLoRA wraps QLinear modules with
        bfloat16 low-rank parameters that the perturbation handler can evolve.
        """
        from ltx_core.model.transformer.scd_model import LTXSCDModel

        from ltx_trainer.model_loader import load_transformer

        logger.info(f"Loading transformer from {self.config.checkpoint}")
        transformer = load_transformer(self.config.checkpoint, device="cpu", dtype=self.dtype)

        # Read dimensions before quantization (patchify_proj may lose .in_features after quanto)
        if hasattr(transformer, "patchify_proj"):
            self._latent_channels = transformer.patchify_proj.in_features
        else:
            self._latent_channels = 128  # LTX-2 default

        # Quantize (no LoRA yet — plain nn.Linear modules, quanto handles them cleanly)
        if self.config.quantization != "none":
            from ltx_trainer.quantization import quantize_model

            logger.info(f"Quantizing with {self.config.quantization}")
            transformer = quantize_model(transformer, self.config.quantization, device="cuda:0")

        # Move to primary device
        transformer = transformer.to(self.device)

        # Wrap with SCD
        self.scd_model = LTXSCDModel(
            base_model=transformer,
            encoder_layers=self.config.encoder_layers,
            decoder_input_combine=self.config.decoder_combine,
        )
        self.scd_model.eval()

        # Inject ManualLoRA on decoder blocks and load checkpoint weights
        if self.config.lora_path:
            self._inject_manual_lora()

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

    def _inject_manual_lora(self) -> None:
        """Inject ManualLoRA wrappers on decoder blocks and load checkpoint weights.

        Parses the PEFT-format LoRA checkpoint to determine:
          - rank (from lora_A tensor shape)
          - which modules to wrap (from checkpoint keys)
          - weight values (loaded into ManualLoRA.lora_A / lora_B)

        Works AFTER quantization — ManualLoRA wraps QLinear modules cleanly.
        """
        from safetensors.torch import load_file

        state_dict = load_file(self.config.lora_path)
        logger.info(f"Loading LoRA from {self.config.lora_path} ({len(state_dict)} tensors)")

        # Detect LoRA rank from any lora_A weight
        rank = None
        alpha = None
        for key, tensor in state_dict.items():
            if "lora_A" in key and "weight" in key:
                rank = tensor.shape[0]
                break
            elif "lora_A" in key and tensor.dim() == 2:
                rank = tensor.shape[0]
                break

        if rank is None:
            raise ValueError(f"Could not detect LoRA rank from {self.config.lora_path}")

        alpha = rank  # Assume alpha == rank (standard for SCD training)
        logger.info(f"Detected LoRA rank={rank}, alpha={alpha}")

        # Parse checkpoint keys to build (block_idx, module_path) → (lora_A, lora_B) mapping
        # SCD training saves keys in multiple formats:
        #   diffusion_model.base_model.transformer_blocks.{idx}.{path}.lora_A.weight
        #   diffusion_model.encoder_blocks.{idx}.{path}.lora_A.weight  (duplicates)
        #   diffusion_model.decoder_blocks.{idx}.{path}.lora_A.weight  (duplicates)
        #   base_model.model.transformer_blocks.{idx}.{path}.lora_A.default.weight  (PEFT)
        # We only use transformer_blocks.N keys (canonical), ignore encoder/decoder duplicates.
        _KEY_PATTERN = re.compile(
            r"(?:diffusion_model\.)?(?:base_model\.(?:model\.)?)?transformer_blocks\.(\d+)\."
            r"(.*?)\.lora_([AB])(?:\.default)?\.weight$"
        )
        # Also match keys without .weight suffix (raw param format)
        _KEY_PATTERN_RAW = re.compile(
            r"(?:diffusion_model\.)?(?:base_model\.(?:model\.)?)?transformer_blocks\.(\d+)\."
            r"(.*?)\.lora_([AB])$"
        )

        lora_weights: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        for key, tensor in state_dict.items():
            m = _KEY_PATTERN.match(key) or _KEY_PATTERN_RAW.match(key)
            if m is None:
                continue
            block_idx = int(m.group(1))
            module_path = m.group(2)  # e.g. "attn.to_q"
            ab = m.group(3)  # "A" or "B"

            pair_key = (block_idx, module_path)
            if pair_key not in lora_weights:
                lora_weights[pair_key] = {}
            lora_weights[pair_key][f"lora_{ab}"] = tensor

        logger.info(f"Found LoRA weights for {len(lora_weights)} modules")

        # Filter to decoder blocks only (>= encoder_layers)
        decoder_start = self.config.encoder_layers
        decoder_pairs = {k: v for k, v in lora_weights.items() if k[0] >= decoder_start}
        encoder_pairs = {k: v for k, v in lora_weights.items() if k[0] < decoder_start}

        logger.info(
            f"  Decoder blocks ({decoder_start}-47): {len(decoder_pairs)} modules"
        )
        if encoder_pairs:
            logger.info(
                f"  Encoder blocks (0-{decoder_start - 1}): {len(encoder_pairs)} modules (loaded but not evolved)"
            )

        # Inject ManualLoRA wrappers — both encoder and decoder blocks get LoRA
        # (encoder LoRA contributes to quality; only decoder LoRA is perturbed)
        injected = 0
        for (block_idx, module_path), weights in sorted(lora_weights.items()):
            if "lora_A" not in weights or "lora_B" not in weights:
                logger.warning(f"Skipping incomplete LoRA pair: block {block_idx}, {module_path}")
                continue

            # Navigate to the target module inside the SCD model
            # Path: scd_model.base_model.transformer_blocks[block_idx].{module_path}
            try:
                block = self.scd_model.base_model.transformer_blocks[block_idx]
                parts = module_path.split(".")
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                child_name = parts[-1]
                base_module = getattr(parent, child_name)
            except (AttributeError, IndexError) as e:
                logger.warning(f"Could not find module: block {block_idx}, {module_path}: {e}")
                continue

            # Create ManualLoRA wrapper
            lora = ManualLoRA(
                base_module=base_module,
                rank=rank,
                alpha=alpha,
                dtype=self.dtype,
            )

            # Load weights (move to same device as model)
            lora.lora_A.data.copy_(weights["lora_A"].to(self.dtype))
            lora.lora_B.data.copy_(weights["lora_B"].to(self.dtype))

            # Move entire ManualLoRA to model device
            lora = lora.to(self.device)

            # Replace the module in the block
            setattr(parent, child_name, lora)
            injected += 1

        logger.info(f"Injected {injected} ManualLoRA wrappers (rank={rank})")

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
        sample_data = torch.load(self.dataset_samples[0]["latent_path"], weights_only=True)
        if isinstance(sample_data, dict):
            # Try common key names: "latents" (ditto), "latent", "video_latent"
            for key in ("latents", "latent", "video_latent"):
                if key in sample_data:
                    sample_latent = sample_data[key]
                    break
            else:
                sample_latent = next(v for v in sample_data.values() if hasattr(v, "shape") and v.dim() >= 4)
        else:
            sample_latent = sample_data

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
            distilled=self.config.distilled,
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

        latent_data = torch.load(sample["latent_path"], weights_only=True)
        if isinstance(latent_data, dict):
            # Try common key names: "latents" (ditto/scrya), "latent", "video_latent"
            for key in ("latents", "latent", "video_latent"):
                if key in latent_data:
                    latent = latent_data[key]
                    break
            else:
                latent = next(v for v in latent_data.values() if hasattr(v, "shape") and v.dim() >= 4)
        else:
            latent = latent_data
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)  # [C,F,H,W] -> [1,C,F,H,W]

        condition = torch.load(sample["condition_path"], weights_only=True)
        if isinstance(condition, dict):
            # Try common key names: "video_prompt_embeds" (ditto/scrya), "prompt_embeds"
            prompt_embeds = condition.get(
                "video_prompt_embeds",
                condition.get("prompt_embeds", next(
                    v for v in condition.values() if hasattr(v, "shape") and v.dim() >= 2
                )),
            )
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

        # Save evolution state (includes normalization for resumption)
        state_dict = self.state.to_dict()
        if self.evaluator and self.evaluator.normalization:
            norm = self.evaluator.normalization
            state_dict["normalization"] = {
                "fm_scale": norm.fm_scale,
                "recon_scale": norm.recon_scale,
                "tcoh_scale": norm.tcoh_scale,
                "lpips_scale": norm.lpips_scale,
                "ssim_scale": norm.ssim_scale,
            }

        state_path = out / f"{name}_state.json"
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info(f"Checkpoint saved: {param_path}")

    def save_full_lora(self, name: str) -> None:
        """Save a full LoRA checkpoint in PEFT-compatible format.

        Converts ManualLoRA parameter names to PEFT format so the checkpoint
        is directly loadable by scd_inference.py (which uses PEFT):

        ManualLoRA: base_model.transformer_blocks.32.attn.to_q.lora_A
        PEFT:       base_model.model.transformer_blocks.32.attn.to_q.lora_A.default.weight
        """
        out = Path(self.config.output_dir) / "checkpoints"
        path = out / f"{name}.safetensors"

        from safetensors.torch import save_file

        # Collect all LoRA parameters and convert to PEFT key format
        lora_state = {}
        for pname, param in self.scd_model.named_parameters():
            if "lora_A" not in pname and "lora_B" not in pname:
                continue
            # Skip base module params inside ManualLoRA (e.g. .base.weight)
            if ".base." in pname:
                continue

            # Convert: base_model.transformer_blocks.N.X.lora_A
            #       → base_model.model.transformer_blocks.N.X.lora_A.default.weight
            peft_key = pname
            if peft_key.startswith("base_model."):
                peft_key = "base_model.model." + peft_key[len("base_model."):]
            peft_key = peft_key + ".default.weight"

            lora_state[peft_key] = param.data.float().cpu()

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
                state_data = json.load(f)
            self.state = EvolutionState.from_dict(state_data)
            logger.info(f"Loaded evolution state: gen={self.state.generation}")

            # Restore normalization if saved
            if "normalization" in state_data and self.evaluator:
                nd = state_data["normalization"]
                self.evaluator.normalization = FitnessNormalization(
                    fm_scale=nd["fm_scale"],
                    recon_scale=nd["recon_scale"],
                    tcoh_scale=nd["tcoh_scale"],
                    lpips_scale=nd.get("lpips_scale", 1.0),
                    ssim_scale=nd.get("ssim_scale", 1.0),
                )
                logger.info("Restored fitness normalization from checkpoint")

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

        # Evaluate baseline (unperturbed) and compute normalization
        baseline = self._evaluate_baseline()
        logger.info(f"Baseline fitness (raw): {baseline}")

        # Compute normalization from baseline so each component ≈ 1.0
        norm = FitnessNormalization.from_baseline(baseline)
        self.evaluator.normalization = norm
        logger.info(
            f"Fitness normalization: fm×{norm.fm_scale:.3f}, recon×{norm.recon_scale:.3f}, "
            f"tcoh×{norm.tcoh_scale:.3f}, lpips×{norm.lpips_scale:.3f}, ssim×{norm.ssim_scale:.3f}"
        )

        # Re-evaluate baseline WITH normalization to show calibrated starting point
        baseline_normed = self._evaluate_baseline()
        logger.info(f"Baseline fitness (normalized): {baseline_normed}")

        if self.wandb_run:
            self.wandb_run.log(
                {
                    "baseline/total": baseline_normed.total,
                    "baseline/total_raw": baseline.total,
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
