"""
Hash-based perturbation for SCD decoder LoRA evolution.

Adapted from PixelGen's EggRoll-style perturbation (egg.c pattern).
Key features:
- No storage of perturbation matrices (regenerate from seeds)
- GPU-native operations (torch.jit compiled)
- Antithetic sampling support (+/-eps pairs)
- Selective targeting of decoder LoRA parameters only
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from ltx_trainer import logger


# ── JIT-compiled hash functions (verbatim from PixelGen) ──


@torch.jit.script
def hash_rng(seed: int, idx: torch.Tensor) -> torch.Tensor:
    """MurmurHash3 finalizer for GPU-native pseudo-random numbers."""
    x = (seed + idx * 0x9E3779B9).to(torch.int64)
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x.to(torch.int32)


@torch.jit.script
def noise_from_hash(seed: int, idx: torch.Tensor) -> torch.Tensor:
    """Uniform [-1, 1] noise from hash values."""
    r = hash_rng(seed, idx)
    sign = torch.where(
        r & 1 == 1,
        torch.ones_like(r, dtype=torch.float32),
        -torch.ones_like(r, dtype=torch.float32),
    )
    magnitude = torch.abs(r.float()) / 2147483647.0
    return sign * magnitude


@torch.jit.script
def gaussian_from_hash(seed: int, idx: torch.Tensor) -> torch.Tensor:
    """Box-Muller Gaussian noise from hash values."""
    r1 = hash_rng(seed, idx)
    r2 = hash_rng(seed + 1000000, idx)
    u1 = (r1.abs().float() / 2147483647.0).clamp(1e-7, 1.0)
    u2 = r2.abs().float() / 2147483647.0
    pi = 3.14159265358979323846
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * pi * u2)


# ── Perturbation metadata ──


@dataclass
class HashPerturbation:
    """Stores perturbation metadata (seed, scale, direction).

    The actual perturbation values are computed on-the-fly from the seed,
    avoiding storage of full perturbation matrices.
    """

    seed: int
    scale: float
    direction: int  # +1 or -1 for antithetic pairs

    def __repr__(self) -> str:
        sign = "+" if self.direction > 0 else "-"
        return f"Perturbation(seed={self.seed}, scale={self.scale:.4f}, dir={sign})"


def generate_perturbation_seeds(
    population_size: int,
    base_seed: int | None = None,
) -> list[int]:
    """Generate unique seeds for a population of perturbations.

    Uses time-based seeding with large prime offsets to ensure independence.
    """
    if base_seed is None:
        base_seed = int(time.time() * 1000) % (2**31)
    return [base_seed + i * 999983 for i in range(population_size)]


def apply_perturbation_to_param(
    param: torch.Tensor,
    seed: int,
    scale: float,
    direction: int,
    use_gaussian: bool = False,
) -> torch.Tensor:
    """Apply hash-based perturbation to a parameter tensor.

    Returns a new tensor (does not modify input in-place).
    """
    flat_size = param.numel()
    idx = torch.arange(flat_size, device=param.device, dtype=torch.int64)
    noise = gaussian_from_hash(seed, idx) if use_gaussian else noise_from_hash(seed, idx)
    noise = noise.view(param.shape).to(param.dtype)
    return param + (noise * scale * direction)


# ── Decoder LoRA parameter selector ──

# Matches transformer_blocks.{32-47}.*.lora_A or lora_B
_DECODER_LORA_PATTERN = re.compile(r"transformer_blocks\.(\d+)\..*lora_[AB]")


def is_decoder_lora_param(name: str, encoder_layers: int = 32) -> bool:
    """Check if a parameter name belongs to a decoder LoRA layer."""
    m = _DECODER_LORA_PATTERN.search(name)
    if m is None:
        return False
    block_idx = int(m.group(1))
    return block_idx >= encoder_layers


class SelectiveLoRAPerturbation:
    """Manages perturbations for SCD decoder LoRA parameters.

    Only targets lora_A / lora_B weights in decoder blocks (index >= encoder_layers).
    Stores original parameter values for revert / ES gradient update.
    Uses Adam-style momentum for gradient smoothing across generations.
    """

    def __init__(
        self,
        model: nn.Module,
        encoder_layers: int = 32,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_gaussian: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        self.model = model
        self.encoder_layers = encoder_layers
        self.device = device
        self.dtype = dtype
        self.use_gaussian = use_gaussian

        # Adam hyperparameters
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.adam_step = 0  # For bias correction

        # Find decoder LoRA parameters
        self.evolvable_params: dict[str, nn.Parameter] = {}
        self.original_values: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if is_decoder_lora_param(name, encoder_layers):
                self.evolvable_params[name] = param
                self.original_values[name] = param.data.clone().detach()

        # Noise cache: seed → {param_name → noise_tensor}
        # Populated during apply_perturbation, consumed in update_from_votes
        self._noise_cache: dict[int, dict[str, torch.Tensor]] = {}

        # Adam state: first and second moment estimates per parameter
        self._m: dict[str, torch.Tensor] = {}  # First moment (momentum)
        self._v: dict[str, torch.Tensor] = {}  # Second moment (RMS)
        for name, val in self.original_values.items():
            self._m[name] = torch.zeros_like(val)
            self._v[name] = torch.zeros_like(val)

        self.num_params = sum(p.numel() for p in self.evolvable_params.values())
        logger.info(
            f"[SelectiveLoRAPerturbation] Evolving {len(self.evolvable_params)} "
            f"parameters ({self.num_params:,} values) in decoder blocks >= {encoder_layers}"
        )

    def apply_perturbation(self, perturbation: HashPerturbation) -> None:
        """Apply perturbation to all evolvable parameters in-place.

        Caches the unit noise (direction-independent) for reuse in update_from_votes.
        Only the +ε direction's noise is cached (noise is symmetric for antithetic pairs).
        """
        cache_noise = perturbation.direction > 0  # Only cache for +ε (first of pair)
        seed_cache: dict[str, torch.Tensor] = {}

        for i, (name, param) in enumerate(self.evolvable_params.items()):
            param_seed = perturbation.seed + i * 10007
            flat_size = self.original_values[name].numel()
            idx = torch.arange(flat_size, device=param.device, dtype=torch.int64)
            noise = (
                gaussian_from_hash(param_seed, idx)
                if self.use_gaussian
                else noise_from_hash(param_seed, idx)
            )
            noise = noise.view(self.original_values[name].shape).to(self.original_values[name].dtype)

            if cache_noise:
                seed_cache[name] = noise  # Unit noise (no scale/direction)

            param.data.copy_(self.original_values[name] + noise * perturbation.scale * perturbation.direction)

        if cache_noise:
            self._noise_cache[perturbation.seed] = seed_cache

    def revert_to_original(self) -> None:
        """Revert all parameters to their pre-perturbation values."""
        for name, param in self.evolvable_params.items():
            param.data.copy_(self.original_values[name])

    def update_from_votes(
        self,
        seeds: list[int],
        fitness_diffs: dict[int, float],
        update_scale: float,
        noise_scale: float,
    ) -> int:
        """Apply ES gradient update with Adam momentum.

        Computes raw ES gradient: g = sum((F+ - F-) * eps / (2*sigma)),
        then applies Adam-style momentum/RMS for smoothing across generations.

        Uses cached noise from apply_perturbation when available, falling back
        to regeneration from hash seeds. Caching avoids ~768 × len(seeds)
        redundant GPU noise generation calls per generation.

        Args:
            seeds: Seeds that were evaluated this generation.
            fitness_diffs: Map seed -> (fitness_pos - fitness_neg).
                Should be pre-shaped (rank-based utilities) for best results.
            update_scale: Learning rate for weight update.
            noise_scale: Perturbation scale (sigma) for gradient normalization.

        Returns:
            Number of contributing seeds (non-zero fitness diff).
        """
        gradient_estimate: dict[str, torch.Tensor] = {
            name: torch.zeros_like(val) for name, val in self.original_values.items()
        }
        num_contributing = 0

        for seed in seeds:
            diff = fitness_diffs.get(seed, 0.0)
            if abs(diff) < 1e-8:
                continue
            num_contributing += 1

            # Try to use cached noise first (much faster than regenerating)
            cached = self._noise_cache.get(seed)

            for i, name in enumerate(self.evolvable_params.keys()):
                if cached is not None and name in cached:
                    noise = cached[name]
                else:
                    param_seed = seed + i * 10007
                    flat_size = self.original_values[name].numel()
                    idx = torch.arange(flat_size, device=self.device, dtype=torch.int64)
                    noise = (
                        gaussian_from_hash(param_seed, idx)
                        if self.use_gaussian
                        else noise_from_hash(param_seed, idx)
                    )
                    noise = noise.view(self.original_values[name].shape).to(self.original_values[name].dtype)
                gradient_estimate[name] += diff * noise / (2.0 * noise_scale)

        # Clear cache after use
        self._noise_cache.clear()

        if num_contributing > 0:
            # Normalize raw gradient by number of contributing seeds
            for name in gradient_estimate:
                gradient_estimate[name] /= num_contributing

            # Adam update
            self.adam_step += 1
            t = self.adam_step
            bc1 = 1.0 - self.beta1 ** t  # Bias correction for first moment
            bc2 = 1.0 - self.beta2 ** t  # Bias correction for second moment

            for name in self.evolvable_params.keys():
                grad = gradient_estimate[name]

                # Update moments
                self._m[name].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                self._v[name].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                # Bias-corrected estimates
                m_hat = self._m[name] / bc1
                v_hat = self._v[name] / bc2

                # Adam step (gradient ASCENT — higher fitness = better)
                step = update_scale * m_hat / (v_hat.sqrt() + self.adam_eps)

                # Clip step to prevent explosion
                step_norm = torch.norm(step).item()
                param_norm = torch.norm(self.original_values[name]).item()
                max_step = max(param_norm * 0.1, 1e-3)  # Cap at 10% of param magnitude
                if step_norm > max_step:
                    step = step * (max_step / step_norm)

                self.original_values[name] += step

            # Copy updated originals to model
            for name, param in self.evolvable_params.items():
                param.data.copy_(self.original_values[name])

        return num_contributing

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return evolved parameter values + Adam state (for checkpointing)."""
        sd = {name: val.clone() for name, val in self.original_values.items()}
        # Save Adam state with prefixed keys
        for name in self._m:
            sd[f"__adam_m__{name}"] = self._m[name].clone()
            sd[f"__adam_v__{name}"] = self._v[name].clone()
        return sd

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load previously evolved parameter values + Adam state."""
        for name, val in state_dict.items():
            if name.startswith("__adam_m__"):
                param_name = name[len("__adam_m__"):]
                if param_name in self._m:
                    self._m[param_name].copy_(val)
            elif name.startswith("__adam_v__"):
                param_name = name[len("__adam_v__"):]
                if param_name in self._v:
                    self._v[param_name].copy_(val)
            elif name in self.original_values:
                self.original_values[name].copy_(val)
        for name, param in self.evolvable_params.items():
            param.data.copy_(self.original_values[name])

    def get_param_stats(self) -> dict[str, dict[str, float]]:
        """Statistics about evolvable parameters."""
        stats = {}
        for name, param in self.evolvable_params.items():
            d = param.data.float()
            stats[name] = {
                "mean": d.mean().item(),
                "std": d.std().item(),
                "numel": param.numel(),
            }
        return stats
