"""
SCD Evolution Strategy — gradient-free fine-tuning for autoregressive quality.

Uses EggRoll-style hash-based perturbation with antithetic sampling to optimize
SCD decoder LoRA weights for multi-frame autoregressive rollout quality.

Phase 1 (prerequisite): Gradient-based SCD LoRA training
Phase 2 (this module): Evolution fine-tuning for AR inference quality
"""

from ltx_trainer.evolution.perturbation import (
    HashPerturbation,
    SelectiveLoRAPerturbation,
    generate_perturbation_seeds,
)
from ltx_trainer.evolution.fitness import ARRolloutEvaluator, FitnessResult
from ltx_trainer.evolution.engine import SCDEvolutionEngine

__all__ = [
    "HashPerturbation",
    "SelectiveLoRAPerturbation",
    "generate_perturbation_seeds",
    "ARRolloutEvaluator",
    "FitnessResult",
    "SCDEvolutionEngine",
]
