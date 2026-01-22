"""Training strategies for different conditioning modes.
This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio)
- Video-to-video training (IC-LoRA mode with reference videos)
- OmniTransfer training (unified spatio-temporal video transfer)
Each strategy encapsulates the specific logic for preparing model inputs and computing loss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    VIDEO_SCALE_FACTORS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy

# Lazy import to avoid circular dependency with omnitransfer
if TYPE_CHECKING:
    from ltx_trainer.omnitransfer.strategy import OmniTransferConfig, OmniTransferStrategy

    # Type alias for all strategy config types (only for type checking)
    TrainingStrategyConfig = TextToVideoConfig | VideoToVideoConfig | OmniTransferConfig


def _get_omnitransfer_types() -> tuple:
    """Lazy import of OmniTransfer types to avoid circular imports."""
    from ltx_trainer.omnitransfer.strategy import OmniTransferConfig, OmniTransferStrategy  # noqa: PLC0415
    return OmniTransferConfig, OmniTransferStrategy

__all__ = [
    "DEFAULT_FPS",
    "VIDEO_SCALE_FACTORS",
    "ModelInputs",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfigBase",
    "VideoToVideoConfig",
    "VideoToVideoStrategy",
    "get_training_strategy",
    # OmniTransfer types: import from ltx_trainer.omnitransfer.strategy directly
    # to avoid circular imports
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.
    The strategy is determined by the `name` field in the configuration.
    Args:
        config: Strategy-specific configuration with a `name` field
    Returns:
        The appropriate training strategy instance
    Raises:
        ValueError: If strategy name is not supported
    """
    # Lazy import OmniTransfer to avoid circular imports
    OmniTransferConfig, OmniTransferStrategy = _get_omnitransfer_types()

    match config:
        case TextToVideoConfig():
            strategy = TextToVideoStrategy(config)
        case VideoToVideoConfig():
            strategy = VideoToVideoStrategy(config)
        case OmniTransferConfig():
            strategy = OmniTransferStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    audio_mode = "(audio enabled 🔈)" if getattr(config, "with_audio", False) else "(audio disabled 🔇)"
    logger.debug(f"🎯 Using {strategy.__class__.__name__} training strategy {audio_mode}")
    return strategy
