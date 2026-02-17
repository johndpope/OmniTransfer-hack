"""OmniTransfer: Unified Spatio-Temporal Video Transfer for LTX-2.

This module implements OmniTransfer (arXiv:2601.14250v1) adapted for LTX-2.
OmniTransfer enables unified spatio-temporal video transfer with three key components:
1. Task-aware Positional Bias (TPB) - RoPE offsets for spatial/temporal tasks
2. Reference-decoupled Causal Learning (RCL) - Separate branches for ref/target
3. Task-adaptive Multimodal Alignment (TMA) - MLLM with MetaQueries for semantic guidance

Quote: "OmniTransfer comprises three key components: 1) Task-aware Positional Bias that
applies distinct positional biases for different task types, 2) Reference-decoupled
Causal Learning that separates reference and target branches for improved efficiency,
3) Task-adaptive Multimodal Alignment that provides task-specific semantic guidance."
(Section 4, OmniTransfer paper)

Includes W&B integration for training visualization:
- Reconstruction comparisons (reference, target, prediction)
- Video comparisons
- Training metrics
"""

from ltx_trainer.omnitransfer.components import (
    OmniTransferTask,
    TaskAwarePositionalBias,
    ReferenceDecoupledCausalLearning,
    TaskAdaptiveMultimodalAlignment,
    MetaQueryBank,
    # Dynamic Identity Anchoring (Movie Weaver-style concept embeddings)
    ConceptEmbedding,
    ConceptEmbeddingConfig,
)
from ltx_trainer.omnitransfer.latent_constructor import ReferenceLatentConstructor
from ltx_trainer.omnitransfer.strategy import (
    OmniTransferConfig,
    OmniTransferStrategy,
    OmniTransferModelInputs,
    OmniTransferStage,
    get_omnitransfer_training_schedule,
)
from ltx_trainer.omnitransfer.visualization import (
    OmniTransferVisualizer,
    OmniTransferWandBCallback,
    ReconstructionSample,
    decode_latents_for_visualization,
)
from ltx_trainer.omnitransfer.training_callback import (
    OmniTransferTrainingCallback,
    create_omnitransfer_callback,
)
from ltx_trainer.omnitransfer.metaquery_tma import (
    MetaQueryTMA,
    MetaQueryTMAConfig,
    MetaQueryFeatureExtractor,
)
from ltx_trainer.omnitransfer.motion_encoder import (
    MotionEncoder,
    DualScaleMotionEncoder,
    MotionAugmenter,
)
from ltx_trainer.omnitransfer.geometric_decoder import (
    GeometricDecoder,
    compute_geometric_loss,
)

# Qwen VL integration for TMA (optional, requires transformers>=4.43)
try:
    from ltx_trainer.omnitransfer.qwen_vl_integration import (
        QwenVLConfig,
        QwenVLFeatureExtractor,
        QwenVLTMAIntegration,
        detect_qwen_model_type,
        get_task_template,
        TASK_TEMPLATES,
        HAS_QWEN_VL,
        HAS_QWEN3_VL,
    )
    _QWEN_VL_AVAILABLE = True
except ImportError:
    _QWEN_VL_AVAILABLE = False
    HAS_QWEN_VL = False
    HAS_QWEN3_VL = False

__all__ = [
    # Core components
    "OmniTransferTask",
    "TaskAwarePositionalBias",
    "ReferenceDecoupledCausalLearning",
    "TaskAdaptiveMultimodalAlignment",
    "MetaQueryBank",
    # Dynamic Identity Anchoring (Movie Weaver CVPR 2025)
    "ConceptEmbedding",
    "ConceptEmbeddingConfig",
    # Latent construction
    "ReferenceLatentConstructor",
    # Strategy
    "OmniTransferConfig",
    "OmniTransferStrategy",
    "OmniTransferModelInputs",
    "OmniTransferStage",
    "get_omnitransfer_training_schedule",
    # Visualization
    "OmniTransferVisualizer",
    "OmniTransferWandBCallback",
    "ReconstructionSample",
    "decode_latents_for_visualization",
    # Training callback
    "OmniTransferTrainingCallback",
    "create_omnitransfer_callback",
    # MetaQuery TMA (Facebook Research integration)
    "MetaQueryTMA",
    "MetaQueryTMAConfig",
    "MetaQueryFeatureExtractor",
    # 3DiMo Motion Encoder (arXiv:2602.03796v2)
    "MotionEncoder",
    "DualScaleMotionEncoder",
    "MotionAugmenter",
    "GeometricDecoder",
    "compute_geometric_loss",
    # Qwen VL TMA integration (optional)
    "QwenVLConfig",
    "QwenVLFeatureExtractor",
    "QwenVLTMAIntegration",
    "detect_qwen_model_type",
    "get_task_template",
    "TASK_TEMPLATES",
    "HAS_QWEN_VL",
    "HAS_QWEN3_VL",
    "_QWEN_VL_AVAILABLE",
]
