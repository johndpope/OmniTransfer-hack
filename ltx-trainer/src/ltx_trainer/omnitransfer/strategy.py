"""OmniTransfer Training Strategy for LTX-2.

This module implements the complete OmniTransfer training strategy that combines:
1. Reference Latent Construction (Section 4.1)
2. Task-aware Positional Bias (Section 4.2)
3. Reference-decoupled Causal Learning (Section 4.3)
4. Task-adaptive Multimodal Alignment (Section 4.4)

Training follows a multi-stage approach:
Quote: "The training process is divided into three sequential stages with distinct
optimization objectives. In the first stage, we train the DiT blocks with TPB and
RCL for 10,000 steps. In the second stage, we freeze the DiT blocks and train only
the TMA connector for 2,000 steps. In the third stage, we jointly fine-tune all
components for 5,000 steps." (Section 5.1)

Includes W&B integration for logging:
- Training metrics (loss, learning rate)
- Reconstruction visualizations (reference, target, prediction)
- Video comparisons at specified intervals

References: OmniTransfer paper (arXiv:2601.14250v1, Jan 20, 2026)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.omnitransfer.components import (
    OmniTransferTask,
    TaskAwarePositionalBias,
    TaskAwarePositionalBiasConfig,
    # Dynamic Identity Anchoring (Movie Weaver CVPR 2025)
    ConceptEmbedding,
    ConceptEmbeddingConfig,
)
from ltx_trainer.omnitransfer.latent_constructor import (
    ConstructedLatents,
    ReferenceLatentConstructor,
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optional LPIPS for perceptual loss
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None

# Optional InsightFace for identity loss
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None

# Optional CLIP for semantic identity loss (Grok recommendation)
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    open_clip = None

# Optional torchvision for VGG style features (Grok recommendation)
try:
    from torchvision import models as tv_models
    from torchvision import transforms as tv_transforms
    VGG_AVAILABLE = True
except ImportError:
    VGG_AVAILABLE = False
    tv_models = None
    tv_transforms = None


class OmniTransferStage(Enum):
    """Training stages for OmniTransfer.

    Quote: "training process is divided into three sequential stages" (Section 5.1)
    """
    # Stage 1: Train DiT blocks with TPB/RCL (10k steps)
    IN_CONTEXT = "in_context"
    # Stage 2: Freeze DiT, train TMA connector (2k steps)
    CONNECTOR = "connector"
    # Stage 3: Joint fine-tuning (5k steps)
    JOINT = "joint"


# =============================================================================
# Task Type Categories (per OmniTransfer paper Table 1)
# =============================================================================
# Appearance Transfer (T2V): Input = V_ref + prompt p, NO target image
# - ID Transfer: Identity from V_ref, scene from prompt
# - Style Transfer: Style from V_ref, content from prompt
# - Movie Weaver: Multi-concept personalization with anchored prompts
APPEARANCE_TASKS = {"identity_preservation", "style_transfer", "id", "style", "movie_weaver"}

# Temporal Transfer (I2V): Input = V_ref + Image I
# - Motion Transfer: Motion from V_ref, starting from image I
# - Camera Movement: Camera motion from V_ref, starting from image I
# - Effect Transfer: Effect from V_ref, starting from image I
TEMPORAL_TASKS = {"motion_transfer", "camera_transfer", "effect_transfer",
                  "motion", "camera", "effect", "pose_reenactment",
                  "action_customization", "scene_composition"}


def is_temporal_task(task_type: str) -> bool:
    """Check if task requires I2V mode (image + motion reference).

    Per paper Table 1:
    - Temporal Transfer (I2V): Motion, Camera, Effect → V_ref + Image I
    - Appearance Transfer (T2V): ID, Style → V_ref + prompt (no image)
    """
    return task_type.lower() in TEMPORAL_TASKS


def is_appearance_task(task_type: str) -> bool:
    """Check if task is T2V mode (reference + prompt only, no target image).

    Per paper Table 1:
    - Appearance Transfer (T2V): ID, Style → V_ref + prompt p
    """
    return task_type.lower() in APPEARANCE_TASKS


class OmniTransferConfig(TrainingStrategyConfigBase):
    """Configuration for OmniTransfer training strategy.

    This configuration controls all aspects of OmniTransfer training including
    task type, component enables, and stage-specific settings.
    """

    name: Literal["omnitransfer"] = "omnitransfer"

    # Task configuration
    task_type: str = Field(
        default="motion_transfer",
        description="Type of transfer task. Options: motion_transfer, pose_reenactment, "
        "action_customization, style_transfer, identity_preservation, scene_composition",
    )

    # Multi-task training configuration
    multi_task_mode: bool = Field(
        default=False,
        description="Enable multi-task training for unified OmniTransfer model. "
        "When enabled, samples tasks from task_types list each batch.",
    )

    task_types: list[str] = Field(
        default=["identity_preservation"],
        description="List of task types to train on in multi-task mode. "
        "Each task should have corresponding data in preprocessed_data_root/<task_type>/",
    )

    task_sampling: str = Field(
        default="uniform",
        description="Task sampling strategy: 'uniform' (equal probability), "
        "'weighted' (use task_weights), 'round_robin' (cycle through tasks)",
    )

    task_weights: dict[str, float] = Field(
        default={},
        description="Optional weights for weighted task sampling. "
        "Tasks not specified default to 1.0. Example: {'identity_preservation': 2.0}",
    )

    # Component enables
    enable_tpb: bool = Field(
        default=True,
        description="Enable Task-aware Positional Bias (TPB) for task-specific "
        "RoPE offsets. Quote: 'Task-aware Positional Bias applies distinct "
        "positional biases based on the task type' (Section 4.2)",
    )

    enable_rcl: bool = Field(
        default=True,
        description="Enable Reference-decoupled Causal Learning (RCL) for "
        "separate ref/target attention branches. Quote: 'RCL decouples the "
        "reference and target branches in attention computation' (Section 4.3)",
    )

    enable_tma: bool = Field(
        default=True,
        description="Enable Task-adaptive Multimodal Alignment (TMA) for "
        "MLLM-guided semantic features. Quote: 'TMA leverages a multimodal LLM "
        "to provide semantic guidance' (Section 4.4)",
    )

    # First frame conditioning
    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    # Data directories
    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Directory name for reference video latents",
    )

    # I2V (Image-to-Video) mode - for pose-free animation
    i2v_mode: bool = Field(
        default=False,
        description="Enable I2V mode for pose-free animation. When True, uses first_frame_latents "
        "as the static image to animate, and reference_latents as the motion source. "
        "Per OmniTransfer website: 'Driven static images by directly injecting fluid, "
        "complex motion from unseen sources without explicit pose extraction.'",
    )

    first_frame_latents_dir: str = Field(
        default="first_frame_latents",
        description="Directory name for first-frame latents (I2V mode). Contains single-frame "
        "latents [128, 1, H, W] representing the static image to animate.",
    )

    # Self-reconstruction mode - video reconstructs itself from first frame
    self_reconstruction_mode: bool = Field(
        default=False,
        description="Enable self-reconstruction training where each video reconstructs itself. "
        "In this mode, reference_latents == latents (same video), and first_frame_latents "
        "provides the starting image. Model learns to animate first frame to match full video. "
        "Useful when cross-subject ground truth is unavailable.",
    )

    # TPB configuration
    tpb_max_pos: list[int] = Field(
        default=[20, 2048, 2048],
        description="Maximum position values [time, height, width] for RoPE normalization",
    )

    tpb_theta: float = Field(
        default=10000.0,
        description="Theta parameter for RoPE frequency computation",
    )

    # RCL configuration
    rcl_ref_timestep: float = Field(
        default=0.0,
        description="Fixed timestep for reference branch (t=0 means noise-free). "
        "Quote: 'reference branch adopts a fixed t=0' (Section 4.3)",
    )

    # TMA configuration
    tma_num_queries: int = Field(
        default=8,
        description="Number of MetaQueries per task for TMA aggregation",
    )

    tma_connector_layers: int = Field(
        default=3,
        description="Number of MLP layers in TMA connector. "
        "Quote: 'projected through a three-layer MLP connector' (Section 4.4)",
    )

    # ==========================================================================
    # Cached TMA Features (for VRAM-efficient training)
    # ==========================================================================
    # Pre-compute Qwen VL features to avoid loading the ~14GB MLLM during training.
    # Run scripts/compute_qwen_vl_features.py first to generate the cached features.
    # ==========================================================================

    use_cached_tma_features: bool = Field(
        default=False,
        description="Use pre-computed Qwen VL features from qwen_vl_features/. "
        "When True, skips loading Qwen VL model entirely, saving ~14GB VRAM. "
        "Requires running scripts/compute_qwen_vl_features.py first.",
    )

    tma_features_dir: str = Field(
        default="qwen_vl_features",
        description="Directory name for cached TMA features (relative to preprocessed_data_root). "
        "Each file should contain: qwen_features [seq_len, hidden_dim], task_type, caption.",
    )

    tma_mllm_hidden_dim: int = Field(
        default=3584,
        description="Hidden dimension of the MLLM used for TMA features. "
        "Qwen 2B: 1536, Qwen2.5-3B: 2048, Qwen 7B: 3584, Qwen 14B/32B: 5120, Qwen 72B: 8192. "
        "Must match the model used in compute_qwen_vl_features.py.",
    )

    # ==========================================================================
    # 3-Stage Training (per OmniTransfer paper Section 5.1)
    # ==========================================================================
    # Quote: "The training process is divided into three sequential stages:
    # Stage 1: Train DiT blocks with TPB and RCL for 10,000 steps
    # Stage 2: Freeze DiT blocks and train only TMA connector for 2,000 steps
    # Stage 3: Jointly fine-tune all components for 5,000 steps"
    # ==========================================================================

    training_stage: int = Field(
        default=1,
        description="Training stage (1-3). "
        "Stage 1: Train DiT + TPB + RCL (10k steps, TMA disabled). "
        "Stage 2: Freeze DiT, train TMA connector only (2k steps). "
        "Stage 3: Joint fine-tuning of all components (5k steps).",
        ge=1,
        le=3,
    )

    # MetaQuery MLLM configuration (Facebook Research integration)
    use_metaquery_mllm: bool = Field(
        default=False,
        description="Use Facebook's MetaQuery MLLM for semantic feature extraction. "
        "If False, uses simple learnable TMA. Requires metaquery package.",
    )

    metaquery_model: str = Field(
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        description="HuggingFace model name for MetaQuery MLLM backbone. "
        "Options: 'llava-hf/llava-onevision-qwen2-7b-ov-hf', 'Qwen/Qwen2.5-VL-7B-Instruct'",
    )

    metaquery_load_in_8bit: bool = Field(
        default=True,
        description="Load MetaQuery MLLM in 8-bit for memory efficiency. "
        "Recommended for GPUs with <48GB VRAM.",
    )

    metaquery_freeze: bool = Field(
        default=True,
        description="Freeze MetaQuery MLLM backbone (inference-only). "
        "Set False to fine-tune MLLM jointly (requires more VRAM).",
    )

    # ==========================================================================
    # Dynamic Identity Anchoring (Movie Weaver CVPR 2025)
    # ==========================================================================
    # Based on "Movie Weaver: Tuning-Free Multi-Concept Video Personalization
    # with Anchored Prompts" (CVPR 2025) which achieved 98.2% vs 90.5% accuracy
    # in identity separation using concept embeddings.
    #
    # Quote: "We apply the same concept embedding to the entire set of vision
    # tokens from one image, rather than different embeddings per token."
    # ==========================================================================

    enable_concept_embeddings: bool = Field(
        default=True,
        description="Enable Movie Weaver-style concept embeddings for Dynamic Identity Anchoring. "
        "Adds learnable embeddings to ALL tokens from reference video, helping the model "
        "know 'these tokens all represent the same identity/concept'. Critical for ID Transfer.",
    )

    concept_embedding_dim: int = Field(
        default=128,
        description="Dimension of concept embeddings. Should match model hidden dimension "
        "after patchification (128 for LTX-2 VAE latents).",
    )

    concept_embedding_task_specific: bool = Field(
        default=True,
        description="Use task-specific concept embeddings. Each task (identity, style, motion, etc.) "
        "gets separate learnable embeddings for better task discrimination.",
    )

    concept_embedding_num_concepts: int = Field(
        default=4,
        description="Number of concept slots for multi-concept scenarios. For single-reference "
        "training, only slot 0 is used. Increase for multi-subject/multi-reference setups.",
    )

    concept_embedding_init_scale: float = Field(
        default=0.02,
        description="Initialization scale for concept embeddings. Smaller values ensure "
        "embeddings don't dominate initial training.",
    )

    # Loss weighting
    target_loss_weight: float = Field(
        default=1.0,
        description="Weight for target reconstruction loss",
    )

    ref_preservation_loss_weight: float = Field(
        default=0.0,
        description="Optional weight for reference preservation regularization",
    )

    # Advanced loss options for faster convergence (per Grok recommendations)
    min_snr_gamma: float | None = Field(
        default=5.0,
        description="Min-SNR gamma for loss weighting. Clips SNR to min(SNR, gamma) "
        "to improve gradient flow at low-SNR timesteps. Set to None to disable.",
    )

    lpips_weight: float = Field(
        default=0.0,
        description="Weight for LPIPS perceptual loss. Recommended 0.1-0.5 for faster "
        "convergence with richer gradients. Requires lpips package.",
        ge=0.0,
    )

    identity_loss_weight: float = Field(
        default=0.0,
        description="Weight for identity embedding cosine similarity loss (ArcFace). "
        "Recommended 0.5-1.0 for identity preservation tasks. Requires insightface.",
        ge=0.0,
    )

    identity_loss_model: str = Field(
        default="arcface",
        description="Identity embedding model: 'arcface' or 'clip'. ArcFace recommended "
        "for face identity, CLIP for general identity.",
    )

    style_loss_weight: float = Field(
        default=0.0,
        description="Weight for Gram matrix style loss. Compares style features between "
        "prediction and reference. Recommended 0.1-1.0 for style transfer tasks. "
        "This is CRITICAL for style transfer - without it, the model ignores the reference.",
        ge=0.0,
    )

    # Performance optimization: Compute expensive pixel-space losses periodically
    perceptual_loss_interval: int = Field(
        default=1,
        description="Compute LPIPS/style/identity losses every N steps instead of every step. "
        "Set to 5-10 for 5-10x speedup with minimal quality impact. "
        "Flow matching loss is still computed every step for stable training.",
        ge=1,
    )

    # Grok-recommended: Use decoded pixels for perceptual losses
    # LPIPS and style loss work MUCH better on decoded RGB pixels than raw latents
    use_decoded_pixels_for_lpips: bool = Field(
        default=True,
        description="[GROK RECOMMENDED] Decode latents to RGB pixels before computing LPIPS. "
        "VGG (used by LPIPS) expects image pixels, not latent space vectors. "
        "Set False only if VAE decoder is not available or for speed testing.",
    )

    use_decoded_pixels_for_style: bool = Field(
        default=True,
        description="[GROK RECOMMENDED] Decode latents to RGB pixels before computing style loss. "
        "Gram matrices work better on decoded pixels where styles are not entangled with content. "
        "Set False for faster but lower quality style matching.",
    )

    use_vgg_style_features: bool = Field(
        default=True,
        description="[GROK RECOMMENDED] Use VGG features (conv layers) for style loss instead of raw pixels. "
        "Multi-layer Gram matrices capture richer style information (textures, colors, patterns). "
        "Requires torchvision.",
    )

    vgg_style_layers: list[str] = Field(
        default=["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
        description="VGG layers to extract features from for style loss. "
        "Earlier layers capture low-level textures, later layers capture higher-level patterns.",
    )

    # Grok-recommended: Use CLIP/SigLIP for identity loss
    use_clip_identity: bool = Field(
        default=False,
        description="[GROK RECOMMENDED] Use CLIP/SigLIP for identity loss instead of mean pooling. "
        "Provides robust semantic features for identity preservation. Requires open_clip package.",
    )

    clip_model_name: str = Field(
        default="ViT-SO400M-14-SigLIP-384",
        description="Vision encoder model. For Qwen2.5/Qwen3-VL compatibility, use SigLIP models: "
        "'ViT-SO400M-14-SigLIP-384' (recommended), 'ViT-SO400M-14-SigLIP', 'ViT-SO400M-14-SigLIP2'. "
        "For standard CLIP: 'ViT-B-32', 'ViT-L-14', 'ViT-H-14', 'ViT-bigG-14'.",
    )

    clip_pretrained: str = Field(
        default="webli",
        description="Pretrained weights. For SigLIP: 'webli'. For CLIP: 'openai', 'laion2b_s34b_b79k'. "
        "For ViT-bigG-14 (matches original Qwen-VL): 'laion2b_s39b_b160k'.",
    )

    # W&B Visualization configuration
    log_reconstructions: bool = Field(
        default=True,
        description="Whether to log reconstruction visualizations to W&B",
    )

    reconstruction_log_interval: int = Field(
        default=500,
        description="Steps between reconstruction visualization logging",
    )

    num_frames_to_visualize: int = Field(
        default=8,
        description="Number of frames to include in visualization grids",
    )

    max_samples_per_log: int = Field(
        default=4,
        description="Maximum batch samples to log per visualization step",
    )

    log_video_comparisons: bool = Field(
        default=True,
        description="Whether to log video comparisons (more expensive)",
    )

    video_log_interval: int = Field(
        default=2000,
        description="Steps between video comparison logging (less frequent than images)",
    )

    save_reconstructions_locally: bool = Field(
        default=False,
        description="Whether to save reconstruction images locally",
    )

    local_reconstruction_dir: str | None = Field(
        default=None,
        description="Directory for local reconstruction saves",
    )

    # Data Augmentation (Grok: Critical for preventing overfitting on small datasets)
    augmentation: dict = Field(
        default={
            "enabled": False,
            "horizontal_flip_p": 0.5,
            "temporal_jitter_p": 0.3,
            "temporal_jitter_max_frames": 1,
            "noise_injection_p": 0.2,
            "noise_injection_scale": 0.02,
        },
        description="Data augmentation settings for latent-space training. "
        "horizontal_flip_p: probability of flipping horizontally. "
        "temporal_jitter_p: probability of shifting frames. "
        "noise_injection_p: probability of adding small noise.",
    )

    # ==========================================================================
    # 3DiMo Implicit Motion Encoder (arXiv:2602.03796v2)
    # ==========================================================================
    # Compresses driving video into compact view-agnostic motion tokens injected
    # via cross-attention. Augmentations on driving video prevent identity leakage.
    # Geometric supervision (annealed) provides auxiliary training signal.
    #
    # Quote: "We introduce an implicit motion encoder that compresses the driving
    # video into compact view-agnostic tokens." (Section 3.1)
    # ==========================================================================

    enable_motion_encoder: bool = Field(
        default=False,
        description="Enable 3DiMo implicit motion encoder. Compresses driving video into "
        "K compact motion tokens injected via cross-attention alongside text tokens.",
    )

    motion_encoder_num_tokens: int = Field(
        default=5,
        description="Number of motion query tokens K (paper uses 5 for body).",
        ge=1,
    )

    motion_encoder_hidden_dim: int = Field(
        default=512,
        description="Internal hidden dimension of motion encoder transformer.",
    )

    motion_encoder_num_layers: int = Field(
        default=4,
        description="Number of transformer encoder layers in motion encoder.",
        ge=1,
    )

    motion_encoder_dual_scale: bool = Field(
        default=False,
        description="Use dual-scale body+hand motion encoders (Section 3.2). "
        "Adds a separate hand encoder for fine-grained finger dynamics.",
    )

    motion_encoder_hand_tokens: int = Field(
        default=3,
        description="Number of hand motion tokens for dual-scale encoder.",
        ge=1,
    )

    # Augmentations applied to driving video BEFORE motion encoding
    motion_augmentation: dict = Field(
        default={
            "enabled": True,
            "channel_jitter_scale": 0.05,
            "spatial_crop_ratio_min": 0.7,
            "spatial_crop_ratio_max": 1.0,
            "temporal_subsample_ratio": 0.8,
        },
        description="Augmentations applied to driving video ONLY before motion encoding. "
        "Prevents identity leakage from driving signal. "
        "Quote: 'augmentations... prevent identity leakage' (Section 3.2)",
    )

    # Geometric supervision with SMPL/MANO (annealed)
    enable_geometric_supervision: bool = Field(
        default=False,
        description="Enable auxiliary geometric loss using SMPL body pose and MANO hand joints. "
        "Requires precomputed skeleton/ data from 4DHumans+HaMeR.",
    )

    geometric_loss_initial_weight: float = Field(
        default=0.1,
        description="Initial weight lambda_0 for geometric loss. Linearly annealed to 0.",
        ge=0.0,
    )

    geometric_loss_anneal_steps: int = Field(
        default=12000,
        description="Number of steps over which geometric loss is annealed to 0.",
        ge=1,
    )

    geometric_pseudo_gt_dir: str = Field(
        default="skeleton",
        description="Directory name for precomputed SMPL/MANO pseudo-GT .pt files.",
    )

    # Self-reconstruction mode for 3DiMo Phase 1
    motion_self_reconstruction: bool = Field(
        default=False,
        description="3DiMo Phase 1: Self-reconstruction where target == driving video. "
        "First frame is used as reference image. Model learns to reconstruct "
        "the driving video from its own first frame + motion tokens.",
    )

    # Cross-view training for 3DiMo Phase 2-3
    motion_cross_view: bool = Field(
        default=False,
        description="Enable cross-view training where driving and target are different views. "
        "Used in 3DiMo Phases 2-3 for view-independent motion transfer.",
    )

    motion_cross_view_ratio: float = Field(
        default=0.5,
        description="Ratio of cross-view samples vs self-reconstruction during mixed training.",
        ge=0.0,
        le=1.0,
    )

    # 3DiMo training phase
    motion_training_phase: int = Field(
        default=1,
        description="3DiMo training phase: "
        "Phase 1: Self-reconstruction + geo supervision (10K steps). "
        "Phase 2: Mixed self-recon + cross-view, geo annealing (15K steps). "
        "Phase 3: Cross-view only, no geo loss (5K steps).",
        ge=1,
        le=3,
    )

    # ======================================================================
    # AMB3R 3D Reconstruction Losses (arXiv:2511.20343)
    # ======================================================================
    enable_depth_3d_loss: bool = Field(
        default=False,
        description="Enable AMB3R-style 3D reconstruction losses (depth, normal, edge). "
        "Requires pre-computed depth_3d/ pseudo-GT from frozen AMB3R or DA3.",
    )
    depth_3d_loss_weight: float = Field(
        default=0.05,
        description="Weight for depth consistency loss (scale-invariant).",
        ge=0.0,
    )
    normal_3d_loss_weight: float = Field(
        default=0.03,
        description="Weight for surface normal consistency loss.",
        ge=0.0,
    )
    edge_3d_loss_weight: float = Field(
        default=0.02,
        description="Weight for edge direction consistency loss.",
        ge=0.0,
    )
    depth_3d_loss_anneal_steps: int = Field(
        default=15000,
        description="Steps over which 3D losses are annealed to zero.",
        ge=1,
    )
    depth_3d_pseudo_gt_dir: str = Field(
        default="depth_3d",
        description="Directory for pre-computed AMB3R depth/points/confidence .pt files.",
    )
    use_confidence_weighted_mse: bool = Field(
        default=False,
        description="Use AMB3R confidence maps to weight the main flow-matching MSE loss. "
        "Focuses training on high-confidence regions.",
    )

    # AMB3R 3D Geometric Token Encoder
    enable_geometric_3d_encoder: bool = Field(
        default=False,
        description="Enable 3D geometric token encoder. Produces K tokens from pre-computed "
        "AMB3R features, injected via cross-attention alongside text/motion tokens.",
    )
    geometric_3d_num_tokens: int = Field(
        default=8,
        description="Number of 3D geometric query tokens.",
        ge=1,
    )
    geometric_3d_hidden_dim: int = Field(
        default=512,
        description="Internal hidden dimension of geometric 3D encoder.",
    )
    geometric_3d_num_layers: int = Field(
        default=4,
        description="Number of transformer layers in geometric 3D encoder.",
        ge=1,
    )
    geometric_3d_feature_dim: int = Field(
        default=7,
        description="Per-frame feature dimension from pseudo-GT (depth=1 + normal=3 + confidence=1 + mean_point=2 = 7 or configurable).",
    )
    geometric_3d_decoder_loss_weight: float = Field(
        default=0.05,
        description="Weight for auxiliary geometric 3D decoder loss.",
        ge=0.0,
    )
    geometric_3d_decoder_anneal_steps: int = Field(
        default=15000,
        description="Steps over which geometric 3D decoder loss anneals to zero.",
        ge=1,
    )

    # ======================================================================
    # MotionStream Track Conditioning (arXiv:2511.01266)
    # ======================================================================
    enable_track_conditioning: bool = Field(
        default=False,
        description="Enable MotionStream-style track conditioning. Encodes 2D point "
        "tracks as cross-attention tokens for explicit geometric motion control.",
    )
    track_num_tokens: int = Field(
        default=8,
        description="Number of track query tokens for cross-attention.",
        ge=1,
    )
    track_hidden_dim: int = Field(
        default=512,
        description="Internal hidden dimension of track conditioner.",
    )
    track_num_layers: int = Field(
        default=4,
        description="Number of transformer layers in track conditioner.",
        ge=1,
    )
    track_sincos_dim: int = Field(
        default=64,
        description="Sinusoidal embedding dimension per coordinate (total PE dim = sincos_dim * 2).",
    )
    track_max_points: int = Field(
        default=128,
        description="Maximum number of track points per sample.",
        ge=1,
    )
    track_decoder_loss_weight: float = Field(
        default=0.05,
        description="Weight for auxiliary track decoder loss.",
        ge=0.0,
    )
    track_decoder_anneal_steps: int = Field(
        default=15000,
        description="Steps over which track decoder loss anneals to zero.",
        ge=1,
    )
    track_consistency_loss_weight: float = Field(
        default=0.02,
        description="Weight for track consistency loss on generated latents.",
        ge=0.0,
    )
    track_pseudo_gt_dir: str = Field(
        default="tracks",
        description="Directory for pre-computed track .pt files.",
    )
    enable_joint_motion_cfg: bool = Field(
        default=False,
        description="Enable joint text-motion classifier-free guidance during training. "
        "Requires randomly dropping text/motion conditions independently.",
    )
    motion_cfg_drop_rate: float = Field(
        default=0.1,
        description="Probability of dropping motion/track conditions for CFG training.",
        ge=0.0, le=1.0,
    )
    text_cfg_drop_rate: float = Field(
        default=0.1,
        description="Probability of dropping text conditions for CFG training.",
        ge=0.0, le=1.0,
    )

    _task_step_counter: int = 0  # For round-robin sampling

    @property
    def task(self) -> OmniTransferTask:
        """Get the OmniTransferTask enum from string config."""
        return self._str_to_task(self.task_type)

    @staticmethod
    def _str_to_task(task_str: str) -> OmniTransferTask:
        """Convert task string to OmniTransferTask enum."""
        task_map = {
            "motion_transfer": OmniTransferTask.MOTION_TRANSFER,
            "pose_reenactment": OmniTransferTask.POSE_REENACTMENT,
            "action_customization": OmniTransferTask.ACTION_CUSTOMIZATION,
            "style_transfer": OmniTransferTask.STYLE_TRANSFER,
            "identity_preservation": OmniTransferTask.IDENTITY_PRESERVATION,
            "scene_composition": OmniTransferTask.SCENE_COMPOSITION,
        }
        return task_map.get(task_str, OmniTransferTask.MOTION_TRANSFER)

    def sample_task(self) -> OmniTransferTask:
        """Sample a task for multi-task training.

        Returns the configured task_type if multi_task_mode is False,
        otherwise samples according to the task_sampling strategy.
        """
        import random

        if not self.multi_task_mode or len(self.task_types) <= 1:
            return self.task

        if self.task_sampling == "round_robin":
            task_str = self.task_types[self._task_step_counter % len(self.task_types)]
            # Note: This counter needs to be incremented externally
            return self._str_to_task(task_str)

        elif self.task_sampling == "weighted":
            weights = [self.task_weights.get(t, 1.0) for t in self.task_types]
            task_str = random.choices(self.task_types, weights=weights, k=1)[0]
            return self._str_to_task(task_str)

        else:  # uniform
            task_str = random.choice(self.task_types)
            return self._str_to_task(task_str)


@dataclass
class OmniTransferModelInputs(ModelInputs):
    """Extended ModelInputs for OmniTransfer with additional metadata.

    Extends base ModelInputs to include OmniTransfer-specific information
    needed for loss computation, component coordination, and visualization.
    """
    # Task information
    task: OmniTransferTask | None = None

    # Reference/target split information
    ref_positions: Tensor | None = None  # [B, 3, ref_seq_len, 2]
    tgt_positions: Tensor | None = None  # [B, 3, tgt_seq_len, 2]

    # Target dimensions for TPB offset computation
    target_width: int | None = None
    target_frames: int | None = None

    # TMA features if computed
    tma_features: Tensor | None = None

    # Raw latents for reconstruction visualization (not patchified)
    # These are stored for W&B logging of source/target/prediction comparisons
    ref_latent_raw: Tensor | None = None      # [B, C, F, H, W] - reference video latent
    tgt_latent_raw: Tensor | None = None      # [B, C, F, H, W] - target video latent (clean)
    tgt_latent_noisy: Tensor | None = None    # [B, C, F, H, W] - noisy target for visualization
    noise: Tensor | None = None               # [B, C, F, H, W] - noise used for training
    sigmas: Tensor | None = None              # [B] - noise levels used
    first_frame_latent_raw: Tensor | None = None  # [B, C, 1, H, W] - I2V target image (for 4-panel viz)

    # Prompts for logging
    prompts: list[str] | None = None

    # 3DiMo motion encoder outputs (for geometric decoder loss)
    motion_hidden: Tensor | None = None        # [B, K, 512] hidden states for geometric decoder
    geometric_pseudo_gt: dict | None = None    # Precomputed SMPL/MANO from skeleton cache

    # AMB3R 3D reconstruction outputs (arXiv:2511.20343)
    geo_3d_hidden: Tensor | None = None        # [B, K, hidden_dim] from Geometric3DEncoder
    depth_3d_pseudo_gt: dict | None = None     # Pre-computed depth/points/confidence/normals

    # MotionStream track conditioning outputs (arXiv:2511.01266)
    track_hidden: Tensor | None = None         # [B, K, hidden_dim] from TrackConditioner
    track_pseudo_gt: dict | None = None        # Pre-computed tracks/visibility


class OmniTransferStrategy(TrainingStrategy):
    """OmniTransfer training strategy for unified spatio-temporal video transfer.

    This strategy implements the complete OmniTransfer training pipeline:
    1. Constructs reference (noise-free) and target (noised) latents
    2. Applies Task-aware Positional Bias to reference positions
    3. Uses Reference-decoupled Causal Learning for attention
    4. Optionally integrates Task-adaptive Multimodal Alignment

    Quote: "OmniTransfer comprises three key components: 1) Task-aware Positional
    Bias that applies distinct positional biases for different task types,
    2) Reference-decoupled Causal Learning that separates reference and target
    branches for improved efficiency, 3) Task-adaptive Multimodal Alignment that
    provides task-specific semantic guidance." (Section 4)
    """

    config: OmniTransferConfig

    def __init__(self, config: OmniTransferConfig):
        """Initialize OmniTransfer strategy.

        Args:
            config: OmniTransfer configuration
        """
        super().__init__(config)

        # Initialize latent constructor
        self._latent_constructor = ReferenceLatentConstructor(
            latent_channels=128,  # LTX-2 video latent channels
            default_task=config.task,
        )

        # Initialize TPB if enabled
        if config.enable_tpb:
            self._tpb = TaskAwarePositionalBias(
                dim=4096,  # LTX-2 model dim
                num_heads=32,
                config=TaskAwarePositionalBiasConfig(
                    max_pos=config.tpb_max_pos,
                    theta=config.tpb_theta,
                ),
            )
        else:
            self._tpb = None

        # Initialize Concept Embeddings for Dynamic Identity Anchoring
        # Based on Movie Weaver (CVPR 2025) which achieved 98.2% accuracy
        # in identity separation using concept embeddings
        if config.enable_concept_embeddings:
            self._concept_embedding = ConceptEmbedding(
                ConceptEmbeddingConfig(
                    embedding_dim=config.concept_embedding_dim,
                    num_concepts=config.concept_embedding_num_concepts,
                    task_specific=config.concept_embedding_task_specific,
                    num_task_types=6,  # identity, style, motion, camera, effect, scene
                    init_scale=config.concept_embedding_init_scale,
                )
            )
            logger.info(
                f"Initialized ConceptEmbedding for Dynamic Identity Anchoring: "
                f"dim={config.concept_embedding_dim}, task_specific={config.concept_embedding_task_specific}"
            )
        else:
            self._concept_embedding = None

        # ==========================================================================
        # Initialize TMA (Task-adaptive Multimodal Alignment) for Stages 2 and 3
        # ==========================================================================
        # TMA uses pre-computed Qwen VL features and a connector MLP to inject
        # semantic guidance into the target branch's cross-attention.
        # Stage 1: TMA disabled (train DiT + TPB + RCL only)
        # Stage 2: TMA enabled, freeze DiT, train connector only
        # Stage 3: TMA enabled, joint fine-tuning of all components
        # ==========================================================================
        self._tma = None
        self._tma_features_dir = None

        if config.enable_tma and config.training_stage >= 2:
            from ltx_trainer.omnitransfer.components import TaskAdaptiveMultimodalAlignment

            # Task type mapping for MetaQuery bank
            # These must match the task types used in the dataset
            task_types = [
                "motion_transfer",
                "pose_reenactment",
                "action_customization",
                "style_transfer",
                "identity_preservation",
                "scene_composition",
            ]

            self._tma = TaskAdaptiveMultimodalAlignment(
                mllm_hidden_dim=config.tma_mllm_hidden_dim,
                output_dim=3840,  # Match Gemma dimension - model does final projection
                num_connector_layers=config.tma_connector_layers,
                num_queries_per_task=config.tma_num_queries,
                dropout=0.1,
            )

            # Build task type to index mapping for runtime lookup
            self._tma_task_to_idx = {t: i for i, t in enumerate(task_types)}

            # Stage 2: Freeze MetaQuery params (only train connector)
            # This saves memory/compute by not calculating unused gradients
            if config.training_stage == 2:
                self._tma.meta_query_bank.requires_grad_(False)
                logger.info("Stage 2: Froze MetaQueryBank parameters (only training connector)")

            if config.use_cached_tma_features:
                logger.info(
                    f"TMA enabled with cached features (stage {config.training_stage}): "
                    f"hidden_dim={config.tma_mllm_hidden_dim}, "
                    f"num_queries={config.tma_num_queries}"
                )
            else:
                logger.info(
                    f"TMA enabled (stage {config.training_stage}): will compute features online"
                )
        elif config.enable_tma:
            logger.info(
                f"TMA configured but disabled for stage {config.training_stage} "
                "(TMA only used in stages 2-3)"
            )

        # ==========================================================================
        # Initialize 3DiMo Motion Encoder (arXiv:2602.03796v2)
        # ==========================================================================
        # Compresses driving video into compact view-agnostic motion tokens
        # injected via cross-attention alongside text/TMA tokens.
        # ==========================================================================
        self._motion_encoder = None
        self._motion_augmenter = None
        self._geometric_decoder = None

        if config.enable_motion_encoder:
            if config.motion_encoder_dual_scale:
                self._motion_encoder = DualScaleMotionEncoder(
                    latent_channels=128,
                    hidden_dim=config.motion_encoder_hidden_dim,
                    output_dim=3840,  # Match Gemma embedding dim
                    body_tokens=config.motion_encoder_num_tokens,
                    hand_tokens=config.motion_encoder_hand_tokens,
                    num_layers=config.motion_encoder_num_layers,
                )
                total_tokens = config.motion_encoder_num_tokens + config.motion_encoder_hand_tokens
            else:
                self._motion_encoder = MotionEncoder(
                    latent_channels=128,
                    hidden_dim=config.motion_encoder_hidden_dim,
                    output_dim=3840,  # Match Gemma embedding dim
                    num_tokens=config.motion_encoder_num_tokens,
                    num_layers=config.motion_encoder_num_layers,
                )
                total_tokens = config.motion_encoder_num_tokens

            # Motion augmenter for identity leakage prevention
            aug_cfg = config.motion_augmentation
            if aug_cfg.get("enabled", True):
                self._motion_augmenter = MotionAugmenter(
                    channel_jitter_scale=aug_cfg.get("channel_jitter_scale", 0.05),
                    spatial_crop_ratio_min=aug_cfg.get("spatial_crop_ratio_min", 0.7),
                    spatial_crop_ratio_max=aug_cfg.get("spatial_crop_ratio_max", 1.0),
                    temporal_subsample_ratio=aug_cfg.get("temporal_subsample_ratio", 0.8),
                )

            # Geometric decoder for auxiliary SMPL/MANO supervision
            if config.enable_geometric_supervision:
                self._geometric_decoder = GeometricDecoder(
                    hidden_dim=config.motion_encoder_hidden_dim,
                    num_tokens=total_tokens,
                )

            param_count = sum(p.numel() for p in self._motion_encoder.parameters())
            geo_info = f", GeometricDecoder" if self._geometric_decoder else ""
            aug_info = ", augmented" if self._motion_augmenter else ""
            logger.info(
                f"Initialized 3DiMo MotionEncoder: "
                f"tokens={total_tokens}, hidden={config.motion_encoder_hidden_dim}, "
                f"layers={config.motion_encoder_num_layers}, params={param_count:,}"
                f"{aug_info}{geo_info}"
            )

        # ======================================================================
        # AMB3R 3D Reconstruction Components (arXiv:2511.20343)
        # ======================================================================
        self._geometric_3d_encoder = None
        self._geometric_3d_decoder = None

        if config.enable_geometric_3d_encoder:
            from ltx_trainer.omnitransfer.geometric_3d_encoder import Geometric3DEncoder
            self._geometric_3d_encoder = Geometric3DEncoder(
                feature_dim=config.geometric_3d_feature_dim,
                hidden_dim=config.geometric_3d_hidden_dim,
                output_dim=3840,
                num_tokens=config.geometric_3d_num_tokens,
                num_layers=config.geometric_3d_num_layers,
            )

            if config.geometric_3d_decoder_loss_weight > 0:
                from ltx_trainer.omnitransfer.geometric_3d_decoder import Geometric3DDecoder
                self._geometric_3d_decoder = Geometric3DDecoder(
                    hidden_dim=config.geometric_3d_hidden_dim,
                    num_tokens=config.geometric_3d_num_tokens,
                )

            param_count = sum(p.numel() for p in self._geometric_3d_encoder.parameters())
            dec_info = ", +Decoder" if self._geometric_3d_decoder else ""
            logger.info(
                f"Initialized AMB3R Geometric3DEncoder: "
                f"tokens={config.geometric_3d_num_tokens}, hidden={config.geometric_3d_hidden_dim}, "
                f"layers={config.geometric_3d_num_layers}, params={param_count:,}{dec_info}"
            )

        # ======================================================================
        # MotionStream Track Conditioning (arXiv:2511.01266)
        # ======================================================================
        self._track_conditioner = None
        self._track_decoder = None

        if config.enable_track_conditioning:
            from ltx_trainer.omnitransfer.track_conditioner import TrackConditioner
            self._track_conditioner = TrackConditioner(
                sincos_dim=config.track_sincos_dim,
                hidden_dim=config.track_hidden_dim,
                output_dim=3840,
                num_tokens=config.track_num_tokens,
                num_layers=config.track_num_layers,
                max_tracks=config.track_max_points,
            )

            if config.track_decoder_loss_weight > 0:
                from ltx_trainer.omnitransfer.track_decoder import TrackDecoder
                self._track_decoder = TrackDecoder(
                    hidden_dim=config.track_hidden_dim,
                    num_tokens=config.track_num_tokens,
                    max_tracks=config.track_max_points,
                )

            param_count = sum(p.numel() for p in self._track_conditioner.parameters())
            dec_info = ", +Decoder" if self._track_decoder else ""
            logger.info(
                f"Initialized TrackConditioner: tokens={config.track_num_tokens}, "
                f"hidden={config.track_hidden_dim}, layers={config.track_num_layers}, "
                f"params={param_count:,}{dec_info}"
            )

        # Log I2V mode if enabled
        i2v_status = "I2V=True (pose-free animation)" if config.i2v_mode else "I2V=False"
        stage_info = f"Stage={config.training_stage}"
        motion_info = f", 3DiMo={config.enable_motion_encoder}" if config.enable_motion_encoder else ""
        logger.info(
            f"Initialized OmniTransfer strategy: task={config.task_type}, "
            f"TPB={config.enable_tpb}, RCL={config.enable_rcl}, TMA={config.enable_tma}, "
            f"{stage_info}, {i2v_status}{motion_info}"
        )

        # VAE decoder for Grok-recommended pixel-space losses
        self._vae_decoder = None

        # VGG feature extractor for style loss (Grok recommended)
        self._vgg_features = None
        self._vgg_layers = None

        # CLIP model for identity loss (Grok recommended)
        self._clip_model = None
        self._clip_preprocess = None

        # Step counter for periodic perceptual loss computation
        self._current_step = 0

    def set_vae_decoder(self, vae_decoder) -> None:
        """Set VAE decoder for pixel-space loss computation.

        [GROK RECOMMENDED] LPIPS and style loss should operate on decoded pixels,
        not raw latents. VGG (used in LPIPS) and Gram matrices work much better
        on RGB images where style/content are properly separated.

        Args:
            vae_decoder: LTX-2 video VAE decoder module
        """
        self._vae_decoder = vae_decoder
        logger.info("VAE decoder set for pixel-space loss computation (Grok recommended)")

        # Diagnostic probe: verify decoder output shape with dummy input
        try:
            param = next(vae_decoder.parameters())
            dev, dt = param.device, param.dtype
            with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=False):
                dummy = torch.randn(1, 128, 1, 14, 26, device=dev, dtype=dt)
                out = vae_decoder(dummy)
            logger.info(
                f"VAE decoder probe: input={dummy.shape} → output={out.shape}, "
                f"patch_size={getattr(vae_decoder, 'patch_size', '?')}, "
                f"type={type(vae_decoder).__name__}"
            )
            if out.shape[1] != 3:
                logger.error(
                    f"VAE decoder probe FAILED: expected 3 channels, got {out.shape[1]}! "
                    f"Pixel-space VGG style loss will fall back to latent-space."
                )
            del dummy, out
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"VAE decoder probe failed: {e}")

    def get_trainable_parameters(self) -> list:
        """Get trainable parameters for OmniTransfer components based on training stage.

        This method returns parameters that should be trained according to the
        3-stage training protocol from the OmniTransfer paper (Section 5.1):

        Stage 1: Train DiT + TPB + RCL (ConceptEmbedding)
            - LoRA parameters (from trainer)
            - TPB parameters (positional bias)
            - ConceptEmbedding parameters (identity anchoring)
            - TMA is NOT trained

        Stage 2: Freeze DiT, train TMA connector only
            - TMA connector MLP parameters only
            - MetaQueries are frozen (learned task-specific queries)

        Stage 3: Joint fine-tuning of all components
            - All parameters from stages 1 + 2

        Returns:
            List of trainable parameter groups for the optimizer
        """
        params = []
        stage = self.config.training_stage

        # Stage 1 and 3: TPB parameters
        if stage in [1, 3] and self._tpb is not None:
            tpb_params = list(self._tpb.parameters())
            if tpb_params:
                params.extend(tpb_params)
                logger.debug(f"Added {len(tpb_params)} TPB parameters")

        # Stage 1 and 3: ConceptEmbedding parameters
        if stage in [1, 3] and self._concept_embedding is not None:
            ce_params = list(self._concept_embedding.parameters())
            if ce_params:
                params.extend(ce_params)
                logger.debug(f"Added {len(ce_params)} ConceptEmbedding parameters")

        # Stage 2: TMA connector only (not MetaQueries)
        # Stage 3: All TMA parameters
        if self._tma is not None:
            if stage == 2:
                # Only connector MLP parameters
                connector_params = list(self._tma.connector.parameters())
                if connector_params:
                    params.extend(connector_params)
                    logger.debug(f"Added {len(connector_params)} TMA connector parameters (stage 2)")
            elif stage == 3:
                # All TMA parameters including MetaQueries
                tma_params = list(self._tma.parameters())
                if tma_params:
                    params.extend(tma_params)
                    logger.debug(f"Added {len(tma_params)} TMA parameters (stage 3)")

        # 3DiMo Motion Encoder parameters (all stages when enabled)
        if self._motion_encoder is not None:
            me_params = list(self._motion_encoder.parameters())
            if me_params:
                params.extend(me_params)
                logger.debug(f"Added {len(me_params)} MotionEncoder parameters")

        # 3DiMo Geometric Decoder parameters (annealed, but trained in stages 1-2)
        if self._geometric_decoder is not None:
            gd_params = list(self._geometric_decoder.parameters())
            if gd_params:
                params.extend(gd_params)
                logger.debug(f"Added {len(gd_params)} GeometricDecoder parameters")

        total_params = sum(p.numel() for p in params)
        logger.info(
            f"OmniTransfer trainable params (stage {stage}): {total_params:,} "
            f"(TPB: {self._tpb is not None}, CE: {self._concept_embedding is not None}, "
            f"TMA: {self._tma is not None}, 3DiMo: {self._motion_encoder is not None})"
        )

        return params

    def get_strategy_state_dict(self) -> dict[str, torch.Tensor]:
        """Get a named state dict for all strategy-owned modules.

        Returns parameters from ConceptEmbedding, TMA, MotionEncoder, and
        GeometricDecoder with a ``strategy.`` prefix so they can be saved
        alongside LoRA weights in the same checkpoint file.

        Returns:
            Dictionary mapping ``strategy.<module>.<param>`` → tensor.
        """
        state_dict: dict[str, torch.Tensor] = {}

        if self._concept_embedding is not None:
            for k, v in self._concept_embedding.state_dict().items():
                state_dict[f"strategy.concept_embedding.{k}"] = v

        if self._tma is not None:
            for k, v in self._tma.state_dict().items():
                state_dict[f"strategy.tma.{k}"] = v

        if self._motion_encoder is not None:
            for k, v in self._motion_encoder.state_dict().items():
                state_dict[f"strategy.motion_encoder.{k}"] = v

        if self._geometric_decoder is not None:
            for k, v in self._geometric_decoder.state_dict().items():
                state_dict[f"strategy.geometric_decoder.{k}"] = v

        return state_dict

    def load_strategy_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Load strategy parameters from a checkpoint state dict.

        Expects keys with the ``strategy.<module>.<param>`` prefix produced by
        :meth:`get_strategy_state_dict`.

        Args:
            state_dict: Full checkpoint state dict (may contain non-strategy keys).
            strict: If True, raise on missing/unexpected keys per module.

        Returns:
            Tuple of (loaded_keys, skipped_keys).
        """
        loaded: list[str] = []
        skipped: list[str] = []

        module_map: dict[str, torch.nn.Module | None] = {
            "concept_embedding": self._concept_embedding,
            "tma": self._tma,
            "motion_encoder": self._motion_encoder,
            "geometric_decoder": self._geometric_decoder,
        }

        for module_name, module in module_map.items():
            prefix = f"strategy.{module_name}."
            sub_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if not sub_dict:
                continue

            if module is None:
                skipped.extend(f"{prefix}{k}" for k in sub_dict)
                logger.warning(
                    f"Checkpoint has {len(sub_dict)} {module_name} params "
                    f"but module is not initialised — skipping"
                )
                continue

            result = module.load_state_dict(sub_dict, strict=strict)
            loaded.extend(f"{prefix}{k}" for k in sub_dict)
            if result.missing_keys:
                logger.warning(f"{module_name}: missing keys {result.missing_keys}")
            if result.unexpected_keys:
                logger.warning(f"{module_name}: unexpected keys {result.unexpected_keys}")
            logger.info(
                f"✅ Loaded {len(sub_dict)} {module_name} params from checkpoint"
            )

        return loaded, skipped

    def _get_task_enum(self, task_str: str) -> OmniTransferTask:
        """Convert task string to OmniTransferTask enum.

        Handles both full names (identity_preservation) and short names (id, identity).

        Args:
            task_str: Task name string

        Returns:
            OmniTransferTask enum value
        """
        # Extended mapping including short names used in multi-task configs
        task_map = {
            # Full names
            "motion_transfer": OmniTransferTask.MOTION_TRANSFER,
            "pose_reenactment": OmniTransferTask.POSE_REENACTMENT,
            "action_customization": OmniTransferTask.ACTION_CUSTOMIZATION,
            "style_transfer": OmniTransferTask.STYLE_TRANSFER,
            "identity_preservation": OmniTransferTask.IDENTITY_PRESERVATION,
            "scene_composition": OmniTransferTask.SCENE_COMPOSITION,
            "movie_weaver": OmniTransferTask.MOVIE_WEAVER,
            # Short names (used in website demos config)
            "motion": OmniTransferTask.MOTION_TRANSFER,
            "camera": OmniTransferTask.MOTION_TRANSFER,  # Camera movement is temporal
            "effect": OmniTransferTask.MOTION_TRANSFER,  # Effect transfer is temporal
            "id": OmniTransferTask.IDENTITY_PRESERVATION,
            "identity": OmniTransferTask.IDENTITY_PRESERVATION,
            "style": OmniTransferTask.STYLE_TRANSFER,
            "weaver": OmniTransferTask.MOVIE_WEAVER,
        }
        return task_map.get(task_str.lower(), OmniTransferTask.MOTION_TRANSFER)

    @staticmethod
    def _create_rcl_attention_mask(
        ref_seq_len: int, tgt_seq_len: int, device: torch.device
    ) -> Tensor:
        """Create the RCL self-attention keep-mask (Section 4.3).

        Reference tokens only self-attend (cannot see the noisy target); target
        tokens attend to themselves AND the reference (clean, t=0).

        Quote: "The reference branch... remains noise-free throughout the diffusion
        process... Attention masking prevents reference from attending to noisy target."

        Returned as a [0, 1] keep-mask of shape (1, T, T) — 1 = attend, 0 = block —
        which is the format TransformerArgsPreprocessor._prepare_self_attention_mask
        expects (it converts it to an additive log-space bias and hands it to the
        block's video self-attention as ``self_attention_mask``).

        Note: no row is fully masked (every ref row keeps the ``ref_seq_len`` ref
        columns), so softmax cannot produce NaNs from an all-masked row.
        """
        total = ref_seq_len + tgt_seq_len
        keep = torch.zeros(1, total, total, dtype=torch.float32, device=device)
        keep[:, :ref_seq_len, :ref_seq_len] = 1.0  # ref rows -> ref cols only
        keep[:, ref_seq_len:, :] = 1.0  # target rows -> everything (target + ref)
        return keep

    def _decode_latents_to_pixels(
        self,
        latents: Tensor,
        sample_frames: int = 4,
        preserve_grad: bool = False,
    ) -> Tensor | None:
        """Decode latents to RGB pixels for perceptual loss computation.

        [GROK RECOMMENDED] LPIPS expects RGB images in [-1, 1], not latent vectors.
        VGG features are meaningless on latents - they need proper image pixels.

        Args:
            latents: Video latents [B, C, F, H, W]
            sample_frames: Number of frames to decode (for efficiency)
            preserve_grad: If True, keep gradient flow through the VAE decoder
                so that losses on decoded pixels can backprop to the model.
                Costs ~2GB extra VRAM for the decoder computation graph.
                Only useful when the decoder is on a separate GPU with headroom.

        Returns:
            Decoded RGB frames [B*sample_frames, 3, H, W] in [-1, 1] or None if decoder unavailable
        """
        if self._vae_decoder is None:
            if not getattr(self, '_vae_decode_warned', False):
                if self.config.use_decoded_pixels_for_lpips or self.config.use_decoded_pixels_for_style:
                    logger.warning(
                        "VAE decoder not set but use_decoded_pixels enabled. "
                        "Call set_vae_decoder() from trainer. Falling back to latent-space loss."
                    )
                self._vae_decode_warned = True
            return None

        try:
            device = latents.device
            dtype = latents.dtype
            batch_size = latents.shape[0]
            num_frames = latents.shape[2]

            # Sample frames for efficiency
            sample_frames = min(sample_frames, num_frames)
            frame_indices = torch.linspace(0, num_frames - 1, sample_frames, device=device).long()

            # Extract sampled frames [B, C, sample_frames, H, W]
            sampled = latents[:, :, frame_indices, :, :]

            vae_param = next(self._vae_decoder.parameters())
            vae_device = vae_param.device
            vae_dtype = vae_param.dtype

            if preserve_grad:
                # Keep gradient flow: move to VAE device without detaching.
                # VAE decoder params are frozen (requires_grad=False) so only
                # the input tensor gets gradients. Costs ~2GB extra VRAM for
                # the decoder computation graph on the VAE device.
                sampled_vae = sampled.to(device=vae_device, dtype=vae_dtype)
            else:
                # Detach + clone to avoid keeping the decoder's computation graph
                # in memory (saves ~2GB VRAM). Pixel-space losses won't backprop.
                sampled_vae = sampled.detach().clone().to(device=vae_device, dtype=vae_dtype)

            # Disable autocast to prevent mixed-precision context from
            # interfering with the decoder's internal operations.
            with torch.amp.autocast(device_type=vae_device.type, enabled=False):
                decoded = self._vae_decoder(sampled_vae)

            # Log shape on first call
            if not getattr(self, '_vae_decode_logged', False):
                logger.debug(
                    f"VAE decode: input={sampled_vae.shape}, output={decoded.shape}, "
                    f"dtype={decoded.dtype}"
                )
                self._vae_decode_logged = True

            # Reshape to [B*F, C, H, W] for LPIPS/VGG
            # Decoder outputs [B, 3, F, H, W]. Permute to [B, F, 3, H, W] then flatten.
            # Keep decoded on VAE device (cuda:1) to avoid VRAM pressure on training GPU.
            # VGG + Gram matrices compute on VAE device; only scalar loss moves back.
            if decoded.dim() == 5:
                decoded = decoded.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                decoded = decoded.reshape(-1, *decoded.shape[2:])  # [B*F, C, H, W]

            # Verify 3 RGB channels (decoder with unpatchify should produce exactly 3)
            if decoded.shape[1] != 3:
                # Try manual unpatchify if decoder returned pre-unpatchify output
                patch_size = getattr(self._vae_decoder, 'patch_size', 4)
                expected_pre_unpatchify = 3 * patch_size * patch_size
                if decoded.shape[1] == expected_pre_unpatchify and decoded.dim() == 4:
                    from ltx_core.model.video_vae.ops import unpatchify
                    logger.warning(
                        f"VAE decoder returned {decoded.shape[1]} channels (pre-unpatchify). "
                        f"Applying manual unpatchify with patch_size={patch_size}."
                    )
                    # unpatchify expects 5D: add temporal dim
                    decoded = decoded.unsqueeze(2)  # [B*F, C, 1, H, W]
                    decoded = unpatchify(decoded, patch_size_hw=patch_size, patch_size_t=1)
                    decoded = decoded.squeeze(2)  # [B*F, 3, H*ps, W*ps]
                else:
                    logger.warning(
                        f"VAE decoder output has {decoded.shape[1]} channels instead of 3 RGB. "
                        f"Full shape: {decoded.shape}. Falling back to latent-space loss."
                    )
                    return None

            # Clamp to [-1, 1] for LPIPS/VGG
            decoded = decoded.clamp(-1, 1)

            return decoded

        except Exception as e:
            if not getattr(self, '_vae_decode_error_logged', False):
                logger.warning(f"Failed to decode latents to pixels: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self._vae_decode_error_logged = True
            return None

    def _get_vgg_features(self, images: Tensor) -> dict[str, Tensor]:
        """Extract VGG features from images for style loss.

        [GROK RECOMMENDED] Multi-layer Gram matrices on VGG features capture
        richer style information than raw pixels or latents.

        Args:
            images: RGB images [B, 3, H, W] in [-1, 1]

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        if not VGG_AVAILABLE:
            logger.warning("torchvision not available for VGG features")
            return {}

        try:
            # Initialize VGG lazily
            if self._vgg_features is None:
                vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
                vgg = vgg.to(images.device).eval()
                for p in vgg.parameters():
                    p.requires_grad = False
                self._vgg_features = vgg

                # Layer name to index mapping for VGG19
                self._vgg_layers = {
                    'relu1_1': 1, 'relu1_2': 3,
                    'relu2_1': 6, 'relu2_2': 8,
                    'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
                    'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
                    'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
                }

            # Resize to 256px for VGG — full resolution (e.g., 448x832) wastes
            # memory and compute; Gram-matrix style loss is scale-invariant.
            _, _, h, w = images.shape
            max_dim = max(h, w)
            if max_dim > 256:
                scale = 256 / max_dim
                new_h = max(1, int(h * scale))
                new_w = max(1, int(w * scale))
                images = torch.nn.functional.interpolate(
                    images, size=(new_h, new_w), mode="bilinear", align_corners=False
                )

            # Normalize images for VGG (expects ImageNet normalization)
            # Input is [-1, 1], convert to [0, 1] then normalize
            images = (images + 1) / 2  # [-1, 1] -> [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            images = (images - mean) / std

            # Run through VGG layers sequentially and grab features at target layers
            features = {}
            x = images.float()  # VGG expects float32
            for i, layer in enumerate(self._vgg_features):
                x = layer(x)
                for name, idx in self._vgg_layers.items():
                    if i == idx and name in self.config.vgg_style_layers:
                        features[name] = x.clone()

            return features

        except Exception as e:
            logger.warning(f"VGG feature extraction failed: {e}")
            return {}

    def _get_clip_features(self, images: Tensor, preserve_grad: bool = False) -> Tensor | None:
        """Extract CLIP/SigLIP features from images for identity loss.

        [GROK RECOMMENDED] CLIP/SigLIP provides robust semantic features that work
        much better than mean pooling for identity preservation.

        For Qwen2.5-VL and Qwen3-VL compatibility, SigLIP models are recommended
        since those models also use SigLIP-trained vision encoders.

        Args:
            images: RGB images [B, 3, H, W] in [-1, 1]
            preserve_grad: If True, don't use inference_mode so gradients can
                flow through the CLIP forward pass back to the input images.
                CLIP weights stay frozen (requires_grad=False) — only the input
                tensor accumulates gradients.

        Returns:
            Features [B, feature_dim] or None if unavailable
        """
        if not CLIP_AVAILABLE:
            logger.warning("open_clip not available for CLIP/SigLIP features")
            return None

        try:
            # Initialize model lazily
            if self._clip_model is None:
                self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                    self.config.clip_model_name,
                    pretrained=self.config.clip_pretrained,
                    device=images.device,
                )
                self._clip_model.eval()
                for p in self._clip_model.parameters():
                    p.requires_grad = False

                # Detect model type for proper preprocessing
                model_name = self.config.clip_model_name.lower()
                self._is_siglip = "siglip" in model_name
                self._clip_image_size = 384 if "384" in model_name else 256 if "256" in model_name else 224

                logger.info(
                    f"Loaded {'SigLIP' if self._is_siglip else 'CLIP'} model: "
                    f"{self.config.clip_model_name} (input size: {self._clip_image_size}px)"
                )

            # Convert [-1, 1] -> [0, 1]
            images = (images + 1) / 2

            # Resize to model's expected size
            images = torch.nn.functional.interpolate(
                images, size=(self._clip_image_size, self._clip_image_size),
                mode='bilinear', align_corners=False
            )

            # Apply appropriate normalization
            if self._is_siglip:
                mean = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
                std = torch.tensor([0.5, 0.5, 0.5], device=images.device).view(1, 3, 1, 1)
            else:
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)

            images = (images - mean) / std

            # Extract features — preserve_grad keeps gradient flow through
            # the CLIP forward pass so identity loss can backprop to the model.
            # CLIP weights are frozen, only the input tensor gets gradients.
            if preserve_grad:
                features = self._clip_model.encode_image(images.float())
            else:
                with torch.inference_mode():
                    features = self._clip_model.encode_image(images.float())

            return features

        except Exception as e:
            logger.warning(f"CLIP/SigLIP feature extraction failed: {e}")
            return None

    def get_data_sources(self) -> dict[str, str]:
        """OmniTransfer requires latents, conditions, and reference latents.

        In I2V mode, also requires first_frame_latents for static image conditioning.
        With TMA enabled (stages 2-3), requires qwen_vl_features for semantic guidance.

        Returns:
            Dictionary mapping data directory names to output keys
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
            self.config.reference_latents_dir: "ref_latents",
        }

        # I2V mode requires first-frame latents for static image conditioning
        if self.config.i2v_mode:
            sources[self.config.first_frame_latents_dir] = "first_frame_latents"

        # TMA requires pre-computed Qwen VL features (stages 2-3)
        if self.config.enable_tma and self.config.use_cached_tma_features and self.config.training_stage >= 2:
            sources[self.config.tma_features_dir] = "qwen_vl_features"

        # 3DiMo geometric supervision requires skeleton pseudo-GT
        if self.config.enable_motion_encoder and self.config.enable_geometric_supervision:
            sources[self.config.geometric_pseudo_gt_dir] = "skeleton"

        # AMB3R 3D reconstruction pseudo-GT
        if self.config.enable_depth_3d_loss or self.config.enable_geometric_3d_encoder:
            sources[self.config.depth_3d_pseudo_gt_dir] = "depth_3d"

        # MotionStream track conditioning pseudo-GT
        if self.config.enable_track_conditioning:
            sources[self.config.track_pseudo_gt_dir] = "tracks"

        return sources

    def _apply_augmentations(
        self,
        target_latents: Tensor,
        ref_latents: Tensor,
        first_frame_latent: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Apply data augmentations to latents for regularization.

        Grok recommendation: Augmentations are critical for preventing overfitting
        on small datasets. We apply the SAME augmentation to all related latents
        to maintain alignment (e.g., if we flip target, we flip ref and first_frame too).

        Args:
            target_latents: Ground truth video latents [B, C, F, H, W]
            ref_latents: Reference video latents [B, C, F, H, W]
            first_frame_latent: Optional first frame latent [B, C, 1, H, W]

        Returns:
            Augmented (target_latents, ref_latents, first_frame_latent)
        """
        import random

        aug_config = self.config.augmentation
        if not aug_config.get("enabled", False):
            return target_latents, ref_latents, first_frame_latent

        # Horizontal flip (flip width dimension)
        if random.random() < aug_config.get("horizontal_flip_p", 0.0):
            target_latents = torch.flip(target_latents, dims=[-1])  # Flip W
            ref_latents = torch.flip(ref_latents, dims=[-1])
            if first_frame_latent is not None:
                first_frame_latent = torch.flip(first_frame_latent, dims=[-1])

        # Temporal jitter (shift frames for reference video only - target stays aligned with GT)
        if random.random() < aug_config.get("temporal_jitter_p", 0.0):
            max_shift = aug_config.get("temporal_jitter_max_frames", 1)
            num_frames = ref_latents.shape[2]
            if num_frames > 2 * max_shift:  # Only if enough frames
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    # Shift reference video frames (circular)
                    ref_latents = torch.roll(ref_latents, shifts=shift, dims=2)

        # Noise injection (add small noise to all latents)
        if random.random() < aug_config.get("noise_injection_p", 0.0):
            noise_scale = aug_config.get("noise_injection_scale", 0.02)
            # Add noise proportional to latent magnitude
            target_std = target_latents.std()
            noise_mag = noise_scale * target_std

            target_latents = target_latents + noise_mag * torch.randn_like(target_latents)
            ref_latents = ref_latents + noise_mag * torch.randn_like(ref_latents)
            if first_frame_latent is not None:
                first_frame_latent = first_frame_latent + noise_mag * torch.randn_like(first_frame_latent)

        return target_latents, ref_latents, first_frame_latent

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> OmniTransferModelInputs:
        """Prepare inputs for OmniTransfer training.

        This method:
        1. Extracts reference and target latents from batch
        2. Constructs latents with task-specific masks
        3. Applies noise to target (reference stays at t=0)
        4. Computes task-aware positions for TPB
        5. Concatenates ref+target for model input

        I2V Mode (pose-free animation):
        - first_frame_latents: Static image to animate [B, C, 1, H, W]
        - reference_latents: Motion source video [B, C, F, H, W]
        - latents: Ground truth animated video [B, C, F, H, W]
        The model learns to animate the static image with motion from reference.

        Quote: "The reference branch adopts a fixed t = 0, meaning it remains
        noise-free throughout the diffusion process." (Section 4.3)

        Args:
            batch: Raw batch data containing latents, conditions, ref_latents
            timestep_sampler: Sampler for generating timesteps

        Returns:
            OmniTransferModelInputs ready for model forward pass
        """
        # Extract latents
        latents_info = batch["latents"]
        target_latents = latents_info["latents"]  # [B, C, F, H, W]
        ref_latents_info = batch["ref_latents"]
        ref_latents = ref_latents_info["latents"]  # [B, C, F, H, W]

        # CRITICAL: Assert reference and target are DIFFERENT (cross-subject transfer)
        # UNLESS self_reconstruction_mode is enabled (where they're intentionally the same)
        if not self.config.self_reconstruction_mode:
            # Compare first frame of reference with first frame of target video
            ref_first = ref_latents[:, :, 0:1, :, :]
            tgt_first = target_latents[:, :, 0:1, :, :]
            mean_diff = (ref_first - tgt_first).abs().mean().item()
            assert mean_diff > 0.1, (
                f"CRITICAL ERROR: Reference and target latents are nearly identical (diff={mean_diff:.4f})! "
                f"This trains identity mapping, not transfer. "
                f"Ensure your dataset has DIFFERENT videos for reference and target, "
                f"OR enable self_reconstruction_mode in config if intentional. "
                f"See AGENTS.md: 'NEVER USE THE SAME VIDEO FOR BOTH REFERENCE AND TARGET!'"
            )

        # I2V mode: Extract first-frame latents for conditioning
        first_frame_latent = None
        if self.config.i2v_mode and "first_frame_latents" in batch:
            first_frame_info = batch["first_frame_latents"]
            first_frame_latent = first_frame_info["latents"]  # [B, C, 1, H, W]
            # Only log once (at first call)
            if not hasattr(self, '_logged_i2v_shape'):
                logger.info(f"I2V mode: Using first_frame_latent shape {first_frame_latent.shape}")
                self._logged_i2v_shape = True

        # Apply data augmentations (Grok: critical for small datasets)
        target_latents, ref_latents, first_frame_latent = self._apply_augmentations(
            target_latents, ref_latents, first_frame_latent
        )

        # Get dimensions
        num_frames = latents_info["num_frames"][0].item()
        height = latents_info["height"][0].item()
        width = latents_info["width"][0].item()

        ref_frames = ref_latents_info["num_frames"][0].item()
        ref_height = ref_latents_info["height"][0].item()
        ref_width = ref_latents_info["width"][0].item()

        # Handle FPS
        fps = latents_info.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values in batch. Found: {fps.tolist()}, using first: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings
        conditions = batch["conditions"]
        prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = target_latents.shape[0]
        device = target_latents.device
        dtype = target_latents.dtype

        # ==========================================================================
        # Process TMA (Task-adaptive Multimodal Alignment) features if available
        # ==========================================================================
        # TMA injects semantic guidance from Qwen VL into the cross-attention context.
        # The features are pre-computed via scripts/compute_qwen_vl_features.py
        # and processed through the TMA connector to match DiT's hidden dimension.
        # ==========================================================================
        tma_context = None
        if self._tma is not None and "qwen_vl_features" in batch:
            try:
                # Load cached Qwen VL features [B, seq_len, mllm_dim]
                qwen_data = batch["qwen_vl_features"]
                qwen_features = qwen_data["qwen_features"].to(device=device, dtype=dtype)

                # Get task indices for MetaQuery lookup
                task_types = qwen_data.get("task_type", [self.config.task_type] * batch_size)
                if isinstance(task_types, str):
                    task_types = [task_types] * batch_size
                elif hasattr(task_types, 'tolist'):
                    task_types = task_types.tolist()

                # Map task strings to indices
                task_indices = torch.tensor([
                    self._tma_task_to_idx.get(t, 0) for t in task_types
                ], device=device)

                # Ensure TMA module is on the correct device (do this once)
                if not hasattr(self, '_tma_device_set'):
                    self._tma.to(device=device, dtype=dtype)
                    self._tma_device_set = True
                    logger.info(f"TMA module moved to {device} with dtype {dtype}")

                # Run TMA forward: MetaQuery cross-attention + Connector MLP
                # Output: [B, num_queries, output_dim=4096]
                tma_context = self._tma(qwen_features, task_indices)

                if not hasattr(self, '_logged_tma_shape'):
                    logger.info(
                        f"TMA context computed: {tma_context.shape} "
                        f"(prepended to prompt_embeds for cross-attention)"
                    )
                    self._logged_tma_shape = True

            except Exception as e:
                logger.warning(f"TMA feature processing failed: {e}")
                tma_context = None

        # If TMA context available, prepend to prompt embeddings for cross-attention
        if tma_context is not None:
            # TMA outputs at 3840 dim to match Gemma embeddings
            # Model's caption_projection will project both together to 4096
            # Prepend TMA context: [B, tma_len, 3840] + [B, prompt_len, 3840]
            prompt_embeds = torch.cat([tma_context, prompt_embeds], dim=1)

            # Update attention mask: [B, tma_len + prompt_len]
            tma_mask = torch.ones(
                batch_size, tma_context.shape[1],
                dtype=prompt_attention_mask.dtype, device=device
            )
            prompt_attention_mask = torch.cat([tma_mask, prompt_attention_mask], dim=1)

        # ==========================================================================
        # 3DiMo Motion Encoding (arXiv:2602.03796v2)
        # ==========================================================================
        # Compress driving video into compact motion tokens via implicit encoder.
        # Tokens are prepended to prompt_embeds for cross-attention injection.
        # Augmentations on driving video prevent identity leakage.
        # ==========================================================================
        motion_hidden = None
        geometric_pseudo_gt = None

        if self._motion_encoder is not None:
            # Use reference latents as driving video for motion encoding
            driving_latents = ref_latents.clone()

            # Apply motion augmentations to driving video ONLY (not target)
            # This prevents identity leakage from driving signal
            if self._motion_augmenter is not None:
                driving_latents = self._motion_augmenter(driving_latents)

            # Move motion encoder to device (once)
            if not hasattr(self, '_motion_encoder_device_set'):
                self._motion_encoder.to(device=device, dtype=dtype)
                self._motion_encoder_device_set = True
                if self._geometric_decoder is not None:
                    self._geometric_decoder.to(device=device, dtype=dtype)
                logger.info(f"3DiMo MotionEncoder moved to {device}")

            # Encode driving video -> motion tokens
            motion_tokens, motion_hidden = self._motion_encoder(driving_latents)
            # motion_tokens: [B, K, 3840] - same dim as Gemma embeddings
            # motion_hidden: [B, K, hidden_dim] - for geometric decoder

            if not hasattr(self, '_logged_motion_shape'):
                logger.info(
                    f"3DiMo motion tokens: {motion_tokens.shape} "
                    f"(prepended to prompt_embeds for cross-attention)"
                )
                self._logged_motion_shape = True

            # Prepend motion tokens to prompt_embeds
            # Final order: [motion_tokens, tma_tokens, text_tokens]
            prompt_embeds = torch.cat([motion_tokens, prompt_embeds], dim=1)

            # Update attention mask
            motion_mask = torch.ones(
                batch_size, motion_tokens.shape[1],
                dtype=prompt_attention_mask.dtype, device=device
            )
            prompt_attention_mask = torch.cat([motion_mask, prompt_attention_mask], dim=1)

            # Load skeleton pseudo-GT if geometric supervision is enabled
            if self._geometric_decoder is not None and "skeleton" in batch:
                try:
                    skeleton_data = batch["skeleton"]
                    geometric_pseudo_gt = {
                        k: v.to(device=device, dtype=dtype) if isinstance(v, Tensor) else v
                        for k, v in skeleton_data.items()
                        if k in ("body_pose", "hand_joints_3d", "body_joints_3d", "body_confidence")
                    }
                except Exception as e:
                    logger.warning(f"Failed to load skeleton pseudo-GT: {e}")
                    geometric_pseudo_gt = None

        # ======================================================================
        # AMB3R 3D Geometric Encoder Processing (arXiv:2511.20343)
        # ======================================================================
        geo_3d_hidden = None
        depth_3d_pseudo_gt = None

        # Load 3D pseudo-GT if available
        if (self.config.enable_depth_3d_loss or self._geometric_3d_encoder is not None) and "depth_3d" in batch:
            try:
                d3d = batch["depth_3d"]
                depth_3d_pseudo_gt = {
                    k: v.to(device=device, dtype=dtype) if isinstance(v, Tensor) else v
                    for k, v in d3d.items()
                    if k in ("depth", "points", "confidence", "normals", "pose", "frame_features")
                }
            except Exception as e:
                logger.debug(f"Failed to load depth_3d pseudo-GT: {e}")
                depth_3d_pseudo_gt = None

        # Encode 3D geometry into tokens for cross-attention
        if self._geometric_3d_encoder is not None and depth_3d_pseudo_gt is not None:
            if not hasattr(self, '_geo_3d_encoder_device_set'):
                self._geometric_3d_encoder.to(device=device, dtype=dtype)
                self._geo_3d_encoder_device_set = True
                if self._geometric_3d_decoder is not None:
                    self._geometric_3d_decoder.to(device=device, dtype=dtype)
                logger.info(f"AMB3R Geometric3DEncoder moved to {device}")

            # Extract per-frame features from pseudo-GT
            frame_features = depth_3d_pseudo_gt.get("frame_features")  # [B, F, feature_dim]
            if frame_features is not None:
                geo_tokens, geo_3d_hidden = self._geometric_3d_encoder(frame_features)
                # geo_tokens: [B, K, 3840]

                if not hasattr(self, '_logged_geo3d_shape'):
                    logger.info(
                        f"AMB3R geo_tokens: {geo_tokens.shape} "
                        f"(prepended to prompt_embeds for cross-attention)"
                    )
                    self._logged_geo3d_shape = True

                # Prepend to prompt_embeds (before motion tokens if present)
                prompt_embeds = torch.cat([geo_tokens, prompt_embeds], dim=1)
                geo_mask = torch.ones(
                    batch_size, geo_tokens.shape[1],
                    device=device, dtype=prompt_attention_mask.dtype,
                )
                prompt_attention_mask = torch.cat([geo_mask, prompt_attention_mask], dim=1)

        # ======================================================================
        # MotionStream Track Conditioning Processing (arXiv:2511.01266)
        # ======================================================================
        track_hidden = None
        track_pseudo_gt = None

        # Load track pseudo-GT if available
        if self._track_conditioner is not None and "tracks" in batch:
            try:
                trk = batch["tracks"]
                track_pseudo_gt = {
                    k: v.to(device=device, dtype=dtype) if isinstance(v, Tensor) else v
                    for k, v in trk.items()
                    if k in ("coords", "visibility", "track_ids")
                }
            except Exception as e:
                logger.debug(f"Failed to load track pseudo-GT: {e}")
                track_pseudo_gt = None

        # Encode tracks into tokens for cross-attention
        if self._track_conditioner is not None and track_pseudo_gt is not None:
            if not hasattr(self, '_track_cond_device_set'):
                self._track_conditioner.to(device=device, dtype=dtype)
                self._track_cond_device_set = True
                if self._track_decoder is not None:
                    self._track_decoder.to(device=device, dtype=dtype)
                logger.info(f"TrackConditioner moved to {device}")

            coords = track_pseudo_gt.get("coords")       # [B, F, N, 2]
            visibility = track_pseudo_gt.get("visibility")  # [B, F, N]

            if coords is not None:
                # Joint motion CFG: randomly drop track conditions
                drop_tracks = False
                if self.config.enable_joint_motion_cfg and self.training:
                    import random
                    drop_tracks = random.random() < self.config.motion_cfg_drop_rate

                if not drop_tracks:
                    track_tokens, track_hidden = self._track_conditioner(coords, visibility)
                    # track_tokens: [B, K, 3840]

                    if not hasattr(self, '_logged_track_shape'):
                        logger.info(
                            f"MotionStream track tokens: {track_tokens.shape} "
                            f"(prepended to prompt_embeds for cross-attention)"
                        )
                        self._logged_track_shape = True

                    # Prepend to prompt_embeds (before motion/geo/text tokens)
                    prompt_embeds = torch.cat([track_tokens, prompt_embeds], dim=1)
                    trk_mask = torch.ones(
                        batch_size, track_tokens.shape[1],
                        device=device, dtype=prompt_attention_mask.dtype,
                    )
                    prompt_attention_mask = torch.cat([trk_mask, prompt_attention_mask], dim=1)

        # 3DiMo self-reconstruction mode: target == driving video
        if self.config.motion_self_reconstruction and self._motion_encoder is not None:
            import random as _random
            # In self-reconstruction, the model reconstructs the driving video from:
            # - First frame as reference image (ref_latents = first frame expanded)
            # - Motion tokens from the full driving video
            # This teaches the motion encoder to capture full video dynamics
            target_latents = ref_latents.clone()  # Target IS the driving video
            # Use first frame as the reference for latent construction
            first_frame_latent = ref_latents[:, :, 0:1, :, :]

            # In Phase 2, mix with cross-view samples
            if self.config.motion_cross_view and _random.random() < self.config.motion_cross_view_ratio:
                # Use original target (cross-view) instead of self-reconstruction
                target_latents = batch["latents"]["latents"]
                first_frame_latent = batch.get("first_frame_latents", {}).get("latents", None)

        # Sample timestep/sigma for target
        # Reference uses fixed t=0 per RCL design
        # Note: Use sample() directly since latents are unpatchified [B, C, F, H, W]
        # sample_for() expects patchified 3D [B, seq_len, C] format
        # Compute seq_length from spatial dims: F * H * W (after patchification this is what it would be)
        _, _, f, h, w = target_latents.shape
        seq_length = f * h * w
        sigmas = timestep_sampler.sample(batch_size, seq_length, device=device)
        noise = torch.randn_like(target_latents)

        # Sample task for this batch (uses multi-task sampling if enabled)
        current_task = self.config.sample_task()
        if self.config.multi_task_mode:
            self.config._task_step_counter += 1  # Increment for round-robin

        # Per paper Table 1: Determine if this task uses I2V (image+motion) or T2V (prompt only)
        # - Temporal Transfer (I2V): Motion, Camera, Effect → V_ref + Image I
        # - Appearance Transfer (T2V): ID, Style → V_ref + prompt p (NO image)
        # Note: current_task is an OmniTransferTask enum, use .value to get string
        task_uses_i2v = is_temporal_task(current_task.value)
        effective_first_frame = first_frame_latent if task_uses_i2v else None

        if not hasattr(self, '_logged_task_mode') or current_task not in self._logged_task_mode:
            if not hasattr(self, '_logged_task_mode'):
                self._logged_task_mode = set()
            mode_str = "I2V (image+motion)" if task_uses_i2v else "T2V (prompt only)"
            logger.info(f"Task '{current_task}' uses {mode_str} mode")
            self._logged_task_mode.add(current_task)

        # Construct latents using ReferenceLatentConstructor
        # In I2V mode (temporal tasks), pass explicit first_frame_latent for conditioning
        # In T2V mode (appearance tasks), first_frame is None - prompt drives generation
        constructed = self._latent_constructor.construct(
            ref_video_latent=ref_latents,
            tgt_video_latent=target_latents,
            task=current_task,
            noise=noise,
            sigma=sigmas,
            first_frame_conditioning=task_uses_i2v,  # Only condition on first frame for temporal tasks
            first_frame_conditioning_prob=self.config.first_frame_conditioning_p if task_uses_i2v else 0.0,
            first_frame_latent=effective_first_frame,
        )

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        ref_latents_patched = self._video_patchifier.patchify(constructed.ref_latent)
        tgt_latents_patched = self._video_patchifier.patchify(constructed.tgt_latent)

        # =====================================================================
        # Dynamic Identity Anchoring via Concept Embeddings (Movie Weaver CVPR 2025)
        # =====================================================================
        if self._concept_embedding is not None:
            if not hasattr(self, '_concept_emb_device_set'):
                self._concept_embedding.to(device=device, dtype=dtype)
                self._concept_emb_device_set = True
                logger.info(f"ConceptEmbedding moved to {device}")

            task_enum = current_task
            ref_latents_patched = self._concept_embedding(
                ref_latents_patched, concept_index=0, task=task_enum,
            )
            if not hasattr(self, '_logged_concept_emb') or current_task not in self._logged_concept_emb:
                if not hasattr(self, '_logged_concept_emb'):
                    self._logged_concept_emb = set()
                logger.info(f"Applied concept embedding for identity anchoring: task={current_task}, ref_tokens={ref_latents_patched.shape[1]}")
                self._logged_concept_emb.add(current_task)

        ref_seq_len = ref_latents_patched.shape[1]
        tgt_seq_len = tgt_latents_patched.shape[1]

        # Create per-token timesteps
        ref_timesteps = torch.full(
            (batch_size, ref_seq_len),
            self.config.rcl_ref_timestep,
            device=device, dtype=dtype
        )

        effective_ff_prob = self.config.first_frame_conditioning_p if task_uses_i2v else 0.0
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=tgt_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=effective_ff_prob,
        )

        tgt_timesteps = self._create_per_token_timesteps(
            conditioning_mask=target_conditioning_mask,
            sampled_sigma=sigmas.squeeze(),
        )

        # Concatenate reference and target latents
        combined_latents = torch.cat([ref_latents_patched, tgt_latents_patched], dim=1)

        # Concatenate timesteps
        combined_timesteps = torch.cat([ref_timesteps, tgt_timesteps], dim=1)

        # Generate positions for reference and target
        ref_positions = self._get_video_positions(
            num_frames=ref_frames,
            height=ref_height,
            width=ref_width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        tgt_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Apply Task-aware Positional Bias if enabled
        # Quote: "we add an offset Δ along the spatial/temporal dimension" (Section 4.2)
        if self._tpb is not None:
            biased_ref_positions = self._tpb.apply_task_bias(
                ref_positions=ref_positions,
                task=current_task,
                target_width=width,
                target_frames=num_frames,
            )
        else:
            biased_ref_positions = ref_positions

        # Concatenate positions
        combined_positions = torch.cat([biased_ref_positions, tgt_positions], dim=2)

        # Reference-decoupled Causal Learning (Section 4.3).
        # The reference branch is held noise-free at t=0 (via combined_timesteps above)
        # and must be DECOUPLED from the noisy target: reference tokens may attend only
        # to the reference, while target tokens attend to target+reference. We express
        # this as a self-attention keep-mask on Modality.attention_mask, which ltx-core
        # converts to the block's self_attention_mask.
        #
        # NOTE: `rcl_split_point` alone does NOT implement RCL — nothing in ltx-core
        # consumes it, so before this the model ran full bidirectional attention and
        # the reference "decoupling" was inactive. The mask is what actually enforces it.
        rcl_split = None
        rcl_attention_mask = None
        if self.config.enable_rcl:
            rcl_split = ref_seq_len
            rcl_attention_mask = self._create_rcl_attention_mask(
                ref_seq_len, tgt_seq_len, combined_latents.device
            )
            if not hasattr(self, "_logged_rcl_split"):
                logger.info(
                    f"RCL decoupled attention (masked): ref={ref_seq_len} | "
                    f"tgt={tgt_seq_len} (total={ref_seq_len + tgt_seq_len} tokens)"
                )
                self._logged_rcl_split = True

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=combined_latents,
            sigma=sigmas.view(-1),
            timesteps=combined_timesteps,
            positions=combined_positions,
            context=prompt_embeds,
            context_mask=prompt_attention_mask,
            attention_mask=rcl_attention_mask,
            rcl_split_point=rcl_split,
        )

        # Compute targets for loss (velocity prediction)
        # v = noise - clean
        targets = noise - target_latents
        targets_patched = self._video_patchifier.patchify(targets)

        # Loss mask: only compute loss on non-conditioning target tokens
        ref_loss_mask = torch.zeros(batch_size, ref_seq_len, dtype=torch.bool, device=device)
        tgt_loss_mask = ~target_conditioning_mask
        video_loss_mask = torch.cat([ref_loss_mask, tgt_loss_mask], dim=1)

        # Extract prompts from conditions for visualization
        prompts = batch.get("prompts", None)
        if prompts is None:
            # Try to get from conditions metadata
            prompts = conditions.get("prompts", [""] * batch_size)

        return OmniTransferModelInputs(
            video=video_modality,
            audio=None,  # OmniTransfer currently video-only
            video_targets=targets_patched,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
            task=current_task,
            ref_positions=ref_positions,
            tgt_positions=tgt_positions,
            target_width=width,
            target_frames=num_frames,
            # TMA context is already prepended to prompt_embeds in the Modality
            tma_features=tma_context.detach() if tma_context is not None else None,
            # Store raw latents for W&B reconstruction visualization
            ref_latent_raw=constructed.ref_latent.detach(),
            tgt_latent_raw=constructed.tgt_clean.detach(),
            tgt_latent_noisy=constructed.tgt_latent.detach(),
            noise=noise.detach(),
            sigmas=sigmas.detach(),
            # I2V mode: store first frame latent for 4-panel visualization
            first_frame_latent_raw=first_frame_latent.detach() if first_frame_latent is not None else None,
            prompts=prompts if isinstance(prompts, list) else [prompts] * batch_size,
            # 3DiMo motion encoder outputs (NOT detached - geometric loss needs gradients)
            motion_hidden=motion_hidden if motion_hidden is not None else None,
            geometric_pseudo_gt=geometric_pseudo_gt,
            # AMB3R 3D reconstruction outputs (NOT detached - decoder loss needs gradients)
            geo_3d_hidden=geo_3d_hidden if geo_3d_hidden is not None else None,
            depth_3d_pseudo_gt=depth_3d_pseudo_gt,
            # MotionStream track conditioning outputs (NOT detached - decoder loss needs gradients)
            track_hidden=track_hidden if track_hidden is not None else None,
            track_pseudo_gt=track_pseudo_gt,
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        _audio_pred: Tensor | None,
        inputs: OmniTransferModelInputs,
    ) -> Tensor:
        """Compute OmniTransfer training loss with advanced options.

        The loss is computed only on the target portion (not reference)
        following the RCL design where reference is noise-free and loss-free.

        Quote: "The reference branch... remains noise-free throughout the diffusion
        process... loss is computed only on the target tokens." (Section 4.3)

        Includes advanced loss options for faster convergence:
        - Min-SNR weighting: Clips SNR to improve gradient flow at low timesteps
        - LPIPS perceptual loss: Richer gradients from perceptual similarity
        - Identity loss: ArcFace embedding similarity for identity preservation

        Args:
            video_pred: Model prediction [B, ref_seq_len + tgt_seq_len, C]
            _audio_pred: Audio prediction (unused for video-only)
            inputs: OmniTransferModelInputs with targets and masks

        Returns:
            Scalar loss tensor
        """
        # Extract target portion of prediction
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]

        # Get target portion of loss mask
        target_loss_mask = inputs.video_loss_mask[:, ref_seq_len:]

        # Compute MSE loss (velocity prediction)
        mse_loss = (target_pred - inputs.video_targets).pow(2)

        # Apply min-SNR weighting if configured
        # This improves gradient flow at low-SNR timesteps
        if self.config.min_snr_gamma is not None and inputs.sigmas is not None:
            # SNR = (1-sigma)^2 / sigma^2 for flow matching
            sigmas = inputs.sigmas.clamp(min=1e-6)
            snr = ((1 - sigmas) / sigmas).pow(2)
            # min-SNR weight: min(SNR, gamma) / SNR
            snr_weight = torch.clamp(snr, max=self.config.min_snr_gamma) / snr
            # Expand for broadcasting [B] -> [B, 1, 1]
            snr_weight = snr_weight.view(-1, 1, 1)
            mse_loss = mse_loss * snr_weight

        # Apply loss mask
        loss_mask = target_loss_mask.unsqueeze(-1).float()

        # Avoid division by zero
        mask_sum = loss_mask.sum()
        if mask_sum > 0:
            loss = mse_loss.mul(loss_mask).sum() / mask_sum
        else:
            loss = mse_loss.mean()

        # Apply target loss weight
        loss = loss * self.config.target_loss_weight

        # Optional reference preservation loss (regularization)
        if self.config.ref_preservation_loss_weight > 0:
            ref_pred = video_pred[:, :ref_seq_len, :]
            # Reference should predict zero velocity (no change)
            ref_loss = ref_pred.pow(2).mean()
            loss = loss + self.config.ref_preservation_loss_weight * ref_loss

        # Increment step counter
        self._current_step += 1

        # Compute expensive pixel-space losses periodically for efficiency
        # Flow matching loss (above) is still computed every step
        compute_perceptual = (self._current_step % self.config.perceptual_loss_interval == 0)

        # Optional LPIPS perceptual loss
        if compute_perceptual and self.config.lpips_weight > 0 and LPIPS_AVAILABLE:
            lpips_loss = self._compute_lpips_loss(target_pred, inputs)
            if lpips_loss is not None:
                # Scale up to compensate for periodic computation
                loss = loss + self.config.lpips_weight * lpips_loss * self.config.perceptual_loss_interval

        # Optional identity embedding loss (for identity preservation task)
        if compute_perceptual and self.config.identity_loss_weight > 0 and self.config.task_type == "identity_preservation":
            identity_loss = self._compute_identity_loss(target_pred, inputs)
            if identity_loss is not None:
                scaled_id_loss = self.config.identity_loss_weight * identity_loss * self.config.perceptual_loss_interval
                loss = loss + scaled_id_loss
                # Log identity loss breakdown to W&B
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "train/identity_loss_raw": identity_loss.item(),
                        "train/identity_loss_scaled": scaled_id_loss.item(),
                        "train/mse_loss": (loss - scaled_id_loss).item(),
                    }, commit=False)

        # Optional style loss (CRITICAL for style transfer)
        if compute_perceptual and self.config.style_loss_weight > 0 and self.config.task_type == "style_transfer":
            style_loss = self._compute_style_loss(target_pred, inputs)
            if style_loss is not None:
                loss = loss + self.config.style_loss_weight * style_loss * self.config.perceptual_loss_interval

        # 3DiMo geometric supervision loss (annealed)
        # Auxiliary SMPL/MANO loss from geometric decoder, linearly annealed to 0
        if (
            self._geometric_decoder is not None
            and inputs.motion_hidden is not None
            and inputs.geometric_pseudo_gt is not None
        ):
            geo_preds = self._geometric_decoder(inputs.motion_hidden.to(device=loss.device))
            geo_loss = compute_geometric_loss(
                preds=geo_preds,
                pseudo_gt=inputs.geometric_pseudo_gt,
                step=self._current_step,
                initial_weight=self.config.geometric_loss_initial_weight,
                anneal_steps=self.config.geometric_loss_anneal_steps,
            )
            if geo_loss is not None:
                loss = loss + geo_loss
                # Log geometric loss weight periodically
                if self._current_step % 500 == 0:
                    weight = max(0.0, self.config.geometric_loss_initial_weight * (
                        1.0 - self._current_step / self.config.geometric_loss_anneal_steps
                    ))
                    logger.info(
                        f"3DiMo geometric loss: {geo_loss.item():.4f} "
                        f"(weight={weight:.4f}, step={self._current_step})"
                    )

        # ======================================================================
        # AMB3R 3D Reconstruction Losses (arXiv:2511.20343)
        # ======================================================================
        if self.config.enable_depth_3d_loss and inputs.depth_3d_pseudo_gt is not None and compute_perceptual:
            from ltx_trainer.omnitransfer.depth_3d_losses import compute_depth_3d_loss
            d3d_loss = compute_depth_3d_loss(
                target_pred=target_pred,
                inputs=inputs,
                step=self._current_step,
                depth_weight=self.config.depth_3d_loss_weight,
                normal_weight=self.config.normal_3d_loss_weight,
                edge_weight=self.config.edge_3d_loss_weight,
                anneal_steps=self.config.depth_3d_loss_anneal_steps,
            )
            if d3d_loss is not None:
                loss = loss + d3d_loss * self.config.perceptual_loss_interval

        # AMB3R Geometric 3D Decoder auxiliary loss
        if (
            self._geometric_3d_decoder is not None
            and inputs.geo_3d_hidden is not None
            and inputs.depth_3d_pseudo_gt is not None
        ):
            from ltx_trainer.omnitransfer.geometric_3d_decoder import compute_geo_3d_decoder_loss
            geo_3d_dec_loss = compute_geo_3d_decoder_loss(
                decoder=self._geometric_3d_decoder,
                hidden=inputs.geo_3d_hidden.to(device=loss.device),
                pseudo_gt=inputs.depth_3d_pseudo_gt,
                step=self._current_step,
                initial_weight=self.config.geometric_3d_decoder_loss_weight,
                anneal_steps=self.config.geometric_3d_decoder_anneal_steps,
            )
            if geo_3d_dec_loss is not None:
                loss = loss + geo_3d_dec_loss

        # ======================================================================
        # MotionStream Track Conditioning Losses (arXiv:2511.01266)
        # ======================================================================
        # Track decoder auxiliary loss
        if (
            self._track_decoder is not None
            and inputs.track_hidden is not None
            and inputs.track_pseudo_gt is not None
        ):
            from ltx_trainer.omnitransfer.track_decoder import compute_track_decoder_loss
            trk_dec_loss = compute_track_decoder_loss(
                decoder=self._track_decoder,
                hidden=inputs.track_hidden.to(device=loss.device),
                gt_tracks=inputs.track_pseudo_gt,
                step=self._current_step,
                initial_weight=self.config.track_decoder_loss_weight,
                anneal_steps=self.config.track_decoder_anneal_steps,
            )
            if trk_dec_loss is not None:
                loss = loss + trk_dec_loss

        # Track consistency loss on generated latents
        if (
            self.config.track_consistency_loss_weight > 0
            and inputs.track_pseudo_gt is not None
            and compute_perceptual
        ):
            from ltx_trainer.omnitransfer.track_losses import compute_track_loss
            trk_loss = compute_track_loss(
                target_pred=target_pred,
                inputs=inputs,
                step=self._current_step,
                weight=self.config.track_consistency_loss_weight,
                anneal_steps=self.config.depth_3d_loss_anneal_steps,  # Reuse anneal schedule
            )
            if trk_loss is not None:
                loss = loss + trk_loss * self.config.perceptual_loss_interval

        return loss

    def _compute_lpips_loss(
        self,
        target_pred: Tensor,
        inputs: OmniTransferModelInputs,
    ) -> Tensor | None:
        """Compute LPIPS perceptual loss between predicted and target.

        [GROK RECOMMENDED] LPIPS should operate on decoded RGB pixels, not latents.
        VGG (used internally by LPIPS) expects image pixels in [-1, 1] range.
        Computing LPIPS on latent vectors is mathematically meaningless since
        VGG was trained on natural images.

        Args:
            target_pred: Predicted target latents [B, seq_len, C]
            inputs: Model inputs containing target ground truth

        Returns:
            LPIPS loss scalar or None if computation fails
        """
        if not LPIPS_AVAILABLE or lpips is None:
            return None

        try:
            # Compute predicted clean latent from velocity prediction
            # For flow matching: clean = noisy - sigma * velocity
            dtype = inputs.tgt_latent_noisy.dtype
            device = inputs.tgt_latent_noisy.device
            sigmas = inputs.sigmas.to(dtype=dtype).view(-1, 1, 1, 1, 1)
            clean_pred = inputs.tgt_latent_noisy - sigmas * target_pred.view_as(inputs.tgt_latent_noisy)
            target_clean = inputs.tgt_latent_raw

            # [GROK RECOMMENDED] Decode to RGB pixels for proper LPIPS computation
            if self.config.use_decoded_pixels_for_lpips:
                pred_pixels = self._decode_latents_to_pixels(clean_pred, sample_frames=4)
                target_pixels = self._decode_latents_to_pixels(target_clean, sample_frames=4)

                if pred_pixels is not None and target_pixels is not None:
                    # Initialize LPIPS on the pixel device (VAE device) to avoid
                    # VRAM pressure on the training GPU
                    pixel_device = pred_pixels.device
                    if not hasattr(self, '_lpips_model'):
                        self._lpips_model = lpips.LPIPS(net='vgg').to(pixel_device)
                        self._lpips_model.eval()
                        for p in self._lpips_model.parameters():
                            p.requires_grad = False

                    # Pixels are already in [-1, 1] with 3 RGB channels
                    # Cast to float32 for LPIPS model (VGG expects float32)
                    lpips_loss = self._lpips_model(pred_pixels.float(), target_pixels.float()).mean()
                    return lpips_loss.to(device)
                else:
                    # Fall through to latent-space fallback
                    logger.debug("VAE decode failed, falling back to latent-space LPIPS")

            # Fallback: Latent-space LPIPS (less accurate but works without VAE)
            # This is NOT recommended per Grok but provides a fallback
            batch_size = clean_pred.shape[0]
            num_frames = clean_pred.shape[2]
            sample_frames = min(4, num_frames)
            device = clean_pred.device
            frame_indices = torch.linspace(0, num_frames - 1, sample_frames, device=device).long()

            # Get sampled frames [B, C, sample_frames, H, W] -> [B*sample_frames, C, H, W]
            pred_frames = clean_pred[:, :, frame_indices].permute(0, 2, 1, 3, 4)
            pred_frames = pred_frames.reshape(-1, *pred_frames.shape[2:])
            target_frames = target_clean[:, :, frame_indices].permute(0, 2, 1, 3, 4)
            target_frames = target_frames.reshape(-1, *target_frames.shape[2:])

            # Normalize to [-1, 1] for LPIPS
            pred_norm = pred_frames / pred_frames.abs().max().clamp(min=1e-6) * 2 - 1
            target_norm = target_frames / target_frames.abs().max().clamp(min=1e-6) * 2 - 1

            # Expand to 3 channels if needed (LPIPS expects RGB)
            if pred_norm.shape[1] != 3:
                if pred_norm.shape[1] > 3:
                    pred_norm = pred_norm[:, :3]
                    target_norm = target_norm[:, :3]
                else:
                    pred_norm = pred_norm.repeat(1, 3, 1, 1)[:, :3]
                    target_norm = target_norm.repeat(1, 3, 1, 1)[:, :3]

            # Lazily initialize LPIPS model on VAE device to avoid VRAM pressure
            # on the training GPU (cuda:0)
            if not hasattr(self, '_lpips_model'):
                lpips_device = self._vae_decoder.device if hasattr(self, '_vae_decoder') and self._vae_decoder is not None else device
                self._lpips_model = lpips.LPIPS(net='vgg').to(lpips_device)
                self._lpips_model.eval()
                for p in self._lpips_model.parameters():
                    p.requires_grad = False

            # Move tensors to LPIPS model device, compute, move result back
            lpips_device = next(self._lpips_model.parameters()).device
            lpips_loss = self._lpips_model(
                pred_norm.float().to(lpips_device),
                target_norm.float().to(lpips_device),
            ).mean()
            return lpips_loss.to(device)

        except Exception as e:
            logger.warning(f"LPIPS loss computation failed: {e}")
            return None

    def _compute_identity_loss(
        self,
        target_pred: Tensor,
        inputs: OmniTransferModelInputs,
    ) -> Tensor | None:
        """Compute identity embedding similarity loss.

        Uses CLIP/SigLIP for semantic identity features with full gradient flow:
        1. Decode predicted latents to RGB pixels (gradient-connected through VAE)
        2. Extract CLIP features (gradient-connected through frozen CLIP)
        3. Compute cosine similarity — gradients flow back to model prediction

        The reference is decoded without gradients (it's a fixed target).
        Only the prediction path needs gradients for backprop.

        Args:
            target_pred: Predicted target latents [B, seq_len, C]
            inputs: Model inputs containing reference latents

        Returns:
            Identity loss (1 - cosine_similarity) or None if unavailable
        """
        try:
            ref_latent = inputs.ref_latent_raw  # [B, C, F, H, W]
            device = ref_latent.device
            dtype = ref_latent.dtype

            # Compute predicted clean from velocity (gradient-connected to model output)
            sigmas = inputs.sigmas.to(dtype=dtype).view(-1, 1, 1, 1, 1)
            tgt_pred_5d = inputs.tgt_latent_noisy - sigmas * target_pred.view_as(inputs.tgt_latent_noisy)

            # Use CLIP for semantic identity features with gradient flow
            if self.config.use_clip_identity and CLIP_AVAILABLE:
                # Reference: no gradients needed (fixed target)
                ref_pixels = self._decode_latents_to_pixels(ref_latent, sample_frames=2, preserve_grad=False)
                # Prediction: gradient-connected so loss backprops to model
                pred_pixels = self._decode_latents_to_pixels(tgt_pred_5d, sample_frames=2, preserve_grad=True)

                if ref_pixels is not None and pred_pixels is not None:
                    # Reference features: no gradient needed
                    ref_features = self._get_clip_features(ref_pixels, preserve_grad=False)
                    # Prediction features: gradient flows through frozen CLIP back to model
                    pred_features = self._get_clip_features(pred_pixels, preserve_grad=True)

                    if ref_features is not None and pred_features is not None:
                        # Normalize for cosine similarity
                        ref_norm = ref_features / ref_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        pred_norm = pred_features / pred_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

                        # Identity loss = 1 - cosine_similarity
                        cosine_sim = (ref_norm * pred_norm).sum(dim=-1)
                        identity_loss = (1 - cosine_sim).mean()

                        # Log on first successful computation
                        if not getattr(self, '_clip_identity_logged', False):
                            logger.info(
                                f"CLIP identity loss active: sim={cosine_sim.mean().item():.4f}, "
                                f"loss={identity_loss.item():.4f}, grad_connected={pred_features.requires_grad}"
                            )
                            self._clip_identity_logged = True

                        return identity_loss.float()
                    else:
                        logger.debug("CLIP feature extraction failed, falling back")
                else:
                    logger.debug("VAE decode failed, falling back to latent-space identity")

            # Fallback: Multi-scale latent-space identity (better than simple mean pooling)
            # Use channel statistics at multiple spatial scales for richer comparison
            ref_5d = ref_latent
            pred_5d = tgt_pred_5d

            features_ref = []
            features_pred = []

            # Scale 1: Global mean (coarse identity)
            features_ref.append(ref_5d.mean(dim=(2, 3, 4)))
            features_pred.append(pred_5d.mean(dim=(2, 3, 4)))

            # Scale 2: Global std (texture/detail signature)
            features_ref.append(ref_5d.std(dim=(2, 3, 4)))
            features_pred.append(pred_5d.std(dim=(2, 3, 4)))

            # Scale 3: Spatial quadrant means (preserves coarse spatial structure)
            h_mid = ref_5d.shape[3] // 2
            w_mid = ref_5d.shape[4] // 2
            for quad in [(slice(None, h_mid), slice(None, w_mid)),
                         (slice(None, h_mid), slice(w_mid, None)),
                         (slice(h_mid, None), slice(None, w_mid)),
                         (slice(h_mid, None), slice(w_mid, None))]:
                features_ref.append(ref_5d[:, :, :, quad[0], quad[1]].mean(dim=(2, 3, 4)))
                features_pred.append(pred_5d[:, :, :, quad[0], quad[1]].mean(dim=(2, 3, 4)))

            # Concatenate all scales: [B, 128*6] = [B, 768]
            ref_features = torch.cat(features_ref, dim=-1)
            pred_features = torch.cat(features_pred, dim=-1)

            # Normalize for cosine similarity
            ref_norm = ref_features / ref_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            pred_norm = pred_features / pred_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            # Identity loss = 1 - cosine_similarity
            cosine_sim = (ref_norm.float() * pred_norm.float()).sum(dim=-1)
            identity_loss = (1 - cosine_sim).mean()

            return identity_loss

        except Exception as e:
            logger.warning(f"Identity loss computation failed: {e}")
            return None

    def _compute_style_loss(
        self,
        target_pred: Tensor,
        inputs: OmniTransferModelInputs,
    ) -> Tensor | None:
        """Compute Gram matrix style loss between prediction and reference.

        [GROK RECOMMENDED] Style loss should use VGG features on decoded pixels:
        1. Decode latents to RGB pixels
        2. Extract multi-layer VGG features (conv1_2, conv2_2, conv3_3, conv4_3)
        3. Compute Gram matrices at each layer
        4. MSE between reference and prediction Gram matrices

        This captures richer style information (textures, colors, patterns) than
        computing Gram matrices directly on latents where style is entangled with content.

        Args:
            target_pred: Predicted target latents [B, seq_len, C]
            inputs: Model inputs containing reference latents

        Returns:
            Style loss (MSE between Gram matrices) or None if computation fails
        """
        try:
            # Get reference latent (style source)
            ref_latent = inputs.ref_latent_raw  # [B, C, F, H, W]
            device = ref_latent.device
            dtype = ref_latent.dtype

            # Compute predicted clean from velocity
            sigmas = inputs.sigmas.to(dtype=dtype).view(-1, 1, 1, 1, 1)
            tgt_pred_5d = inputs.tgt_latent_noisy - sigmas * target_pred.view_as(inputs.tgt_latent_noisy)

            # [GROK RECOMMENDED] Use VGG features on decoded pixels
            if self.config.use_decoded_pixels_for_style:
                ref_pixels = self._decode_latents_to_pixels(ref_latent, sample_frames=3)
                pred_pixels = self._decode_latents_to_pixels(tgt_pred_5d, sample_frames=3)

                if ref_pixels is not None and pred_pixels is not None:
                    if not getattr(self, '_style_pixel_logged', False):
                        logger.info(
                            f"Style loss: using PIXEL-SPACE VGG features "
                            f"(ref={ref_pixels.shape}, pred={pred_pixels.shape}, "
                            f"grad={pred_pixels.requires_grad})"
                        )
                        self._style_pixel_logged = True

                    # Use VGG features for multi-layer Gram matrices
                    if self.config.use_vgg_style_features and VGG_AVAILABLE:
                        ref_features = self._get_vgg_features(ref_pixels)
                        pred_features = self._get_vgg_features(pred_pixels)

                        if ref_features and pred_features:
                            # Accumulate on the feature device (VAE device, cuda:1)
                            # to avoid VRAM pressure on training GPU (cuda:0).
                            feat_device = next(iter(ref_features.values())).device
                            style_loss = torch.tensor(0.0, device=feat_device, dtype=torch.float32)
                            num_layers = 0

                            for layer_name in ref_features:
                                if layer_name in pred_features:
                                    ref_feat = ref_features[layer_name]  # [B*F, C, H, W]
                                    pred_feat = pred_features[layer_name]

                                    # Compute Gram matrices
                                    b, c, h, w = ref_feat.shape
                                    ref_flat = ref_feat.view(b, c, -1)  # [B*F, C, H*W]
                                    pred_flat = pred_feat.view(b, c, -1)

                                    n_elements = float(h * w)
                                    ref_gram = torch.bmm(ref_flat, ref_flat.transpose(1, 2)) / n_elements
                                    pred_gram = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / n_elements

                                    layer_loss = torch.nn.functional.mse_loss(pred_gram, ref_gram)
                                    style_loss = style_loss + layer_loss
                                    num_layers += 1

                            if num_layers > 0:
                                # Move scalar loss back to training device
                                return (style_loss / num_layers).to(device)
                        else:
                            logger.debug("VGG feature extraction returned empty, falling back")

                    # Fallback: Gram on decoded pixels without VGG (still better than latents)
                    b = ref_pixels.shape[0]
                    c = ref_pixels.shape[1]
                    ref_flat = ref_pixels.view(b, c, -1)
                    pred_flat = pred_pixels.view(b, c, -1)

                    n_elements = float(ref_flat.shape[2])
                    ref_gram = torch.bmm(ref_flat, ref_flat.transpose(1, 2)) / n_elements
                    pred_gram = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / n_elements

                    style_loss = torch.nn.functional.mse_loss(pred_gram.float(), ref_gram.float())
                    return style_loss.to(device)
                else:
                    if not getattr(self, '_style_latent_logged', False):
                        logger.info("Style loss: VAE decode returned None, using LATENT-SPACE Gram matrices")
                        self._style_latent_logged = True

            # Latent-space Gram matrices — gradient-connected alternative to pixel-space VGG.
            # Less rich than VGG features but gradients flow through to model parameters.
            if not getattr(self, '_style_latent_direct_logged', False):
                logger.info(
                    "Style loss: using LATENT-SPACE Gram matrices "
                    f"(128-ch, gradient-connected, ref_grad={ref_latent.requires_grad}, "
                    f"pred_grad={tgt_pred_5d.requires_grad})"
                )
                self._style_latent_direct_logged = True
            num_frames = ref_latent.shape[2]
            sample_frames = min(3, num_frames)
            frame_indices = torch.linspace(0, num_frames - 1, sample_frames, device=device).long()

            # Get sampled frames [B, C, sample_frames, H, W]
            ref_frames = ref_latent[:, :, frame_indices, :, :].to(dtype=dtype)
            pred_frames = tgt_pred_5d[:, :, frame_indices, :, :].to(dtype=dtype)

            # Compute Gram matrices for style comparison
            b, c = ref_frames.shape[:2]
            ref_flat = ref_frames.reshape(b, c, -1)  # [B, C, N]
            pred_flat = pred_frames.reshape(b, c, -1)  # [B, C, N]

            n_elements = float(ref_flat.shape[2])
            ref_gram = torch.bmm(ref_flat, ref_flat.transpose(1, 2)) / n_elements  # [B, C, C]
            pred_gram = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / n_elements  # [B, C, C]

            style_loss = torch.nn.functional.mse_loss(pred_gram.float(), ref_gram.float())
            return style_loss

        except Exception as e:
            logger.warning(f"Style loss computation failed: {e}")
            return None

    def compute_predicted_clean_latent(
        self,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
    ) -> Tensor:
        """Compute predicted clean latent from model velocity prediction.

        For flow matching, the model predicts velocity v = noise - clean.
        We can recover the predicted clean latent as: clean_pred = noise - v_pred

        Args:
            video_pred: Model prediction [B, ref_seq_len + tgt_seq_len, C]
            inputs: OmniTransferModelInputs with noise and sigmas

        Returns:
            Predicted clean target latent [B, C, F, H, W]
        """
        ref_seq_len = inputs.ref_seq_len

        # Extract target portion of prediction
        target_pred_patched = video_pred[:, ref_seq_len:, :]  # [B, tgt_seq_len, C]

        # Unpatchify to get back to [B, C, F, H, W]
        # Need to infer shape from stored raw latents
        batch_size = inputs.tgt_latent_raw.shape[0]
        channels = inputs.tgt_latent_raw.shape[1]
        frames = inputs.tgt_latent_raw.shape[2]
        height = inputs.tgt_latent_raw.shape[3]
        width = inputs.tgt_latent_raw.shape[4]

        # Reshape from [B, F*H*W, C] to [B, C, F, H, W]
        target_pred = target_pred_patched.view(batch_size, frames, height, width, channels)
        target_pred = target_pred.permute(0, 4, 1, 2, 3)  # [B, C, F, H, W]

        # For velocity prediction: v = noise - clean
        # So: clean_pred = noise - v_pred
        predicted_clean = inputs.noise - target_pred

        return predicted_clean

    def log_reconstructions_to_wandb(
        self,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
        step: int,
        vae_decoder: torch.nn.Module | None = None,
        prefix: str = "train",
    ) -> dict[str, Any]:
        """Log reconstruction visualizations to W&B.

        Creates and logs comparison grids showing:
        - Reference (source) video frames
        - Target (ground truth) video frames
        - Prediction (model output) video frames

        Args:
            video_pred: Model prediction tensor
            inputs: OmniTransferModelInputs with raw latents
            step: Current training step
            vae_decoder: Optional VAE decoder for pixel-space visualization
            prefix: W&B metric prefix

        Returns:
            Dictionary of logged metrics
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return {}

        if not self.config.log_reconstructions:
            return {}

        # Import visualization module
        from ltx_trainer.omnitransfer.visualization import (
            OmniTransferVisualizer,
            ReconstructionSample,
            decode_latents_for_visualization,
        )

        # Initialize visualizer
        visualizer = OmniTransferVisualizer(
            log_to_wandb=True,
            log_interval=self.config.reconstruction_log_interval,
            num_frames_to_log=self.config.num_frames_to_visualize,
            save_local=self.config.save_reconstructions_locally,
            local_save_dir=Path(self.config.local_reconstruction_dir)
            if self.config.local_reconstruction_dir else None,
        )

        # Compute predicted clean latents
        predicted_clean = self.compute_predicted_clean_latent(video_pred, inputs)

        # Get latents for visualization
        ref_latents = inputs.ref_latent_raw
        tgt_latents = inputs.tgt_latent_raw
        pred_latents = predicted_clean
        first_frame_latents = inputs.first_frame_latent_raw  # None if not I2V mode

        # Decode to pixel space if VAE decoder provided
        if vae_decoder is not None:
            with torch.inference_mode():
                ref_decoded = decode_latents_for_visualization(ref_latents, vae_decoder)
                tgt_decoded = decode_latents_for_visualization(tgt_latents, vae_decoder)
                pred_decoded = decode_latents_for_visualization(pred_latents, vae_decoder)
                # I2V mode: also decode first frame latent for 4-panel visualization
                first_frame_decoded = None
                if first_frame_latents is not None:
                    first_frame_decoded = decode_latents_for_visualization(first_frame_latents, vae_decoder)
        else:
            # Use latents directly (less interpretable but still useful)
            # Normalize for visualization
            def normalize_latent(x):
                x = x - x.min()
                x = x / (x.max() + 1e-8)
                return x

            ref_decoded = normalize_latent(ref_latents)
            tgt_decoded = normalize_latent(tgt_latents)
            pred_decoded = normalize_latent(pred_latents)
            first_frame_decoded = normalize_latent(first_frame_latents) if first_frame_latents is not None else None

        # Ensure float32 for visualization (PIL/torchvision require float32 or uint8)
        ref_decoded = ref_decoded.float()
        tgt_decoded = tgt_decoded.float()
        pred_decoded = pred_decoded.float()
        if first_frame_decoded is not None:
            first_frame_decoded = first_frame_decoded.float()

        # Log batch reconstructions
        num_samples = min(self.config.max_samples_per_log, ref_decoded.shape[0])
        tasks = [self.config.task] * num_samples
        prompts = inputs.prompts[:num_samples] if inputs.prompts else [""] * num_samples

        log_dict = visualizer.log_batch_reconstructions(
            references=ref_decoded[:num_samples],
            targets=tgt_decoded[:num_samples],
            predictions=pred_decoded[:num_samples],
            tasks=tasks,
            prompts=prompts,
            step=step,
            losses=None,  # Could compute per-sample losses
            max_samples=num_samples,
            prefix=prefix,
        )

        # Log video comparisons less frequently
        if (self.config.log_video_comparisons and
                step % self.config.video_log_interval == 0):
            sample = ReconstructionSample(
                reference=ref_decoded[0],
                target=tgt_decoded[0],
                prediction=pred_decoded[0],
                task=self.config.task,
                prompt=prompts[0] if prompts else "",
                step=step,
                # I2V mode: include target image for 4-panel visualization
                target_image=first_frame_decoded[0] if first_frame_decoded is not None else None,
            )
            video_logs = visualizer.log_video_comparison(sample, prefix=prefix)
            log_dict.update(video_logs)

        return log_dict

    def get_wandb_metrics(
        self,
        loss: Tensor,
        video_pred: Tensor,
        inputs: OmniTransferModelInputs,
        step: int,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        """Get metrics dictionary for W&B logging.

        Args:
            loss: Training loss tensor
            video_pred: Model prediction
            inputs: Model inputs
            step: Training step
            learning_rate: Current learning rate

        Returns:
            Dictionary of metrics for W&B
        """
        metrics = {
            "train/loss": loss.item(),
            "train/task": self.config.task.value,
            "train/step": step,
        }

        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate

        # Add per-token loss statistics
        ref_seq_len = inputs.ref_seq_len
        target_pred = video_pred[:, ref_seq_len:, :]
        per_token_loss = (target_pred - inputs.video_targets).pow(2).mean(dim=-1)

        metrics["train/target_loss_mean"] = per_token_loss.mean().item()
        metrics["train/target_loss_std"] = per_token_loss.std().item()

        # Add sigma statistics
        if inputs.sigmas is not None:
            metrics["train/sigma_mean"] = inputs.sigmas.mean().item()
            metrics["train/sigma_std"] = inputs.sigmas.std().item()

        return metrics


def get_omnitransfer_training_schedule(
    stage: OmniTransferStage,
    base_lr: float = 1e-5,
) -> dict[str, Any]:
    """Get training configuration for each OmniTransfer stage.

    Quote: "In the first stage, we train the DiT blocks with TPB and RCL for
    10,000 steps. In the second stage, we freeze the DiT blocks and train only
    the TMA connector for 2,000 steps. In the third stage, we jointly fine-tune
    all components for 5,000 steps." (Section 5.1)

    Args:
        stage: Training stage
        base_lr: Base learning rate

    Returns:
        Dictionary with stage-specific training configuration
    """
    schedules = {
        OmniTransferStage.IN_CONTEXT: {
            "steps": 10000,
            "lr": base_lr,
            "freeze_modules": ["tma"],
            "train_modules": ["dit", "tpb", "rcl"],
            "description": "Stage 1: Train DiT blocks with TPB and RCL",
        },
        OmniTransferStage.CONNECTOR: {
            "steps": 2000,
            "lr": base_lr,
            "freeze_modules": ["dit", "tpb", "rcl"],
            "train_modules": ["tma"],
            "description": "Stage 2: Train TMA connector only",
        },
        OmniTransferStage.JOINT: {
            "steps": 5000,
            "lr": base_lr * 0.5,  # Lower LR for joint fine-tuning
            "freeze_modules": [],
            "train_modules": ["dit", "tpb", "rcl", "tma"],
            "description": "Stage 3: Joint fine-tuning of all components",
        },
    }
    return schedules.get(stage, schedules[OmniTransferStage.IN_CONTEXT])
