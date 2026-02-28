"""MetaQuery-based Task-adaptive Multimodal Alignment (TMA) for OmniTransfer.

This module integrates Facebook Research's MetaQuery MLLM for semantic feature
extraction in OmniTransfer's TMA component.

MetaQuery provides a frozen MLLM that extracts rich semantic features from
multimodal inputs (images + text), which are then used to guide the diffusion
process in a task-adaptive manner.

References:
- MetaQuery: https://github.com/facebookresearch/metaquery
- OmniTransfer paper: arXiv:2601.14250v1, Section 4.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from PIL import Image

# Lazy import to avoid circular dependency
# from ltx_trainer.omnitransfer.components import MetaQueryBank, OmniTransferTask

if TYPE_CHECKING:
    from torch import Tensor
    from ltx_trainer.omnitransfer.components import MetaQueryBank, OmniTransferTask

logger = logging.getLogger(__name__)


def _get_components():
    """Lazy import to avoid circular dependency."""
    from ltx_trainer.omnitransfer.components import MetaQueryBank, OmniTransferTask  # noqa: PLC0415
    return MetaQueryBank, OmniTransferTask


# Lazy imports for MetaQuery (may not be installed)
_METAQUERY_AVAILABLE = False
_MLLMInContext = None
_MetaQueryConfig = None


def _check_metaquery_available() -> bool:
    """Check if MetaQuery is available and import it."""
    global _METAQUERY_AVAILABLE, _MLLMInContext, _MetaQueryConfig

    if _METAQUERY_AVAILABLE:
        return True

    try:
        from models.model import MLLMInContext  # noqa: PLC0415
        from transformers import PretrainedConfig  # noqa: PLC0415

        _MLLMInContext = MLLMInContext
        _MetaQueryConfig = PretrainedConfig
        _METAQUERY_AVAILABLE = True
        logger.info("MetaQuery MLLM loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"MetaQuery not available: {e}. Using fallback TMA.")
        return False


@dataclass
class MetaQueryTMAConfig:
    """Configuration for MetaQuery-based TMA.

    Attributes:
        mllm_model_name: HuggingFace model name for the MLLM backbone
            Options: "llava-hf/llava-onevision-qwen2-7b-ov-hf",
                     "Qwen/Qwen2.5-VL-7B-Instruct"
        diffusion_model: Diffusion backbone type ("sana_1600M" or "sd1.5")
        connector_hidden_dim: Hidden dimension for the connector MLP
        output_dim: Output dimension (should match LTX-2 cross-attention dim)
        num_queries_per_task: Number of MetaQueries per task type
        freeze_mllm: Whether to freeze the MLLM backbone
        load_in_8bit: Whether to load MLLM in 8-bit for memory efficiency
        device: Device to load the model on
    """
    mllm_model_name: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    diffusion_model: str = "sana_1600M"
    connector_hidden_dim: int = 2048
    output_dim: int = 4096  # LTX-2 cross-attention dim
    num_queries_per_task: int = 8
    freeze_mllm: bool = True
    load_in_8bit: bool = True
    device: str = "cuda"


class MetaQueryFeatureExtractor(nn.Module):
    """Extract semantic features using MetaQuery's MLLM.

    This module wraps MetaQuery's MLLMInContext to extract semantic features
    from reference frames and text prompts for OmniTransfer's TMA.

    Quote from OmniTransfer paper (Section 4.4):
    "Task-adaptive Multimodal Alignment (TMA) leverages a multimodal LLM (MLLM)
    to provide semantic guidance that is adaptive to different transfer tasks.
    The MLLM takes as input the first-frame tokens of the target video, the
    reference video tokens, template tokens specific to the task type, and
    the text prompt tokens."
    """

    def __init__(self, config: MetaQueryTMAConfig):
        super().__init__()
        self.config = config
        self._mllm = None
        self._processor = None
        self._initialized = False

    def _lazy_init(self) -> bool:
        """Lazily initialize the MLLM to avoid loading at import time."""
        if self._initialized:
            return self._mllm is not None

        self._initialized = True

        if not _check_metaquery_available():
            logger.warning("MetaQuery not available, feature extractor disabled")
            return False

        try:
            from transformers import AutoProcessor  # noqa: PLC0415

            # Load processor for tokenization
            self._processor = AutoProcessor.from_pretrained(
                self.config.mllm_model_name,
                trust_remote_code=True,
            )

            # Load MLLM model
            logger.info(f"Loading MetaQuery MLLM: {self.config.mllm_model_name}")

            # Use MLLMInContext from metaquery
            self._mllm = _MLLMInContext.from_pretrained(
                self.config.mllm_model_name,
                torch_dtype=torch.bfloat16,
                load_in_8bit=self.config.load_in_8bit,
                device_map=self.config.device if not self.config.load_in_8bit else "auto",
                trust_remote_code=True,
            )

            if self.config.freeze_mllm:
                for param in self._mllm.parameters():
                    param.requires_grad = False
                self._mllm.eval()
                logger.info("MLLM frozen for inference-only mode")

            return True

        except Exception as e:
            logger.error(f"Failed to load MetaQuery MLLM: {e}")
            self._mllm = None
            return False

    def _create_task_template(self, task: OmniTransferTask) -> str:
        """Create task-specific template prompt.

        Quote: "template tokens specific to the task type" (Section 4.4)
        """
        templates = {
            OmniTransferTask.IDENTITY_PRESERVATION: (
                "Transfer the identity of the person in the reference image "
                "to the target scene while preserving their facial features, "
                "hair style, and distinctive characteristics."
            ),
            OmniTransferTask.STYLE_TRANSFER: (
                "Apply the artistic style from the reference image to the target, "
                "including color palette, brush strokes, and visual aesthetics."
            ),
            OmniTransferTask.MOTION_TRANSFER: (
                "Transfer the motion pattern from the reference video to the target, "
                "preserving the temporal dynamics and movement characteristics."
            ),
            OmniTransferTask.POSE_REENACTMENT: (
                "Reenact the pose and body position from the reference onto the "
                "target subject while maintaining their identity."
            ),
            OmniTransferTask.ACTION_CUSTOMIZATION: (
                "Customize the action in the target to match the reference action, "
                "adapting the movement to the target context."
            ),
            OmniTransferTask.SCENE_COMPOSITION: (
                "Compose the scene by integrating elements from the reference "
                "into the target environment coherently."
            ),
        }
        return templates.get(task, templates[OmniTransferTask.IDENTITY_PRESERVATION])

    @torch.inference_mode()
    def extract_features(
        self,
        reference_frame: Image.Image | Tensor,
        target_first_frame: Image.Image | Tensor | None,
        text_prompt: str,
        task: OmniTransferTask,
    ) -> Tensor | None:
        """Extract semantic features from multimodal inputs.

        Args:
            reference_frame: Reference video first frame (PIL Image or tensor)
            target_first_frame: Target video first frame (optional)
            text_prompt: Text description/prompt
            task: The transfer task type

        Returns:
            Semantic features [1, seq_len, hidden_dim] or None if extraction fails
        """
        if not self._lazy_init():
            return None

        try:
            # Convert tensors to PIL if needed
            if isinstance(reference_frame, Tensor):
                reference_frame = self._tensor_to_pil(reference_frame)
            if isinstance(target_first_frame, Tensor):
                target_first_frame = self._tensor_to_pil(target_first_frame)

            # Create task-specific prompt
            task_template = self._create_task_template(task)
            full_prompt = f"{task_template}\n\nUser prompt: {text_prompt}"

            # Prepare images list
            images = [reference_frame]
            if target_first_frame is not None:
                images.append(target_first_frame)

            # Tokenize using MetaQuery's method
            inputs = _MLLMInContext.tokenize(
                processor=self._processor,
                text=full_prompt,
                images=images,
            )

            # Move to device
            inputs = {k: v.to(self._mllm.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            # Extract features using encode_condition
            features, attention_mask = self._mllm.encode_condition(**inputs)

            return features

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    @staticmethod
    def _tensor_to_pil(tensor: Tensor) -> Image.Image:
        """Convert tensor [C, H, W] or [B, C, H, W] to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first batch
        if tensor.dim() == 3:
            # Assume [C, H, W], normalize to [0, 255]
            tensor = tensor.float()
            if tensor.max() <= 1.0:
                tensor = tensor * 255
            tensor = tensor.clamp(0, 255).byte()
            tensor = tensor.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(tensor)
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


class MetaQueryTMA(nn.Module):
    """Task-adaptive Multimodal Alignment using MetaQuery MLLM.

    This is the full TMA implementation that:
    1. Uses MetaQuery's MLLM to extract semantic features
    2. Aggregates features using task-specific MetaQueries
    3. Projects to LTX-2's cross-attention dimension

    Falls back to simple learnable TMA if MetaQuery is unavailable.
    """

    def __init__(
        self,
        config: MetaQueryTMAConfig | None = None,
        mllm_hidden_dim: int = 2048,
        output_dim: int = 4096,
        num_queries_per_task: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config or MetaQueryTMAConfig(
            connector_hidden_dim=mllm_hidden_dim,
            output_dim=output_dim,
            num_queries_per_task=num_queries_per_task,
        )

        # MetaQuery feature extractor (lazy loaded)
        self.feature_extractor = MetaQueryFeatureExtractor(self.config)

        # MetaQuery bank for task-specific queries (lazy import)
        MetaQueryBank, _ = _get_components()
        self.meta_query_bank = MetaQueryBank(
            num_tasks=6,
            num_queries_per_task=self.config.num_queries_per_task,
            query_dim=self.config.connector_hidden_dim,
        )

        # Cross-attention for MetaQuery aggregation
        self.query_attn = nn.MultiheadAttention(
            embed_dim=self.config.connector_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Connector MLP (3 layers as per paper)
        self.connector = nn.Sequential(
            nn.Linear(self.config.connector_hidden_dim, self.config.connector_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.connector_hidden_dim, self.config.connector_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.connector_hidden_dim, self.config.output_dim),
        )

        # Normalization
        self.input_norm = nn.LayerNorm(self.config.connector_hidden_dim)
        self.output_norm = nn.LayerNorm(self.config.output_dim)

    def forward(
        self,
        mllm_features: Tensor | None,
        task: OmniTransferTask,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute task-adaptive aligned features.

        Args:
            mllm_features: MLLM output features [B, seq_len, hidden_dim]
                          If None, uses zero features (fallback mode)
            task: The transfer task type
            attention_mask: Optional attention mask [B, seq_len]

        Returns:
            Task-aligned features [B, num_queries, output_dim]
        """
        batch_size = mllm_features.shape[0] if mllm_features is not None else 1
        device = mllm_features.device if mllm_features is not None else "cuda"

        # Fallback: create zero features if MLLM features not available
        if mllm_features is None:
            mllm_features = torch.zeros(
                batch_size, 1, self.config.connector_hidden_dim,
                device=device, dtype=torch.bfloat16
            )

        # Normalize input
        mllm_features = self.input_norm(mllm_features)

        # Get task-specific MetaQueries
        meta_queries = self.meta_query_bank(task, batch_size)
        meta_queries = meta_queries.to(device=device, dtype=mllm_features.dtype)

        # Aggregate using cross-attention
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        aligned_features, _ = self.query_attn(
            query=meta_queries,
            key=mllm_features,
            value=mllm_features,
            key_padding_mask=key_padding_mask,
        )

        # Project through connector
        aligned_features = self.connector(aligned_features)
        aligned_features = self.output_norm(aligned_features)

        return aligned_features

    def extract_and_align(
        self,
        reference_frame: Image.Image | Tensor,
        target_first_frame: Image.Image | Tensor | None,
        text_prompt: str,
        task: OmniTransferTask,
    ) -> Tensor:
        """Full pipeline: extract features and compute aligned output.

        This is the main entry point for TMA during training/inference.

        Args:
            reference_frame: Reference video first frame
            target_first_frame: Target video first frame (optional)
            text_prompt: Text description
            task: The transfer task type

        Returns:
            Task-aligned features [1, num_queries, output_dim]
        """
        # Extract MLLM features
        mllm_features = self.feature_extractor.extract_features(
            reference_frame=reference_frame,
            target_first_frame=target_first_frame,
            text_prompt=text_prompt,
            task=task,
        )

        # Compute aligned features
        return self.forward(mllm_features, task)
