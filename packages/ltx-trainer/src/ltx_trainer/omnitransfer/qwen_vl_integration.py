"""Qwen3-VL Integration for TMA (Task-adaptive Multimodal Alignment).

This module integrates Qwen3-VL as the MLLM backbone for OmniTransfer's TMA component.
It replaces the original Qwen-2.5-VL mentioned in the paper with the newer Qwen3-VL.

Key references:
- OmniTransfer paper Section 4.4: TMA architecture
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL

Quote from paper: "Task-adaptive Multimodal Alignment (TMA) leverages a multimodal LLM
(MLLM) to provide semantic guidance that is adaptive to different transfer tasks."
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import transformers components
try:
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    )
    # Qwen3-VL requires transformers >= 4.43
    try:
        from transformers import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMoeForConditionalGeneration,
        )
        HAS_QWEN3_VL = True
    except ImportError:
        HAS_QWEN3_VL = False
        logger.warning("Qwen3-VL not available. Install transformers>=4.43 for Qwen3-VL support.")
    HAS_QWEN_VL = True
except ImportError:
    HAS_QWEN_VL = False
    HAS_QWEN3_VL = False
    logger.warning("Qwen VL models not available. Install transformers with Qwen VL support.")


# Hidden dimensions for different Qwen VL model sizes
QWEN_HIDDEN_DIMS = {
    # Qwen2-VL
    "qwen2-vl-2b": 1536,
    "qwen2-vl-7b": 3584,
    "qwen2-vl-72b": 8192,
    # Qwen2.5-VL
    "qwen2.5-vl-3b": 2048,
    "qwen2.5-vl-7b": 3584,
    "qwen2.5-vl-32b": 5120,
    "qwen2.5-vl-72b": 8192,
    # Qwen3-VL (estimated based on Qwen3 architecture)
    "qwen3-vl-2b": 1536,
    "qwen3-vl-7b": 3584,
    "qwen3-vl-14b": 5120,
    "qwen3-vl-32b": 5120,
    "qwen3-vl-72b": 8192,
}


@dataclass
class QwenVLConfig:
    """Configuration for Qwen VL feature extractor."""
    model_path: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    max_pixels: int = 28 * 28 * 576  # Default max pixels for images
    min_pixels: int = 28 * 28 * 16   # Default min pixels
    video_max_frames: int = 8
    video_fps: float = 2.0
    freeze_mllm: bool = True  # Stage 1: freeze MLLM, Stage 2+: train connector


def detect_qwen_model_type(model_path: str) -> Tuple[str, int]:
    """Detect Qwen VL model type and hidden dimension from path.

    Args:
        model_path: Path to model or HuggingFace model ID

    Returns:
        Tuple of (model_type, hidden_dim) where model_type is
        'qwen2vl', 'qwen2.5vl', 'qwen3vl', or 'qwen3vl_moe'
    """
    model_name = Path(model_path).name.lower() if "/" not in model_path else model_path.lower()

    # Default hidden dimension
    hidden_dim = 3584  # Common for 7B models

    # Detect model type
    if "qwen3" in model_name:
        # Check for MoE variant (has "a" suffix like "Qwen3-VL-32B-A3B")
        if "-a" in model_name or "_a" in model_name:
            model_type = "qwen3vl_moe"
        else:
            model_type = "qwen3vl"
    elif "qwen2.5" in model_name or "qwen2_5" in model_name:
        model_type = "qwen2.5vl"
    else:
        model_type = "qwen2vl"

    # Detect hidden dimension from model size
    for key, dim in QWEN_HIDDEN_DIMS.items():
        if key.replace("-", "").replace(".", "") in model_name.replace("-", "").replace(".", ""):
            hidden_dim = dim
            break

    logger.info(f"Detected Qwen VL model type: {model_type}, hidden_dim: {hidden_dim}")
    return model_type, hidden_dim


class QwenVLFeatureExtractor(nn.Module):
    """Feature extractor using Qwen VL models for TMA.

    This class wraps Qwen2-VL, Qwen2.5-VL, or Qwen3-VL to extract
    semantic features for Task-adaptive Multimodal Alignment.

    Quote: "The MLLM takes as input the first-frame tokens of the target video,
    the reference video tokens, template tokens specific to the task type, and
    the text prompt tokens." (Section 4.4)
    """

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.config = config

        if not HAS_QWEN_VL:
            raise ImportError(
                "Qwen VL models not available. Install transformers with: "
                "pip install transformers>=4.43"
            )

        # Detect model type and hidden dimension
        self.model_type, self.hidden_dim = detect_qwen_model_type(config.model_path)

        # Load model based on type
        self._load_model()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(config.model_path)

        # Freeze MLLM if requested (Stage 1 training)
        if config.freeze_mllm:
            self._freeze_model()

    def _load_model(self):
        """Load the appropriate Qwen VL model."""
        config = self.config

        # Common loading kwargs
        load_kwargs = {
            "torch_dtype": config.dtype,
            "device_map": config.device if not config.load_in_8bit else "auto",
        }

        if config.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        if config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        # Load model based on detected type
        if self.model_type == "qwen3vl_moe":
            if not HAS_QWEN3_VL:
                raise ImportError("Qwen3-VL MoE requires transformers>=4.43")
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                config.model_path, **load_kwargs
            )
        elif self.model_type == "qwen3vl":
            if not HAS_QWEN3_VL:
                raise ImportError("Qwen3-VL requires transformers>=4.43")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_path, **load_kwargs
            )
        elif self.model_type == "qwen2.5vl":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_path, **load_kwargs
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.model_path, **load_kwargs
            )

        # Disable cache for training
        self.model.config.use_cache = False

        logger.info(f"Loaded {self.model.__class__.__name__} from {config.model_path}")

    def _freeze_model(self):
        """Freeze all MLLM parameters for Stage 1 training."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("MLLM parameters frozen for Stage 1 training")

    def unfreeze_connector(self):
        """Unfreeze vision-language connector for Stage 2 training.

        Quote: "Stage 2: Train the connector MLP while keeping the MLLM
        backbone frozen. This allows the model to learn task-specific
        projections efficiently." (Training procedure)
        """
        # Unfreeze the merger/connector in the vision module
        if hasattr(self.model, 'visual') and hasattr(self.model.visual, 'merger'):
            for param in self.model.visual.merger.parameters():
                param.requires_grad = True
            logger.info("Vision-language connector (merger) unfrozen for Stage 2")

    def unfreeze_all(self):
        """Unfreeze all parameters for Stage 3 fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All MLLM parameters unfrozen for Stage 3 fine-tuning")

    def prepare_inputs(
        self,
        reference_frames: torch.Tensor,
        target_frame: torch.Tensor,
        prompt: str,
        task_template: Optional[str] = None,
    ) -> dict:
        """Prepare inputs for MLLM feature extraction.

        Args:
            reference_frames: Reference video frames [B, T, C, H, W] or [B, C, H, W]
            target_frame: Target first frame [B, C, H, W]
            prompt: Text prompt describing the transfer
            task_template: Optional task-specific template text

        Returns:
            Dictionary of model inputs
        """
        batch_size = reference_frames.shape[0]
        device = reference_frames.device

        # Convert tensors to PIL images for processor
        # Note: In production, we'd handle this more efficiently
        ref_images = []
        tgt_images = []

        for b in range(batch_size):
            # Handle reference (video or image)
            if reference_frames.dim() == 5:  # Video: [B, T, C, H, W]
                # Sample frames for video
                ref_frames = reference_frames[b]  # [T, C, H, W]
                # Convert to list of PIL images
                for t in range(min(ref_frames.shape[0], self.config.video_max_frames)):
                    frame = ref_frames[t].cpu()
                    frame = (frame * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    ref_images.append(Image.fromarray(frame))
            else:  # Image: [B, C, H, W]
                frame = reference_frames[b].cpu()
                frame = (frame * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                ref_images.append(Image.fromarray(frame))

            # Target first frame
            frame = target_frame[b].cpu()
            frame = (frame * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            tgt_images.append(Image.fromarray(frame))

        # Construct message for Qwen VL
        # Quote: "template tokens specific to the task type"
        if task_template:
            full_prompt = f"{task_template}\n{prompt}"
        else:
            full_prompt = prompt

        # Build conversation format for Qwen VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Reference
                    {"type": "image"},  # Target
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        # Process inputs
        # Note: This is a simplified version; production code would handle
        # batching and video inputs more efficiently
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process with images
        inputs = self.processor(
            text=[text] * batch_size,
            images=list(zip(ref_images, tgt_images)),
            videos=None,  # Handle video separately if needed
            padding=True,
            return_tensors="pt",
        )

        return {k: v.to(device) for k, v in inputs.items()}

    @torch.inference_mode()
    def extract_features(
        self,
        reference_frames: torch.Tensor,
        target_frame: torch.Tensor,
        prompt: str,
        task_template: Optional[str] = None,
    ) -> torch.Tensor:
        """Extract semantic features from MLLM for TMA.

        Quote: "The MLLM outputs are aggregated using task-specific MetaQueries
        through cross-attention." (Section 4.4)

        Args:
            reference_frames: Reference video/image frames
            target_frame: Target first frame
            prompt: Text prompt
            task_template: Optional task template

        Returns:
            MLLM hidden states [B, seq_len, hidden_dim]
        """
        # Prepare inputs
        inputs = self.prepare_inputs(
            reference_frames, target_frame, prompt, task_template
        )

        # Forward pass to get hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get last hidden state
        # Shape: [B, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]

        return hidden_states

    def forward(
        self,
        reference_frames: torch.Tensor,
        target_frame: torch.Tensor,
        prompt: str,
        task_template: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass for TMA feature extraction.

        Args:
            reference_frames: Reference video/image [B, T, C, H, W] or [B, C, H, W]
            target_frame: Target first frame [B, C, H, W]
            prompt: Text prompt
            task_template: Optional task template

        Returns:
            MLLM features [B, seq_len, hidden_dim]
        """
        return self.extract_features(
            reference_frames, target_frame, prompt, task_template
        )


class QwenVLTMAIntegration(nn.Module):
    """Integration module connecting Qwen VL to TMA component.

    This module combines the Qwen VL feature extractor with the TMA
    MetaQuery cross-attention and connector MLP.

    Architecture:
    1. Qwen VL extracts semantic features from reference + target + prompt
    2. MetaQueries (learnable, task-specific) attend to MLLM features
    3. Connector MLP projects to LTX-2 dimension
    4. Output is injected into target branch cross-attention
    """

    def __init__(
        self,
        qwen_config: QwenVLConfig,
        output_dim: int = 4096,  # LTX-2 cross-attention dim
        num_queries_per_task: int = 8,
        num_connector_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initialize Qwen VL feature extractor
        self.qwen_vl = QwenVLFeatureExtractor(qwen_config)
        mllm_hidden_dim = self.qwen_vl.hidden_dim

        # Import TMA component
        from .components import MetaQueryBank, OmniTransferTask

        # MetaQuery bank
        self.meta_query_bank = MetaQueryBank(
            num_tasks=6,
            num_queries_per_task=num_queries_per_task,
            query_dim=mllm_hidden_dim,
        )

        # Cross-attention for MetaQuery aggregation
        self.query_attn = nn.MultiheadAttention(
            embed_dim=mllm_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Three-layer MLP connector
        connector_layers = []
        dims = [mllm_hidden_dim, mllm_hidden_dim, mllm_hidden_dim, output_dim]

        for i in range(num_connector_layers):
            connector_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_connector_layers - 1:
                connector_layers.append(nn.GELU())
                connector_layers.append(nn.Dropout(dropout))

        self.connector = nn.Sequential(*connector_layers)

        # Normalization layers
        self.input_norm = nn.LayerNorm(mllm_hidden_dim)
        self.output_norm = nn.LayerNorm(output_dim)

        logger.info(
            f"Initialized QwenVLTMAIntegration: "
            f"mllm_dim={mllm_hidden_dim}, output_dim={output_dim}"
        )

    def forward(
        self,
        reference_frames: torch.Tensor,
        target_frame: torch.Tensor,
        prompt: str,
        task: "OmniTransferTask",
        task_template: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract and align MLLM features for TMA injection.

        Args:
            reference_frames: Reference video/image
            target_frame: Target first frame
            prompt: Text prompt
            task: OmniTransfer task type
            task_template: Optional task-specific template
            attention_mask: Optional attention mask

        Returns:
            Task-aligned features [B, num_queries, output_dim]
        """
        # Extract MLLM features
        mllm_features = self.qwen_vl(
            reference_frames, target_frame, prompt, task_template
        )

        batch_size = mllm_features.shape[0]

        # Normalize
        mllm_features = self.input_norm(mllm_features)

        # Get task-specific MetaQueries
        meta_queries = self.meta_query_bank(task, batch_size)
        meta_queries = meta_queries.to(mllm_features.device, mllm_features.dtype)

        # Cross-attention: MetaQueries attend to MLLM features
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


# Task templates for different OmniTransfer tasks
TASK_TEMPLATES = {
    "motion_transfer": (
        "Analyze the motion patterns in the reference video. "
        "The motion should be transferred to animate the target subject while "
        "preserving its identity and appearance."
    ),
    "pose_reenactment": (
        "Extract the pose sequence from the reference video. "
        "Apply this pose sequence to the target subject while maintaining "
        "the subject's identity and visual characteristics."
    ),
    "style_transfer": (
        "Identify the visual style characteristics in the reference video "
        "including color palette, artistic effects, and rendering style. "
        "Apply this style to the target while preserving its content structure."
    ),
    "identity_preservation": (
        "Focus on the identity features of the subject in the reference. "
        "Ensure these identity characteristics are preserved when the subject "
        "is placed in the new scene or context from the target."
    ),
    "action_customization": (
        "Analyze the specific action being performed in the reference video. "
        "Transfer this action to the target subject while adapting to their "
        "physical characteristics."
    ),
    "scene_composition": (
        "Extract scene elements and composition from the reference. "
        "Compose the target subject into this scene context while maintaining "
        "visual coherence and proper integration."
    ),
}


def get_task_template(task_type: str) -> str:
    """Get the task-specific template for TMA.

    Args:
        task_type: Task type string (e.g., 'motion_transfer')

    Returns:
        Task-specific template text
    """
    return TASK_TEMPLATES.get(task_type, "")
