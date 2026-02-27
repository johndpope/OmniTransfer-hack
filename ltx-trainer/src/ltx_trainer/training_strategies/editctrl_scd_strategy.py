"""EditCtrl + SCD training strategy for LTX-2.

Combines two complementary papers:
  1. EditCtrl (arXiv:2312.03052) — "EditCtrl: Learning Controllable Video Editing"
     Proposes a mask-and-fill paradigm for targeted video editing where an edit mask
     defines WHICH spatial tokens should be regenerated (the edit region), while
     unmasked tokens retain clean source content. Introduces:
       - LocalContextModule (LCM): Processes sparse source tokens at edit-region
         boundary to produce a local control signal injected into the decoder.
         (EditCtrl Paper, Section 3.2: "Local Context Extraction")
       - GlobalContextEmbedder (GCE): Processes downsampled background tokens for
         global scene context, prepended to text context for cross-attention.
         (EditCtrl Paper, Section 3.3: "Global Context Embedding")
       - Mask dilation in latent space for boundary context.
         (EditCtrl Paper, Section 3.2: "We dilate the binary mask by d pixels in
         the latent space to include boundary tokens as local context")
       - Loss masking: Loss computed ONLY on edit-region tokens.
         (EditCtrl Paper, Section 3.4: "Masked Loss Formulation")
       - Two-phase training: Phase 1 trains LCM + LoRA, Phase 2 freezes LCM and
         trains GCE. (EditCtrl Paper, Section 4.1: "Training Procedure")

  2. Separable Causal Diffusion (SCD) (arXiv:2602.10095) — see scd_strategy.py
     Splits the DiT into an encoder (clean latents, causal mask, run once) and a
     decoder (noisy latents, run N times). This strategy uses SCD's encoder to
     extract temporal features from the clean source video, then feeds those
     shifted features into the decoder alongside EditCtrl's local/global controls.

Combined EditCtrl+SCD pipeline (per training step):
  1. SCD Encoder (layers 0-31): Clean source latents + causal mask -> encoder features
     (SCD Paper Section 3.4: "The encoder processes clean video with causal attention")
  2. Shift encoder features by 1 frame (SCD Section 3.4: frame t-1 context -> frame t)
  3. Generate edit mask (random rectangles/ellipses or semantic object masks)
  4. Dilate mask for boundary context (EditCtrl Section 3.2)
  5. LCM: sparse source tokens at dilated boundary -> local control signal
     (EditCtrl Section 3.2: "The LCM attends over boundary tokens to produce
     a control embedding that guides the decoder's inpainting of the edit region")
  6. (Phase 2) GCE: background tokens -> global context prepended to text
     (EditCtrl Section 3.3: "The GCE provides holistic scene understanding")
  7. Construct noisy latents: noise at masked positions, clean source at unmasked
     (EditCtrl Section 3.1: "Mask-and-Fill — unmasked tokens remain noise-free")
  8. SCD Decoder (layers 32-47): noisy latents + shifted features + controls -> velocity
  9. Loss: MSE only on edit-masked tokens (EditCtrl Section 3.4)

Paired Edit Mode (inspired by Ditto-1M, arXiv:2401.12945):
  - In "reconstruction" mode, source and target are the same video (self-supervised).
  - In "paired_edit" mode, the decoder target is a ground-truth edited video, NOT
    the source. The encoder still sees the clean source, and the edit instruction
    text replaces the source caption for the decoder's cross-attention.
  - This enables training on real edit datasets (source -> edited pairs) instead
    of only reconstruction-based self-supervision.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_core.model.transformer.editctrl_modules import (
    GlobalContextEmbedder,
    LocalContextModule,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.scd_model import (
    LTXSCDModel,
    shift_encoder_features,
)
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
)
from ltx_trainer.training_strategies.scd_strategy import (
    SCDTrainingConfig,
    SCDTrainingStrategy,
)

# Add mask utils path — these utilities implement the mask generation and manipulation
# operations described in EditCtrl Paper Section 3.1-3.2:
#   - generate_random_token_masks: Synthetic rectangle/ellipse masks in token space
#     (EditCtrl Paper Section 4.2: "We use random rectangular and elliptical masks
#     during training for generalization")
#   - dilate_token_mask: Morphological dilation of edit mask in latent space
#     (EditCtrl Paper Section 3.2: "dilate by d pixels for boundary context")
#   - gather_masked_tokens / scatter_masked_tokens: Sparse token extraction at
#     dilated positions for efficient LCM processing
#     (EditCtrl Paper Section 3.2: "Only boundary tokens are processed by the LCM")
#   - prepare_background_latents: Downsample non-edit-region tokens for GCE
#     (EditCtrl Paper Section 3.3: "Background tokens are spatially downsampled")
#   - SemanticMaskLibrary / generate_semantic_masks: Object-aware mask generation
#     using pre-computed Qwen VL detections (extension beyond original EditCtrl paper)
sys.path.insert(0, "/home/johndpope/Documents/GitHub/sparse-causal-diffusion")
from scd.utils.mask_utils import (
    SemanticMaskLibrary,
    dilate_token_mask,
    gather_masked_tokens,
    generate_random_token_masks,
    generate_semantic_masks,
    prepare_background_latents,
    scatter_masked_tokens,
)


class EditCtrlSCDConfig(SCDTrainingConfig):
    """Configuration for EditCtrl + SCD training strategy.

    EditCtrl Paper Section 4.1 (Training Procedure):
        Training is split into two phases to stabilize learning:
        - Phase 1: Train LCM + EditCtrl LoRA only (local boundary context learning).
          The LCM learns to extract edit-boundary features from sparse source tokens.
          The LoRA adapters learn to incorporate these features during decoding.
        - Phase 2: Freeze LCM, add and train GCE (global scene context learning).
          With local context already learned, the GCE learns to provide complementary
          global scene understanding (lighting, layout, style) from the background.

    Inherits from SCDTrainingConfig to reuse the SCD encoder/decoder split settings
    (encoder_layers, decoder_input_combine, first_frame_conditioning_p, etc.).
    """

    name: Literal["editctrl_scd"] = "editctrl_scd"

    # =====================================================================
    # EditCtrl Paper Section 4.1: Two-phase training
    # Phase 1: LCM + LoRA learn local edit-boundary context
    # Phase 2: GCE learns global scene context (LCM frozen)
    # =====================================================================
    editctrl_phase: int = Field(
        default=1,
        description="Training phase: 1=local context only, 2=local+global context",
        ge=1,
        le=2,
    )

    # EditCtrl Paper Section 3.2 (Local Context Extraction):
    # "The LCM is a lightweight transformer with N_L blocks that processes
    # sparse source tokens at dilated mask positions."
    # 4 blocks found to be a good balance between capacity and compute.
    local_num_blocks: int = Field(
        default=4,
        description="Number of transformer blocks in LocalContextModule",
        ge=1,
        le=8,
    )

    # EditCtrl Paper Section 3.2: "We dilate the binary edit mask by d pixels
    # in the latent space to include boundary tokens as local context for the LCM.
    # Dilation provides the LCM with a 'halo' of clean source information
    # surrounding the edit region, enabling smooth inpainting at boundaries."
    # d=2 in latent space corresponds to ~16 pixels in pixel space (8x VAE downsampling).
    mask_dilation_latent: int = Field(
        default=2,
        description="Dilation of edit mask in latent token space (provides boundary context)",
        ge=0,
    )

    # EditCtrl Paper Section 3.3 (Global Context Embedding):
    # "Background tokens are spatially downsampled to N_g tokens, providing
    # a compressed representation of the global scene context (lighting,
    # layout, non-edited regions)."
    global_context_num_tokens: int = Field(
        default=256,
        description="Number of background tokens for GlobalContextEmbedder",
    )

    # EditCtrl Paper Section 4.2 (Training Data):
    # "Random rectangular and elliptical masks with area fraction in [5%, 60%]
    # are used during training for generalization across different edit sizes."
    # Masks too small (<5%) provide insufficient training signal.
    # Masks too large (>60%) remove too much context, making the task trivially
    # close to unconditional generation rather than editing.
    mask_min_area: float = Field(
        default=0.05,
        description="Minimum mask area fraction for synthetic masks",
        ge=0.01,
        le=0.5,
    )

    mask_max_area: float = Field(
        default=0.6,
        description="Maximum mask area fraction for synthetic masks",
        ge=0.1,
        le=0.95,
    )

    # EditCtrl Paper Section 3.2: LCM transformer dimension.
    # Default 0 means "match the base DiT's inner_dim" (4096 for LTX-2).
    # A smaller inner_dim reduces LCM compute but may limit expressivity.
    local_inner_dim: int = Field(
        default=0,
        description="Inner dim for LocalContextModule (0 = use transformer inner_dim 4096)",
    )

    local_heads: int = Field(
        default=0,
        description="Number of attention heads in LCM (0 = auto: inner_dim / 64)",
    )

    local_dim_head: int = Field(
        default=64,
        description="Dimension per attention head in LCM",
    )

    # EditCtrl Paper Section 3.3: "The edit region in background latents is filled
    # with a constant value (0.5) to indicate 'unknown' content."
    # This prevents the GCE from seeing edited content and forces it to rely
    # only on the surrounding background for global context.
    background_fill_value: float = Field(
        default=0.5,
        description="Value to fill masked (edit) region in background latents (paper: 0.5)",
        ge=0.0,
        le=1.0,
    )

    # EditCtrl Paper Section 3.2: "The LCM input is the concatenation of
    # encoded source tokens E(V_b) and the downsampled edit mask V_m, giving
    # the LCM both spatial content and awareness of the edit boundary."
    # C = [E(V_b), V_m↓] means the mask is appended as an extra channel.
    mask_channel_concat: bool = Field(
        default=True,
        description="Concatenate edit mask as extra channel to LCM input (paper: C=[E(V_b),V_m↓])",
    )

    # EditCtrl Paper Section 3.2: "The local control signal can be injected
    # either once before the decoder blocks ('pre_decoder') or at selected
    # decoder block outputs ('per_layer') for finer-grained control."
    # Per-layer injection is more expressive — the control signal adapts to
    # different abstraction levels within the decoder hierarchy.
    local_control_injection: str = Field(
        default="per_layer",
        description="How to inject local control: 'pre_decoder' (once before blocks) or 'per_layer' (after FFN at selected blocks)",
    )

    local_control_layers: list[int] | None = Field(
        default=None,
        description="Which decoder block indices to inject local control at (None = all blocks). Only used when local_control_injection='per_layer'",
    )

    # EditCtrl Paper Section 4.1: During Phase 1, the base SCD LoRA weights
    # (if any exist from prior SCD training) should be frozen so that only
    # the new EditCtrl-specific LoRA adapters and LCM are trained.
    # This prevents catastrophic forgetting of the base model's video generation
    # capabilities while learning edit-specific skills.
    freeze_base_lora: bool = Field(
        default=True,
        description="Freeze the base SCD LoRA weights (only train EditCtrl LoRA + modules)",
    )

    gradient_checkpointing_local: bool = Field(
        default=True,
        description="Enable gradient checkpointing on LocalContextModule blocks",
    )

    # =====================================================================
    # TMA (Task-adaptive Multimodal Alignment) — from OmniTransfer (arXiv:2601.14250v1)
    # Section 4.3: "TMA uses a multimodal LLM (MLLM) with learnable MetaQuery
    # tokens to extract task-specific semantic features from the reference/source."
    #
    # In the EditCtrl+SCD context, TMA provides semantic understanding of the
    # source scene via Qwen2.5-VL features. The MetaQuery tokens cross-attend
    # to these features, producing compact semantic embeddings that are prepended
    # to the text context for the decoder's cross-attention. This gives the
    # decoder richer scene understanding beyond what the text caption provides,
    # e.g., object relationships, spatial layout, and visual attributes that
    # are difficult to capture in text alone.
    # =====================================================================
    use_tma: bool = Field(
        default=False,
        description="Enable TMA with pre-computed Qwen VL features for semantic guidance",
    )

    # OmniTransfer Paper Section 4.3: "We use Qwen2.5-VL as the MLLM backbone."
    # The hidden dimension depends on the Qwen VL model variant.
    tma_mllm_hidden_dim: int = Field(
        default=3584,
        description="Qwen VL hidden dimension (7B=3584, 3B=2048)",
    )

    # OmniTransfer Paper Section 4.3: TMA output must match the text encoder's
    # embedding dimension so TMA tokens can be seamlessly concatenated with
    # text embeddings for the decoder's cross-attention layers.
    # For LTX-2 with Gemma text encoder, this is 3840.
    tma_output_dim: int = Field(
        default=3840,
        description="TMA output dim — must match Gemma embedding dim (3840 for LTX-2)",
    )

    # OmniTransfer Paper Section 4.3: "We use N_q learnable MetaQuery tokens
    # that cross-attend to the MLLM's output features. Each task type has its
    # own set of MetaQuery tokens to capture task-specific semantics."
    tma_num_queries: int = Field(
        default=8,
        description="Number of learnable MetaQuery tokens per task type",
    )

    tma_connector_layers: int = Field(
        default=3,
        description="Number of layers in the TMA connector MLP",
    )

    tma_features_dir: str = Field(
        default="qwen_vl_features",
        description="Subdirectory name for cached Qwen VL features under data root",
    )

    # =====================================================================
    # Mask source settings
    #
    # EditCtrl Paper Section 4.2 (Training Data — Mask Generation):
    # "We use random rectangular and elliptical masks during training for
    # generalization across different edit sizes and shapes."
    #
    # Extension beyond original EditCtrl paper: We additionally support
    # semantic object-level masks from Qwen2.5-VL detections. This produces
    # more realistic edit regions (e.g., "replace this person" rather than
    # "replace this random rectangle"), improving the model's ability to
    # handle real-world editing tasks with irregular object boundaries.
    #
    # 'mixed' mode blends random and semantic masks within each batch,
    # providing both diversity and semantic realism.
    # =====================================================================
    mask_source: str = Field(
        default="random",
        description="Mask generation mode: 'random' (rectangles/ellipses), "
                    "'semantic' (per-sample Qwen VL object masks), 'mixed' (blend both)",
    )

    semantic_mask_dir: str | None = Field(
        default=None,
        description="Path to pre-computed per-sample semantic masks (output of compute_semantic_masks.py). "
                    "Each {idx:03d}.pt contains 'masks' [N_objects, H, W] and 'objects' list.",
    )

    semantic_mask_ratio: float = Field(
        default=0.5,
        description="Fraction of samples using semantic masks in 'mixed' mode",
        ge=0.0,
        le=1.0,
    )

    # =====================================================================
    # Paired edit training fields
    #
    # Inspired by Ditto-1M (arXiv:2401.12945) and InstructPix2Pix (arXiv:2211.09800):
    # Instead of only self-supervised reconstruction (source = target), paired edit
    # mode trains the model on actual source->edited video pairs where:
    #   - The SCD encoder sees the CLEAN SOURCE video (before edit)
    #   - The decoder's denoising target is the GROUND-TRUTH EDITED video (after edit)
    #   - The decoder's text context is the EDIT INSTRUCTION (not the source caption)
    #
    # This is crucial for learning instruction-following behavior. With reconstruction
    # mode alone, the model only learns to reconstruct the source (identity mapping
    # within the edit region). Paired edit mode teaches the model to:
    #   (a) Understand edit instructions ("remove the car", "change color to red")
    #   (b) Generate plausible edited content that matches the instruction
    #   (c) Preserve non-edit regions faithfully
    #
    # 'mixed' mode blends both reconstruction and paired edit samples within each
    # batch, providing both strong reconstruction grounding and edit-following ability.
    # =====================================================================
    data_mode: str = Field(
        default="reconstruction",
        description="Training data mode: 'reconstruction' (source=target), "
                    "'paired_edit' (source→edited target), 'mixed' (blend both based on paired_edit_ratio)",
    )

    edited_latents_dir: str | None = Field(
        default=None,
        description="Path to edited_latents/ dir with per-sample .pt files. "
                    "Files named {edit_id:04d}_{sample_id:03d}.pt",
    )

    # Edit conditions contain the text embeddings for the EDIT INSTRUCTION
    # (e.g., "remove the person on the left"), NOT the source video's caption.
    # This follows InstructPix2Pix's approach where the decoder is conditioned
    # on the edit instruction while the encoder provides source visual context.
    edit_conditions_dir: str | None = Field(
        default=None,
        description="Path to edit_conditions/ dir with per-edit caption embeddings. "
                    "Files named {edit_id:04d}.pt in conditions_final format.",
    )

    paired_edit_ratio: float = Field(
        default=0.5,
        description="In 'mixed' data_mode, fraction of samples using paired edit targets vs reconstruction. "
                    "In 'paired_edit' mode, this is ignored (always 1.0).",
        ge=0.0,
        le=1.0,
    )


class EditCtrlSCDTrainingStrategy(SCDTrainingStrategy):
    """EditCtrl + SCD training strategy.

    EditCtrl Paper (arXiv:2312.03052), Section 3 (Method Overview):
        The core idea is a "mask-and-fill" paradigm: given a source video and an edit
        mask defining the region to modify, the model regenerates content ONLY within
        the masked region while preserving unmasked content exactly. Two auxiliary
        modules provide context for coherent inpainting:
        - LocalContextModule (LCM): Boundary-aware local context from sparse source tokens
        - GlobalContextEmbedder (GCE): Scene-level global context from downsampled background

    Combined with SCD (arXiv:2602.10095):
        The SCD encoder-decoder split provides an efficient backbone:
        1. SCD Encoder: Runs once on clean source video with causal attention mask,
           extracting temporal features. (SCD Paper Section 3.4)
        2. 1-frame shift: Frame t's decoder gets frame (t-1)'s encoder features.
           (SCD Paper Section 3.4: causal conditioning)
        3. SCD Decoder: Runs with noisy latents (noise in edit region, clean elsewhere)
           plus shifted encoder features + LCM local control + GCE global context.

    The strategy inherits SCD's encoder pass and extends the decoder with EditCtrl's
    mask-conditioned controls. Loss is computed ONLY on edit-region tokens
    (EditCtrl Paper Section 3.4), unlike SCD's full-sequence loss.

    Training phases (EditCtrl Paper Section 4.1):
        Phase 1: Train LCM + EditCtrl LoRA. The LCM learns boundary-aware local
                 context extraction. The LoRA adapters learn to incorporate the
                 local control signal during decoding. Base SCD weights are frozen.
        Phase 2: Freeze LCM. Train GCE + (optionally) fine-tune LoRA. The GCE
                 learns complementary global scene context from the background.
    """

    config: EditCtrlSCDConfig

    def __init__(self, config: EditCtrlSCDConfig):
        super().__init__(config)
        # EditCtrl Paper Section 3.2: LocalContextModule — a lightweight transformer
        # that processes sparse source tokens at the dilated edit-mask boundary.
        # Set by trainer after instantiation via set_local_context_module().
        self._local_context_module: LocalContextModule | None = None
        # EditCtrl Paper Section 3.3: GlobalContextEmbedder — processes downsampled
        # background tokens for global scene context. Only active in Phase 2.
        # Set by trainer via set_global_embedder().
        self._global_embedder: GlobalContextEmbedder | None = None
        # OmniTransfer Paper Section 4.3: TMA module for semantic guidance via
        # Qwen VL features. Optional enhancement beyond the core EditCtrl approach.
        self._tma: torch.nn.Module | None = None
        self._mask_library: SemanticMaskLibrary | None = None
        # Per-sample pre-computed semantic masks from Qwen VL object detections.
        # Keys are sample indices, values contain 'masks' [N_obj, H, W] and 'objects' list.
        self._per_sample_masks: dict[int, dict] | None = None

        # Paired edit data caches (Ditto-1M-inspired paired edit mode)
        # Maps sample_idx -> list of available edit variants (different edits of same source)
        self._edit_index: dict[int, list[dict]] | None = None  # sample_idx -> list of edit variants
        self._edit_conditions_cache: dict[int, dict] = {}  # edit_id -> conditions dict

        # Load per-sample semantic masks from compute_semantic_masks.py output.
        # These masks are generated by running Qwen2.5-VL object detection on the
        # source video frames, producing per-object bounding box masks that can be
        # used as semantically meaningful edit regions (e.g., "edit this person").
        if config.mask_source in ("semantic", "mixed") and config.semantic_mask_dir:
            mask_dir = config.semantic_mask_dir
            self._per_sample_masks = {}
            count = 0
            for pt_file in sorted(Path(mask_dir).glob("*.pt")):
                try:
                    idx = int(pt_file.stem)
                    data = torch.load(pt_file, weights_only=False, map_location="cpu")
                    if data.get("num_objects", 0) > 0:
                        self._per_sample_masks[idx] = data
                        count += 1
                except (ValueError, RuntimeError):
                    continue
            logger.info(
                f"EditCtrl mask_source={config.mask_source}, "
                f"loaded {count} per-sample semantic masks from {mask_dir}, "
                f"ratio={config.semantic_mask_ratio}"
            )

        # Load paired edit index if data_mode requires it.
        # The index maps each source sample to its available edit variants,
        # enabling random selection of edit pairs during training.
        if config.data_mode in ("paired_edit", "mixed") and config.edited_latents_dir:
            self._load_edit_index(config.edited_latents_dir)

    def set_local_context_module(self, module: LocalContextModule) -> None:
        """Set the LocalContextModule. Called by trainer after instantiation.

        EditCtrl Paper Section 3.2: The LCM is a lightweight transformer that
        processes sparse source tokens gathered at the dilated edit-mask boundary.
        It produces a local control signal of shape [B, seq_len, D] that guides
        the decoder's inpainting of the edit region. The LCM is the primary
        trainable component in Phase 1.
        """
        self._local_context_module = module

    def set_global_embedder(self, module: GlobalContextEmbedder) -> None:
        """Set the GlobalContextEmbedder. Called by trainer for Phase 2.

        EditCtrl Paper Section 3.3: The GCE processes spatially downsampled
        background tokens (non-edit-region) to provide global scene context
        (lighting, layout, style). Its output is prepended to the text context
        for the decoder's cross-attention, complementing the LCM's local boundary
        context with holistic scene understanding.
        """
        self._global_embedder = module

    def set_tma(self, module: torch.nn.Module) -> None:
        """Set the TMA module. Called by trainer when use_tma=True.

        OmniTransfer Paper Section 4.3: TMA (Task-adaptive Multimodal Alignment)
        uses learnable MetaQuery tokens to cross-attend over Qwen2.5-VL features,
        producing compact semantic embeddings prepended to text context.
        """
        self._tma = module

    def _load_edit_index(self, edited_latents_dir: str) -> None:
        """Load the edit index mapping sample_idx -> available edit variants.

        Paired edit mode (inspired by Ditto-1M, arXiv:2401.12945):
            The edit index is a JSON file mapping each source sample to its
            available edit variants. Each variant specifies:
              - 'path': filename of the pre-encoded edited latent .pt file
              - 'edit_id': ID for looking up the edit instruction's text embedding
              - 'object_idx': (optional) which detected object was edited, for
                aligning the edit mask with the ground-truth edited region

            This enables diverse training: a single source video can have multiple
            edit variants (e.g., "remove person", "change car color", "add tree"),
            and we randomly sample one variant per training step.
        """
        index_path = Path(edited_latents_dir) / "_index.json"
        if not index_path.exists():
            logger.warning(
                f"Paired edit index not found at {index_path}. "
                f"Falling back to reconstruction mode."
            )
            return

        with open(index_path) as f:
            raw_index = json.load(f)

        # Convert string keys to int
        self._edit_index = {}
        for sample_key, variants in raw_index.items():
            self._edit_index[int(sample_key)] = variants

        total_variants = sum(len(v) for v in self._edit_index.values())
        logger.info(
            f"Loaded paired edit index: {len(self._edit_index)} samples, "
            f"{total_variants} total edit variants from {edited_latents_dir}"
        )

    def _load_edited_latents(
        self, sample_idx: int, edit_variant: dict, device: torch.device, dtype: torch.dtype,
    ) -> Tensor | None:
        """Load pre-encoded edited latents for a specific sample + edit variant.

        Ditto-1M / InstructPix2Pix paradigm:
            These are VAE-encoded latents of the EDITED video (ground truth after edit).
            The decoder learns to denoise toward this target instead of the source video.
            The source video is still used for:
              (a) The SCD encoder pass (temporal feature extraction from clean source)
              (b) Filling unmasked regions in the noisy decoder input
              (c) LCM boundary context extraction

        Returns patchified latents [1, seq_len, C] or None on failure.
        """
        edited_latents_dir = self.config.edited_latents_dir
        if not edited_latents_dir:
            return None

        path_name = edit_variant["path"]
        full_path = Path(edited_latents_dir) / path_name
        if not full_path.exists():
            return None

        try:
            data = torch.load(full_path, weights_only=False, map_location="cpu")
            latents = data["latents"]  # [128, F', H', W']
            # Add batch dim and patchify: [1, C, F, H, W] → [1, seq_len, C]
            latents = latents.unsqueeze(0).to(device=device, dtype=dtype)
            latents = self._video_patchifier.patchify(latents)
            return latents
        except Exception as e:
            logger.warning(f"Failed to load edited latents from {full_path}: {e}")
            return None

    def _load_edit_conditions(
        self, edit_id: int, device: torch.device, dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor] | None:
        """Load pre-computed edit caption embeddings.

        InstructPix2Pix (arXiv:2211.09800) paradigm:
            In paired edit mode, the decoder's text context is the EDIT INSTRUCTION
            (e.g., "remove the person and replace with grass"), NOT the source video's
            caption. This teaches the model to follow editing instructions.

            The edit conditions are pre-computed Gemma text embeddings of the edit
            instruction, stored separately from the source caption embeddings because
            not all samples have paired edits (reconstruction samples reuse source captions).

        Returns (video_prompt_embeds [1, seq, dim], prompt_attention_mask [1, seq]) or None.
        """
        # Check cache first
        if edit_id in self._edit_conditions_cache:
            cached = self._edit_conditions_cache[edit_id]
            return (
                cached["video_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype),
                cached["prompt_attention_mask"].unsqueeze(0).to(device=device),
            )

        edit_conditions_dir = self.config.edit_conditions_dir
        if not edit_conditions_dir:
            return None

        cond_path = Path(edit_conditions_dir) / f"{edit_id:04d}.pt"
        if not cond_path.exists():
            return None

        try:
            data = torch.load(cond_path, weights_only=False, map_location="cpu")
            self._edit_conditions_cache[edit_id] = data
            return (
                data["video_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype),
                data["prompt_attention_mask"].unsqueeze(0).to(device=device),
            )
        except Exception as e:
            logger.warning(f"Failed to load edit conditions from {cond_path}: {e}")
            return None

    def _get_paired_edit_for_sample(
        self, sample_idx: int, object_idx: int | None = None,
    ) -> dict | None:
        """Pick a random edit variant for a sample, optionally matching object_idx.

        When object_idx is provided (from semantic mask selection), we prefer edit
        variants that target the same object. This ensures consistency between the
        edit mask (which object is being edited) and the edited ground truth (which
        object was actually changed). If no matching variant exists, any variant
        is chosen randomly.
        """
        if self._edit_index is None:
            return None

        variants = self._edit_index.get(sample_idx)
        if not variants:
            return None

        # If object_idx specified, prefer matching variants
        if object_idx is not None:
            matching = [v for v in variants if v.get("object_idx") == object_idx]
            if matching:
                return random.choice(matching)

        return random.choice(variants)

    def get_data_sources(self) -> dict[str, str]:
        """Return data source subdirectory mapping.

        EditCtrl inherits SCD's standard data sources (latents/ + conditions/).
        When TMA is enabled (OmniTransfer Paper Section 4.3), adds qwen_vl_features
        as an additional data source so the dataloader loads pre-computed Qwen2.5-VL
        features alongside latents/conditions.

        Note: edited_latents and edit_conditions are NOT added as standard data sources
        because they don't exist for all samples — only samples with paired edit
        variants have them. They are loaded on-demand by the strategy during
        prepare_training_inputs() when a paired edit sample is selected.
        """
        sources = super().get_data_sources() if hasattr(super(), "get_data_sources") else {}
        if self.config.use_tma:
            sources[self.config.tma_features_dir] = "qwen_vl_features"
        return sources

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return EditCtrl + TMA module parameters for the optimizer.

        EditCtrl Paper Section 4.1 (Training Procedure):
            Phase 1 trainable: LCM parameters + EditCtrl LoRA adapters
            Phase 2 trainable: GCE parameters (LCM frozen, optionally LoRA unfrozen)

        These parameters are ADDITIONAL to the LoRA parameters managed by the
        trainer. The trainer adds these to its optimizer alongside LoRA params.
        """
        params = []
        if self._local_context_module is not None:
            params.extend(p for p in self._local_context_module.parameters() if p.requires_grad)
        if self._global_embedder is not None:
            params.extend(p for p in self._global_embedder.parameters() if p.requires_grad)
        if self._tma is not None:
            params.extend(p for p in self._tma.parameters() if p.requires_grad)
        return params

    def _generate_per_sample_masks(
        self,
        batch: dict[str, Any],
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        tokens_per_frame: int,
        video_seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Generate edit masks from per-sample Qwen VL object detections.

        Extension beyond EditCtrl Paper Section 4.2 (Training Data):
            The original EditCtrl paper uses random rectangular/elliptical masks.
            This method extends that with SEMANTIC masks: pre-computed per-object
            bounding boxes from Qwen2.5-VL detection, producing realistic edit
            regions aligned with actual objects (people, cars, furniture, etc.).

            Semantic masks improve generalization to real editing tasks where users
            select specific objects to edit, rather than arbitrary rectangles.

        For each sample in the batch:
        1. Look up its index in the pre-computed semantic masks
        2. Pick a random detected object that fits area constraints
           (EditCtrl Paper Section 4.2: "mask area fraction in [5%, 60%]")
        3. Resize its bounding box mask to the current latent grid
        4. Broadcast across all frames (static scene = same mask per frame)

        Falls back to random token masks for samples without detections
        or when in 'mixed' mode and the coin flip says random.
        """
        import torch.nn.functional as F

        edit_mask = torch.zeros(batch_size, video_seq_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # In mixed mode, randomly choose semantic vs random
            if self.config.mask_source == "mixed" and random.random() >= self.config.semantic_mask_ratio:
                # Random mask for this sample
                m = generate_random_token_masks(
                    batch_size=1, seq_len=video_seq_len,
                    tokens_per_frame=tokens_per_frame,
                    min_area=self.config.mask_min_area,
                    max_area=self.config.mask_max_area,
                    device=device, height=height, width=width,
                )
                edit_mask[b] = m[0]
                continue

            # Try to get per-sample mask data
            # Dataset stores idx at top level of batch (see datasets.py line 243)
            sample_idx = None
            if "idx" in batch:
                idx_val = batch["idx"]
                sample_idx = idx_val[b].item() if isinstance(idx_val, Tensor) else int(idx_val)
            elif "latents" in batch and "idx" in batch["latents"]:
                sample_idx = batch["latents"]["idx"][b].item()

            mask_data = None
            if sample_idx is not None and self._per_sample_masks is not None:
                mask_data = self._per_sample_masks.get(sample_idx)

            if mask_data is None:
                # No mask data for this sample — try random index from available
                if self._per_sample_masks:
                    fallback_idx = random.choice(list(self._per_sample_masks.keys()))
                    mask_data = self._per_sample_masks[fallback_idx]

            if mask_data is None or mask_data.get("num_objects", 0) == 0:
                # Fallback to random mask
                m = generate_random_token_masks(
                    batch_size=1, seq_len=video_seq_len,
                    tokens_per_frame=tokens_per_frame,
                    min_area=self.config.mask_min_area,
                    max_area=self.config.mask_max_area,
                    device=device, height=height, width=width,
                )
                edit_mask[b] = m[0]
                continue

            # Pick a random object from detected objects
            object_masks = mask_data["masks"]  # [N_objects, stored_h, stored_w]
            objects_list = mask_data["objects"]
            n_objects = object_masks.shape[0]

            # Filter by area constraints
            valid_indices = []
            for i in range(n_objects):
                bbox = objects_list[i]["bbox_norm"]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if self.config.mask_min_area <= area <= self.config.mask_max_area:
                    valid_indices.append(i)

            if not valid_indices:
                # No objects in area range — use any object
                valid_indices = list(range(n_objects))

            obj_idx = random.choice(valid_indices)
            obj_mask = object_masks[obj_idx]  # [stored_h, stored_w]
            obj_name = objects_list[obj_idx]["name"]

            # Resize to current latent dims [height, width]
            obj_mask_resized = F.interpolate(
                obj_mask.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, stored_h, stored_w]
                size=(height, width),
                mode="nearest",
            ).squeeze()  # [height, width]

            # Broadcast across all frames → [num_frames, height, width]
            frame_mask = obj_mask_resized.unsqueeze(0).expand(num_frames, -1, -1)

            # Flatten to token sequence [seq_len]
            token_mask = frame_mask.reshape(-1) > 0.5  # [num_frames * height * width]
            edit_mask[b] = token_mask.to(device)

        return edit_mask

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare EditCtrl + SCD training inputs.

        This is the main training step method that implements the combined
        EditCtrl+SCD pipeline described in the module docstring.

        EditCtrl Paper Section 3 (Method) + SCD Paper Section 3.4:
        Overrides the SCD strategy's prepare_training_inputs to add:
        1. Edit mask generation in token space (EditCtrl Section 3.1: mask-and-fill)
        2. Mask dilation for boundary context (EditCtrl Section 3.2)
        3. (Optional) Paired edit target/caption swapping (Ditto-1M paradigm)
        4. Noisy latent construction: noise ONLY at masked positions, clean source
           at unmasked positions (EditCtrl Section 3.1: "unmasked tokens remain
           noise-free, providing the decoder with exact source context")
        5. LocalContextModule forward pass on sparse boundary tokens
           (EditCtrl Section 3.2: local context extraction)
        6. (Phase 2) GlobalContextEmbedder forward pass on background tokens
           (EditCtrl Section 3.3: global context embedding)
        7. Edit-region-only loss mask (EditCtrl Section 3.4: masked loss)

        The SCD encoder pass is inherited from the parent strategy:
        - Clean source latents (sigma=0) + causal attention mask -> encoder features
        - 1-frame shift (SCD Section 3.4) -> shifted features for decoder conditioning

        Key difference from standard SCD:
        - SCD: ALL tokens are noisy, loss on ALL tokens (except I2V conditioning)
        - EditCtrl+SCD: Only EDIT REGION tokens are noisy, loss ONLY on edit region
        """
        # =====================================================================
        # Step 0: Extract and patchify clean source video latents
        # =====================================================================
        # These are VAE-encoded, noise-free ground truth video latents (x_0).
        # The source video serves as the "before edit" content. In the EditCtrl
        # paradigm, the SCD encoder will process these clean latents to extract
        # temporal features, and unmasked positions will retain these clean values
        # in the decoder's noisy input.
        latents = batch["latents"]
        video_latents = latents["latents"]  # [B, C, F, H, W]

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify: [B, C, F, H, W] -> [B, seq_len, C]
        # Flattens spatial+temporal dims into a token sequence.
        # seq_len = num_frames * height * width; C = 128 (latent channels for LTX-2)
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS metadata (used for temporal position embeddings / RoPE)
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values in batch: {fps.tolist()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-computed text embeddings (Gemma text encoder output).
        # In reconstruction mode: source video caption.
        # In paired edit mode: will be swapped to edit instruction for paired samples.
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # =====================================================================
        # Step 0.5 (Optional): TMA — Prepend Qwen VL semantic tokens to text context
        # =====================================================================
        # OmniTransfer Paper Section 4.3 (Task-adaptive Multimodal Alignment):
        # "TMA uses a multimodal LLM with learnable MetaQuery tokens to extract
        # task-specific semantic features from the source scene."
        #
        # The MetaQuery tokens [B, num_queries, 3840] cross-attend over Qwen2.5-VL
        # features and are PREPENDED to the text context. This gives the decoder
        # richer visual understanding of the source scene beyond what the text
        # caption alone provides — spatial layout, object attributes, lighting, etc.
        #
        # For EditCtrl, task_index=0 (editing) is used for all samples. The TMA
        # tokens help the decoder understand what the edit region should look like
        # based on the surrounding visual context from the MLLM's understanding.
        if self._tma is not None and "qwen_vl_features" in batch:
            qwen_data = batch["qwen_vl_features"]
            # qwen_features: [B, seq_len_qwen, hidden_dim] — variable-length Qwen outputs
            qwen_features = qwen_data["qwen_features"].to(device=device, dtype=dtype)
            # Use task index 0 (editing) for all samples
            task_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            # TMA forward: MetaQueries cross-attend over Qwen features
            # Output: [B, num_queries, output_dim] = [B, 8, 3840]
            tma_context = self._tma(qwen_features, task_indices)  # [B, 8, 3840]

            # Prepend TMA tokens to text context so decoder cross-attention
            # sees: [TMA_tokens | text_tokens] — TMA tokens come first for
            # priority in attention (similar to how OmniTransfer prepends
            # MetaQuery outputs before text embeddings)
            video_prompt_embeds = torch.cat([tma_context, video_prompt_embeds], dim=1)
            tma_mask = torch.ones(
                batch_size, tma_context.shape[1],
                dtype=prompt_attention_mask.dtype, device=device,
            )
            prompt_attention_mask = torch.cat([tma_mask, prompt_attention_mask], dim=1)

        tokens_per_frame = video_seq_len // num_frames

        # =====================================================================
        # Step 1: Generate edit masks in token space
        # =====================================================================
        # EditCtrl Paper Section 3.1 (Mask-and-Fill Paradigm):
        # "Given a source video V_s and an edit mask M defining the region to
        # modify, we regenerate content ONLY within M while preserving the
        # unmasked region exactly."
        #
        # The edit mask is a boolean tensor [B, seq_len] where True = edit region
        # (will be filled with noise for the decoder) and False = preserve region
        # (will retain clean source content in the decoder input).
        #
        # EditCtrl Paper Section 4.2: "Random rectangular and elliptical masks
        # with area fraction in [5%, 60%] are used during training."
        # We extend this with semantic object masks from Qwen VL detections.
        use_semantic = (
            self._per_sample_masks is not None
            and len(self._per_sample_masks) > 0
            and self.config.mask_source in ("semantic", "mixed")
        )
        if use_semantic:
            edit_mask = self._generate_per_sample_masks(
                batch, batch_size, num_frames, height, width,
                tokens_per_frame, video_seq_len, device,
            )
        else:
            edit_mask = generate_random_token_masks(
                batch_size=batch_size,
                seq_len=video_seq_len,
                tokens_per_frame=tokens_per_frame,
                min_area=self.config.mask_min_area,
                max_area=self.config.mask_max_area,
                device=device,
                height=height,
                width=width,
            )  # [B, seq_len] boolean

        # =====================================================================
        # Step 2: Dilate mask for boundary context
        # =====================================================================
        # EditCtrl Paper Section 3.2: "We dilate the binary edit mask by d pixels
        # in the latent space to include boundary tokens as local context."
        #
        # The dilated mask is a SUPERSET of the edit mask: it includes the edit
        # region itself PLUS a 'halo' of surrounding source tokens. The LCM
        # processes tokens at DILATED positions — this gives it both:
        #   (a) Clean source tokens at the boundary (for coherent inpainting)
        #   (b) Knowledge of which tokens are inside the edit region
        # The edit mask itself (non-dilated) is used for:
        #   (a) Determining which tokens get noise vs clean in the decoder input
        #   (b) Loss masking (only edit-region tokens contribute to loss)
        dilated_mask = dilate_token_mask(
            edit_mask,
            tokens_per_frame=tokens_per_frame,
            dilation=self.config.mask_dilation_latent,
            height=height,
            width=width,
        )  # [B, seq_len] boolean — superset of edit_mask

        # =====================================================================
        # Step 3: Paired edit mode — swap decoder target and text for edit samples
        # =====================================================================
        # Ditto-1M (arXiv:2401.12945) / InstructPix2Pix (arXiv:2211.09800) paradigm:
        #
        # In RECONSTRUCTION mode (default):
        #   - decoder_target_latents = source_latents (the model learns to reconstruct)
        #   - decoder_text = source caption
        #   - This is self-supervised: the model learns to inpaint the edit region
        #     to match the original source, building strong reconstruction ability.
        #
        # In PAIRED EDIT mode:
        #   - decoder_target_latents = EDITED latents (ground truth after edit)
        #   - decoder_text = EDIT INSTRUCTION (e.g., "remove the car")
        #   - The SCD encoder STILL sees the clean source video (before edit)
        #   - The decoder learns: given source context + edit instruction, generate
        #     the edited version within the mask region
        #
        # In MIXED mode:
        #   - Each sample in the batch randomly uses either reconstruction or
        #     paired edit, controlled by paired_edit_ratio
        #   - This provides both strong reconstruction grounding AND edit-following
        #
        # CRITICAL: source_latents is ALWAYS the clean source (never swapped).
        # It provides unmasked-region content and LCM boundary context regardless
        # of whether the decoder target is reconstruction or edited.
        source_latents = video_latents  # Always clean source — used for unmasked fill + LCM
        decoder_target_latents = video_latents.clone()  # Default: reconstruction target
        decoder_text = video_prompt_embeds.clone()  # Clone! In-place edits must NOT corrupt encoder text
        decoder_text_mask = prompt_attention_mask.clone()
        is_paired = torch.zeros(batch_size, dtype=torch.bool, device=device)

        use_paired = (
            self.config.data_mode in ("paired_edit", "mixed")
            and self._edit_index is not None
            and len(self._edit_index) > 0
        )

        if use_paired:
            for b in range(batch_size):
                # In 'mixed' mode, randomly decide whether this sample uses
                # paired edit or reconstruction (controlled by paired_edit_ratio)
                if self.config.data_mode == "mixed":
                    if random.random() >= self.config.paired_edit_ratio:
                        continue  # Skip -> use reconstruction for this sample

                # Get sample index for looking up available edit variants
                sample_idx = None
                if "idx" in batch:
                    idx_val = batch["idx"]
                    sample_idx = idx_val[b].item() if isinstance(idx_val, Tensor) else int(idx_val)

                if sample_idx is None:
                    continue

                # Pick a random edit variant for this sample from the index
                edit_variant = self._get_paired_edit_for_sample(sample_idx)
                if edit_variant is None:
                    continue

                # Load edited latents (ground-truth AFTER edit, VAE-encoded)
                edited = self._load_edited_latents(sample_idx, edit_variant, device, dtype)
                if edited is None:
                    continue

                # Load edit text embeddings (edit INSTRUCTION, not source caption)
                edit_id = edit_variant["edit_id"]
                edit_cond = self._load_edit_conditions(edit_id, device, dtype)

                # Swap decoder target: denoising target is now the EDITED video
                # (InstructPix2Pix paradigm: decoder learns to produce edited output)
                decoder_target_latents[b] = edited[0]  # [seq_len, C]
                is_paired[b] = True

                # Swap decoder text: replace source caption with EDIT INSTRUCTION
                # for this sample's decoder cross-attention
                if edit_cond is not None:
                    edit_embeds, edit_mask_text = edit_cond  # [1, seq, dim], [1, seq]
                    # If TMA was prepended, edit conditions need TMA tokens too.
                    # TMA tokens come from the SOURCE scene (Qwen VL features of source)
                    # and provide visual context about what the scene looks like BEFORE
                    # the edit. We keep TMA tokens unchanged and only swap the text portion.
                    if self._tma is not None and "qwen_vl_features" in batch:
                        # decoder_text layout: [TMA_tokens | text_tokens]
                        # Only swap the text_tokens portion, keep TMA from source
                        tma_len = self.config.tma_num_queries
                        decoder_text[b, tma_len:] = edit_embeds[0]
                        decoder_text_mask[b, tma_len:] = edit_mask_text[0]
                    else:
                        decoder_text[b] = edit_embeds[0]
                        decoder_text_mask[b] = edit_mask_text[0]

                # For paired edits with known object_idx, replace the random/semantic
                # mask with the GROUND-TRUTH object mask from Qwen VL detections.
                # This ensures the edit mask precisely covers the object that was
                # actually edited in the ground-truth edited video, preventing
                # mask-target misalignment that would confuse the decoder.
                obj_idx = edit_variant.get("object_idx")
                if obj_idx is not None and self._per_sample_masks is not None:
                    mask_data = self._per_sample_masks.get(sample_idx)
                    if mask_data is not None and obj_idx < mask_data["masks"].shape[0]:
                        import torch.nn.functional as F
                        obj_mask = mask_data["masks"][obj_idx]
                        obj_mask_resized = F.interpolate(
                            obj_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(height, width), mode="nearest",
                        ).squeeze()
                        frame_mask = obj_mask_resized.unsqueeze(0).expand(num_frames, -1, -1)
                        token_mask = frame_mask.reshape(-1) > 0.5
                        edit_mask[b] = token_mask.to(device)

            num_paired = is_paired.sum().item()
            if num_paired > 0:
                logger.debug(
                    f"Paired edit batch: {num_paired}/{batch_size} samples using edited targets"
                )

        # =====================================================================
        # Step 4: Sample noise and construct noisy latents
        # =====================================================================
        # Flow matching formulation (SCD Paper Section 5.2):
        #   x_t = (1 - sigma) * x_0 + sigma * epsilon
        #   v   = epsilon - x_0  (velocity target)
        #
        # EditCtrl Paper Section 3.1 (Mask-and-Fill — Critical Difference from SCD):
        # Unlike standard SCD where ALL tokens are noisy, EditCtrl applies noise
        # ONLY to edit-region tokens. Unmasked tokens retain clean source content.
        # This is the "fill" part of mask-and-fill: the decoder sees exact source
        # context outside the edit region, enabling precise preservation of
        # unedited content and coherent boundary transitions.
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)
        sigmas_expanded = sigmas.view(-1, 1, 1)

        # Noise is applied toward the DECODER TARGET:
        #   - Reconstruction mode: target = source latents (self-supervised)
        #   - Paired edit mode: target = edited latents (instruction-following)
        noisy_video = (1 - sigmas_expanded) * decoder_target_latents + sigmas_expanded * video_noise
        edit_mask_expanded = edit_mask.unsqueeze(-1)  # [B, seq_len, 1]
        # EditCtrl Paper Section 3.1: "Unmasked tokens retain clean source content."
        # torch.where(mask, noisy, clean) = noisy at edit positions, clean at preserved positions.
        # CRITICAL: Unmasked region always shows clean SOURCE (not edited target),
        # even in paired edit mode. This ensures the decoder can rely on exact
        # source context outside the edit region.
        noisy_video = torch.where(edit_mask_expanded, noisy_video, source_latents)

        # Velocity target: v = epsilon - x_0 where x_0 = decoder target
        # In reconstruction mode, this is noise - source (standard SCD).
        # In paired edit mode, this is noise - edited (teaches model to generate edit).
        video_targets = video_noise - decoder_target_latents



        # Video positions (RoPE temporal+spatial position embeddings)
        video_positions = self._get_video_positions(
            num_frames=num_frames, height=height, width=width,
            batch_size=batch_size, fps=fps, device=device, dtype=dtype,
        )

        # =====================================================================
        # Step 5: SCD Encoder Pass (inherited from SCD strategy)
        # =====================================================================
        # SCD Paper Section 3.4: "The encoder processes clean video with causal
        # attention, extracting temporal features that are reusable across
        # denoising steps."
        #
        # CRITICAL: The encoder ALWAYS sees the clean SOURCE video (not edited),
        # even in paired edit mode. This is by design:
        #   - The encoder's job is to extract temporal context from the source
        #   - The decoder uses this context to understand what the scene looks
        #     like BEFORE the edit, enabling coherent editing
        #   - In the EditCtrl paradigm, the encoder provides "what's there now"
        #     while the decoder learns "what should be there after the edit"
        if self._scd_model is None:
            raise RuntimeError("EditCtrl requires SCD model wrapper. Set via set_scd_model().")

        # SCD Paper Section 5.1: Encoder sees clean latents with timestep=0 (sigma=0).
        # All tokens are noise-free, telling the model there is no noise to denoise.
        encoder_timesteps = torch.zeros(batch_size, video_seq_len, device=device, dtype=dtype)
        encoder_modality = Modality(
            enabled=True,
            latent=video_latents,  # Clean source latents x_0 (NOT noisy, NOT edited)
            timesteps=encoder_timesteps,  # All zeros: sigma=0 means "clean signal"
            positions=video_positions,
            context=video_prompt_embeds,  # Source caption (or TMA-prepended)
            context_mask=prompt_attention_mask,
        )

        # SCD Paper Section 3.4: Encoder with frame-level causal attention mask.
        # Frame t can only attend to frames <= t (causal — no future information).
        encoder_video_args, encoder_audio_args = self._scd_model.forward_encoder(
            video=encoder_modality, audio=None, perturbations=None,
            tokens_per_frame=tokens_per_frame,
        )

        # SCD Paper Section 3.4: Shift encoder features by 1 frame.
        # Frame t's decoder receives frame (t-1)'s encoder features.
        # Frame 0 gets zero features (bootstrap — no preceding context).
        # CRITICAL: Do NOT detach! Gradients must flow back through the encoder
        # for end-to-end training (SCD Paper Section 5.1).
        encoder_features = encoder_video_args.x  # [B, seq_len, D]
        shifted_features = shift_encoder_features(encoder_features, tokens_per_frame, num_frames)

        # =====================================================================
        # Step 6: LocalContextModule (LCM) forward pass
        # =====================================================================
        # EditCtrl Paper Section 3.2 (Local Context Extraction):
        # "The LCM is a lightweight transformer that processes sparse source
        # tokens gathered at dilated mask positions. It produces a local control
        # signal that guides the decoder's inpainting of the edit region."
        #
        # Architecture: N_L transformer blocks (default 4) with:
        #   - Self-attention over sparse boundary tokens
        #   - Cross-attention to text embeddings
        #   - Timestep conditioning via AdaLN
        #
        # Input: Source tokens gathered at DILATED mask positions (boundary halo)
        #   - Edit region tokens: help LCM understand what needs to be generated
        #   - Boundary tokens (dilation halo): provide clean context for smooth transition
        #   - Optional mask channel (EditCtrl Paper: C = [E(V_b), V_m_down])
        #     appended as extra dim to distinguish edit vs boundary tokens
        #
        # Output: local_control [B, seq_len, D] — a dense control signal that is
        # injected into the decoder (either once pre-decoder or per-layer via
        # additive injection after FFN at selected blocks).
        local_control = None
        if self._local_context_module is not None:
            # EditCtrl Paper Section 3.2: "C = [E(V_b), V_m↓]" — concatenate
            # the edit mask as an extra channel to source tokens so the LCM knows
            # which tokens are inside the edit region (mask=1) vs boundary context (mask=0)
            if getattr(self._local_context_module, 'mask_channel_concat', False):
                edit_mask_channel = edit_mask.unsqueeze(-1).to(dtype=dtype)  # [B, seq_len, 1]
                lcm_input = torch.cat([video_latents, edit_mask_channel], dim=-1)  # [B, seq_len, C+1]
            else:
                lcm_input = video_latents

            # EditCtrl Paper Section 3.2: "Only tokens at dilated mask positions
            # are processed by the LCM" — gather_masked_tokens extracts a compact
            # set of sparse tokens, making the LCM efficient (processes ~10-60%
            # of tokens, not the full sequence).
            sparse_tokens, sparse_lengths = gather_masked_tokens(lcm_input, dilated_mask)

            # Get timestep embedding from the base model's Adaptive Layer Norm (AdaLN).
            # The LCM needs to know the current noise level (sigma) because its control
            # signal should adapt to the denoising stage — early steps (high sigma)
            # need stronger structural guidance, later steps (low sigma) need finer detail.
            # adaln_single returns (shift_scale_gate [B, 6*D], embedded_timestep [B, D])
            _, embedded_timestep = self._scd_model.base_model.adaln_single(
                sigmas.flatten(),  # [B] — adaln expects 1D timesteps
                None,  # hidden_dtype
            )
            timestep_emb = embedded_timestep.unsqueeze(1)  # [B, 1, base_model_dim]

            # LCM forward: sparse tokens + text context + timestep -> local control
            # The output is a DENSE [B, seq_len, D] tensor (scattered back from sparse)
            # that can be directly added to decoder features at matching positions.
            local_control = self._local_context_module(
                source_tokens=sparse_tokens,
                mask_indices=dilated_mask,
                text_context=video_prompt_embeds,
                text_mask=prompt_attention_mask,
                timestep_emb=timestep_emb,
                seq_len=video_seq_len,
            )

        # =====================================================================
        # Step 7 (Phase 2 only): GlobalContextEmbedder (GCE) forward pass
        # =====================================================================
        # EditCtrl Paper Section 3.3 (Global Context Embedding):
        # "While the LCM captures local boundary context, the GCE provides
        # holistic scene understanding from the background (non-edit region)."
        #
        # The GCE processes spatially DOWNSAMPLED background tokens where the
        # edit region is filled with a constant value (default 0.5) to prevent
        # information leakage from edited content. The output is a sequence of
        # global context tokens that can be prepended to the decoder's text
        # context for cross-attention, similar to how TMA tokens are prepended.
        #
        # EditCtrl Paper Section 4.1: "In Phase 2, the LCM is frozen and
        # the GCE is trained. This two-phase approach prevents the GCE from
        # learning to replicate local context (already handled by LCM) and
        # instead focuses on complementary global information."
        global_context = None
        if self._global_embedder is not None and self.config.editctrl_phase >= 2:
            # EditCtrl Paper Section 3.3: "Background tokens are prepared by
            # filling the edit region with fill_value (0.5) and spatially
            # downsampling to N_g tokens (default 256)."
            bg_tokens = prepare_background_latents(
                source_latents=video_latents,
                edit_mask=edit_mask,
                target_num_tokens=self.config.global_context_num_tokens,
                fill_value=self.config.background_fill_value,
            )
            global_context = self._global_embedder(bg_tokens)

        # =====================================================================
        # Step 8: Build decoder modality for the SCD decoder pass
        # =====================================================================
        # The decoder receives noisy latents (noise at edit region, clean elsewhere)
        # and produces a velocity prediction v_hat = epsilon_hat - x_0_hat.
        #
        # Key differences from standard SCD decoder setup:
        # - Noisy latents have noise ONLY at edit-region positions (Step 4 above)
        # - Text context may be edit instruction (paired mode) instead of source caption
        # - No first-frame conditioning: EditCtrl does NOT use SCD's I2V first-frame
        #   conditioning because the edit mask already handles which regions are clean.
        #   Passing zeros_like(edit_mask) as conditioning mask means no tokens get
        #   special I2V treatment (all tokens use the sampled sigma for timesteps).
        decoder_timesteps = self._create_per_token_timesteps(
            torch.zeros_like(edit_mask),  # No first-frame conditioning for EditCtrl
            sigmas.squeeze(),
        )

        decoder_modality = Modality(
            enabled=True,
            latent=noisy_video,  # Noise at edit region, clean source elsewhere
            timesteps=decoder_timesteps,  # All tokens use sampled sigma (no I2V conditioning)
            positions=video_positions,
            context=decoder_text,  # Edit caption for paired, source caption for reconstruction
            context_mask=decoder_text_mask,
        )

        # =====================================================================
        # Step 9: Loss mask — ONLY edit-region tokens contribute to loss
        # =====================================================================
        # EditCtrl Paper Section 3.4 (Masked Loss Formulation):
        # "Loss is computed only on tokens within the edit mask M. Unmasked tokens
        # are not noisy and should not contribute to the denoising loss."
        #
        # This is fundamentally different from SCD's loss masking:
        #   SCD: Loss on ALL tokens except I2V first-frame conditioning tokens
        #   EditCtrl: Loss ONLY on edit-region tokens (typically 5-60% of all tokens)
        #
        # Masked loss has two benefits:
        # 1. Prevents the model from wasting capacity on trivially predicting
        #    velocity for clean (unmasked) tokens where v = epsilon - x_0 is trivial
        # 2. Focuses learning on the hard part: generating plausible content within
        #    the edit region that is coherent with surrounding source context
        video_loss_mask = edit_mask

        # Package everything into ModelInputs for the trainer's forward pass.
        # The trainer will call scd_model.forward_decoder() with the decoder modality,
        # shifted encoder features, and EditCtrl controls (local + global).
        model_inputs = ModelInputs(
            video=decoder_modality,
            audio=None,
            video_targets=video_targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            shared_noise=video_noise,
            shared_sigmas=sigmas,
        )

        # Attach SCD-specific data for the trainer's forward pass.
        # These are accessed by the trainer to call scd_model.forward_decoder()
        # with the correct encoder features (SCD Paper Section 5.1).
        # CRITICAL: shifted_features are NOT detached — gradients flow back
        # through encoder for end-to-end training.
        model_inputs._encoder_features = shifted_features
        model_inputs._scd_model = self._scd_model
        model_inputs._encoder_audio_args = encoder_audio_args
        model_inputs._raw_video_latents = batch["latents"]["latents"]  # For reconstruction vis

        # Attach EditCtrl-specific data for the trainer's forward pass:
        # - local_control: LCM output [B, seq_len, D] injected into decoder blocks
        #   (EditCtrl Paper Section 3.2)
        # - global_context: GCE output prepended to decoder text context
        #   (EditCtrl Paper Section 3.3, Phase 2 only)
        # - edit_mask: Boolean [B, seq_len] for loss computation and visualization
        #   (EditCtrl Paper Section 3.4)
        model_inputs._local_control = local_control
        model_inputs._global_context = global_context
        model_inputs._edit_mask = edit_mask

        return model_inputs

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        """Compute masked MSE loss for EditCtrl training.

        EditCtrl Paper Section 3.4 (Masked Loss Formulation):
            "The training loss is computed only on tokens within the edit mask M,
            normalized by the number of masked tokens:

                L = sum_{i in M} ||v_hat_i - v_i||^2 / |M|

            where v = epsilon - x_0 is the velocity target and v_hat is the
            decoder's velocity prediction."

        Key difference from SCD's compute_loss (parent class):
            SCD normalizes by mean of the loss mask (fraction of active tokens),
            which keeps the effective loss magnitude constant regardless of mask size.
            EditCtrl normalizes by the COUNT of masked tokens (num_masked), which
            gives each masked token equal weight regardless of how many are masked.

            This means: small edit masks (5% of tokens) produce the same per-token
            loss magnitude as large edit masks (60% of tokens), but the total gradient
            magnitude scales with mask size. This is intentional — larger edits
            should produce stronger gradients to reflect their greater complexity.
        """
        edit_mask = getattr(inputs, "_edit_mask", inputs.video_loss_mask)

        # Velocity prediction loss: MSE between predicted and target velocity
        # video_pred: [B, seq_len, C] — decoder's velocity prediction v_hat
        # inputs.video_targets: [B, seq_len, C] — target velocity v = epsilon - x_0
        video_loss = (video_pred - inputs.video_targets).pow(2)  # [B, seq_len, C]

        # EditCtrl Paper Section 3.4: Apply edit mask — only edit-region tokens
        # contribute to the loss. Unmasked tokens are clean and should NOT
        # contribute (their velocity prediction is trivially correct since
        # the decoder sees exact clean content at those positions).
        mask_float = edit_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        masked_loss = video_loss * mask_float

        # Normalize by masked token count (not fraction) to give each masked
        # token equal weight. Clamp to avoid division by zero when mask is empty.
        num_masked = mask_float.sum().clamp(min=1.0)
        loss = masked_loss.sum() / num_masked

        return loss

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Include EditCtrl + TMA + paired edit metadata in checkpoints.

        This metadata is saved alongside the checkpoint weights and is needed
        at inference time to reconstruct the EditCtrl modules (LCM, GCE, TMA)
        with the correct architecture configuration. Without this metadata,
        the inference script would need to guess the number of LCM blocks,
        dilation settings, etc.
        """
        meta = super().get_checkpoint_metadata()
        meta.update({
            "editctrl_phase": self.config.editctrl_phase,
            "editctrl_local_num_blocks": self.config.local_num_blocks,
            "editctrl_mask_dilation": self.config.mask_dilation_latent,
            "editctrl_global_num_tokens": self.config.global_context_num_tokens,
            "editctrl_background_fill": self.config.background_fill_value,
            "editctrl_mask_channel_concat": self.config.mask_channel_concat,
            "editctrl_local_control_injection": self.config.local_control_injection,
            "editctrl_local_control_layers": self.config.local_control_layers,
            "use_tma": self.config.use_tma,
            "data_mode": self.config.data_mode,
            "paired_edit_ratio": self.config.paired_edit_ratio,
        })
        if self.config.use_tma:
            meta.update({
                "tma_mllm_hidden_dim": self.config.tma_mllm_hidden_dim,
                "tma_output_dim": self.config.tma_output_dim,
                "tma_num_queries": self.config.tma_num_queries,
                "tma_connector_layers": self.config.tma_connector_layers,
            })
        return meta

    def get_strategy_state_dict(self) -> dict[str, Tensor]:
        """Save EditCtrl + TMA module weights alongside LoRA checkpoint.

        EditCtrl Paper Section 4.1: At checkpoint time, we save the trained
        weights for all EditCtrl-specific modules (LCM, GCE, TMA) with a
        'strategy.' prefix so they can be distinguished from LoRA weights
        and loaded separately at inference time.

        Module weights saved:
          - strategy.local_context_module.* — LCM transformer blocks (Phase 1+2)
          - strategy.global_embedder.* — GCE projection layers (Phase 2)
          - strategy.tma.* — TMA MetaQuery + connector (when enabled)
        """
        state = {}
        if self._local_context_module is not None:
            for k, v in self._local_context_module.state_dict().items():
                state[f"strategy.local_context_module.{k}"] = v
        if self._global_embedder is not None:
            for k, v in self._global_embedder.state_dict().items():
                state[f"strategy.global_embedder.{k}"] = v
        if self._tma is not None:
            for k, v in self._tma.state_dict().items():
                state[f"strategy.tma.{k}"] = v
        return state

    def load_strategy_state_dict(
        self, state_dict: dict[str, Tensor]
    ) -> tuple[list[str], list[str]]:
        """Load EditCtrl + TMA module weights from checkpoint.

        EditCtrl Paper Section 4.1 (Phase Transitions):
            When transitioning from Phase 1 to Phase 2, the Phase 1 checkpoint
            contains LCM weights but NO GCE weights. The GCE is initialized
            fresh for Phase 2 training. This method handles this gracefully:
            - LCM weights: loaded if present in checkpoint AND module exists
            - GCE weights: loaded if present (Phase 2 checkpoint), skipped otherwise
            - TMA weights: loaded if present and TMA module is instantiated

            strict=False is used because the module architecture might have changed
            slightly between training runs (e.g., added a layer). Missing keys
            will be randomly initialized, extra keys will be ignored with a warning.

        Returns:
            (loaded_keys, skipped_keys): Lists of weight keys that were
            successfully loaded vs. skipped (no matching module instantiated).
        """
        loaded = []
        skipped = []

        # Extract and load LocalContextModule weights (Phase 1+2)
        lcm_prefix = "strategy.local_context_module."
        lcm_dict = {
            k[len(lcm_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(lcm_prefix)
        }
        if lcm_dict and self._local_context_module is not None:
            self._local_context_module.load_state_dict(lcm_dict, strict=False)
            loaded.extend(lcm_dict.keys())
        elif lcm_dict:
            skipped.extend(lcm_dict.keys())

        # Extract and load GlobalContextEmbedder weights (Phase 2 only)
        ge_prefix = "strategy.global_embedder."
        ge_dict = {
            k[len(ge_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(ge_prefix)
        }
        if ge_dict and self._global_embedder is not None:
            self._global_embedder.load_state_dict(ge_dict, strict=False)
            loaded.extend(ge_dict.keys())
        elif ge_dict:
            skipped.extend(ge_dict.keys())

        # Extract and load TMA weights (optional OmniTransfer enhancement)
        tma_prefix = "strategy.tma."
        tma_dict = {
            k[len(tma_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(tma_prefix)
        }
        if tma_dict and self._tma is not None:
            self._tma.load_state_dict(tma_dict, strict=False)
            loaded.extend(tma_dict.keys())
            logger.info(f"Loaded TMA weights: {len(tma_dict)} tensors")
        elif tma_dict:
            skipped.extend(tma_dict.keys())

        return loaded, skipped
