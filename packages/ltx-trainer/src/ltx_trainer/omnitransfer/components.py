"""OmniTransfer Core Components for LTX-2.

This module implements the three key components of OmniTransfer:
1. Task-aware Positional Bias (TPB) - Section 4.2 of the paper
2. Reference-decoupled Causal Learning (RCL) - Section 4.3 of the paper
3. Task-adaptive Multimodal Alignment (TMA) - Section 4.4 of the paper

References: OmniTransfer paper (arXiv:2601.14250v1, Jan 20, 2026)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ltx_core.model.transformer.rope import (
    LTXRopeType,
    apply_rotary_emb,
    precompute_freqs_cis,
)


class OmniTransferTask(Enum):
    """Task types for OmniTransfer spatio-temporal video transfer.

    Quote: "For temporal reference tasks (e.g., motion transfer, pose reenactment),
    the reference video provides temporal/motion cues that should align with the
    target's spatial layout. For appearance reference tasks (e.g., style transfer,
    identity preservation), the reference provides spatial/appearance cues that
    should flow into the target's temporal progression." (Section 4.2)
    """
    # Temporal reference tasks - reference provides motion/temporal cues
    MOTION_TRANSFER = "motion_transfer"
    POSE_REENACTMENT = "pose_reenactment"
    ACTION_CUSTOMIZATION = "action_customization"

    # Appearance reference tasks - reference provides spatial/appearance cues
    STYLE_TRANSFER = "style_transfer"
    IDENTITY_PRESERVATION = "identity_preservation"
    SCENE_COMPOSITION = "scene_composition"

    @property
    def is_temporal(self) -> bool:
        """Whether this is a temporal reference task."""
        return self in {
            OmniTransferTask.MOTION_TRANSFER,
            OmniTransferTask.POSE_REENACTMENT,
            OmniTransferTask.ACTION_CUSTOMIZATION,
        }

    @property
    def is_appearance(self) -> bool:
        """Whether this is an appearance reference task."""
        return not self.is_temporal

    @property
    def task_flag(self) -> int:
        """Task flag value for latent construction.

        Quote: "We set mref to task-specific flags: -1 for temporal tasks,
        -2 for identity-based tasks, -3 for style tasks." (Section 4.1)
        """
        if self == OmniTransferTask.MOTION_TRANSFER:
            return -1
        elif self == OmniTransferTask.POSE_REENACTMENT:
            return -1
        elif self == OmniTransferTask.ACTION_CUSTOMIZATION:
            return -1
        elif self == OmniTransferTask.IDENTITY_PRESERVATION:
            return -2
        elif self == OmniTransferTask.STYLE_TRANSFER:
            return -3
        elif self == OmniTransferTask.SCENE_COMPOSITION:
            return -3
        return -1


@dataclass
class TaskAwarePositionalBiasConfig:
    """Configuration for Task-aware Positional Bias.

    Quote: "For temporal reference tasks, we add an offset Δ = (0, w_tgt, 0) along
    the spatial (width) dimension... For appearance reference tasks, we add an
    offset Δ = (f, 0, 0) along the temporal dimension." (Section 4.2)
    """
    # Maximum position values for RoPE normalization [time, height, width]
    max_pos: list[int] | None = None
    # RoPE theta parameter
    theta: float = 10000.0
    # Whether to use middle indices for fractional positions
    use_middle_indices: bool = True


class TaskAwarePositionalBias(nn.Module):
    """Task-aware Positional Bias (TPB) for OmniTransfer.

    TPB applies distinct positional biases based on task type to enable
    effective in-context learning for different spatio-temporal transfer tasks.

    Quote: "To address this, we propose Task-aware Positional Bias (TPB), which
    applies distinct positional biases based on the task type. The key insight
    is that different tasks require different positional relationships between
    reference and target." (Section 4.2)

    For LTX-2 adaptation:
    - LTX-2 uses 3D RoPE with positions in format [B, 3, seq_len, 2] for video
    - We modify the reference positions based on task type before computing RoPE
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 32,
        config: TaskAwarePositionalBiasConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.config = config or TaskAwarePositionalBiasConfig()
        self.rope_type = rope_type

        if self.config.max_pos is None:
            # Default max positions for LTX-2: [time, height, width]
            self.config.max_pos = [20, 2048, 2048]

    def compute_task_offset(
        self,
        task: OmniTransferTask,
        target_width: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Compute positional offset based on task type.

        Quote: "R*_θ(·) = { R_θ(·, Δ=(0, w_tgt, 0)) for temporal reference;
                          R_θ(·, Δ=(f, 0, 0)) for appearance reference }" (Section 4.2)

        Args:
            task: The transfer task type
            target_width: Width of target video in latent space
            target_frames: Number of frames in target video

        Returns:
            Offset tensor of shape [3] for (time, height, width)
        """
        if task.is_temporal:
            # Temporal tasks: offset along width dimension
            # Quote: "we add an offset Δ = (0, w_tgt, 0) along the spatial (width) dimension"
            return torch.tensor([0.0, 0.0, float(target_width)])
        else:
            # Appearance tasks: offset along temporal dimension
            # Quote: "we add an offset Δ = (f, 0, 0) along the temporal dimension"
            return torch.tensor([float(target_frames), 0.0, 0.0])

    def apply_task_bias(
        self,
        ref_positions: torch.Tensor,
        task: OmniTransferTask,
        target_width: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Apply task-aware positional bias to reference positions.

        Args:
            ref_positions: Reference positions [B, 3, ref_seq_len, 2]
            task: The transfer task type
            target_width: Width of target in latent space
            target_frames: Number of target frames

        Returns:
            Biased positions [B, 3, ref_seq_len, 2]
        """
        offset = self.compute_task_offset(task, target_width, target_frames)
        offset = offset.to(ref_positions.device, ref_positions.dtype)

        # Apply offset to both start and end coordinates
        # positions shape: [B, 3, seq_len, 2] where last dim is [start, end]
        biased_positions = ref_positions.clone()
        for dim_idx in range(3):
            biased_positions[:, dim_idx, :, :] += offset[dim_idx]

        return biased_positions

    def compute_biased_rope(
        self,
        ref_positions: torch.Tensor,
        tgt_positions: torch.Tensor,
        task: OmniTransferTask,
        target_width: int,
        target_frames: int,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Compute RoPE embeddings with task-aware bias for reference.

        Args:
            ref_positions: Reference positions [B, 3, ref_seq_len, 2]
            tgt_positions: Target positions [B, 3, tgt_seq_len, 2]
            task: The transfer task type
            target_width: Width of target in latent space
            target_frames: Number of target frames
            dtype: Output dtype

        Returns:
            Tuple of (ref_freqs_cis, tgt_freqs_cis) where each is (cos, sin)
        """
        # Apply task-aware bias to reference positions
        biased_ref_positions = self.apply_task_bias(
            ref_positions, task, target_width, target_frames
        )

        # Compute RoPE for biased reference
        # Need to transpose for precompute_freqs_cis: [B, 3, seq_len, 2] -> [B, seq_len, 3, 2]
        ref_pos_transposed = biased_ref_positions.permute(0, 2, 1, 3)
        ref_freqs_cis = precompute_freqs_cis(
            indices_grid=ref_pos_transposed,
            dim=self.dim,
            out_dtype=dtype,
            theta=self.config.theta,
            max_pos=self.config.max_pos,
            use_middle_indices_grid=self.config.use_middle_indices,
            num_attention_heads=self.num_heads,
            rope_type=self.rope_type,
        )

        # Compute standard RoPE for target
        tgt_pos_transposed = tgt_positions.permute(0, 2, 1, 3)
        tgt_freqs_cis = precompute_freqs_cis(
            indices_grid=tgt_pos_transposed,
            dim=self.dim,
            out_dtype=dtype,
            theta=self.config.theta,
            max_pos=self.config.max_pos,
            use_middle_indices_grid=self.config.use_middle_indices,
            num_attention_heads=self.num_heads,
            rope_type=self.rope_type,
        )

        return ref_freqs_cis, tgt_freqs_cis

    def forward(
        self,
        q_ref: torch.Tensor,
        k_ref: torch.Tensor,
        q_tgt: torch.Tensor,
        k_tgt: torch.Tensor,
        ref_positions: torch.Tensor,
        tgt_positions: torch.Tensor,
        task: OmniTransferTask,
        target_width: int,
        target_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply task-aware RoPE to Q/K tensors.

        Args:
            q_ref, k_ref: Reference Q/K tensors [B, ref_seq_len, dim]
            q_tgt, k_tgt: Target Q/K tensors [B, tgt_seq_len, dim]
            ref_positions: Reference positions [B, 3, ref_seq_len, 2]
            tgt_positions: Target positions [B, 3, tgt_seq_len, 2]
            task: The transfer task type
            target_width: Width of target in latent space
            target_frames: Number of target frames

        Returns:
            Tuple of (q_ref_rotated, k_ref_rotated, q_tgt_rotated, k_tgt_rotated)
        """
        ref_freqs_cis, tgt_freqs_cis = self.compute_biased_rope(
            ref_positions, tgt_positions, task,
            target_width, target_frames, q_ref.dtype
        )

        # Apply RoPE
        q_ref_rotated = apply_rotary_emb(q_ref, ref_freqs_cis, self.rope_type)
        k_ref_rotated = apply_rotary_emb(k_ref, ref_freqs_cis, self.rope_type)
        q_tgt_rotated = apply_rotary_emb(q_tgt, tgt_freqs_cis, self.rope_type)
        k_tgt_rotated = apply_rotary_emb(k_tgt, tgt_freqs_cis, self.rope_type)

        return q_ref_rotated, k_ref_rotated, q_tgt_rotated, k_tgt_rotated


class ReferenceDecoupledCausalLearning(nn.Module):
    """Reference-decoupled Causal Learning (RCL) for OmniTransfer.

    RCL separates the attention computation into reference and target branches,
    with the target branch attending to both itself and the reference.

    Quote: "We propose Reference-decoupled Causal Learning (RCL), which decouples
    the reference and target branches in attention computation. The reference branch
    performs self-attention independently, while the target branch attends to both
    itself and the reference through concatenated keys and values." (Section 4.3)

    Key insight: Reference branch uses t=0 (clean, noise-free) while target uses
    sampled timestep t.

    Quote: "Critically, the reference branch adopts a fixed t = 0, meaning it remains
    noise-free throughout the diffusion process. This design ensures that the reference
    information is always clean and reliable." (Section 4.3)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 32,
        head_dim: int = 128,
        qkv_bias: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # Separate Q/K/V projections for reference and target
        # Quote: "Project to Q/K/V separately for ref/tgt" (Eq. 3, 4)
        self.to_q_ref = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k_ref = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v_ref = nn.Linear(dim, self.inner_dim, bias=qkv_bias)

        self.to_q_tgt = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_k_tgt = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_v_tgt = nn.Linear(dim, self.inner_dim, bias=qkv_bias)

        # Output projections
        self.to_out_ref = nn.Linear(self.inner_dim, dim, bias=True)
        self.to_out_tgt = nn.Linear(self.inner_dim, dim, bias=True)

        # Q/K normalization (following LTX-2 pattern)
        self.q_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)

        self.scale = head_dim ** -0.5

    def forward(
        self,
        x_ref: torch.Tensor,
        x_tgt: torch.Tensor,
        ref_pe: Tuple[torch.Tensor, torch.Tensor] | None = None,
        tgt_pe: Tuple[torch.Tensor, torch.Tensor] | None = None,
        ref_biased_pe: Tuple[torch.Tensor, torch.Tensor] | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute decoupled attention for reference and target.

        Quote: "Attn_ref = Attn(R_θ(Q_ref), R_θ(K_ref), V_ref)" (Eq. 3)
        Quote: "Attn_tgt = Attn(R_θ(Q_tgt), [R_θ(K_tgt); R*_θ(K_ref)], [V_tgt; V_ref])" (Eq. 4)

        Args:
            x_ref: Reference hidden states [B, ref_seq_len, dim]
            x_tgt: Target hidden states [B, tgt_seq_len, dim]
            ref_pe: Standard RoPE for reference (cos, sin)
            tgt_pe: Standard RoPE for target (cos, sin)
            ref_biased_pe: Task-biased RoPE for reference (cos, sin)
            rope_type: Type of RoPE to apply

        Returns:
            Tuple of (ref_output, tgt_output) each [B, seq_len, dim]
        """
        batch_size = x_ref.shape[0]
        ref_seq_len = x_ref.shape[1]
        tgt_seq_len = x_tgt.shape[1]

        # Project reference
        q_ref = self.q_norm(self.to_q_ref(x_ref))
        k_ref = self.k_norm(self.to_k_ref(x_ref))
        v_ref = self.to_v_ref(x_ref)

        # Project target
        q_tgt = self.q_norm(self.to_q_tgt(x_tgt))
        k_tgt = self.k_norm(self.to_k_tgt(x_tgt))
        v_tgt = self.to_v_tgt(x_tgt)

        # Apply RoPE if provided
        if ref_pe is not None:
            q_ref = apply_rotary_emb(q_ref, ref_pe, rope_type)
            k_ref_self = apply_rotary_emb(k_ref, ref_pe, rope_type)
        else:
            k_ref_self = k_ref

        if tgt_pe is not None:
            q_tgt = apply_rotary_emb(q_tgt, tgt_pe, rope_type)
            k_tgt = apply_rotary_emb(k_tgt, tgt_pe, rope_type)

        # For target attending to reference, use biased RoPE on reference K
        if ref_biased_pe is not None:
            k_ref_cross = apply_rotary_emb(k_ref, ref_biased_pe, rope_type)
        else:
            k_ref_cross = k_ref_self

        # Reshape for attention: [B, seq_len, heads, head_dim]
        q_ref = q_ref.view(batch_size, ref_seq_len, self.num_heads, self.head_dim)
        k_ref_self = k_ref_self.view(batch_size, ref_seq_len, self.num_heads, self.head_dim)
        v_ref = v_ref.view(batch_size, ref_seq_len, self.num_heads, self.head_dim)

        q_tgt = q_tgt.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim)
        k_tgt = k_tgt.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim)
        v_tgt = v_tgt.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim)
        k_ref_cross = k_ref_cross.view(batch_size, ref_seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [B, heads, seq_len, head_dim]
        q_ref = q_ref.transpose(1, 2)
        k_ref_self = k_ref_self.transpose(1, 2)
        v_ref_t = v_ref.transpose(1, 2)

        q_tgt = q_tgt.transpose(1, 2)
        k_tgt = k_tgt.transpose(1, 2)
        v_tgt_t = v_tgt.transpose(1, 2)
        k_ref_cross = k_ref_cross.transpose(1, 2)

        # Reference self-attention (Eq. 3)
        # Quote: "Attn_ref = Attn(R_θ(Q_ref), R_θ(K_ref), V_ref)"
        attn_ref = F.scaled_dot_product_attention(
            q_ref, k_ref_self, v_ref_t,
            dropout_p=0.0, is_causal=False
        )

        # Target attention with concatenated reference (Eq. 4)
        # Quote: "Attn_tgt = Attn(R_θ(Q_tgt), [R_θ(K_tgt); R*_θ(K_ref)], [V_tgt; V_ref])"
        # Concatenate K and V from target and reference
        k_concat = torch.cat([k_tgt, k_ref_cross], dim=2)  # [B, heads, tgt+ref, head_dim]
        v_concat = torch.cat([v_tgt_t, v_ref_t], dim=2)    # [B, heads, tgt+ref, head_dim]

        attn_tgt = F.scaled_dot_product_attention(
            q_tgt, k_concat, v_concat,
            dropout_p=0.0, is_causal=False
        )

        # Reshape and project outputs
        attn_ref = attn_ref.transpose(1, 2).reshape(batch_size, ref_seq_len, self.inner_dim)
        attn_tgt = attn_tgt.transpose(1, 2).reshape(batch_size, tgt_seq_len, self.inner_dim)

        out_ref = self.to_out_ref(attn_ref)
        out_tgt = self.to_out_tgt(attn_tgt)

        return out_ref, out_tgt


class MetaQueryBank(nn.Module):
    """Learnable MetaQuery bank for Task-adaptive Multimodal Alignment.

    Quote: "We introduce a set of learnable tokens, MetaQueries, dedicated to each
    task type. These MetaQueries aggregate task-specific information from the MLLM
    outputs through cross-attention." (Section 4.4)
    """

    def __init__(
        self,
        num_tasks: int = 6,
        num_queries_per_task: int = 8,
        query_dim: int = 4096,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_queries_per_task = num_queries_per_task
        self.query_dim = query_dim

        # Learnable MetaQueries for each task type
        # Shape: [num_tasks, num_queries_per_task, query_dim]
        self.meta_queries = nn.Parameter(
            torch.randn(num_tasks, num_queries_per_task, query_dim) * 0.02
        )

        # Task to index mapping
        self._task_to_idx = {
            OmniTransferTask.MOTION_TRANSFER: 0,
            OmniTransferTask.POSE_REENACTMENT: 1,
            OmniTransferTask.ACTION_CUSTOMIZATION: 2,
            OmniTransferTask.STYLE_TRANSFER: 3,
            OmniTransferTask.IDENTITY_PRESERVATION: 4,
            OmniTransferTask.SCENE_COMPOSITION: 5,
        }

    def get_queries(self, task: OmniTransferTask) -> torch.Tensor:
        """Get MetaQueries for a specific task.

        Args:
            task: The transfer task type

        Returns:
            MetaQueries tensor [num_queries_per_task, query_dim]
        """
        idx = self._task_to_idx.get(task, 0)
        return self.meta_queries[idx]

    def forward(self, task: OmniTransferTask, batch_size: int) -> torch.Tensor:
        """Get batched MetaQueries for a task.

        Args:
            task: The transfer task type
            batch_size: Batch size

        Returns:
            MetaQueries tensor [batch_size, num_queries_per_task, query_dim]
        """
        queries = self.get_queries(task)
        return queries.unsqueeze(0).expand(batch_size, -1, -1)


class TaskAdaptiveMultimodalAlignment(nn.Module):
    """Task-adaptive Multimodal Alignment (TMA) for OmniTransfer.

    TMA uses an MLLM to provide task-specific semantic guidance through
    learnable MetaQueries and a connector MLP.

    Quote: "Task-adaptive Multimodal Alignment (TMA) leverages a multimodal LLM
    (MLLM) to provide semantic guidance that is adaptive to different transfer
    tasks. The MLLM takes as input the first-frame tokens of the target video,
    the reference video tokens, template tokens specific to the task type, and
    the text prompt tokens." (Section 4.4)

    For LTX-2 adaptation:
    - Uses Gemma as the MLLM backbone (replacing Qwen-2.5-VL from original)
    - Connector projects MLLM outputs to LTX-2's cross-attention dimension
    - Outputs are injected into target branch cross-attention only

    Quote: "The aligned features are then injected solely into the target branch's
    cross-attention mechanism, ensuring that the reference branch remains unaffected
    by potentially task-specific semantic modifications." (Section 4.4)
    """

    def __init__(
        self,
        mllm_hidden_dim: int = 3072,  # Gemma-3-12B hidden size
        output_dim: int = 4096,        # LTX-2 cross-attention dim
        num_connector_layers: int = 3,
        num_queries_per_task: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mllm_hidden_dim = mllm_hidden_dim
        self.output_dim = output_dim

        # MetaQuery bank for task-specific queries
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
        # Quote: "The aggregated features are projected through a three-layer MLP
        # connector to match the cross-attention dimension" (Section 4.4)
        connector_layers = []
        dims = [mllm_hidden_dim, mllm_hidden_dim, mllm_hidden_dim, output_dim]

        for i in range(num_connector_layers):
            connector_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_connector_layers - 1:
                connector_layers.append(nn.GELU())
                connector_layers.append(nn.Dropout(dropout))

        self.connector = nn.Sequential(*connector_layers)

        # Layer norm for stability
        self.input_norm = nn.LayerNorm(mllm_hidden_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        mllm_features: torch.Tensor,
        task: OmniTransferTask,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute task-adaptive aligned features.

        Quote: "MLLM takes as input the first-frame tokens of the target video,
        the reference video tokens, template tokens, and prompt tokens. The outputs
        are aggregated using task-specific MetaQueries through cross-attention."
        (Section 4.4)

        Args:
            mllm_features: MLLM output features [B, seq_len, mllm_hidden_dim]
            task: The transfer task type
            attention_mask: Optional attention mask [B, seq_len]

        Returns:
            Task-aligned features [B, num_queries, output_dim]
        """
        batch_size = mllm_features.shape[0]

        # Normalize input
        mllm_features = self.input_norm(mllm_features)

        # Get task-specific MetaQueries
        meta_queries = self.meta_query_bank(task, batch_size)

        # Aggregate MLLM features using MetaQueries through cross-attention
        # Q: MetaQueries, K/V: MLLM features
        # Quote: "MetaQueries aggregate task-specific information from the MLLM
        # outputs through cross-attention" (Section 4.4)
        if attention_mask is not None:
            # Convert to attention mask format (True = attend, False = ignore)
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        aligned_features, _ = self.query_attn(
            query=meta_queries,
            key=mllm_features,
            value=mllm_features,
            key_padding_mask=key_padding_mask,
        )

        # Project through connector MLP
        aligned_features = self.connector(aligned_features)
        aligned_features = self.output_norm(aligned_features)

        return aligned_features


class OmniTransferDiTBlockWrapper(nn.Module):
    """Wrapper for LTX-2 DiT block with OmniTransfer components.

    This wrapper integrates TPB, RCL, and TMA into a standard LTX-2
    BasicAVTransformerBlock for OmniTransfer training.

    Quote: "OmniTransfer comprises three key components that work together:
    1) Task-aware Positional Bias for position-task alignment,
    2) Reference-decoupled Causal Learning for efficient decoupled attention,
    3) Task-adaptive Multimodal Alignment for semantic guidance." (Section 4)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 32,
        head_dim: int = 128,
        context_dim: int = 4096,
        norm_eps: float = 1e-6,
        enable_tpb: bool = True,
        enable_rcl: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.enable_tpb = enable_tpb
        self.enable_rcl = enable_rcl

        # Task-aware Positional Bias
        if enable_tpb:
            self.tpb = TaskAwarePositionalBias(
                dim=dim,
                num_heads=num_heads,
            )

        # Reference-decoupled Causal Learning
        if enable_rcl:
            self.rcl = ReferenceDecoupledCausalLearning(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                norm_eps=norm_eps,
            )

        # Cross-attention for TMA features injection
        self.tma_cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        # Gating for TMA injection
        self.tma_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_ref: torch.Tensor,
        x_tgt: torch.Tensor,
        ref_positions: torch.Tensor,
        tgt_positions: torch.Tensor,
        task: OmniTransferTask,
        target_width: int,
        target_frames: int,
        tma_features: torch.Tensor | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with OmniTransfer components.

        Args:
            x_ref: Reference hidden states [B, ref_seq_len, dim]
            x_tgt: Target hidden states [B, tgt_seq_len, dim]
            ref_positions: Reference positions [B, 3, ref_seq_len, 2]
            tgt_positions: Target positions [B, 3, tgt_seq_len, 2]
            task: The transfer task type
            target_width: Width of target in latent space
            target_frames: Number of target frames
            tma_features: TMA aligned features [B, num_queries, dim] or None
            rope_type: Type of RoPE to apply

        Returns:
            Tuple of (ref_output, tgt_output)
        """
        # Compute RoPE embeddings
        if self.enable_tpb:
            ref_freqs_cis, tgt_freqs_cis = self.tpb.compute_biased_rope(
                ref_positions, tgt_positions, task,
                target_width, target_frames, x_ref.dtype
            )
            # Also compute standard RoPE for reference self-attention
            ref_pos_transposed = ref_positions.permute(0, 2, 1, 3)
            from ltx_core.model.transformer.rope import precompute_freqs_cis
            ref_self_freqs_cis = precompute_freqs_cis(
                indices_grid=ref_pos_transposed,
                dim=self.dim,
                out_dtype=x_ref.dtype,
                theta=self.tpb.config.theta,
                max_pos=self.tpb.config.max_pos,
                use_middle_indices_grid=self.tpb.config.use_middle_indices,
                num_attention_heads=self.tpb.num_heads,
                rope_type=rope_type,
            )
        else:
            ref_self_freqs_cis = None
            ref_freqs_cis = None
            tgt_freqs_cis = None

        # Apply Reference-decoupled Causal Learning
        if self.enable_rcl:
            out_ref, out_tgt = self.rcl(
                x_ref, x_tgt,
                ref_pe=ref_self_freqs_cis,
                tgt_pe=tgt_freqs_cis,
                ref_biased_pe=ref_freqs_cis,
                rope_type=rope_type,
            )
        else:
            out_ref = x_ref
            out_tgt = x_tgt

        # Inject TMA features into target only
        # Quote: "aligned features are injected solely into the target branch"
        if tma_features is not None:
            tma_out, _ = self.tma_cross_attn(
                query=out_tgt,
                key=tma_features,
                value=tma_features,
            )
            # Gated addition
            out_tgt = out_tgt + torch.sigmoid(self.tma_gate) * tma_out

        return out_ref, out_tgt
