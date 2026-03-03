"""AMB3R 3D Geometric Token Encoder for OmniTransfer.

Encodes pre-computed 3D geometry features (from frozen AMB3R/DA3) into
compact tokens for cross-attention injection alongside text/motion tokens.
Follows the MotionEncoder architecture (motion_encoder.py:31-191) exactly.

Key design decisions:
- Zero-init on output projection (AMB3R's ZeroConvBlock trick, blocks.py:13-15)
  ensures the encoder starts with zero output → no disruption to pretrained DiT
- Learnable gate_scale allows the model to control contribution magnitude
- Per-frame temporal processing captures temporal depth/pose evolution
- Cross-attention with learnable queries compresses F frames → K tokens

Architecture:
    Input: [B, F, feature_dim] per-frame 3D features
    → Linear projection to hidden_dim
    → Add temporal position embeddings
    → TransformerEncoder (temporal reasoning)
    → Cross-attention: K geo_queries attend to F frame features
    → Output projection with zero-init → [B, K, output_dim=3840]

References:
    AMB3R (arXiv:2511.20343)
    MotionEncoder pattern (motion_encoder.py)
    ZeroConvBlock (amb3r/blocks.py:5-21)
"""

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer import logger


class Geometric3DEncoder(nn.Module):
    """3D Geometric Token Encoder following MotionEncoder pattern.

    Compresses per-frame 3D geometry features (depth statistics, mean normals,
    confidence scores, etc.) into K compact tokens for cross-attention injection
    into the DiT. The zero-initialized output ensures smooth training start.

    Args:
        feature_dim: Per-frame input feature dimension (default 7:
            depth=1 + normal=3 + confidence=1 + mean_point_xy=2)
        hidden_dim: Internal hidden dimension (default 512)
        output_dim: Output dimension matching text embedding dim (3840 for Gemma)
        num_tokens: Number of geometric query tokens K (default 8)
        num_layers: Number of transformer encoder layers (default 4)
        num_heads: Number of attention heads (default 8)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        feature_dim: int = 7,
        hidden_dim: int = 512,
        output_dim: int = 3840,
        num_tokens: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Input projection: per-frame features → hidden_dim
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Temporal position embedding (per-frame, max 128 frames)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 128, hidden_dim) * 0.02
        )

        # Transformer encoder for temporal processing across frames
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Learnable geometric queries for cross-attention aggregation
        self.geo_queries = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )

        # Cross-attention: geo queries attend to frame features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        # Output projection to match text embedding dimension
        # Zero-init on the final linear (AMB3R's ZeroConvBlock trick)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Learnable gate scale (starts at 1.0, model learns optimal magnitude)
        self.gate_scale = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: trunc_normal(0.02), zeros on biases, zero-init output."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init the final linear in output_proj (index -2 is the Linear before final LN)
        # This is the AMB3R ZeroConvBlock trick: output starts at zero,
        # gradually "fading in" the encoder's contribution during training
        final_linear = self.output_proj[-2]  # Linear(hidden*2, output_dim)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        frame_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode per-frame 3D features into geometric tokens.

        Args:
            frame_features: Per-frame 3D geometry features [B, F, feature_dim]

        Returns:
            Tuple of:
                - geo_tokens: [B, K, output_dim] for cross-attention injection
                - hidden_states: [B, K, hidden_dim] for geometric 3D decoder
        """
        B, F, D = frame_features.shape

        # Project to hidden dim
        x = self.input_proj(frame_features)  # [B, F, hidden_dim]

        # Add temporal position embeddings
        temporal_pos = self.temporal_pos_embed[:, :F, :]  # [1, F, hidden_dim]
        x = x + temporal_pos

        # Transformer encoder for temporal reasoning across frames
        x = self.transformer(x)  # [B, F, hidden_dim]

        # Cross-attention: K geometric queries attend to F frame features
        queries = self.geo_queries.expand(B, -1, -1)  # [B, K, hidden_dim]
        queries = self.cross_attn_norm(queries)
        kv = self.kv_norm(x)

        hidden_states, _ = self.cross_attn(
            query=queries,
            key=kv,
            value=kv,
        )  # [B, K, hidden_dim]

        # Residual connection for queries
        hidden_states = hidden_states + queries

        # Project to output dimension with zero-init and gate scaling
        geo_tokens = self.output_proj(hidden_states)  # [B, K, output_dim]
        geo_tokens = geo_tokens * self.gate_scale

        return geo_tokens, hidden_states
