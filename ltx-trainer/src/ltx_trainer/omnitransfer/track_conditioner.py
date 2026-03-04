"""MotionStream Track Conditioning Encoder for OmniTransfer.

Implements track conditioning from MotionStream (arXiv:2511.01266):
Encodes 2D point tracks (from CoTracker/TAPIR/VGGT) into cross-attention tokens
for explicit geometric motion control. This *complements* the existing MotionEncoder
(semantic motion via spatial mean-pooling) with fine-grained trajectory control.

Key design decisions:
- Sinusoidal 2D positional encoding preserves spatial layout (unlike MotionEncoder)
- Per-track ID embeddings distinguish different tracked points
- Visibility masking zeros out occluded track features before processing
- Zero-init on output projection (AMB3R's ZeroConvBlock trick) ensures
  the encoder starts with zero output -> no disruption to pretrained DiT
- Cross-attention with learnable queries compresses F*N track features -> K tokens

Architecture:
    Input: tracks [B, F, N, 2] + visibility [B, F, N]
    -> Sinusoidal PE of (x,y) coordinates -> [B, F, N, sincos_dim*2]
    -> Linear projection to hidden_dim
    -> Add track_id_embed + temporal_pos_embed
    -> Mask by visibility
    -> Reshape [B, F*N, hidden_dim]
    -> TransformerEncoder (temporal-spatial reasoning)
    -> Cross-attention: K track_queries attend to F*N features
    -> Output projection with zero-init -> [B, K, output_dim=3840]

References:
    MotionStream (arXiv:2511.01266)
    VGGT track PE (thirdparty/vggt/heads/track_modules/utils.py:90-121)
    MotionEncoder pattern (motion_encoder.py:31-191)
    Geometric3DEncoder pattern (geometric_3d_encoder.py:35-181)
"""

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer import logger


def sinusoidal_track_embedding(xy: Tensor, C: int) -> Tensor:
    """Sinusoidal 2D positional encoding of (x,y) coordinates.

    Ported from VGGT get_2d_embedding (track_modules/utils.py:90-121).
    Uses separate sin/cos basis for x and y, concatenated into a single vector.
    This creates a smooth, continuous representation where nearby coordinates
    map to similar embeddings.

    Args:
        xy: Coordinate tensor with last dim = 2.
            Supports [B, N, 2], [B, F, N, 2], or any shape [..., 2].
        C: Embedding dimension per coordinate. Output dim = C * 2
            (C for x-component, C for y-component).

    Returns:
        Positional encoding with same leading dims and last dim = C * 2.
        E.g., input [B, F, N, 2] -> output [B, F, N, C*2].
    """
    assert xy.shape[-1] == 2, f"Last dim must be 2 (x,y), got {xy.shape[-1]}"

    # Store original shape for reshape at end
    orig_shape = xy.shape[:-1]  # [...] without the trailing 2

    # Flatten to [M, 2] for uniform processing
    flat = xy.reshape(-1, 2)
    M = flat.shape[0]

    x = flat[:, 0:1]  # [M, 1]
    y = flat[:, 1:2]  # [M, 1]

    # Frequency divisions matching VGGT: div_term = arange(0, C, 2) * (1000 / C)
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32)
        * (1000.0 / C)
    ).reshape(1, C // 2)  # [1, C//2]

    pe_x = torch.zeros(M, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(M, C, device=xy.device, dtype=torch.float32)

    pe_x[:, 0::2] = torch.sin(x * div_term)
    pe_x[:, 1::2] = torch.cos(x * div_term)

    pe_y[:, 0::2] = torch.sin(y * div_term)
    pe_y[:, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=1)  # [M, C*2]

    # Reshape back to original leading dims
    return pe.reshape(*orig_shape, C * 2).to(dtype=xy.dtype)


class TrackConditioner(nn.Module):
    """MotionStream Track Conditioning Encoder.

    Encodes 2D point tracks into K compact tokens for cross-attention injection
    alongside text/motion/geometric tokens. Preserves spatial layout via sinusoidal
    PE (unlike MotionEncoder which discards it).

    Args:
        sincos_dim: Sinusoidal embedding dimension per coordinate (default 64,
            total PE dim = sincos_dim * 2 = 128).
        hidden_dim: Internal hidden dimension (default 512).
        output_dim: Output dimension matching text embedding dim (3840 for Gemma).
        num_tokens: Number of track query tokens K (default 8).
        num_layers: Number of transformer encoder layers (default 4).
        num_heads: Number of attention heads (default 8).
        max_tracks: Maximum number of track points N (default 128).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        sincos_dim: int = 64,
        hidden_dim: int = 512,
        output_dim: int = 3840,
        num_tokens: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        max_tracks: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sincos_dim = sincos_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.max_tracks = max_tracks

        # Input projection: sinusoidal PE -> hidden_dim
        # PE dim = sincos_dim * 2 (separate sin/cos for x and y)
        pe_dim = sincos_dim * 2
        self.input_proj = nn.Linear(pe_dim, hidden_dim)

        # Learnable per-track ID embedding (distinguishes different tracked points)
        self.track_id_embed = nn.Parameter(
            torch.randn(1, 1, max_tracks, hidden_dim) * 0.02
        )

        # Temporal position embedding (per-frame, max 128 frames)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 128, hidden_dim) * 0.02
        )

        # Transformer encoder for joint temporal-spatial reasoning over all tracks
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

        # Learnable track queries for cross-attention aggregation
        self.track_queries = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )

        # Cross-attention: track queries attend to F*N track features
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
        """Initialize weights: trunc_normal(0.02), zeros biases, zero-init output."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init the final linear in output_proj (index -2 is Linear before final LN)
        # This is the AMB3R ZeroConvBlock trick: output starts at zero,
        # gradually "fading in" the encoder's contribution during training
        final_linear = self.output_proj[-2]  # Linear(hidden*2, output_dim)
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        tracks: Tensor,
        visibility: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode 2D point tracks into track tokens for cross-attention.

        Args:
            tracks: Per-frame 2D track coordinates [B, F, N, 2] in pixel space.
            visibility: Per-frame visibility flags [B, F, N] (0=occluded, 1=visible).
                If None, all tracks are treated as visible.

        Returns:
            Tuple of:
                - track_tokens: [B, K, output_dim] for cross-attention injection
                - hidden_states: [B, K, hidden_dim] for track decoder
        """
        B, F, N, _ = tracks.shape

        # Sinusoidal PE of (x,y) coordinates: [B, F, N, 2] -> [B, F, N, sincos_dim*2]
        track_pe = sinusoidal_track_embedding(tracks, self.sincos_dim)

        # Project to hidden dim: [B, F, N, sincos_dim*2] -> [B, F, N, hidden_dim]
        x = self.input_proj(track_pe)

        # Add per-track ID embedding (broadcast across frames)
        # track_id_embed: [1, 1, max_tracks, hidden_dim] -> slice to N tracks
        x = x + self.track_id_embed[:, :, :N, :]

        # Add temporal position embedding (broadcast across tracks)
        # temporal_pos_embed: [1, F, hidden_dim] -> [1, F, 1, hidden_dim]
        temporal_pos = self.temporal_pos_embed[:, :F, :].unsqueeze(2)  # [1, F, 1, hidden_dim]
        x = x + temporal_pos

        # Apply visibility mask: zero out features for occluded tracks
        if visibility is not None:
            vis_mask = visibility.unsqueeze(-1)  # [B, F, N, 1]
            x = x * vis_mask

        # Reshape to sequence: [B, F, N, hidden_dim] -> [B, F*N, hidden_dim]
        x = x.reshape(B, F * N, self.hidden_dim)

        # Transformer encoder for joint temporal-spatial reasoning
        x = self.transformer(x)  # [B, F*N, hidden_dim]

        # Cross-attention: K track queries attend to F*N track features
        queries = self.track_queries.expand(B, -1, -1)  # [B, K, hidden_dim]
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
        track_tokens = self.output_proj(hidden_states)  # [B, K, output_dim]
        track_tokens = track_tokens * self.gate_scale

        return track_tokens, hidden_states
