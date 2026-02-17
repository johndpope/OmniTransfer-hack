"""3DiMo Implicit Motion Encoder for OmniTransfer.

Implements the motion encoding pipeline from 3DiMo (arXiv:2602.03796v2):
1. MotionEncoder - Compresses driving video into compact view-agnostic motion tokens
2. DualScaleMotionEncoder - Body + hand dual-scale variant (Section 3.2)
3. MotionAugmenter - Latent-space augmentations to prevent identity leakage

Key design principle: Spatial mean-pooling discards spatial layout, forcing the
encoder to capture only motion dynamics (not identity/appearance). Motion tokens
are injected via cross-attention alongside text tokens, leveraging the DiT's
existing 3D priors for spatial reasoning.

Quote: "We introduce an implicit motion encoder that compresses the driving video
into compact view-agnostic tokens... injected via cross-attention to leverage
the pretrained model's 3D priors." (Section 3.1, 3DiMo)

References: 3DiMo paper (arXiv:2602.03796v2)
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer import logger


class MotionEncoder(nn.Module):
    """Implicit Motion Encoder from 3DiMo.

    Compresses driving video VAE latents into K compact motion tokens that
    capture view-agnostic motion dynamics. The key operation is spatial
    mean-pooling per frame, which discards spatial layout and forces the
    encoder to learn pure motion representations.

    Architecture:
        Input: [B, C, F, H, W] VAE latents
        -> Patchify + project to hidden_dim
        -> Add temporal position embeddings
        -> Transformer encoder (num_layers)
        -> Spatial mean-pool per frame -> [B, F, hidden_dim]
        -> Cross-attention with K learnable motion queries
        -> Output projection -> [B, K, output_dim]

    Quote: "The spatial mean-pooling... ensures that the motion tokens
    are view-agnostic, capturing temporal dynamics without encoding
    spatial layout information." (Section 3.1)

    Args:
        latent_channels: Number of VAE latent channels (128 for LTX-2)
        hidden_dim: Internal hidden dimension (default 512)
        output_dim: Output dimension matching text embedding dim (3840 for Gemma)
        num_tokens: Number of motion query tokens K (default 5)
        num_layers: Number of transformer encoder layers (default 4)
        num_heads: Number of attention heads (default 8)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        latent_channels: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 3840,
        num_tokens: int = 5,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        # Input projection: flatten spatial dims and project
        # VAE latents are [B, C=128, F, H, W], patchified to [B, F*H*W, C]
        self.input_proj = nn.Linear(latent_channels, hidden_dim)

        # Temporal position embedding (per-frame, max 128 frames)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, 128, hidden_dim) * 0.02
        )

        # Transformer encoder for temporal processing
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

        # Learnable motion queries for cross-attention aggregation
        self.motion_queries = nn.Parameter(
            torch.randn(1, num_tokens, hidden_dim) * 0.02
        )

        # Cross-attention: motion queries attend to frame features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)

        # Output projection to match text embedding dimension
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        driving_latents: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode driving video latents into motion tokens.

        Args:
            driving_latents: VAE-encoded driving video [B, C, F, H, W]

        Returns:
            Tuple of:
                - motion_tokens: [B, K, output_dim] for cross-attention injection
                - hidden_states: [B, K, hidden_dim] for geometric decoder
        """
        B, C, F, H, W = driving_latents.shape

        # Patchify: [B, C, F, H, W] -> [B, F*H*W, C]
        x = driving_latents.permute(0, 2, 3, 4, 1)  # [B, F, H, W, C]
        x = x.reshape(B, F * H * W, C)

        # Project to hidden dim
        x = self.input_proj(x)  # [B, F*H*W, hidden_dim]

        # Add temporal position embeddings (same pos for all spatial locs in a frame)
        # Create frame indices [F*H*W] where each spatial loc gets its frame's pos
        frame_idx = torch.arange(F, device=x.device).repeat_interleave(H * W)
        temporal_pos = self.temporal_pos_embed[:, frame_idx, :]  # [1, F*H*W, hidden_dim]
        x = x + temporal_pos

        # Transformer encoder for temporal reasoning
        x = self.transformer(x)  # [B, F*H*W, hidden_dim]

        # Spatial mean-pool per frame: [B, F*H*W, hidden_dim] -> [B, F, hidden_dim]
        # This is THE key operation for view-agnosticism: discards spatial layout
        x = x.reshape(B, F, H * W, self.hidden_dim)
        frame_features = x.mean(dim=2)  # [B, F, hidden_dim]

        # Cross-attention: K motion queries attend to F frame features
        queries = self.motion_queries.expand(B, -1, -1)  # [B, K, hidden_dim]
        queries = self.cross_attn_norm(queries)
        kv = self.kv_norm(frame_features)

        hidden_states, _ = self.cross_attn(
            query=queries,
            key=kv,
            value=kv,
        )  # [B, K, hidden_dim]

        # Residual connection for queries
        hidden_states = hidden_states + queries

        # Project to output dimension
        motion_tokens = self.output_proj(hidden_states)  # [B, K, output_dim]

        return motion_tokens, hidden_states


class DualScaleMotionEncoder(nn.Module):
    """Dual-scale Motion Encoder for body + hand motion (Section 3.2 of 3DiMo).

    Uses separate encoders for coarse body motion and fine-grained hand dynamics,
    then concatenates the output tokens.

    Quote: "We employ a dual-scale architecture with separate encoders for body
    and hand motion... The body encoder captures coarse motion while the hand
    encoder focuses on fine-grained finger dynamics." (Section 3.2)

    Args:
        latent_channels: Number of VAE latent channels (128 for LTX-2)
        hidden_dim: Internal hidden dimension
        output_dim: Output dimension matching text embedding dim
        body_tokens: Number of body motion tokens (default 5)
        hand_tokens: Number of hand motion tokens (default 3)
        num_layers: Number of transformer layers per encoder
    """

    def __init__(
        self,
        latent_channels: int = 128,
        hidden_dim: int = 512,
        output_dim: int = 3840,
        body_tokens: int = 5,
        hand_tokens: int = 3,
        num_layers: int = 4,
    ):
        super().__init__()
        self.body_encoder = MotionEncoder(
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_tokens=body_tokens,
            num_layers=num_layers,
        )
        self.hand_encoder = MotionEncoder(
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_tokens=hand_tokens,
            num_layers=num_layers,
        )
        self.total_tokens = body_tokens + hand_tokens

    def forward(
        self,
        driving_latents: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode driving video with dual-scale body + hand encoders.

        Args:
            driving_latents: VAE-encoded driving video [B, C, F, H, W]

        Returns:
            Tuple of:
                - motion_tokens: [B, body_tokens + hand_tokens, output_dim]
                - hidden_states: [B, body_tokens + hand_tokens, hidden_dim]
        """
        body_tokens, body_hidden = self.body_encoder(driving_latents)
        hand_tokens, hand_hidden = self.hand_encoder(driving_latents)

        motion_tokens = torch.cat([body_tokens, hand_tokens], dim=1)
        hidden_states = torch.cat([body_hidden, hand_hidden], dim=1)

        return motion_tokens, hidden_states


class MotionAugmenter:
    """Latent-space augmentations for driving video (Section 3.2 of 3DiMo).

    Applied to driving video ONLY (not target) before motion encoding to prevent
    identity leakage. By augmenting the driving signal, the motion encoder is
    forced to extract motion rather than appearance/identity information.

    Quote: "We apply augmentations including color jitter, random crop, and
    temporal subsampling to the driving video to prevent identity leakage
    from the driving signal to the generated output." (Section 3.2)

    Augmentations operate in latent space (not pixel space):
    - channel_jitter: Per-channel additive noise (color jitter analog)
    - spatial_crop_resize: Random crop + resize (perspective analog)
    - temporal_subsample: Random frame drop + interpolation

    Args:
        channel_jitter_scale: Scale of per-channel noise (default 0.05)
        spatial_crop_ratio_min: Minimum crop ratio (default 0.7)
        spatial_crop_ratio_max: Maximum crop ratio (default 1.0)
        temporal_subsample_ratio: Fraction of frames to keep (default 0.8)
    """

    def __init__(
        self,
        channel_jitter_scale: float = 0.05,
        spatial_crop_ratio_min: float = 0.7,
        spatial_crop_ratio_max: float = 1.0,
        temporal_subsample_ratio: float = 0.8,
    ):
        self.channel_jitter_scale = channel_jitter_scale
        self.spatial_crop_ratio_min = spatial_crop_ratio_min
        self.spatial_crop_ratio_max = spatial_crop_ratio_max
        self.temporal_subsample_ratio = temporal_subsample_ratio

    def __call__(self, latents: Tensor) -> Tensor:
        """Apply augmentations to driving video latents.

        Args:
            latents: Driving video latents [B, C, F, H, W]

        Returns:
            Augmented latents [B, C, F, H, W] (same shape, contents altered)
        """
        latents = self.channel_jitter(latents)
        latents = self.spatial_crop_resize(latents)
        latents = self.temporal_subsample(latents)
        return latents

    def channel_jitter(self, latents: Tensor) -> Tensor:
        """Per-channel additive noise (color jitter analog in latent space).

        Adds different random noise to each channel, disrupting color/appearance
        information while preserving spatial-temporal structure.

        Args:
            latents: [B, C, F, H, W]

        Returns:
            Jittered latents [B, C, F, H, W]
        """
        if self.channel_jitter_scale <= 0:
            return latents

        b, c = latents.shape[:2]
        # Per-channel noise: [B, C, 1, 1, 1] broadcast across spatial/temporal dims
        channel_noise = torch.randn(b, c, 1, 1, 1, device=latents.device, dtype=latents.dtype)
        channel_noise = channel_noise * self.channel_jitter_scale * latents.std()
        return latents + channel_noise

    def spatial_crop_resize(self, latents: Tensor) -> Tensor:
        """Random spatial crop + resize back to original size (perspective analog).

        Crops a random region and resizes back, simulating viewpoint changes.
        Forces encoder to ignore absolute spatial positions.

        Args:
            latents: [B, C, F, H, W]

        Returns:
            Cropped and resized latents [B, C, F, H, W]
        """
        if self.spatial_crop_ratio_min >= 1.0:
            return latents

        b, c, nf, h, w = latents.shape
        ratio = random.uniform(self.spatial_crop_ratio_min, self.spatial_crop_ratio_max)
        crop_h = max(1, int(h * ratio))
        crop_w = max(1, int(w * ratio))

        # Random crop position
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        # Crop [B, C, F, crop_h, crop_w]
        cropped = latents[:, :, :, top:top + crop_h, left:left + crop_w]

        # Resize back to original spatial dims
        # Reshape for F.interpolate: [B*F, C, crop_h, crop_w] -> [B*F, C, H, W]
        cropped = cropped.reshape(b * nf, c, crop_h, crop_w)
        resized = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
        return resized.reshape(b, c, nf, h, w)

    def temporal_subsample(self, latents: Tensor) -> Tensor:
        """Random frame drop + interpolation (temporal augmentation).

        Drops random frames and interpolates back to original frame count.
        Forces encoder to be robust to temporal sampling variations.

        Args:
            latents: [B, C, F, H, W]

        Returns:
            Temporally augmented latents [B, C, F, H, W]
        """
        b, c, nf, h, w = latents.shape
        keep_frames = max(2, int(nf * self.temporal_subsample_ratio))

        if keep_frames >= nf:
            return latents

        # Select random subset of frames (sorted to preserve order)
        frame_indices = sorted(random.sample(range(nf), keep_frames))
        frame_indices = torch.tensor(frame_indices, device=latents.device)

        # Extract subset [B, C, keep_frames, H, W]
        subset = latents[:, :, frame_indices, :, :]

        # Interpolate back to original frame count
        # Reshape to [B*C, keep_frames, H*W], interpolate over frames, reshape back
        hw = h * w
        subset_flat = subset.reshape(b * c, keep_frames, hw)  # [B*C, keep, H*W]
        subset_flat = subset_flat.permute(0, 2, 1)  # [B*C, H*W, keep]
        interpolated = F.interpolate(
            subset_flat, size=nf, mode="linear", align_corners=False
        )  # [B*C, H*W, F]
        interpolated = interpolated.permute(0, 2, 1)  # [B*C, F, H*W]
        return interpolated.reshape(b, c, nf, h, w)
