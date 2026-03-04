"""MotionStream Track Decoder for OmniTransfer.

Auxiliary decoder for track coordinate and visibility prediction from
TrackConditioner hidden states. Follows the Geometric3DDecoder pattern
(geometric_3d_decoder.py:31-252) exactly.

Used only during early training phases as an auxiliary loss that encourages
the conditioner's hidden states to retain meaningful track information.
Annealed to zero weight over training (linear schedule).

Architecture:
    Input: hidden_states [B, K, hidden_dim] from TrackConditioner
    -> Flatten: [B, K * hidden_dim]
    -> coord_decoder MLP -> [B, N * 2] -> reshape to [B, N, 2] (mean track coords)
    -> visibility_decoder MLP -> [B, N] (mean visibility scores)

References:
    MotionStream (arXiv:2511.01266)
    Geometric3DDecoder pattern (geometric_3d_decoder.py)
    GeometricDecoder pattern (geometric_decoder.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer import logger


class TrackDecoder(nn.Module):
    """Auxiliary track decoder for coordinate/visibility supervision.

    Predicts aggregate track properties from conditioner hidden states.
    Used only during early training as auxiliary loss, then annealed to zero.

    Args:
        hidden_dim: Input hidden dimension from TrackConditioner (default 512).
        num_tokens: Number of track tokens K (default 8).
        max_tracks: Maximum number of track points N (default 128).
        mlp_hidden_dim: Hidden dimension of decoder MLPs (default 1024).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_tokens: int = 8,
        max_tracks: int = 128,
        mlp_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.max_tracks = max_tracks

        input_dim = num_tokens * hidden_dim

        # Coordinate decoder: predicts mean (x,y) per track point
        self.coord_decoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, max_tracks * 2),
        )

        # Visibility decoder: predicts mean visibility per track point
        self.visibility_decoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, max_tracks),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights near zero for smooth start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        track_hidden: Tensor,
    ) -> dict[str, Tensor]:
        """Predict track properties from conditioner hidden states.

        Args:
            track_hidden: Track conditioner hidden states [B, K, hidden_dim].

        Returns:
            Dictionary with:
                - mean_coords: [B, N, 2] predicted mean (x,y) per track
                - mean_visibility: [B, N] predicted mean visibility per track (sigmoid)
        """
        B = track_hidden.shape[0]

        # Flatten track tokens: [B, K, hidden_dim] -> [B, K * hidden_dim]
        x = track_hidden.reshape(B, -1)

        # Predict mean coordinates
        coords = self.coord_decoder(x)  # [B, N*2]
        coords = coords.reshape(B, self.max_tracks, 2)

        # Predict mean visibility (sigmoid for [0, 1] range)
        visibility = torch.sigmoid(self.visibility_decoder(x))  # [B, N]

        return {
            "mean_coords": coords,
            "mean_visibility": visibility,
        }


def compute_track_decoder_loss(
    decoder: TrackDecoder,
    hidden: Tensor,
    gt_tracks: dict[str, Tensor],
    step: int,
    initial_weight: float = 0.05,
    anneal_steps: int = 15000,
) -> Tensor | None:
    """Compute auxiliary track decoder loss with linear annealing.

    Follows the same annealing pattern as compute_geo_3d_decoder_loss.
    Weight decreases linearly to zero, then returns None to skip computation.

    Args:
        decoder: TrackDecoder module.
        hidden: Conditioner hidden states [B, K, hidden_dim].
        gt_tracks: Pre-computed track pseudo-GT dictionary containing:
            - coords: [B, F, N, 2] per-frame track coordinates
            - visibility: [B, F, N] per-frame visibility flags
        step: Current training step.
        initial_weight: Starting loss weight.
        anneal_steps: Steps over which to anneal to zero.

    Returns:
        Weighted loss tensor, or None if annealing complete.
    """
    weight = max(0.0, initial_weight * (1.0 - step / anneal_steps))
    if weight <= 0.0:
        return None

    preds = decoder(hidden)
    device = hidden.device
    total_loss = torch.tensor(0.0, device=device)
    loss_count = 0

    # Coordinate loss: smooth L1 between predicted and actual mean coordinates
    gt_coords = gt_tracks.get("coords")
    if gt_coords is not None:
        try:
            # Compute mean coordinates across frames: [B, F, N, 2] -> [B, N, 2]
            gt_mean_coords = gt_coords.mean(dim=1).to(device=device)

            # Slice to match decoder's max_tracks
            N = min(gt_mean_coords.shape[1], decoder.max_tracks)
            pred_coords = preds["mean_coords"][:, :N, :]
            gt_mean_coords = gt_mean_coords[:, :N, :]

            coord_loss = F.smooth_l1_loss(pred_coords, gt_mean_coords)
            total_loss = total_loss + coord_loss
            loss_count += 1
        except Exception as e:
            logger.debug(f"Track coord decoder loss failed: {e}")

    # Visibility loss: BCE between predicted and actual mean visibility
    gt_vis = gt_tracks.get("visibility")
    if gt_vis is not None:
        try:
            # Compute mean visibility across frames: [B, F, N] -> [B, N]
            gt_mean_vis = gt_vis.float().mean(dim=1).to(device=device)

            # Slice to match decoder's max_tracks
            N = min(gt_mean_vis.shape[1], decoder.max_tracks)
            pred_vis = preds["mean_visibility"][:, :N]
            gt_mean_vis = gt_mean_vis[:, :N]

            vis_loss = F.binary_cross_entropy(
                pred_vis.clamp(1e-6, 1 - 1e-6), gt_mean_vis,
            )
            total_loss = total_loss + vis_loss
            loss_count += 1
        except Exception as e:
            logger.debug(f"Track visibility decoder loss failed: {e}")

    if loss_count == 0:
        return None

    return weight * (total_loss / loss_count)
