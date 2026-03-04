"""AMB3R 3D Geometric Decoder for OmniTransfer.

Auxiliary decoder for depth distribution, mean surface normal, and mean
confidence prediction from Geometric3DEncoder hidden states. Follows the
GeometricDecoder pattern (geometric_decoder.py:33-139) exactly.

Used only during early training phases as an auxiliary loss that encourages
the encoder's hidden states to retain meaningful 3D information. Annealed
to zero weight over training.

Architecture:
    Input: hidden_states [B, K, hidden_dim] from Geometric3DEncoder
    → Flatten: [B, K * hidden_dim]
    → depth_decoder MLP → [B, num_depth_bins] (discretized depth distribution)
    → normal_decoder MLP → [B, 3] (mean surface normal direction)
    → confidence_decoder MLP → [B, 1] (mean confidence score)

References:
    AMB3R (arXiv:2511.20343)
    GeometricDecoder pattern (geometric_decoder.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer import logger


class Geometric3DDecoder(nn.Module):
    """Auxiliary 3D geometric decoder for depth/normal/confidence supervision.

    Predicts aggregate 3D properties from geometric encoder hidden states.
    Used only during early training as auxiliary loss, then annealed to zero.

    Args:
        hidden_dim: Input hidden dimension from Geometric3DEncoder (default 512)
        num_tokens: Number of geometric tokens K (default 8)
        num_depth_bins: Number of depth histogram bins (default 64)
        mlp_hidden_dim: Hidden dimension of decoder MLPs (default 1024)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_tokens: int = 8,
        num_depth_bins: int = 64,
        mlp_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_depth_bins = num_depth_bins

        input_dim = num_tokens * hidden_dim

        # Depth distribution decoder: predicts discretized depth histogram
        self.depth_decoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, num_depth_bins),
        )

        # Normal decoder: predicts mean surface normal direction
        self.normal_decoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, 3),
        )

        # Confidence decoder: predicts mean confidence score
        self.confidence_decoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, 1),
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
        geo_hidden: Tensor,
    ) -> dict[str, Tensor]:
        """Predict 3D geometric quantities from encoder hidden states.

        Args:
            geo_hidden: Geometric encoder hidden states [B, K, hidden_dim]

        Returns:
            Dictionary with:
                - depth_dist: [B, num_depth_bins] depth distribution (log-softmax)
                - mean_normal: [B, 3] mean surface normal (L2-normalized)
                - mean_confidence: [B, 1] mean confidence score (sigmoid)
        """
        B = geo_hidden.shape[0]

        # Flatten geometric tokens: [B, K, hidden_dim] → [B, K * hidden_dim]
        x = geo_hidden.reshape(B, -1)

        # Predict depth distribution
        depth_dist = self.depth_decoder(x)  # [B, num_depth_bins]
        depth_dist = F.log_softmax(depth_dist, dim=-1)

        # Predict mean surface normal (L2-normalized)
        mean_normal = self.normal_decoder(x)  # [B, 3]
        mean_normal = F.normalize(mean_normal, dim=-1)

        # Predict mean confidence (sigmoid for [0, 1] range)
        mean_confidence = torch.sigmoid(self.confidence_decoder(x))  # [B, 1]

        return {
            "depth_dist": depth_dist,
            "mean_normal": mean_normal,
            "mean_confidence": mean_confidence,
        }


def compute_geo_3d_decoder_loss(
    decoder: Geometric3DDecoder,
    hidden: Tensor,
    pseudo_gt: dict[str, Tensor],
    step: int,
    initial_weight: float = 0.05,
    anneal_steps: int = 15000,
) -> Tensor | None:
    """Compute auxiliary 3D decoder loss with linear annealing.

    Follows the same annealing pattern as compute_geometric_loss from
    geometric_decoder.py. Weight decreases linearly to zero, then returns
    None to skip computation.

    Args:
        decoder: Geometric3DDecoder module
        hidden: Encoder hidden states [B, K, hidden_dim]
        pseudo_gt: Pre-computed 3D pseudo-GT dictionary containing:
            - depth: [F, H, W] per-frame depth maps
            - normals: [F, H, W, 3] per-pixel normals (optional)
            - confidence: [F, H, W] per-pixel confidence (optional)
        step: Current training step
        initial_weight: Starting loss weight
        anneal_steps: Steps over which to anneal to zero

    Returns:
        Weighted loss tensor, or None if annealing complete
    """
    weight = max(0.0, initial_weight * (1.0 - step / anneal_steps))
    if weight <= 0.0:
        return None

    preds = decoder(hidden)
    device = hidden.device
    total_loss = torch.tensor(0.0, device=device)
    loss_count = 0

    # Depth distribution loss: KL divergence against GT histogram
    gt_depth = pseudo_gt.get("depth")
    if gt_depth is not None:
        try:
            # Create GT depth histogram from actual depth values
            gt_depth_flat = gt_depth[gt_depth > 0]  # Valid depths only
            if gt_depth_flat.numel() > 0:
                # Discretize GT depth into bins
                d_min = gt_depth_flat.min()
                d_max = gt_depth_flat.max().clamp(min=d_min + 1e-6)
                num_bins = decoder.num_depth_bins

                # Compute histogram (soft assignment via linear interpolation)
                normalized = (gt_depth_flat - d_min) / (d_max - d_min)  # [0, 1]
                bin_idx = (normalized * (num_bins - 1)).long().clamp(0, num_bins - 1)
                gt_hist = torch.zeros(num_bins, device=device)
                gt_hist.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=gt_hist.dtype))
                gt_hist = gt_hist / gt_hist.sum().clamp(min=1.0)

                # Expand to batch size
                B = preds["depth_dist"].shape[0]
                gt_hist = gt_hist.unsqueeze(0).expand(B, -1)

                # KL divergence: sum(gt * (log(gt) - log(pred)))
                gt_log = (gt_hist + 1e-8).log()
                kl_loss = F.kl_div(
                    preds["depth_dist"], gt_hist, reduction="batchmean", log_target=False,
                )
                total_loss = total_loss + kl_loss
                loss_count += 1
        except Exception as e:
            logger.debug(f"Depth decoder loss failed: {e}")

    # Normal direction loss: cosine similarity with mean GT normal
    gt_normals = pseudo_gt.get("normals")
    if gt_normals is not None:
        try:
            # Compute mean GT normal across spatial dims
            if gt_normals.dim() >= 3:
                gt_mean_normal = gt_normals.reshape(-1, 3).mean(dim=0)
                gt_mean_normal = F.normalize(gt_mean_normal.unsqueeze(0), dim=-1)

                B = preds["mean_normal"].shape[0]
                gt_mean_normal = gt_mean_normal.expand(B, -1).to(device=device)

                # 1 - cosine similarity
                cos_sim = F.cosine_similarity(preds["mean_normal"], gt_mean_normal, dim=-1)
                normal_loss = (1.0 - cos_sim).mean()
                total_loss = total_loss + normal_loss
                loss_count += 1
        except Exception as e:
            logger.debug(f"Normal decoder loss failed: {e}")

    # Confidence loss: MSE against mean GT confidence
    gt_conf = pseudo_gt.get("confidence")
    if gt_conf is not None:
        try:
            # Compute mean GT confidence (normalized)
            gt_conf_flat = gt_conf[gt_conf > 1.0]  # Valid confidence only
            if gt_conf_flat.numel() > 0:
                # Normalize: (conf - 1) / conf → [0, 1]
                gt_conf_norm = ((gt_conf_flat - 1.0) / gt_conf_flat).mean()

                B = preds["mean_confidence"].shape[0]
                gt_target = gt_conf_norm.expand(B, 1).to(device=device)

                conf_loss = F.mse_loss(preds["mean_confidence"], gt_target)
                total_loss = total_loss + conf_loss
                loss_count += 1
        except Exception as e:
            logger.debug(f"Confidence decoder loss failed: {e}")

    if loss_count == 0:
        return None

    return weight * (total_loss / loss_count)
