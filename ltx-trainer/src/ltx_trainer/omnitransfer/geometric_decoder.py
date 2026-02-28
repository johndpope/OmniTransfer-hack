"""Geometric Decoder for 3DiMo auxiliary supervision.

Implements the auxiliary geometric supervision from 3DiMo (arXiv:2602.03796v2,
Section 3.3). Predicts SMPL body pose and MANO hand joints from motion tokens
as an auxiliary training signal, annealed to zero during training.

Key design principles:
- Excludes global root orientation to ensure view-agnostic learning
- Uses linear annealing: lambda decreases from initial_weight to 0
- After annealing completes, computation is skipped entirely for efficiency

Quote: "We employ auxiliary geometric supervision... the loss weight is annealed
progressively as training proceeds... completely removed for remaining steps of
stage 2 and entirety of stage 3." (Section 3.3)

Pseudo-GT format (precomputed .pt files from 4DHumans + HaMeR):
    body_pose: [T, 23, 3, 3]       - SMPL rotation matrices (no global orient)
    hand_joints_3d: [T, 21, 3]     - MANO 3D hand joints
    body_joints_3d: [T, 44, 3]     - Full 3D body joints (optional)
    body_confidence: [T, 44]        - Per-joint confidence (optional)

References: 3DiMo paper (arXiv:2602.03796v2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer import logger


class GeometricDecoder(nn.Module):
    """Auxiliary geometric decoder for SMPL/MANO supervision.

    Predicts body pose (SMPL) and hand joints (MANO) from motion encoder
    hidden states. Used only during early training phases as auxiliary loss,
    then annealed to zero weight.

    Architecture:
        Input: motion hidden states [B, K, hidden_dim]
        -> Flatten: [B, K * hidden_dim]
        -> Body decoder MLP -> [B, 23 * 9] (SMPL body pose as rotation matrices)
        -> Hand decoder MLP -> [B, 21 * 3 * 2] (MANO hand joints, both hands)

    Note: Global root orientation is excluded per paper to prevent the motion
    encoder from learning viewpoint-dependent features.

    Args:
        hidden_dim: Input hidden dimension from motion encoder (default 512)
        num_tokens: Number of motion tokens K (default 5)
        num_body_joints: Number of SMPL body joints (23, excluding root)
        num_hand_joints: Number of MANO hand joints per hand (21)
        mlp_hidden_dim: Hidden dimension of decoder MLPs (default 1024)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_tokens: int = 5,
        num_body_joints: int = 23,
        num_hand_joints: int = 21,
        mlp_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_body_joints = num_body_joints
        self.num_hand_joints = num_hand_joints

        input_dim = num_tokens * hidden_dim

        # Body pose decoder: predicts SMPL rotation matrices (flattened)
        # 23 joints * 9 (3x3 rotation matrix flattened) = 207 outputs
        body_output_dim = num_body_joints * 9
        self.body_decoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, body_output_dim),
        )

        # Hand joints decoder: predicts 3D hand joints for both hands
        # 21 joints * 3 coords * 2 hands = 126 outputs
        hand_output_dim = num_hand_joints * 3 * 2
        self.hand_decoder = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, hand_output_dim),
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
        motion_hidden: Tensor,
    ) -> dict[str, Tensor]:
        """Predict geometric quantities from motion hidden states.

        Args:
            motion_hidden: Motion encoder hidden states [B, K, hidden_dim]

        Returns:
            Dictionary with:
                - body_pose: [B, 23, 9] SMPL rotation matrices (flattened per joint)
                - hand_joints: [B, 2, 21, 3] MANO hand joints (both hands)
        """
        B = motion_hidden.shape[0]

        # Flatten motion tokens: [B, K, hidden_dim] -> [B, K * hidden_dim]
        x = motion_hidden.reshape(B, -1)

        # Predict body pose
        body_pose = self.body_decoder(x)  # [B, 23 * 9]
        body_pose = body_pose.reshape(B, self.num_body_joints, 9)

        # Predict hand joints
        hand_joints = self.hand_decoder(x)  # [B, 21 * 3 * 2]
        hand_joints = hand_joints.reshape(B, 2, self.num_hand_joints, 3)

        return {
            "body_pose": body_pose,
            "hand_joints": hand_joints,
        }


def compute_geometric_loss(
    preds: dict[str, Tensor],
    pseudo_gt: dict[str, Tensor],
    step: int,
    initial_weight: float = 0.1,
    anneal_steps: int = 12000,
) -> Tensor | None:
    """Compute geometric supervision loss with linear annealing.

    The loss weight decreases linearly from initial_weight to 0 over
    anneal_steps. After annealing completes, returns None to skip
    computation entirely.

    Quote: "auxiliary loss weight annealed progressively as training proceeds...
    completely removed for remaining steps of stage 2 and entirety of stage 3"
    (Section 3.3)

    Args:
        preds: Predictions from GeometricDecoder
            - body_pose: [B, 23, 9]
            - hand_joints: [B, 2, 21, 3]
        pseudo_gt: Pseudo ground truth from 4DHumans/HaMeR
            - body_pose: [T, 23, 3, 3] or [B, 23, 3, 3] SMPL rotation matrices
            - hand_joints_3d: [T, 21, 3] or [B, 21, 3] per-hand 3D joints
        step: Current training step
        initial_weight: Starting loss weight lambda_0
        anneal_steps: Number of steps to anneal to zero

    Returns:
        Weighted geometric loss tensor, or None if annealing complete
    """
    # Compute annealing weight
    weight = max(0.0, initial_weight * (1.0 - step / anneal_steps))
    if weight <= 0.0:
        return None

    total_loss = torch.tensor(0.0, device=preds["body_pose"].device)
    loss_count = 0

    # Body pose loss
    if "body_pose" in pseudo_gt and pseudo_gt["body_pose"] is not None:
        gt_body = pseudo_gt["body_pose"]
        pred_body = preds["body_pose"]

        # Handle shape mismatch: GT may be [T, 23, 3, 3], need [B, 23, 9]
        if gt_body.dim() == 4 and gt_body.shape[-2:] == (3, 3):
            # Flatten rotation matrices: [B, 23, 3, 3] -> [B, 23, 9]
            B = pred_body.shape[0]
            # If GT has more frames than batch, take middle frame
            if gt_body.shape[0] > B:
                mid = gt_body.shape[0] // 2
                gt_body = gt_body[mid:mid + B]
            elif gt_body.shape[0] < B:
                # Repeat last frame to match batch size
                gt_body = gt_body[-1:].expand(B, -1, -1, -1)
            gt_body = gt_body.reshape(B, -1, 9)

        # Ensure same device/dtype
        gt_body = gt_body.to(device=pred_body.device, dtype=pred_body.dtype)

        # Match shapes (take first B if needed)
        if gt_body.shape[0] != pred_body.shape[0]:
            min_b = min(gt_body.shape[0], pred_body.shape[0])
            gt_body = gt_body[:min_b]
            pred_body = pred_body[:min_b]

        body_loss = F.mse_loss(pred_body, gt_body)
        total_loss = total_loss + body_loss
        loss_count += 1

    # Hand joints loss
    if "hand_joints_3d" in pseudo_gt and pseudo_gt["hand_joints_3d"] is not None:
        gt_hands = pseudo_gt["hand_joints_3d"]
        pred_hands = preds["hand_joints"]  # [B, 2, 21, 3]

        # Handle shape: GT may be [T, 21, 3] for single hand
        B = pred_hands.shape[0]
        gt_hands = gt_hands.to(device=pred_hands.device, dtype=pred_hands.dtype)

        if gt_hands.dim() == 3:
            # Single hand [T, 21, 3] -> expand to both hands [B, 2, 21, 3]
            if gt_hands.shape[0] > B:
                mid = gt_hands.shape[0] // 2
                gt_hands = gt_hands[mid:mid + B]
            elif gt_hands.shape[0] < B:
                gt_hands = gt_hands[-1:].expand(B, -1, -1)
            gt_hands = gt_hands.unsqueeze(1).expand(-1, 2, -1, -1)
        elif gt_hands.dim() == 4:
            # Both hands [T, 2, 21, 3] or [B, 2, 21, 3]
            if gt_hands.shape[0] > B:
                mid = gt_hands.shape[0] // 2
                gt_hands = gt_hands[mid:mid + B]
            elif gt_hands.shape[0] < B:
                gt_hands = gt_hands[-1:].expand(B, -1, -1, -1)

        hand_loss = F.mse_loss(pred_hands, gt_hands)
        total_loss = total_loss + hand_loss
        loss_count += 1

    if loss_count == 0:
        return None

    return weight * (total_loss / loss_count)
