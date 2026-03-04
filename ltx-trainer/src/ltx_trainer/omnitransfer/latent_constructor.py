"""Reference Latent Construction for OmniTransfer.

This module implements the Reference Latent Construction component described
in Section 4.1 of the OmniTransfer paper.

Quote: "Reference Latent Construction builds separate latent representations for
the reference video l_ref and target video l_tgt. For the reference, we set the
noise level to zero (z0_ref is noise-free) and use task-specific flags in the
mask channel m_ref." (Section 4.1)

References: OmniTransfer paper (arXiv:2601.14250v1, Jan 20, 2026)
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ltx_trainer.omnitransfer.components import OmniTransferTask


@dataclass
class ConstructedLatents:
    """Container for constructed reference and target latents.

    Quote: "l_ref ∈ R^{f × h_ref × w_ref × (2n+4)} = [c_ref, m_ref, z0_ref]" (Section 4.1)

    Attributes:
        ref_latent: Reference video latent [B, C, F, H, W] (noise-free)
        tgt_latent: Target video latent [B, C, F, H, W] (noised)
        ref_mask: Reference mask with task flags [B, 1, F, H, W]
        tgt_mask: Target mask (0/1 for conditioning) [B, 1, F, H, W]
        ref_clean: Clean reference for loss-free branch [B, C, F, H, W]
        tgt_clean: Clean target for loss computation [B, C, F, H, W]
        task: The transfer task type
    """
    ref_latent: torch.Tensor
    tgt_latent: torch.Tensor
    ref_mask: torch.Tensor
    tgt_mask: torch.Tensor
    ref_clean: torch.Tensor
    tgt_clean: torch.Tensor
    task: OmniTransferTask

    @property
    def ref_seq_len(self) -> int:
        """Sequence length of reference after patchification."""
        # [B, C, F, H, W] -> F * H * W
        return self.ref_latent.shape[2] * self.ref_latent.shape[3] * self.ref_latent.shape[4]

    @property
    def tgt_seq_len(self) -> int:
        """Sequence length of target after patchification."""
        return self.tgt_latent.shape[2] * self.tgt_latent.shape[3] * self.tgt_latent.shape[4]


class ReferenceLatentConstructor(nn.Module):
    """Constructs reference and target latents for OmniTransfer training.

    Quote: "In OmniTransfer, we construct separate latent representations for
    the reference and target videos. The reference latent l_ref consists of:
    - c_ref: VAE-encoded visual features from the reference video
    - m_ref: Task-specific mask flags (-1 temporal, -2 identity, -3 style)
    - z0_ref: Noise-free latent (set to zero or clean encoding)

    The target latent l_tgt follows the standard diffusion process:
    - c_tgt: VAE-encoded features from target first frame (if conditioning)
    - m_tgt: Binary mask (1 for preserved regions, 0 for generation)
    - z_t: Noised latent at timestep t" (Section 4.1)

    For LTX-2 adaptation:
    - Latents have shape [B, 128, F, H, W] for video
    - We construct masks as separate tensors (not concatenated to latent)
    - Reference always uses t=0 (noise-free) per RCL requirements
    """

    def __init__(
        self,
        latent_channels: int = 128,
        default_task: OmniTransferTask = OmniTransferTask.MOTION_TRANSFER,
    ):
        """Initialize the constructor.

        Args:
            latent_channels: Number of channels in VAE latent (128 for LTX-2)
            default_task: Default task type if not specified
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.default_task = default_task

    def create_task_mask(
        self,
        shape: Tuple[int, ...],
        task: OmniTransferTask,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create task-specific mask for reference latent.

        Quote: "We set m_ref to task-specific flags: -1 for temporal tasks,
        -2 for identity-based tasks, -3 for style tasks." (Section 4.1)

        Args:
            shape: Shape for the mask [B, 1, F, H, W]
            task: The transfer task type
            device: Target device
            dtype: Target dtype

        Returns:
            Task mask tensor filled with task flag value
        """
        task_flag = task.task_flag
        return torch.full(shape, task_flag, device=device, dtype=dtype)

    def create_conditioning_mask(
        self,
        shape: Tuple[int, ...],
        first_frame_conditioning: bool,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create conditioning mask for target latent.

        For first-frame conditioning, the first frame tokens are set to 1
        (preserved), rest to 0 (to be generated).

        Args:
            shape: Shape for the mask [B, 1, F, H, W]
            first_frame_conditioning: Whether to condition on first frame
            height: Latent height
            width: Latent width
            device: Target device
            dtype: Target dtype

        Returns:
            Conditioning mask tensor
        """
        mask = torch.zeros(shape, device=device, dtype=dtype)

        if first_frame_conditioning:
            # First frame is preserved (mask = 1)
            mask[:, :, 0, :, :] = 1.0

        return mask

    def construct(
        self,
        ref_video_latent: torch.Tensor,
        tgt_video_latent: torch.Tensor,
        task: OmniTransferTask | None = None,
        noise: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        first_frame_conditioning: bool = True,
        first_frame_conditioning_prob: float = 0.1,
        first_frame_latent: torch.Tensor | None = None,
    ) -> ConstructedLatents:
        """Construct reference and target latents for training.

        Quote: "The reference branch adopts a fixed t = 0, meaning it remains
        noise-free throughout the diffusion process. This design ensures that
        the reference information is always clean and reliable." (Section 4.3)

        I2V Mode (pose-free animation):
        When first_frame_latent is provided, it's used as the conditioning image
        instead of extracting the first frame from tgt_video_latent. This enables
        training where a static image is animated with motion from reference video.

        Args:
            ref_video_latent: Pre-encoded reference video latent [B, C, F, H, W]
            tgt_video_latent: Pre-encoded target video latent [B, C, F, H, W]
            task: Transfer task type (uses default if None)
            noise: Optional pre-sampled noise for target [B, C, F, H, W]
            sigma: Noise level/timestep for target [B] or [B, 1, 1, 1, 1]
            first_frame_conditioning: Whether to condition on first frame
            first_frame_conditioning_prob: Probability of first frame conditioning
            first_frame_latent: Explicit first-frame latent for I2V [B, C, 1, H, W]

        Returns:
            ConstructedLatents containing all constructed tensors
        """
        task = task or self.default_task
        batch_size = ref_video_latent.shape[0]
        device = ref_video_latent.device
        dtype = ref_video_latent.dtype

        # Reference dimensions
        _, _, ref_frames, ref_height, ref_width = ref_video_latent.shape
        # Target dimensions
        _, _, tgt_frames, tgt_height, tgt_width = tgt_video_latent.shape

        # Create task mask for reference
        # Quote: "m_ref to task-specific flags" (Section 4.1)
        ref_mask = self.create_task_mask(
            shape=(batch_size, 1, ref_frames, ref_height, ref_width),
            task=task,
            device=device,
            dtype=dtype,
        )

        # Reference latent is noise-free (t=0)
        # Quote: "reference branch adopts a fixed t = 0" (Section 4.3)
        ref_latent = ref_video_latent.clone()
        ref_clean = ref_video_latent.clone()

        # I2V mode: If explicit first_frame_latent provided, always apply conditioning
        # Otherwise, apply stochastically based on probability
        if first_frame_latent is not None:
            apply_first_frame = True
        else:
            apply_first_frame = first_frame_conditioning and (
                torch.rand(1).item() < first_frame_conditioning_prob
            )

        # Create conditioning mask for target
        tgt_mask = self.create_conditioning_mask(
            shape=(batch_size, 1, tgt_frames, tgt_height, tgt_width),
            first_frame_conditioning=apply_first_frame,
            height=tgt_height,
            width=tgt_width,
            device=device,
            dtype=dtype,
        )

        # Store clean target for loss computation
        tgt_clean = tgt_video_latent.clone()

        # Apply noise to target latent
        if noise is None:
            noise = torch.randn_like(tgt_video_latent)

        if sigma is None:
            # Will be set by timestep sampler in strategy
            tgt_latent = tgt_video_latent.clone()
        else:
            # Expand sigma for broadcasting
            if sigma.dim() == 1:
                sigma = sigma.view(-1, 1, 1, 1, 1)

            # Flow matching noise application: x_t = (1 - sigma) * x_0 + sigma * noise
            tgt_latent = (1 - sigma) * tgt_video_latent + sigma * noise

            # If first frame conditioning, keep first frame clean
            if apply_first_frame:
                if first_frame_latent is not None:
                    # I2V mode: Use explicit first-frame latent as conditioning
                    # first_frame_latent is [B, C, 1, H, W], extract and use
                    tgt_latent[:, :, 0, :, :] = first_frame_latent[:, :, 0, :, :]
                else:
                    # Standard mode: Use first frame from target video
                    tgt_latent[:, :, 0, :, :] = tgt_video_latent[:, :, 0, :, :]

        return ConstructedLatents(
            ref_latent=ref_latent,
            tgt_latent=tgt_latent,
            ref_mask=ref_mask,
            tgt_mask=tgt_mask,
            ref_clean=ref_clean,
            tgt_clean=tgt_clean,
            task=task,
        )

    def construct_for_inference(
        self,
        ref_video_latent: torch.Tensor,
        tgt_first_frame_latent: torch.Tensor | None,
        task: OmniTransferTask | None = None,
        num_frames: int = 97,
        height: int = 18,
        width: int = 30,
    ) -> ConstructedLatents:
        """Construct latents for inference/generation.

        Args:
            ref_video_latent: Pre-encoded reference video latent [B, C, F, H, W]
            tgt_first_frame_latent: Optional first frame latent [B, C, 1, H, W]
            task: Transfer task type
            num_frames: Number of frames to generate
            height: Latent height
            width: Latent width

        Returns:
            ConstructedLatents for inference
        """
        task = task or self.default_task
        batch_size = ref_video_latent.shape[0]
        device = ref_video_latent.device
        dtype = ref_video_latent.dtype

        _, _, ref_frames, ref_height, ref_width = ref_video_latent.shape

        # Reference mask
        ref_mask = self.create_task_mask(
            shape=(batch_size, 1, ref_frames, ref_height, ref_width),
            task=task,
            device=device,
            dtype=dtype,
        )

        # Target starts as pure noise
        tgt_latent = torch.randn(
            batch_size, self.latent_channels, num_frames, height, width,
            device=device, dtype=dtype
        )

        # Target mask - first frame preserved if provided
        has_first_frame = tgt_first_frame_latent is not None
        tgt_mask = self.create_conditioning_mask(
            shape=(batch_size, 1, num_frames, height, width),
            first_frame_conditioning=has_first_frame,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )

        # If first frame provided, inject it
        if has_first_frame:
            tgt_latent[:, :, 0:1, :, :] = tgt_first_frame_latent

        return ConstructedLatents(
            ref_latent=ref_video_latent.clone(),
            tgt_latent=tgt_latent,
            ref_mask=ref_mask,
            tgt_mask=tgt_mask,
            ref_clean=ref_video_latent.clone(),
            tgt_clean=tgt_latent.clone(),  # For inference, clean = initial noise
            task=task,
        )


class AudioReferenceLatentConstructor(nn.Module):
    """Constructs audio reference latents for audio-video OmniTransfer.

    For LTX-2's dual audio-video stream, we also need to handle audio
    references for tasks involving audio transfer.

    This extends the video constructor with audio-specific handling.
    """

    def __init__(
        self,
        audio_latent_channels: int = 8,
        video_constructor: ReferenceLatentConstructor | None = None,
    ):
        super().__init__()
        self.audio_latent_channels = audio_latent_channels
        self.video_constructor = video_constructor or ReferenceLatentConstructor()

    def construct_audio(
        self,
        ref_audio_latent: torch.Tensor | None,
        tgt_audio_latent: torch.Tensor | None,
        task: OmniTransferTask,
        sigma: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Construct audio latents.

        For most visual transfer tasks, audio is generated fresh.
        For audio-specific tasks, reference audio may be used.

        Args:
            ref_audio_latent: Reference audio latent [B, C, T, F] or None
            tgt_audio_latent: Target audio latent [B, C, T, F] or None
            task: Transfer task type
            sigma: Noise level for target audio

        Returns:
            Tuple of (ref_audio, tgt_audio) latents
        """
        if ref_audio_latent is None:
            return None, None

        # Audio reference is also noise-free (following RCL design)
        ref_audio = ref_audio_latent.clone()

        if tgt_audio_latent is None:
            return ref_audio, None

        # Apply noise to target audio
        if sigma is not None:
            if sigma.dim() == 1:
                sigma = sigma.view(-1, 1, 1, 1)
            noise = torch.randn_like(tgt_audio_latent)
            tgt_audio = (1 - sigma) * tgt_audio_latent + sigma * noise
        else:
            tgt_audio = tgt_audio_latent.clone()

        return ref_audio, tgt_audio
