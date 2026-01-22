#!/usr/bin/env python3
"""OmniTransfer Inference Script for LTX-2.

Generate videos using a trained OmniTransfer model with reference video guidance.

Quote: "OmniTransfer enables unified spatio-temporal video transfer through
in-context learning, using a reference video to guide the generation of
target videos." (Section 1, OmniTransfer paper)

Supported task types:
- Temporal tasks: motion_transfer, pose_reenactment, action_customization
  (reference provides motion/temporal cues)
- Appearance tasks: style_transfer, identity_preservation, scene_composition
  (reference provides spatial/appearance cues)

Usage:
    python scripts/omnitransfer_inference.py \\
        --model-path /path/to/ltx2_model.safetensors \\
        --lora-path outputs/omnitransfer_stage3/checkpoint-final \\
        --reference-video /path/to/reference.mp4 \\
        --prompt "A person dancing gracefully" \\
        --task-type motion_transfer \\
        --output output.mp4
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from ltx_trainer import logger
from ltx_trainer.model_loader import load_model
from ltx_trainer.omnitransfer.components import OmniTransferTask
from ltx_trainer.omnitransfer.latent_constructor import ReferenceLatentConstructor
from ltx_trainer.video_utils import load_video_frames, resize_for_vae, save_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="OmniTransfer inference for video generation with reference guidance"
    )

    # Model paths
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to LTX-2 model checkpoint",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=Path,
        required=True,
        help="Path to Gemma text encoder",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Path to trained OmniTransfer LoRA checkpoint",
    )

    # Input
    parser.add_argument(
        "--reference-video",
        type=Path,
        required=True,
        help="Path to reference video",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt",
    )
    parser.add_argument(
        "--first-frame-image",
        type=Path,
        default=None,
        help="Optional first frame image for conditioning",
    )

    # Task configuration
    parser.add_argument(
        "--task-type",
        type=str,
        default="motion_transfer",
        choices=[
            "motion_transfer",
            "pose_reenactment",
            "action_customization",
            "style_transfer",
            "identity_preservation",
            "scene_composition",
        ],
        help="Type of transfer task",
    )

    # Generation parameters
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Output video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=544,
        help="Output video height",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=97,
        help="Number of frames to generate (must satisfy frames %% 8 == 1)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output video frame rate",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output video path",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )

    return parser.parse_args()


def get_task_enum(task_type: str) -> OmniTransferTask:
    """Convert task type string to enum."""
    task_map = {
        "motion_transfer": OmniTransferTask.MOTION_TRANSFER,
        "pose_reenactment": OmniTransferTask.POSE_REENACTMENT,
        "action_customization": OmniTransferTask.ACTION_CUSTOMIZATION,
        "style_transfer": OmniTransferTask.STYLE_TRANSFER,
        "identity_preservation": OmniTransferTask.IDENTITY_PRESERVATION,
        "scene_composition": OmniTransferTask.SCENE_COMPOSITION,
    }
    return task_map.get(task_type, OmniTransferTask.MOTION_TRANSFER)


def main():
    args = parse_args()

    # Validate frame count
    if args.num_frames % 8 != 1:
        raise ValueError(
            f"num_frames must satisfy frames %% 8 == 1 for LTX-2, "
            f"got {args.num_frames}. Valid values: 1, 9, 17, 25, ..., 97, ..."
        )

    # Setup
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # Set seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    logger.info(f"Running OmniTransfer inference with task: {args.task_type}")

    # Load model components
    logger.info("Loading model components...")
    components = load_model(
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        dtype=dtype,
        device=device,
    )

    # Load LoRA if provided
    if args.lora_path is not None:
        logger.info(f"Loading LoRA from {args.lora_path}")
        from peft import PeftModel
        components.transformer = PeftModel.from_pretrained(
            components.transformer,
            args.lora_path,
        )

    # Initialize latent constructor
    latent_constructor = ReferenceLatentConstructor(
        latent_channels=128,
        default_task=get_task_enum(args.task_type),
    )

    # Load and encode reference video
    logger.info("Processing reference video...")
    ref_frames = load_video_frames(args.reference_video)
    ref_frames = resize_for_vae(
        ref_frames,
        target_width=args.width,
        target_height=args.height,
    )
    ref_frames = ref_frames.to(device, dtype=dtype).unsqueeze(0)

    with torch.inference_mode():
        ref_latent = components.vae_encoder.encode(ref_frames)

    # Encode first frame if provided
    first_frame_latent = None
    if args.first_frame_image is not None:
        logger.info("Processing first frame image...")
        from PIL import Image
        import torchvision.transforms as T

        img = Image.open(args.first_frame_image).convert("RGB")
        img = img.resize((args.width, args.height))
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        first_frame = transform(img).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
        first_frame = first_frame.to(device, dtype=dtype)

        with torch.inference_mode():
            first_frame_latent = components.vae_encoder.encode(first_frame)

    # Compute latent dimensions
    latent_height = args.height // 32
    latent_width = args.width // 32
    latent_frames = (args.num_frames - 1) // 8 + 1

    # Construct inference latents
    logger.info("Constructing inference latents...")
    constructed = latent_constructor.construct_for_inference(
        ref_video_latent=ref_latent,
        tgt_first_frame_latent=first_frame_latent,
        task=get_task_enum(args.task_type),
        num_frames=latent_frames,
        height=latent_height,
        width=latent_width,
    )

    # Encode prompt
    logger.info("Encoding prompt...")
    with torch.inference_mode():
        prompt_embeds = components.text_encoder.encode(args.prompt)
        negative_embeds = components.text_encoder.encode(args.negative_prompt)

    # Setup scheduler
    from ltx_core.pipeline.components.schedulers import LTX2Scheduler
    from ltx_core.pipeline.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.pipeline.components.guiders import CFGGuider

    scheduler = LTX2Scheduler(num_inference_steps=args.num_inference_steps)
    diffusion_step = EulerDiffusionStep()
    guider = CFGGuider(guidance_scale=args.guidance_scale)

    # Get timesteps
    timesteps = scheduler.get_timesteps()

    logger.info(f"Running {args.num_inference_steps} denoising steps...")

    # Denoising loop
    # Note: This is a simplified loop - full implementation would integrate
    # with the OmniTransfer components (TPB, RCL, TMA) through the model forward
    latent = constructed.tgt_latent

    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Prepare model input with reference
        # In full implementation, this would:
        # 1. Apply TPB to reference positions
        # 2. Use RCL attention for decoupled ref/target
        # 3. Inject TMA features into target

        # For now, simplified forward pass
        with torch.inference_mode():
            # This would be replaced with full OmniTransfer forward
            # including reference conditioning
            noise_pred = components.transformer(
                latent=latent,
                timestep=t,
                context=prompt_embeds.video_encoding,
            )

            # Euler step
            latent = diffusion_step(
                x_t=latent,
                noise_pred=noise_pred,
                sigma=t,
            )

    # Decode latent
    logger.info("Decoding video...")
    with torch.inference_mode():
        # VideoDecoder uses forward(), not decode()
        video = components.vae_decoder(latent)

    # Save video
    save_video(
        video.squeeze(0),
        args.output,
        fps=args.fps,
    )

    logger.info(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
