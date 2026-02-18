#!/usr/bin/env python3
"""Progressive Overfit: Teacher-guided style transfer via OmniTransfer.

This script orchestrates the full pipeline for overfitting a LoRA adapter to
reproduce a specific style transformation between an input image and a target
image (e.g., from Grok Imagine API).

Pipeline stages (sequential for VRAM safety):
  A. VAE encode both images → reference_latents/ and latents/
  B. Text encode prompt → conditions/
  C. Generate per-pair YAML config
  D. Train via existing train.py in 100-step chunks, checking convergence

Usage:
    python scripts/progressive_overfit.py \
        --pair-id t2_isometric \
        --input-image /home/johndpope/Desktop/t2.webp \
        --target-image /home/johndpope/Desktop/t2-isometric-3d-view-photorealistic.jpg \
        --prompt "isometric 3D view of this scene, photorealistic miniature diorama" \
        --max-steps 500 --convergence-loss 0.005

Output: JSON to stdout with {converged, final_loss, steps_taken, checkpoint_path}
"""

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Add scripts directory to path so we can import sibling scripts
_SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import torch

from ltx_trainer import logger


# Default model paths (can be overridden via CLI)
DEFAULT_MODEL_PATH = "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev-fp8.safetensors"
DEFAULT_TEXT_ENCODER_PATH = "/media/2TB/ltx-models/gemma"
DEFAULT_CONFIG_TEMPLATE = Path(__file__).parent.parent / "configs" / "ltx2_progressive_overfit.yaml"

# Training chunk size
STEPS_PER_CHUNK = 100


def encode_images(
    input_image: Path,
    target_image: Path,
    output_dir: Path,
    model_path: Path,
    device: str,
    target_h: int,
    target_w: int,
) -> None:
    """Stage A: VAE encode both images into latent tensors.

    Loads the VAE once and encodes both images to avoid double load/unload.
    """
    from encode_single_image import load_and_prepare_image

    from ltx_trainer.model_loader import load_video_vae_encoder

    ref_dir = output_dir / "reference_latents"
    lat_dir = output_dir / "latents"
    ref_dir.mkdir(parents=True, exist_ok=True)
    lat_dir.mkdir(parents=True, exist_ok=True)

    ref_path = ref_dir / "0.pt"
    lat_path = lat_dir / "0.pt"

    # Skip if both already encoded
    if ref_path.exists() and lat_path.exists():
        logger.info("[Stage A] Both images already encoded, skipping VAE loading")
    else:
        # Load VAE once for both images
        logger.info(f"[Stage A] Loading VAE encoder from {model_path}")
        dtype = torch.bfloat16
        vae_encoder = load_video_vae_encoder(model_path, dtype=dtype)
        vae_encoder = vae_encoder.to(device)
        vae_encoder.eval()

        for img_path, out_path, label in [
            (input_image, ref_path, "input (→ reference_latents)"),
            (target_image, lat_path, "target (→ latents)"),
        ]:
            if out_path.exists():
                logger.info(f"[Stage A] Skipping {label} (already encoded): {out_path}")
                continue

            logger.info(f"[Stage A] Encoding {label}: {img_path}")
            img_tensor = load_and_prepare_image(img_path, target_h, target_w)
            img_tensor = img_tensor.to(device, dtype=dtype)

            with torch.inference_mode():
                latent = vae_encoder(img_tensor)

            latent = latent.cpu()
            torch.save({
                "latents": latent.squeeze(0),  # [C, 1, H_lat, W_lat]
                "num_frames": torch.tensor([1]),
                "height": torch.tensor([latent.shape[3]]),
                "width": torch.tensor([latent.shape[4]]),
            }, out_path)
            logger.info(f"[Stage A] Saved latent {latent.squeeze(0).shape} to {out_path}")

        # Cleanup VAE
        del vae_encoder
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Stage A] VAE encoder unloaded")

    # Verify images are different
    ref_data = torch.load(ref_path, weights_only=False)
    lat_data = torch.load(lat_path, weights_only=False)
    diff = (ref_data["latents"] - lat_data["latents"]).abs().mean().item()
    logger.info(f"[Stage A] Reference vs target diff: {diff:.4f}")
    if diff < 0.01:
        logger.warning("Reference and target are very similar! Style transfer may not learn anything.")


def compute_text_embedding(
    prompt: str,
    output_dir: Path,
    model_path: Path,
    text_encoder_path: Path,
    device: str,
    load_in_8bit: bool = True,
) -> None:
    """Stage B: Compute raw AND final (post-connector) text embeddings.

    Produces two directories:
      - conditions/0.pt:       raw Gemma embeddings [1024, 3840]
      - conditions_final/0.pt: post-connector embeddings [seq_len, 4096]

    The final embeddings allow training to skip loading the text encoder entirely,
    saving ~28GB VRAM on the training GPU.
    """
    from ltx_trainer.model_loader import load_text_encoder

    cond_dir = output_dir / "conditions"
    final_dir = output_dir / "conditions_final"
    cond_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    cond_path = cond_dir / "0.pt"
    final_path = final_dir / "0.pt"

    # Skip if both already exist
    if cond_path.exists() and final_path.exists():
        logger.info(f"[Stage B] Skipping text embedding (already computed): {cond_path}")
        logger.info(f"[Stage B] Skipping final embedding (already computed): {final_path}")
        return

    # Load text encoder (with full Gemma + connectors)
    logger.info(f"[Stage B] Loading text encoder (8-bit={load_in_8bit}) on {device}")
    text_encoder = load_text_encoder(
        checkpoint_path=model_path,
        gemma_model_path=text_encoder_path,
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
    )
    text_encoder.eval()

    # Step 1: Compute raw embeddings (pre-connector)
    if not cond_path.exists():
        logger.info(f"[Stage B] Computing raw text embedding for: {prompt[:80]}...")
        with torch.inference_mode():
            prompt_embeds, prompt_attention_mask = text_encoder._preprocess_text(
                prompt, padding_side="left"
            )
        raw_data = {
            "prompt_embeds": prompt_embeds[0].cpu().contiguous(),
            "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
            "caption": prompt,
        }
        torch.save(raw_data, cond_path)
        logger.info(f"[Stage B] Saved raw embedding {raw_data['prompt_embeds'].shape} to {cond_path}")
    else:
        # Load existing raw embeddings for connector pass
        logger.info(f"[Stage B] Raw embedding already exists: {cond_path}")
        raw_data = torch.load(cond_path, map_location="cpu", weights_only=True)
        prompt_embeds = raw_data["prompt_embeds"].unsqueeze(0).to(device)
        prompt_attention_mask = raw_data["prompt_attention_mask"].unsqueeze(0).to(device)

    # Step 2: Run connectors to produce final embeddings
    if not final_path.exists():
        logger.info("[Stage B] Computing final (post-connector) embeddings...")
        # Ensure we have batched tensors on device
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0).to(device)
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0).to(device)
        elif prompt_embeds.device != torch.device(device):
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_mask = prompt_attention_mask.to(device)

        with torch.inference_mode():
            video_embeds, audio_embeds, attention_mask = text_encoder._run_connectors(
                prompt_embeds, prompt_attention_mask
            )

        final_data = {
            "video_prompt_embeds": video_embeds[0].cpu().contiguous(),
            "audio_prompt_embeds": audio_embeds[0].cpu().contiguous(),
            "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
            "is_final_embedding": True,
        }
        torch.save(final_data, final_path)
        logger.info(
            f"[Stage B] Saved final embedding video={final_data['video_prompt_embeds'].shape} "
            f"audio={final_data['audio_prompt_embeds'].shape} to {final_path}"
        )
    else:
        logger.info(f"[Stage B] Final embedding already exists: {final_path}")

    # Cleanup text encoder
    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("[Stage B] Text encoder unloaded")


def generate_config(
    template_path: Path,
    output_dir: Path,
    data_root: Path,
    pair_id: str,
    model_path: Path,
    text_encoder_path: Path,
    checkpoint_path: Path | None = None,
    steps: int = STEPS_PER_CHUNK,
    learning_rate: float = 1e-4,
    style_loss_weight: float = 0.5,
    lpips_weight: float = 0.1,
    lora_rank: int = 64,
    encode_device: str = "cuda:1",
    train_device: str = "cuda:0",
) -> Path:
    """Stage C: Generate per-pair YAML config from template.

    Reads the template, modifies key fields, and writes a new config.
    """
    import yaml

    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Model paths
    config["model"]["model_path"] = str(model_path)
    config["model"]["text_encoder_path"] = str(text_encoder_path)
    config["model"]["load_checkpoint"] = str(checkpoint_path) if checkpoint_path else None

    # LoRA
    config["lora"]["rank"] = lora_rank
    config["lora"]["alpha"] = lora_rank

    # Training strategy losses
    config["training_strategy"]["style_loss_weight"] = style_loss_weight
    config["training_strategy"]["lpips_weight"] = lpips_weight

    # Optimization
    config["optimization"]["steps"] = steps
    config["optimization"]["learning_rate"] = learning_rate

    # Hardware devices — multi-GPU: text encoder + VAE on encode_device,
    # transformer on train_device. The trainer handles device placement.
    config["hardware"]["devices"]["text_encoder"] = encode_device
    config["hardware"]["devices"]["vae_encoder"] = encode_device
    config["hardware"]["devices"]["vae_decoder"] = encode_device
    config["hardware"]["devices"]["transformer"] = train_device
    config["hardware"]["devices"]["audio_vae"] = train_device
    config["hardware"]["devices"]["vocoder"] = train_device

    # Data root + final embeddings (skip text encoder during training)
    config["data"]["preprocessed_data_root"] = str(data_root)
    config["data"]["use_cached_final_embeddings"] = True
    config["data"]["final_embeddings_dir"] = "conditions_final"

    # Output directory
    pair_output = output_dir / pair_id / "output"
    config["output_dir"] = str(pair_output)

    # W&B run name
    config["wandb"]["tags"] = ["style-transfer", "overfit", pair_id]

    config_path = output_dir / pair_id / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"[Stage C] Config written to: {config_path}")
    return config_path


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest checkpoint in the output directory."""
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    checkpoints = sorted(ckpt_dir.glob("lora_weights_step_*.safetensors"))
    if not checkpoints:
        return None

    return checkpoints[-1]


def parse_loss_from_output(output: str) -> float | None:
    """Parse the last reported loss value from training output.

    The trainer logs lines like:
        Step 80/100 - Loss: 0.0423, LR: 1.00e-04, ...
    """
    pattern = r"Loss:\s*([\d.]+)"
    matches = re.findall(pattern, output)
    if matches:
        return float(matches[-1])
    return None


def run_training_chunk(
    config_path: Path,
    disable_progress_bars: bool = True,
    timeout: int = 7200,
) -> tuple[bool, float | None, str]:
    """Stage D: Run one training chunk via subprocess.

    Runs the existing train.py as a subprocess with the generated config.
    The config specifies multi-GPU device placement (text encoder on one GPU,
    transformer on another), so no CUDA_VISIBLE_DEVICES isolation is needed.

    Args:
        config_path: Path to the YAML config file.
        disable_progress_bars: Suppress progress bars in subprocess.
        timeout: Timeout in seconds (default: 7200 = 2 hours).

    Returns:
        (success, final_loss, output_text)
    """
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train.py"),
        str(config_path),
    ]
    if disable_progress_bars:
        cmd.append("--disable-progress-bars")

    # Log to file so we can monitor progress during training
    log_path = config_path.parent / "train_chunk.log"

    logger.info(f"[Stage D] Running: {' '.join(cmd)}")
    logger.info(f"[Stage D] Log file: {log_path}")

    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

    output = log_path.read_text()

    if result.returncode != 0:
        logger.error(f"Training failed with code {result.returncode}")
        # Show last 2000 chars of log
        logger.error(f"Log tail: {output[-2000:]}")
        return False, None, output

    loss = parse_loss_from_output(output)
    return True, loss, output


def progressive_overfit(
    pair_id: str,
    input_image: Path,
    target_image: Path,
    prompt: str,
    max_steps: int = 500,
    convergence_loss: float = 0.005,
    model_path: Path = Path(DEFAULT_MODEL_PATH),
    text_encoder_path: Path = Path(DEFAULT_TEXT_ENCODER_PATH),
    config_template: Path = DEFAULT_CONFIG_TEMPLATE,
    work_dir: Path = Path("/tmp/progressive_overfit"),
    checkpoint: Path | None = None,
    target_h: int = 448,
    target_w: int = 832,
    learning_rate: float = 1e-4,
    style_loss_weight: float = 0.5,
    lpips_weight: float = 0.1,
    lora_rank: int = 64,
    encode_device: str = "cuda:1",
    train_device: str = "cuda:0",
    load_in_8bit: bool = True,
) -> dict:
    """Run the full progressive overfitting pipeline.

    Args:
        pair_id: Unique identifier for this image pair.
        input_image: Path to style source image.
        target_image: Path to target (ground truth) image.
        prompt: Text prompt describing the transformation.
        max_steps: Maximum total training steps.
        convergence_loss: Stop if loss drops below this value.
        model_path: Path to LTX-2 checkpoint.
        text_encoder_path: Path to Gemma model directory.
        config_template: Path to YAML config template.
        work_dir: Working directory for data and outputs.
        checkpoint: Optional initial checkpoint to resume from.
        target_h: Target image height (divisible by 32).
        target_w: Target image width (divisible by 32).
        learning_rate: Learning rate for training.
        style_loss_weight: Weight for VGG Gram matrix loss.
        lpips_weight: Weight for LPIPS perceptual loss.
        lora_rank: LoRA adapter rank.
        encode_device: CUDA device for encoding stages.
        train_device: CUDA device for training.
        load_in_8bit: Load text encoder in 8-bit mode.

    Returns:
        Dict with {converged, final_loss, steps_taken, checkpoint_path, pair_id}.
    """
    pair_dir = work_dir / pair_id
    data_root = pair_dir / "data"
    output_dir = pair_dir / "output"

    logger.info("=" * 60)
    logger.info(f"Progressive Overfit: {pair_id}")
    logger.info(f"  Input:  {input_image}")
    logger.info(f"  Target: {target_image}")
    logger.info(f"  Prompt: {prompt[:80]}...")
    logger.info(f"  Max steps: {max_steps}, Convergence: {convergence_loss}")
    logger.info("=" * 60)

    start_time = time.time()

    # ================================================================
    # Stage A: VAE encode both images
    # ================================================================
    encode_images(
        input_image=input_image,
        target_image=target_image,
        output_dir=data_root,
        model_path=model_path,
        device=encode_device,
        target_h=target_h,
        target_w=target_w,
    )

    # Force GPU cleanup after VAE encoding
    torch.cuda.empty_cache()
    gc.collect()

    # ================================================================
    # Stage B: Text encode prompt
    # ================================================================
    compute_text_embedding(
        prompt=prompt,
        output_dir=data_root,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        device=encode_device,
        load_in_8bit=load_in_8bit,
    )

    # Force GPU cleanup after text encoding
    torch.cuda.empty_cache()
    gc.collect()

    # ================================================================
    # Stage C+D: Iterative training chunks with convergence checking
    # ================================================================
    total_steps = 0
    final_loss = None
    converged = False
    current_checkpoint = checkpoint
    max_chunks = max_steps // STEPS_PER_CHUNK

    for chunk_idx in range(max_chunks):
        chunk_num = chunk_idx + 1
        logger.info(f"\n{'='*40} Chunk {chunk_num}/{max_chunks} {'='*40}")

        # Stage C: Generate config for this chunk
        config_path = generate_config(
            template_path=config_template,
            output_dir=work_dir,
            data_root=data_root,
            pair_id=pair_id,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            checkpoint_path=current_checkpoint,
            steps=STEPS_PER_CHUNK,
            learning_rate=learning_rate,
            style_loss_weight=style_loss_weight,
            lpips_weight=lpips_weight,
            lora_rank=lora_rank,
            encode_device=encode_device,
            train_device=train_device,
        )

        # Stage D: Run training chunk
        success, loss, output = run_training_chunk(config_path)

        if not success:
            logger.error(f"Chunk {chunk_num} failed! Stopping.")
            break

        total_steps += STEPS_PER_CHUNK
        if loss is not None:
            final_loss = loss

        # Find the latest checkpoint
        latest_ckpt = find_latest_checkpoint(output_dir)
        if latest_ckpt:
            current_checkpoint = latest_ckpt
            logger.info(f"Chunk {chunk_num}: loss={loss}, checkpoint={latest_ckpt.name}")
        else:
            logger.warning(f"Chunk {chunk_num}: No checkpoint found after training!")

        # Check convergence
        if final_loss is not None and final_loss < convergence_loss:
            converged = True
            logger.info(f"Converged! Loss {final_loss:.6f} < {convergence_loss}")
            break

        logger.info(f"Not converged yet: loss={final_loss}, threshold={convergence_loss}")

    elapsed = time.time() - start_time

    result = {
        "pair_id": pair_id,
        "converged": converged,
        "final_loss": final_loss,
        "steps_taken": total_steps,
        "checkpoint_path": str(current_checkpoint) if current_checkpoint else None,
        "elapsed_seconds": round(elapsed, 1),
        "input_image": str(input_image),
        "target_image": str(target_image),
        "prompt": prompt,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Result: {'CONVERGED' if converged else 'NOT CONVERGED'}")
    logger.info(f"  Steps: {total_steps}, Loss: {final_loss}, Time: {elapsed:.0f}s")
    logger.info(f"  Checkpoint: {current_checkpoint}")
    logger.info(f"{'='*60}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Progressive overfit: Teacher-guided style transfer via OmniTransfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with T2 image pair
    python scripts/progressive_overfit.py \\
        --pair-id t2_isometric \\
        --input-image /home/johndpope/Desktop/t2.webp \\
        --target-image /home/johndpope/Desktop/t2-isometric-3d-view-photorealistic.jpg \\
        --prompt "isometric 3D view of this scene, photorealistic miniature diorama"

    # With custom convergence settings
    python scripts/progressive_overfit.py \\
        --pair-id my_style \\
        --input-image input.jpg --target-image target.jpg \\
        --prompt "apply artistic watercolor style" \\
        --max-steps 1000 --convergence-loss 0.01

    # Resume from a previous checkpoint
    python scripts/progressive_overfit.py \\
        --pair-id t2_isometric \\
        --input-image input.jpg --target-image target.jpg \\
        --prompt "style transfer" \\
        --checkpoint /path/to/lora_weights_step_00100.safetensors
        """,
    )

    # Required arguments
    parser.add_argument("--pair-id", type=str, required=True, help="Unique ID for this image pair")
    parser.add_argument("--input-image", type=Path, required=True, help="Style source image")
    parser.add_argument("--target-image", type=Path, required=True, help="Target (ground truth) image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing transformation")

    # Training parameters
    parser.add_argument("--max-steps", type=int, default=500, help="Max total training steps (default: 500)")
    parser.add_argument("--convergence-loss", type=float, default=0.005, help="Stop if loss < this (default: 0.005)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--style-loss-weight", type=float, default=0.5, help="VGG style loss weight (default: 0.5)")
    parser.add_argument("--lpips-weight", type=float, default=0.1, help="LPIPS loss weight (default: 0.1)")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (default: 64)")

    # Model paths
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL_PATH), help="LTX-2 checkpoint")
    parser.add_argument("--text-encoder-path", type=Path, default=Path(DEFAULT_TEXT_ENCODER_PATH), help="Gemma dir")
    parser.add_argument("--config-template", type=Path, default=DEFAULT_CONFIG_TEMPLATE, help="YAML template")

    # Image dimensions
    parser.add_argument("--target-height", type=int, default=448, help="Target height, div by 32 (default: 448)")
    parser.add_argument("--target-width", type=int, default=832, help="Target width, div by 32 (default: 832)")

    # Device configuration
    parser.add_argument("--encode-device", type=str, default="cuda:1", help="Device for VAE/text encoding (default: cuda:1 = RTX PRO 4000)")
    parser.add_argument("--train-device", type=str, default="cuda:0", help="Device for training (default: cuda:0 = RTX 5090)")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit text encoder loading")

    # Paths
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/progressive_overfit"), help="Working directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume from checkpoint")

    # Output
    parser.add_argument("--json-output", action="store_true", help="Print JSON result to stdout")

    args = parser.parse_args()

    # Validate
    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image not found: {args.input_image}")
    if not args.target_image.exists():
        raise FileNotFoundError(f"Target image not found: {args.target_image}")
    if args.target_height % 32 != 0 or args.target_width % 32 != 0:
        raise ValueError(f"Dimensions must be divisible by 32: {args.target_width}x{args.target_height}")

    result = progressive_overfit(
        pair_id=args.pair_id,
        input_image=args.input_image,
        target_image=args.target_image,
        prompt=args.prompt,
        max_steps=args.max_steps,
        convergence_loss=args.convergence_loss,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        config_template=args.config_template,
        work_dir=args.work_dir,
        checkpoint=args.checkpoint,
        target_h=args.target_height,
        target_w=args.target_width,
        learning_rate=args.learning_rate,
        style_loss_weight=args.style_loss_weight,
        lpips_weight=args.lpips_weight,
        lora_rank=args.lora_rank,
        encode_device=args.encode_device,
        train_device=args.train_device,
        load_in_8bit=not args.no_8bit,
    )

    if args.json_output:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
