#!/usr/bin/env python3
"""OmniTransfer Live — Watch Folder + Overfit + Inference.

Watches a folder for new MP4+TXT pairs, preprocesses them (VAE encode + text
encode), overfits the model on each sample until it can reproduce the target,
then runs inference to generate the output video.

Three sequential VRAM phases per sample (32GB constraint):
  Phase 1: VAE Encode  (~10GB) — video → latents, first frame → ref latent
  Phase 2: Text Encode (~28GB) — text → conditions_final embedding
  Phase 3: Train+Infer (~20GB) — overfit loop → inference → decode

Usage:
    cd ~/Documents/GitHub/ltx2-omnitransfer/ltx-trainer
    uv run python scripts/omnitransfer_live.py \
        --watch-dir /path/to/sync_folder \
        --checkpoint /media/2TB/omnitransfer/output/isometric_omnitransfer/checkpoints/lora_weights_step_02000.safetensors \
        --output-dir /media/2TB/omnitransfer/inference/live \
        --overfit-steps 200 \
        --overfit-lr 1e-4 \
        --loss-threshold 0.01
"""

import argparse
import gc
import time
from dataclasses import replace
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import load_file

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
TEXT_ENCODER_PATH = Path("/media/2TB/ltx-models/gemma")

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_WIDTH, DEFAULT_HEIGHT = 768, 1152  # Fallback if auto-detect fails
TARGET_FRAMES = 25               # frames%8==1 → 25 frames (~1s at 25fps)
FPS = 25.0
LORA_RANK = 64
LORA_ALPHA = 64
LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]

# ── Prompt templates (must match training caption format) ────────────────────
# Training data used: "{BASE}. {ACTION}. {SUFFIX}."
# The 4 base patterns from isometric_identity training metadata:
BASE_PROMPTS = [
    "Static camera, fixed isometric viewpoint",
    "Fixed isometric angle, no camera motion",
    "Isometric 3D view, camera stays completely still",
    "3D isometric scene with static camera",
]
DEFAULT_BASE_PROMPT = BASE_PROMPTS[0]
PROMPT_SUFFIX = "No camera movement."


def format_training_prompt(raw_prompt: str, base_prompt: str | None = None) -> str:
    """Format a raw prompt into the training caption style.

    Training captions follow: "{base}. {action}. {suffix}."
    e.g. "Static camera, fixed isometric viewpoint. Fidgeting nervously. No camera movement."

    If the raw prompt already looks like a full scene description (starts with "A 3D",
    "A photo", etc.), we still wrap it — the model was trained with the base prefix.

    Args:
        raw_prompt: The text from the .txt file (action prompt or scene description)
        base_prompt: Override base prompt. If None, uses DEFAULT_BASE_PROMPT.
    """
    base = base_prompt or DEFAULT_BASE_PROMPT
    action = raw_prompt.strip().rstrip(".-")  # Clean trailing punctuation
    return f"{base}. {action}. {PROMPT_SUFFIX}"


def snap_resolution(w: int, h: int, divisor: int = 32) -> tuple[int, int]:
    """Snap width and height to nearest values divisible by divisor."""
    return (round(w / divisor) * divisor, round(h / divisor) * divisor)


def get_video_resolution(video_path: Path) -> tuple[int, int, int, float]:
    """Probe video for resolution and frame count.

    Returns:
        (width, height, num_frames, fps)
    """
    import subprocess, json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(video_path)],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            w = int(stream["width"])
            h = int(stream["height"])
            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "24/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            # Frame count
            nb = stream.get("nb_frames", "0")
            try:
                nf = int(nb)
            except (ValueError, TypeError):
                nf = 0
            return w, h, nf, fps
    return DEFAULT_WIDTH, DEFAULT_HEIGHT, 0, 24.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OmniTransfer Live — watch + overfit + infer")
    p.add_argument("--watch-dir", type=Path, required=True, help="Folder to watch for MP4+TXT pairs")
    p.add_argument("--checkpoint", type=Path, required=True, help="Base LoRA checkpoint (.safetensors)")
    p.add_argument("--output-dir", type=Path, default=Path("/media/2TB/omnitransfer/inference/live"))
    p.add_argument("--overfit-steps", type=int, default=200, help="Max overfit steps per sample")
    p.add_argument("--overfit-lr", type=float, default=1e-4, help="Learning rate for overfit loop")
    p.add_argument("--loss-threshold", type=float, default=0.01, help="Stop overfit when loss drops below this")
    p.add_argument("--inference-steps", type=int, default=30, help="Denoising steps for inference")
    p.add_argument("--cfg", type=float, default=4.0, help="Classifier-free guidance scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-quantize", action="store_true", help="Skip int8 quantization")
    p.add_argument("--oneshot", action="store_true", help="Process one file and exit (no watch loop)")
    p.add_argument("--num-output-frames", type=int, default=TARGET_FRAMES,
                   help="Number of output video frames (must satisfy frames%%8==1)")
    p.add_argument("--width", type=int, default=None,
                   help="Force output width (must be ÷32). Default: auto-detect from source")
    p.add_argument("--height", type=int, default=None,
                   help="Force output height (must be ÷32). Default: auto-detect from source")
    p.add_argument("--base-prompt", type=str, default=None,
                   help="Base prompt prefix for training format. "
                        f"Default: '{DEFAULT_BASE_PROMPT}'. "
                        "Set to empty string '' to use raw prompts as-is.")
    p.add_argument("--device-transformer", type=str, default="cuda:0")
    p.add_argument("--device-vae", type=str, default="cuda:1")
    return p.parse_args()


# =============================================================================
# Phase 1: VAE Encode — video → target latent + first-frame reference latent
# =============================================================================

def preprocess_video(
    video_path: Path,
    device: torch.device,
    target_frames: int = TARGET_FRAMES,
    force_w: int | None = None,
    force_h: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Encode video → target latents + extract first-frame reference latent.

    Auto-detects source resolution and snaps to nearest LTX-2-valid (÷32),
    unless force_w/force_h are specified.

    Returns:
        (target_latent, ref_latent, pixel_w, pixel_h) — latents on CPU.
        target_latent: [1, 128, lat_F, lat_H, lat_W]
        ref_latent:    [1, 128, 1, lat_H, lat_W]
        pixel_w, pixel_h: The pixel resolution used for encoding
    """
    from ltx_trainer.model_loader import load_video_vae_encoder
    from ltx_trainer.video_utils import read_video

    # Determine target resolution
    src_w, src_h, src_nf, src_fps = get_video_resolution(video_path)
    if force_w and force_h:
        target_w, target_h = force_w, force_h
    else:
        target_w, target_h = snap_resolution(src_w, src_h, divisor=32)
        # Clamp to reasonable range for VRAM
        target_w = min(target_w, 800)
        target_h = min(target_h, 1200)
    print(f"[Phase 1] Source: {src_w}x{src_h} → encoding at {target_w}x{target_h}")

    print(f"   Loading VAE encoder on {device}...")
    vae_encoder = load_video_vae_encoder(MODEL_PATH).to(device)

    # Read video: [F, C, H, W] in [0, 1]
    video_tensor, fps = read_video(str(video_path))
    num_frames = video_tensor.shape[0]
    print(f"   Video: {video_path.name} — {num_frames} frames @ {fps:.1f}fps, "
          f"shape {video_tensor.shape}")

    # Trim/select frames to satisfy frames%8==1
    if num_frames > target_frames:
        video_tensor = video_tensor[:target_frames]
    elif num_frames < target_frames:
        # Pad by repeating last frame
        pad_count = target_frames - num_frames
        last_frame = video_tensor[-1:].expand(pad_count, -1, -1, -1)
        video_tensor = torch.cat([video_tensor, last_frame], dim=0)

    actual_frames = video_tensor.shape[0]
    assert actual_frames % 8 == 1, f"Frame count {actual_frames} doesn't satisfy frames%8==1"

    # Resize to target resolution
    video_resized = torch.nn.functional.interpolate(
        video_tensor,  # [F, C, H, W]
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    # Convert to [-1, 1] and add batch dim: [1, C, F, H, W]
    video_input = (video_resized * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
    video_input = video_input.to(device, dtype=torch.float32)

    print(f"   Encoding {actual_frames} frames at {target_w}x{target_h}...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        target_latent = vae_encoder(video_input)  # [1, 128, lat_F, lat_H, lat_W]

    print(f"   Target latent: {target_latent.shape}")

    # Extract first frame as reference: [1, 128, 1, lat_H, lat_W]
    ref_latent = target_latent[:, :, :1, :, :].clone()
    print(f"   Ref latent (first frame): {ref_latent.shape}")

    # Move to CPU to free GPU
    target_latent = target_latent.cpu()
    ref_latent = ref_latent.cpu()

    del vae_encoder, video_input, video_tensor, video_resized
    torch.cuda.empty_cache()
    gc.collect()
    print(f"   VAE encoder unloaded. GPU freed.")

    return target_latent, ref_latent, target_w, target_h


# =============================================================================
# Phase 2: Text Encode — text → conditions_final embedding
# =============================================================================

def preprocess_text(
    text_path: Path,
    device: torch.device,
    base_prompt: str | None = None,
) -> dict[str, torch.Tensor]:
    """Encode text → final embeddings matching training format.

    Args:
        text_path: Path to .txt file with raw prompt
        device: GPU device for text encoder
        base_prompt: If provided (non-empty), wraps raw prompt in training format:
                     "{base_prompt}. {raw_prompt}. No camera movement."
                     If None, uses DEFAULT_BASE_PROMPT. If empty string, uses raw prompt.

    Returns dict with:
        video_prompt_embeds: [1024, dim] on CPU
        prompt_attention_mask: [1024] on CPU
    """
    from ltx_trainer.model_loader import load_text_encoder

    raw_prompt = text_path.read_text().strip()
    # Format prompt to match training caption style
    if base_prompt == "":
        # Explicit empty string = use raw prompt as-is
        prompt = raw_prompt
    else:
        prompt = format_training_prompt(raw_prompt, base_prompt=base_prompt)
    print(f"[Phase 2] Loading text encoder (Gemma) on {device}...")
    print(f"   Raw:      '{raw_prompt[:70]}{'...' if len(raw_prompt) > 70 else ''}'")
    print(f"   Formatted: '{prompt[:70]}{'...' if len(prompt) > 70 else ''}'")

    text_encoder = load_text_encoder(
        checkpoint_path=MODEL_PATH,
        gemma_model_path=TEXT_ENCODER_PATH,
        device=device,
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )
    text_encoder.eval()

    # Full forward pass: Gemma + connectors → final embeddings
    with torch.inference_mode():
        video_embeds, audio_embeds, attention_mask = text_encoder(prompt)

    # video_embeds: [1, seq_len, dim], attention_mask: [1, seq_len]
    conditions = {
        "video_prompt_embeds": video_embeds[0].cpu().contiguous(),
        "prompt_attention_mask": attention_mask[0].cpu().contiguous(),
        "is_final_embedding": True,
    }

    print(f"   Embeddings: {conditions['video_prompt_embeds'].shape} "
          f"(dtype={conditions['video_prompt_embeds'].dtype})")

    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()
    print(f"   Text encoder unloaded. GPU freed.")

    return conditions


# =============================================================================
# Phase 3a: Overfit Training Loop
# =============================================================================

def get_video_positions(
    num_frames: int, height: int, width: int,
    batch_size: int, fps: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Compute video positions [B, 3, seq_len, 2] matching training code."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    latent_coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            frames=num_frames, height=height, width=width,
            batch=batch_size, channels=128,
        ),
        device=device,
    )
    pixel_coords = get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=SpatioTemporalScaleFactors.default(),
        causal_fix=True,
    ).to(dtype)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
    return pixel_coords


def load_transformer_with_lora(
    checkpoint_path: Path,
    device: torch.device,
    quantize: bool = True,
) -> tuple:
    """Load transformer + LoRA + strategy components from checkpoint.

    Returns:
        (transformer, concept_embedding, full_state_dict)
    """
    from ltx_trainer.model_loader import load_transformer
    from ltx_trainer.omnitransfer.components import (
        ConceptEmbedding,
        ConceptEmbeddingConfig,
    )
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, set_peft_model_state_dict

    dtype = torch.bfloat16

    print(f"[Phase 3] Loading transformer...")
    transformer = load_transformer(MODEL_PATH)

    if quantize:
        print(f"   Quantizing to int8-quanto...")
        from ltx_trainer.quantization import quantize_model
        quantize_model(transformer, precision="int8-quanto")

    transformer = transformer.to(device)

    # Apply LoRA
    print(f"   Applying LoRA (rank={LORA_RANK})...")
    lora_config = PeftLoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        init_lora_weights=True,
    )
    transformer = get_peft_model(transformer, lora_config)

    # Enable gradient checkpointing — CRITICAL for 32GB VRAM
    # Must call on base model (unwrapped from PEFT)
    base = transformer.get_base_model()
    base.set_gradient_checkpointing(True)
    print(f"   Gradient checkpointing: enabled")

    # Load checkpoint weights
    print(f"   Loading checkpoint: {checkpoint_path.name}")
    full_state_dict = load_file(str(checkpoint_path))

    # Split LoRA vs strategy params
    lora_dict = {k.replace("diffusion_model.", "", 1): v
                 for k, v in full_state_dict.items()
                 if not k.startswith("strategy.")}
    set_peft_model_state_dict(transformer.get_base_model(), lora_dict)

    # Load ConceptEmbedding
    concept_embedding = None
    ce_keys = {k[len("strategy.concept_embedding."):]: v
               for k, v in full_state_dict.items()
               if k.startswith("strategy.concept_embedding.")}
    if ce_keys:
        config = ConceptEmbeddingConfig(embedding_dim=128, task_specific=True)
        concept_embedding = ConceptEmbedding(config)
        concept_embedding.load_state_dict(ce_keys, strict=False)
        concept_embedding = concept_embedding.to(device=device, dtype=dtype)
        print(f"   ConceptEmbedding loaded ({len(ce_keys)} params)")

    print(f"   GPU: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")
    return transformer, concept_embedding, full_state_dict


def overfit_train(
    target_latent: torch.Tensor,
    ref_latent: torch.Tensor,
    conditions: dict[str, torch.Tensor],
    checkpoint_path: Path,
    device: torch.device,
    steps: int = 200,
    lr: float = 1e-4,
    threshold: float = 0.01,
    quantize: bool = True,
    seed: int = 42,
) -> tuple:
    """Continue training from checkpoint until loss < threshold.

    Returns:
        (transformer, concept_embedding) — still loaded for inference
    """
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.model.transformer.modality import Modality
    from ltx_trainer.omnitransfer.components import (
        OmniTransferTask,
        TaskAwarePositionalBias,
    )

    dtype = torch.bfloat16
    task = OmniTransferTask.IDENTITY_PRESERVATION

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load model
    transformer, concept_embedding, _ = load_transformer_with_lora(
        checkpoint_path, device, quantize=quantize,
    )

    # Move latents to device
    target_lat = target_latent.to(device, dtype)
    ref_lat = ref_latent.to(device, dtype)

    # Get dimensions from latents
    _, _, lat_f, lat_h, lat_w = target_lat.shape
    _, _, ref_f, ref_h, ref_w = ref_lat.shape

    # Patchify
    patchifier = VideoLatentPatchifier(patch_size=1)
    tpb = TaskAwarePositionalBias(dim=128)

    target_patched = patchifier.patchify(target_lat)  # [1, tgt_seq, 128]
    ref_patched_clean = patchifier.patchify(ref_lat)  # [1, ref_seq, 128]

    ref_seq_len = ref_patched_clean.shape[1]
    tgt_seq_len = target_patched.shape[1]

    # Compute positions
    ref_positions = get_video_positions(ref_f, ref_h, ref_w, 1, FPS, device, dtype)
    tgt_positions = get_video_positions(lat_f, lat_h, lat_w, 1, FPS, device, dtype)

    # Apply TPB to reference positions (appearance task → temporal offset)
    biased_ref_pos = tpb.apply_task_bias(ref_positions, task, lat_w, lat_f)

    # Concatenate positions
    combined_positions = torch.cat([biased_ref_pos, tgt_positions], dim=2)

    # Load text embeddings
    prompt_embeds = conditions["video_prompt_embeds"].unsqueeze(0).to(device, dtype)
    prompt_mask = conditions["prompt_attention_mask"].unsqueeze(0).to(device)

    # Setup optimizer — only train LoRA parameters
    # Use 8-bit AdamW to halve optimizer state memory (~1.7GB savings)
    transformer.train()
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=0.01)
        print(f"   Optimizer: AdamW8bit (memory-efficient)")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        print(f"   Optimizer: AdamW (standard)")

    if concept_embedding is not None:
        concept_embedding.train()
        ce_params = list(concept_embedding.parameters())
        optimizer.add_param_group({"params": ce_params, "lr": lr})
        total_trainable = sum(p.numel() for p in trainable_params) + sum(p.numel() for p in ce_params)
    else:
        total_trainable = sum(p.numel() for p in trainable_params)

    print(f"\n   Overfit training: {steps} steps, lr={lr}, threshold={threshold}")
    print(f"   Trainable params: {total_trainable:,}")
    print(f"   Target: {tgt_seq_len} tokens, Ref: {ref_seq_len} tokens")

    # Velocity target: v = noise - clean (flow matching convention)
    # We precompute the clean target patched for loss computation
    clean_target = target_patched.detach()

    best_loss = float("inf")
    for step in range(steps):
        optimizer.zero_grad()

        # Sample sigma from shifted logit-normal (matching training)
        u = torch.randn(1, device=device) * 1.0  # std=1.0
        sigma = torch.sigmoid(u)  # → (0, 1)
        sigma = sigma.view(1, 1, 1)  # [1, 1, 1] for broadcasting

        # Create noise and noisy target
        noise = torch.randn_like(clean_target)
        noisy_target = sigma * noise + (1.0 - sigma) * clean_target

        # Apply ConceptEmbedding to reference tokens
        ref_patched = ref_patched_clean.clone()
        if concept_embedding is not None:
            ref_patched = concept_embedding(ref_patched, concept_index=0, task=task)

        # Concatenate ref + noisy target
        combined_latent = torch.cat([ref_patched, noisy_target], dim=1)

        # RCL timesteps: ref=0, target=sigma
        ref_timesteps = torch.zeros(1, ref_seq_len, device=device, dtype=dtype)
        tgt_timesteps = torch.full((1, tgt_seq_len), sigma.item(), device=device, dtype=dtype)
        combined_timesteps = torch.cat([ref_timesteps, tgt_timesteps], dim=1)

        # Build Modality
        video = Modality(
            enabled=True,
            latent=combined_latent,
            timesteps=combined_timesteps,
            positions=combined_positions,
            context=prompt_embeds,
            context_mask=prompt_mask,
        )

        # Forward pass
        with torch.autocast("cuda", dtype=dtype):
            video_pred, _ = transformer(video=video, audio=None, perturbations=None)

        # Extract target prediction (skip ref tokens)
        target_pred = video_pred[:, ref_seq_len:, :]

        # Velocity target: v = noise - clean
        velocity_target = noise - clean_target

        # MSE loss on target tokens only
        loss = (target_pred.float() - velocity_target.float()).pow(2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        loss_val = loss.item()
        best_loss = min(best_loss, loss_val)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"   Step {step + 1:4d}/{steps} | loss={loss_val:.6f} | "
                  f"best={best_loss:.6f} | sigma={sigma.item():.3f}")

        if loss_val < threshold:
            print(f"   Converged at step {step + 1}! loss={loss_val:.6f} < {threshold}")
            break

    print(f"   Overfit complete. Best loss: {best_loss:.6f}")

    # Switch to eval mode for inference
    transformer.eval()
    if concept_embedding is not None:
        concept_embedding.eval()

    del optimizer, noise, noisy_target, velocity_target, video_pred, target_pred
    torch.cuda.empty_cache()

    return transformer, concept_embedding


# =============================================================================
# Phase 3b: Inference — generate video from reference + text
# =============================================================================

def run_inference(
    transformer: torch.nn.Module,
    concept_embedding,
    conditions: dict[str, torch.Tensor],
    ref_latent: torch.Tensor,
    output_path: Path,
    device_transformer: torch.device,
    device_vae: torch.device,
    pixel_w: int,
    pixel_h: int,
    num_frames: int = TARGET_FRAMES,
    steps: int = 30,
    cfg: float = 4.0,
    seed: int = 42,
) -> None:
    """Generate video from first-frame reference + text prompt."""
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import VideoLatentShape
    from ltx_trainer.model_loader import load_video_vae_decoder
    from ltx_trainer.omnitransfer.components import (
        OmniTransferTask,
        TaskAwarePositionalBias,
    )

    dtype = torch.bfloat16
    task = OmniTransferTask.IDENTITY_PRESERVATION

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    patchifier = VideoLatentPatchifier(patch_size=1)
    tpb = TaskAwarePositionalBias(dim=128)

    # Compute latent dimensions for output from pixel dimensions
    lat_f = (num_frames - 1) // 8 + 1  # e.g., 25 → 4
    lat_h = pixel_h // 32              # e.g., 704 → 22
    lat_w = pixel_w // 32              # e.g., 448 → 14

    ref_lat = ref_latent.to(device_transformer, dtype)
    _, _, ref_f, ref_h, ref_w = ref_lat.shape

    # Patchify reference
    ref_patched = patchifier.patchify(ref_lat)
    ref_seq_len = ref_patched.shape[1]

    # Apply ConceptEmbedding
    if concept_embedding is not None:
        with torch.inference_mode():
            ref_patched = concept_embedding(ref_patched, concept_index=0, task=task)

    # Create noisy target (pure noise)
    generator = torch.Generator(device=device_transformer).manual_seed(seed)
    tgt_noise = torch.randn(1, 128, lat_f, lat_h, lat_w,
                            device=device_transformer, dtype=dtype, generator=generator)
    tgt_patched = patchifier.patchify(tgt_noise)
    tgt_seq_len = tgt_patched.shape[1]

    # Compute positions
    ref_positions = get_video_positions(ref_f, ref_h, ref_w, 1, FPS, device_transformer, dtype)
    tgt_positions = get_video_positions(lat_f, lat_h, lat_w, 1, FPS, device_transformer, dtype)

    # Apply TPB to reference
    biased_ref_pos = tpb.apply_task_bias(ref_positions, task, lat_w, lat_f)

    # Load text embeddings
    prompt_embeds = conditions["video_prompt_embeds"].unsqueeze(0).to(device_transformer, dtype)
    prompt_mask = conditions["prompt_attention_mask"].unsqueeze(0).to(device_transformer)
    neg_embeds = torch.zeros_like(prompt_embeds)

    # Setup denoising
    print(f"\n   Inference: {steps} steps, CFG={cfg}, output={lat_f}×{lat_h}×{lat_w} latent "
          f"→ {num_frames}×{pixel_h}×{pixel_w} video")

    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=steps).to(device_transformer).float()
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(cfg)
    x0_model = X0Model(transformer)

    tgt_state = tgt_patched.clone()

    # Denoise masks for RCL: ref=0 (noise-free), target=1 (will be scaled by sigma)
    # Shape [B, T, 1] needed for broadcasting with [B, T, D=128] in X0Model
    ref_denoise_mask = torch.zeros(1, ref_seq_len, 1, device=device_transformer, dtype=torch.float32)
    tgt_denoise_mask = torch.ones(1, tgt_seq_len, 1, device=device_transformer, dtype=torch.float32)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for step_idx, sigma in enumerate(sigmas[:-1]):
            combined_latent = torch.cat([ref_patched, tgt_state], dim=1)

            # RCL timesteps: ref=0 * sigma = 0, target=1 * sigma = sigma
            # Shape [B, T, 1] for broadcasting with [B, T, D=128]
            combined_denoise = torch.cat([ref_denoise_mask, tgt_denoise_mask], dim=1)
            combined_timesteps = sigma * combined_denoise

            combined_positions = torch.cat([biased_ref_pos, tgt_positions], dim=2)

            video = Modality(
                enabled=True,
                latent=combined_latent,
                timesteps=combined_timesteps,
                positions=combined_positions,
                context=prompt_embeds,
                context_mask=prompt_mask,
            )

            pos_video, _ = x0_model(video=video, audio=None, perturbations=None)

            # CFG: negative pass with zero embeddings
            video_neg = replace(video, context=neg_embeds)
            neg_video, _ = x0_model(video=video_neg, audio=None, perturbations=None)

            # CFG combine: cond + (scale-1) * (cond - uncond)
            denoised = pos_video + cfg_guider.delta(pos_video, neg_video)
            denoised_tgt = denoised[:, ref_seq_len:]

            tgt_state = stepper.step(
                sample=tgt_state, denoised_sample=denoised_tgt,
                sigmas=sigmas, step_index=step_idx,
            )

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                print(f"   Step {step_idx + 1}/{steps} | sigma={sigma:.4f}")

    print("   Denoising complete!")

    # Unpatchify → [1, 128, F, H, W]
    tgt_decoded_latent = patchifier.unpatchify(
        tgt_state,
        output_shape=VideoLatentShape(
            frames=lat_f, height=lat_h, width=lat_w, batch=1, channels=128,
        ),
    )

    # Free transformer
    del transformer, x0_model
    torch.cuda.empty_cache()
    gc.collect()

    # Decode with VAE
    print(f"   Loading VAE decoder on {device_vae}...")
    vae_decoder = load_video_vae_decoder(MODEL_PATH).to(device_vae)
    tgt_decoded_latent = tgt_decoded_latent.to(device_vae, dtype)

    with torch.inference_mode():
        pixels = vae_decoder(tgt_decoded_latent)  # [1, 3, F, H, W]

    pixels = ((pixels + 1.0) / 2.0).clamp(0.0, 1.0)

    del vae_decoder, tgt_decoded_latent
    torch.cuda.empty_cache()
    gc.collect()

    # Save video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from ltx_trainer.video_utils import save_video
    # save_video expects [C, F, H, W]
    video_out = pixels[0].float().cpu()  # [3, F, H, W]
    save_video(video_out, output_path, fps=FPS)
    print(f"   Video saved: {output_path}")

    # Save comparison: first output frame as PNG
    out_first = video_out[:, 0]  # [3, H, W]
    save_comparison(out_first, output_path, pixel_w, pixel_h)


def save_comparison(
    out_frame: torch.Tensor,
    video_path: Path,
    pixel_w: int,
    pixel_h: int,
) -> None:
    """Save first output frame as PNG for quick visual check."""
    frame_path = video_path.parent / f"{video_path.stem}_frame0.png"
    out_img = T.ToPILImage()(out_frame.clamp(0, 1))
    out_img.save(frame_path)
    print(f"   First frame saved: {frame_path}")


# =============================================================================
# Main Watch Loop
# =============================================================================

def process_single_file(
    mp4_path: Path,
    txt_path: Path,
    args: argparse.Namespace,
) -> None:
    """Process a single MP4+TXT pair through all 3 phases."""
    device_transformer = torch.device(args.device_transformer)
    device_vae = torch.device(args.device_vae)

    print(f"\n{'='*70}")
    print(f"Processing: {mp4_path.name}")
    print(f"{'='*70}")

    start_time = time.time()

    # Phase 1: VAE Encode (auto-detects resolution, or uses --width/--height)
    target_latent, ref_latent, pixel_w, pixel_h = preprocess_video(
        mp4_path, device_vae,
        target_frames=args.num_output_frames,
        force_w=args.width,
        force_h=args.height,
    )
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 2: Text Encode
    conditions = preprocess_text(txt_path, device_vae, base_prompt=args.base_prompt)
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 3a: Overfit Training
    transformer, concept_embedding = overfit_train(
        target_latent=target_latent,
        ref_latent=ref_latent,
        conditions=conditions,
        checkpoint_path=args.checkpoint,
        device=device_transformer,
        steps=args.overfit_steps,
        lr=args.overfit_lr,
        threshold=args.loss_threshold,
        quantize=not args.no_quantize,
        seed=args.seed,
    )

    # Phase 3b: Inference
    output_path = args.output_dir / f"{mp4_path.stem}_out.mp4"
    run_inference(
        transformer=transformer,
        concept_embedding=concept_embedding,
        conditions=conditions,
        ref_latent=ref_latent,
        output_path=output_path,
        device_transformer=device_transformer,
        device_vae=device_vae,
        pixel_w=pixel_w,
        pixel_h=pixel_h,
        num_frames=args.num_output_frames,
        steps=args.inference_steps,
        cfg=args.cfg,
        seed=args.seed,
    )

    # Cleanup
    del transformer, concept_embedding, target_latent, ref_latent, conditions
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - start_time
    print(f"\n   Total time: {elapsed / 60:.1f} min")
    print(f"   Output: {output_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Validate checkpoint exists
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Validate frame count
    if args.num_output_frames % 8 != 1:
        valid = [f for f in range(1, 130) if f % 8 == 1]
        raise ValueError(
            f"--num-output-frames={args.num_output_frames} doesn't satisfy frames%8==1. "
            f"Valid values: {valid}"
        )

    print(f"OmniTransfer Live")
    print(f"  Watch dir:  {args.watch_dir}")
    print(f"  Checkpoint: {args.checkpoint.name}")
    print(f"  Output dir: {args.output_dir}")
    if args.width and args.height:
        print(f"  Resolution: {args.width}x{args.height} (forced)")
    else:
        print(f"  Resolution: auto-detect (snap to nearest ÷32)")
    print(f"  Frames:     {args.num_output_frames}")
    print(f"  Overfit:    {args.overfit_steps} steps, lr={args.overfit_lr}, threshold={args.loss_threshold}")
    print(f"  Inference:  {args.inference_steps} steps, CFG={args.cfg}")
    bp = args.base_prompt
    if bp == "":
        print(f"  Prompt:     raw (no base prefix)")
    elif bp is None:
        print(f"  Prompt:     '{DEFAULT_BASE_PROMPT}. <action>. {PROMPT_SUFFIX}'")
    else:
        print(f"  Prompt:     '{bp}. <action>. {PROMPT_SUFFIX}'")
    print()

    processed: set[str] = set()

    while True:
        # Find all MP4 files with matching TXT
        mp4_files = sorted(args.watch_dir.glob("*.mp4"))

        for mp4_path in mp4_files:
            if mp4_path.name in processed:
                continue
            # Prefer _combined.txt (Qwen VL scene + action), fall back to raw .txt
            combined_path = mp4_path.parent / f"{mp4_path.stem}_combined.txt"
            raw_path = mp4_path.with_suffix(".txt")
            if combined_path.exists():
                txt_path = combined_path
            elif raw_path.exists():
                txt_path = raw_path
            else:
                continue

            processed.add(mp4_path.name)

            try:
                process_single_file(mp4_path, txt_path, args)
            except Exception as e:
                print(f"\n   ERROR processing {mp4_path.name}: {e}")
                import traceback
                traceback.print_exc()
                # Ensure GPU is cleaned up on error
                torch.cuda.empty_cache()
                gc.collect()

        if args.oneshot:
            break

        time.sleep(5)


if __name__ == "__main__":
    main()
