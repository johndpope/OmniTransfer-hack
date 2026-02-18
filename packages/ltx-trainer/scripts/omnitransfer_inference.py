#!/usr/bin/env python3
"""OmniTransfer Style Transfer Inference — single image output.

Generates an isometric 3D view from a movie scene using the trained LoRA
plus ConceptEmbedding and TMA strategy parameters.

Usage:
    cd ~/Documents/GitHub/ltx2-omnitransfer/packages/ltx-trainer
    uv run python scripts/omnitransfer_inference.py \
        --reference /media/2TB/movie_dioramas/blade_runner_rooftop/scene.jpg \
        --lora /media/2TB/training_output/isometric_phase3/checkpoints/lora_weights_step_01000.safetensors \
        --output /media/2TB/inference_output/blade_runner_isometric.png

The script mirrors the training pipeline:
  1. Encode reference image → latent (VAE on cuda:1)
  2. Load transformer + LoRA (int8-quanto on cuda:0)
  3. Load strategy params (ConceptEmbedding, TMA) from checkpoint
  4. Construct reference + noisy target latents
  5. Apply ConceptEmbedding to reference tokens
  6. Apply TMA context to prompt embeddings
  7. Apply TPB positional bias
  8. Run denoising loop (X0 prediction with CFG)
  9. Decode target latent → pixel image (VAE on cuda:1)
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
CACHED_EMBEDDINGS_DIR = Path("/media/2TB/diorama_training/conditions_final")
CACHED_QWEN_DIR = Path("/media/2TB/diorama_training/qwen_vl_features")

# ── Defaults ─────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 832, 448          # Must match training resolution
FPS = 25.0
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 4.0
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="OmniTransfer isometric inference")
    p.add_argument("--reference", type=Path, required=True, help="Reference scene image")
    p.add_argument("--output", type=Path, default=Path("/media/2TB/inference_output/result.png"))
    p.add_argument("--lora", type=Path, required=True, help="Path to LoRA checkpoint (.safetensors)")
    p.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    p.add_argument("--cfg", type=float, default=GUIDANCE_SCALE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-quantize", action="store_true", help="Skip int8 quantization")
    p.add_argument("--no-strategy", action="store_true",
                    help="Skip loading ConceptEmbedding/TMA (use LoRA only)")
    return p.parse_args()


def load_and_prepare_image(path: Path, target_w: int, target_h: int) -> torch.Tensor:
    """Load image, resize to target, return [1, 3, 1, H, W] in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)
    import torchvision.transforms as T
    tensor = T.ToTensor()(img)           # [3, H, W] in [0, 1]
    tensor = tensor * 2.0 - 1.0          # → [-1, 1]
    return tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]


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


def load_strategy_components(
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Load ConceptEmbedding and TMA from strategy.* keys in checkpoint.

    Returns:
        (concept_embedding, tma) — either can be None if not in checkpoint.
    """
    from ltx_trainer.omnitransfer.components import (
        ConceptEmbedding,
        ConceptEmbeddingConfig,
        TaskAdaptiveMultimodalAlignment,
    )

    concept_embedding = None
    tma = None

    # Check for ConceptEmbedding params
    ce_keys = {k[len("strategy.concept_embedding."):]: v
               for k, v in state_dict.items()
               if k.startswith("strategy.concept_embedding.")}
    if ce_keys:
        config = ConceptEmbeddingConfig(embedding_dim=128, task_specific=True)
        concept_embedding = ConceptEmbedding(config)
        concept_embedding.load_state_dict(ce_keys, strict=False)
        concept_embedding = concept_embedding.to(device=device, dtype=dtype).eval()
        print(f"   ✅ ConceptEmbedding loaded ({len(ce_keys)} params)")

    # Check for TMA params
    tma_keys = {k[len("strategy.tma."):]: v
                for k, v in state_dict.items()
                if k.startswith("strategy.tma.")}
    if tma_keys:
        # Match training config: mllm_hidden_dim=3584, output_dim=3840, 8 queries
        tma = TaskAdaptiveMultimodalAlignment(
            mllm_hidden_dim=3584,
            output_dim=3840,  # matches Gemma embedding dim
            num_connector_layers=3,
            num_queries_per_task=8,
        )
        tma.load_state_dict(tma_keys, strict=False)
        tma = tma.to(device=device, dtype=dtype).eval()
        print(f"   ✅ TMA loaded ({len(tma_keys)} params)")

    return concept_embedding, tma


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device_transformer = torch.device("cuda:0")  # RTX 5090
    device_vae = torch.device("cuda:1")           # RTX PRO 4000
    dtype = torch.bfloat16

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── Step 1: Encode reference image with VAE ──────────────────────────────
    print("📸 Loading VAE encoder and encoding reference image...")
    from ltx_trainer.model_loader import load_video_vae_encoder, load_video_vae_decoder

    vae_encoder = load_video_vae_encoder(MODEL_PATH).to(device_vae)
    ref_pixels = load_and_prepare_image(args.reference, WIDTH, HEIGHT).to(device_vae, torch.float32)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        ref_latent = vae_encoder(ref_pixels)  # [1, 128, 1, 14, 26]
    print(f"   Reference latent: {ref_latent.shape}")

    del vae_encoder
    torch.cuda.empty_cache()

    # ── Step 2: Load transformer + LoRA ──────────────────────────────────────
    print("🔧 Loading transformer (this takes ~2 min with quantization)...")
    from ltx_trainer.model_loader import load_transformer

    transformer = load_transformer(MODEL_PATH)  # loads to CPU

    if not args.no_quantize:
        print("   Quantizing to int8-quanto...")
        from ltx_trainer.quantization import quantize_model
        quantize_model(transformer, precision="int8-quanto")

    transformer = transformer.to(device_transformer)

    # Apply LoRA
    print(f"   Loading LoRA from {args.lora.name}...")
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, set_peft_model_state_dict

    lora_config = PeftLoraConfig(
        r=64, lora_alpha=64, lora_dropout=0.0,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights=True,
    )
    transformer = get_peft_model(transformer, lora_config)

    # Load full checkpoint (LoRA + strategy params)
    full_state_dict = load_file(str(args.lora))

    # Split: LoRA params vs strategy params
    lora_dict = {k.replace("diffusion_model.", "", 1): v
                 for k, v in full_state_dict.items()
                 if not k.startswith("strategy.")}
    set_peft_model_state_dict(transformer.get_base_model(), lora_dict)
    transformer.eval()
    print(f"   ✅ LoRA loaded. GPU: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

    # ── Step 2b: Load strategy components (ConceptEmbedding, TMA) ────────────
    concept_embedding = None
    tma = None
    has_strategy = any(k.startswith("strategy.") for k in full_state_dict)

    if has_strategy and not args.no_strategy:
        print("🧩 Loading strategy components from checkpoint...")
        concept_embedding, tma = load_strategy_components(
            full_state_dict, device_transformer, dtype
        )
    elif not has_strategy:
        print("ℹ️  No strategy params in checkpoint (old format) — using LoRA only")

    # ── Step 3: Load cached text embeddings ──────────────────────────────────
    print("📝 Loading cached text embeddings...")
    cond = torch.load(CACHED_EMBEDDINGS_DIR / "0.pt", map_location="cpu", weights_only=True)
    prompt_embeds = cond["video_prompt_embeds"].unsqueeze(0).to(device_transformer, dtype)
    prompt_mask = cond["prompt_attention_mask"].unsqueeze(0).to(device_transformer)

    # For CFG, create unconditional embeddings (zeros)
    neg_embeds = torch.zeros_like(prompt_embeds)

    # ── Step 3b: Apply TMA to prompt embeddings ─────────────────────────────
    if tma is not None and CACHED_QWEN_DIR.exists():
        print("🔗 Applying TMA (Qwen VL features → cross-attention context)...")
        # Load sample 0's Qwen features as proxy (all have same style task)
        qwen_data = torch.load(CACHED_QWEN_DIR / "0.pt", map_location="cpu", weights_only=True)
        qwen_features = qwen_data["qwen_features"].unsqueeze(0).to(device_transformer, dtype)

        from ltx_trainer.omnitransfer.components import OmniTransferTask
        task_idx = torch.tensor([OmniTransferTask.STYLE_TRANSFER.task_index],
                                device=device_transformer)

        with torch.inference_mode():
            tma_context = tma(qwen_features, task_idx)  # [1, 8, 3840]

        # Prepend TMA context to prompt embeddings (same as training)
        prompt_embeds = torch.cat([tma_context, prompt_embeds], dim=1)
        tma_mask = torch.ones(1, tma_context.shape[1],
                              dtype=prompt_mask.dtype, device=device_transformer)
        prompt_mask = torch.cat([tma_mask, prompt_mask], dim=1)

        # Also extend negative embeddings to match
        neg_embeds = torch.zeros_like(prompt_embeds)
        print(f"   TMA context: {tma_context.shape} prepended → prompt_embeds: {prompt_embeds.shape}")
    else:
        print(f"   Prompt embeds: {prompt_embeds.shape}")

    # ── Step 4: Construct latents ────────────────────────────────────────────
    print("🏗️  Constructing reference + target latents...")
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.transformer.modality import Modality
    from ltx_trainer.omnitransfer.components import OmniTransferTask, TaskAwarePositionalBias

    patchifier = VideoLatentPatchifier(patch_size=1)
    ref_latent = ref_latent.to(device_transformer, dtype)

    lat_h, lat_w, lat_f = 14, 26, 1  # 448/32, 832/32, 1 frame

    # Patchify reference: [1, 128, 1, 14, 26] → [1, 364, 128]
    ref_patched = patchifier.patchify(ref_latent)
    ref_seq_len = ref_patched.shape[1]

    # Apply ConceptEmbedding to reference tokens (identity anchoring)
    if concept_embedding is not None:
        with torch.inference_mode():
            ref_patched = concept_embedding(
                ref_patched, concept_index=0,
                task=OmniTransferTask.STYLE_TRANSFER,
            )
        print(f"   ✅ ConceptEmbedding applied to {ref_seq_len} ref tokens")

    # Create noisy target (pure noise)
    generator = torch.Generator(device=device_transformer).manual_seed(args.seed)
    tgt_noise = torch.randn(1, 128, lat_f, lat_h, lat_w,
                            device=device_transformer, dtype=dtype, generator=generator)
    tgt_patched = patchifier.patchify(tgt_noise)
    tgt_seq_len = tgt_patched.shape[1]

    # Compute positions
    ref_positions = get_video_positions(lat_f, lat_h, lat_w, 1, FPS, device_transformer, dtype)
    tgt_positions = get_video_positions(lat_f, lat_h, lat_w, 1, FPS, device_transformer, dtype)

    # Apply TPB: style_transfer = appearance task → offset along temporal dim
    tpb = TaskAwarePositionalBias(dim=128)
    biased_ref_pos = tpb.apply_task_bias(
        ref_positions, OmniTransferTask.STYLE_TRANSFER,
        target_width=lat_w, target_frames=lat_f,
    )

    # ── Step 5: Denoising loop ───────────────────────────────────────────────
    print(f"🔄 Running {args.steps}-step denoising (CFG={args.cfg})...")
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=args.steps).to(device_transformer).float()
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(args.cfg)
    x0_model = X0Model(transformer)

    ref_denoise_mask = torch.zeros(1, ref_seq_len, 1, device=device_transformer, dtype=torch.float32)
    tgt_denoise_mask = torch.ones(1, tgt_seq_len, 1, device=device_transformer, dtype=torch.float32)

    tgt_state = tgt_patched.clone()

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for step_idx, sigma in enumerate(sigmas[:-1]):
            combined_latent = torch.cat([ref_patched, tgt_state], dim=1)
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

            video_neg = replace(video, context=neg_embeds)
            neg_video, _ = x0_model(video=video_neg, audio=None, perturbations=None)

            denoised = pos_video + cfg_guider.delta(pos_video, neg_video)
            denoised_tgt = denoised[:, ref_seq_len:]

            tgt_state = stepper.step(
                sample=tgt_state, denoised_sample=denoised_tgt,
                sigmas=sigmas, step_index=step_idx,
            )

            if (step_idx + 1) % 5 == 0 or step_idx == 0:
                print(f"   Step {step_idx + 1}/{args.steps} | σ={sigma:.4f}")

    print("✅ Denoising complete!")

    # ── Step 6: Decode to pixels ─────────────────────────────────────────────
    print("🖼️  Decoding latent to image...")
    from ltx_core.types import VideoLatentShape

    tgt_decoded_latent = patchifier.unpatchify(
        tgt_state,
        output_shape=VideoLatentShape(
            frames=lat_f, height=lat_h, width=lat_w, batch=1, channels=128,
        ),
    )

    del transformer, x0_model
    torch.cuda.empty_cache()

    vae_decoder = load_video_vae_decoder(MODEL_PATH).to(device_vae)
    tgt_decoded_latent = tgt_decoded_latent.to(device_vae, dtype)

    with torch.inference_mode():
        pixels = vae_decoder(tgt_decoded_latent)  # [1, 3, 1, 448, 832]

    pixels = ((pixels + 1.0) / 2.0).clamp(0.0, 1.0)
    pixels = pixels[0, :, 0].float().cpu()  # [3, 448, 832]

    import torchvision.transforms as T
    img = T.ToPILImage()(pixels)
    img.save(args.output)
    print(f"💾 Saved to {args.output}")

    # Side-by-side comparison
    ref_img = Image.open(args.reference).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
    comparison = Image.new("RGB", (WIDTH * 2, HEIGHT))
    comparison.paste(ref_img, (0, 0))
    comparison.paste(img, (WIDTH, 0))
    comp_path = args.output.parent / f"{args.output.stem}_comparison{args.output.suffix}"
    comparison.save(comp_path)
    print(f"📊 Comparison saved to {comp_path}")


if __name__ == "__main__":
    main()
