#!/usr/bin/env python3
"""Diagnostic: Test single-frame denoising with the FULL 48-block model (no SCD).

This verifies that the base LTX-2 model + our Euler solver + sigma schedule
produces coherent output. If this works but SCD doesn't, the SCD split is the issue.
"""

import argparse
import time

import torch
from PIL import Image
from safetensors.torch import load_file

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors")
    parser.add_argument("--cached-embedding", required=True)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--quantization", default="int8-quanto", choices=["int8-quanto", "fp8-quanto", "none"])
    parser.add_argument("--output", default="/media/2TB/omnitransfer/inference/test_base_model.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Latent dimensions
    latent_h = args.height // 32
    latent_w = args.width // 32
    latent_channels = 128
    tokens_per_frame = latent_h * latent_w
    num_frames = 1  # Single frame test

    print(f"Resolution: {args.width}x{args.height} (latent {latent_w}x{latent_h})")
    print(f"Tokens per frame: {tokens_per_frame}")

    # 1. Load cached embedding
    print("\n[1/4] Loading cached embedding...")
    embed_data = torch.load(args.cached_embedding, weights_only=True)
    prompt_embeds = embed_data.get("video_prompt_embeds", embed_data.get("prompt_embeds"))
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    prompt_mask = embed_data.get("prompt_attention_mask")
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_mask is not None:
        prompt_mask = prompt_mask.to(device=device)
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask.unsqueeze(0)
    print(f"  Shape: {prompt_embeds.shape}")

    # Create null embeddings for CFG
    use_cfg = args.guidance_scale > 1.0
    null_embeds = torch.zeros_like(prompt_embeds) if use_cfg else None
    null_mask = torch.zeros_like(prompt_mask) if use_cfg and prompt_mask is not None else None

    # 2. Load transformer (full model, no SCD, no LoRA)
    print("\n[2/4] Loading transformer (full 48-block model, NO SCD, NO LoRA)...")
    from ltx_trainer.model_loader import load_transformer

    transformer = load_transformer(args.checkpoint)

    if args.quantization != "none":
        from ltx_trainer.quantization import quantize_model
        print(f"  Quantizing ({args.quantization})...")
        quantize_model(transformer, args.quantization)

    transformer = transformer.to(device)
    transformer.eval()

    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"  GPU memory: {mem_gb:.1f} GB")

    # 3. Setup patchifier, positions, sigma schedule
    print("\n[3/4] Setting up inference components...")
    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    patchifier = VideoLatentPatchifier(patch_size=1)
    scale_factors = SpatioTemporalScaleFactors.default()

    # Compute positions for 1 frame
    coords = patchifier.get_patch_grid_bounds(
        output_shape=VideoLatentShape(
            frames=num_frames, height=latent_h, width=latent_w,
            batch=1, channels=latent_channels,
        ),
        device=device,
    )
    positions = get_pixel_coords(latent_coords=coords, scale_factors=scale_factors, causal_fix=True).to(dtype)
    positions[:, 0, ...] = positions[:, 0, ...] / 24.0  # Normalize temporal by FPS

    # Sigma schedule (matching validation_sampler: use full-frame token count)
    dummy_latent = torch.empty(1, 1, num_frames, latent_h, latent_w)
    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=args.num_inference_steps, latent=dummy_latent).to(device=device, dtype=dtype)
    print(f"  Sigma schedule: [{sigmas[0]:.4f} → {sigmas[-2]:.4f} → {sigmas[-1]:.4f}]")
    print(f"  Token count for shift: {num_frames * latent_h * latent_w}")
    print(f"  CFG: {'enabled' if use_cfg else 'disabled'} (scale={args.guidance_scale})")

    # 4. Denoise single frame
    print("\n[4/4] Denoising single frame...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_t = torch.randn(1, latent_channels, num_frames, latent_h, latent_w,
                       device=device, dtype=dtype, generator=generator)

    t_start = time.time()
    for step in range(args.num_inference_steps):
        sigma = sigmas[step]
        sigma_next = sigmas[step + 1]

        # Patchify
        noisy_patch = patchifier.patchify(x_t)
        seq_len = noisy_patch.shape[1]

        # Build modality
        video_modality = Modality(
            enabled=True,
            latent=noisy_patch,
            timesteps=torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype),
            positions=positions,
            context=prompt_embeds,
            context_mask=prompt_mask,
        )

        # Forward pass — full 48-block model
        video_pred, _ = transformer(video=video_modality, audio=None, perturbations=None)

        velocity = video_pred  # Model predicts velocity in latent patch space

        # CFG
        if use_cfg:
            uncond_modality = Modality(
                enabled=True,
                latent=noisy_patch,
                timesteps=torch.full((1, seq_len), sigma.item(), device=device, dtype=dtype),
                positions=positions,
                context=null_embeds,
                context_mask=null_mask,
            )
            velocity_uncond, _ = transformer(video=uncond_modality, audio=None, perturbations=None)
            velocity = velocity_uncond + args.guidance_scale * (velocity - velocity_uncond)

        # Euler step (float32 accumulation)
        dt = sigma_next - sigma
        noisy_patch = (noisy_patch.float() + velocity.float() * dt.float()).to(dtype)

        # Debug stats
        if step in (0, 1, 5, 14, 28, 29):
            v_std = velocity.float().std().item()
            x_std = noisy_patch.float().std().item()
            print(f"  Step {step:2d}: sigma={sigma.item():.4f}→{sigma_next.item():.4f} "
                  f"v_std={v_std:.4f} x_std={x_std:.4f}")

        # Unpatchify
        x_t = patchifier.unpatchify(
            noisy_patch,
            output_shape=VideoLatentShape(
                frames=num_frames, height=latent_h, width=latent_w,
                batch=1, channels=latent_channels,
            ),
        )

    gen_time = time.time() - t_start
    print(f"\n  Denoised in {gen_time:.1f}s ({gen_time / args.num_inference_steps:.2f}s/step)")

    # 5. Decode with VAE
    print("\n  Decoding with VAE...")
    from ltx_trainer.model_loader import load_video_vae_decoder

    vae_decoder = load_video_vae_decoder(args.checkpoint)
    vae_decoder = vae_decoder.to("cuda:1").eval()

    with torch.inference_mode():
        x_t_dec = x_t.to("cuda:1")
        pixels = vae_decoder(x_t_dec)
        if hasattr(pixels, "sample"):
            pixels = pixels.sample
        pixels = pixels.float().clamp(0, 1)
        print(f"  Pixel shape: {pixels.shape}, range: [{pixels.min():.3f}, {pixels.max():.3f}]")

    # Save as image
    frame = pixels[0, :, 0].permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(frame)
    img.save(args.output)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
