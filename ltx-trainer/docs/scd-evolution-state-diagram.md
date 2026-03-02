# SCD + Evolution + PEFT Bypass — State Diagram

## Overview

Two-phase pipeline for long-form autoregressive video quality optimization.

## Phase 1: Gradient-Based SCD LoRA Training

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: GRADIENT TRAINING                          │
│                    (trainer.py + scd_strategy.py)                      │
│                                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐ │
│  │ Load     │    │ Quantize │    │ Apply    │    │ Wrap with        │ │
│  │ bf16     │───▶│ (quanto) │───▶│ PEFT     │───▶│ LTXSCDModel      │ │
│  │ weights  │    │ nn.Linear│    │ LoRA     │    │ (32 enc + 16 dec)│ │
│  │ (CPU)    │    │ → QLinear│    │ on QLinr │    │                  │ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘ │
│                                       ▲                    │           │
│                                       │                    ▼           │
│                               ┌───────┴───────┐  ┌─────────────────┐ │
│                               │ PEFT creates  │  │ Per-frame decoder│ │
│                               │ LoRA adapters │  │ training loop    │ │
│                               │ on QLinear    │  │ (teacher forcing)│ │
│                               │ modules       │  │                  │ │
│                               │               │  │ Loss → backprop  │ │
│                               │ ⚠️ This works │  │ through LoRA     │ │
│                               │ in the trainer│  │ params only      │ │
│                               │ because PEFT  │  └────────┬────────┘ │
│                               │ v0.14+ CAN    │           │           │
│                               │ wrap QLinear  │           ▼           │
│                               │ (fixed 2024)  │  ┌─────────────────┐ │
│                               └───────────────┘  │ Save LoRA       │ │
│                                                   │ checkpoint      │ │
│                                                   │ (.safetensors)  │ │
│                                                   └────────┬────────┘ │
└────────────────────────────────────────────────────────────┼──────────┘
                                                              │
                              LoRA checkpoint                 │
                              (PEFT format)                   │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: EVOLUTION (engine.py)                       │
│                    Gradient-free AR quality fine-tuning                 │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MODEL LOADING — PEFT BYPASS via ManualLoRA                     │   │
│  │                                                                  │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │   │
│  │  │ 1. Load  │    │ 2. Quant │    │ 3. Wrap  │    │ 4. Inject│  │   │
│  │  │ bf16     │───▶│ (quanto) │───▶│ SCD      │───▶│ Manual   │  │   │
│  │  │ weights  │    │ int8     │    │ Model    │    │ LoRA     │  │   │
│  │  │ (CPU)    │    │ QLinear  │    │          │    │ wrappers │  │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │   │
│  │                                                       │         │   │
│  │  WHY NOT PEFT?                                        ▼         │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │ Circular dependency:                                     │   │   │
│  │  │                                                          │   │   │
│  │  │  Approach A: Quantize first, then PEFT                   │   │   │
│  │  │    nn.Linear → QLinear → PEFT(QLinear) = ❌ CRASH        │   │   │
│  │  │    PEFT can't determine in/out features of QLinear       │   │   │
│  │  │                                                          │   │   │
│  │  │  Approach B: PEFT first, then quantize                   │   │   │
│  │  │    nn.Linear → LoraLayer{base_layer, lora_A, lora_B}    │   │   │
│  │  │    quanto quantize(block) → finds lora_A (nn.Linear)    │   │   │
│  │  │    → QUANTIZES LoRA PARAMS! = ❌ BROKEN                  │   │   │
│  │  │    (evolution needs bf16 LoRA params for perturbation)   │   │   │
│  │  │                                                          │   │   │
│  │  │  Solution: ManualLoRA (no PEFT)                          │   │   │
│  │  │    nn.Linear → QLinear (quanto, clean)                   │   │   │
│  │  │    QLinear → ManualLoRA{base=QLinear, lora_A, lora_B}    │   │   │
│  │  │    ManualLoRA.forward = base(x) + B(A(x)) * scaling     │   │   │
│  │  │    LoRA params stay bf16 ✅ Base stays quantized ✅       │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  EVOLUTION LOOP (ES gradient-free optimization)                 │   │
│  │                                                                  │   │
│  │  for generation in range(300):                                   │   │
│  │                                                                  │   │
│  │    ┌─────────────────────────────────────────────────────────┐   │   │
│  │    │  for seed in antithetic_pairs(population_size):         │   │   │
│  │    │                                                         │   │   │
│  │    │  ┌─────────┐   ┌──────────────────┐   ┌────────────┐  │   │   │
│  │    │  │ +ε pert │──▶│ AR Rollout       │──▶│ Fitness+   │  │   │   │
│  │    │  │ (seed,  │   │ (4 frames, each  │   │ (latent +  │  │   │   │
│  │    │  │  +1)    │   │  denoised 8 or   │   │  pixel     │  │   │   │
│  │    │  └─────────┘   │  15 steps)       │   │  metrics)  │  │   │   │
│  │    │       │        └──────────────────┘   └────────────┘  │   │   │
│  │    │       │                                      │         │   │   │
│  │    │       ▼                                      │         │   │   │
│  │    │  ┌─────────┐                                 │         │   │   │
│  │    │  │ Revert  │◀────────────────────────────────┘         │   │   │
│  │    │  │ to orig │                                           │   │   │
│  │    │  └─────────┘                                           │   │   │
│  │    │       │                                                │   │   │
│  │    │       ▼                                                │   │   │
│  │    │  ┌─────────┐   ┌──────────────────┐   ┌────────────┐ │   │   │
│  │    │  │ -ε pert │──▶│ AR Rollout       │──▶│ Fitness-   │ │   │   │
│  │    │  │ (seed,  │   │ (same samples)   │   │            │ │   │   │
│  │    │  │  -1)    │   │                  │   │            │ │   │   │
│  │    │  └─────────┘   └──────────────────┘   └────────────┘ │   │   │
│  │    │       │                                      │        │   │   │
│  │    │       ▼                                      │        │   │   │
│  │    │  ┌─────────┐                                 │        │   │   │
│  │    │  │ Revert  │◀────────────────────────────────┘        │   │   │
│  │    │  │ to orig │                                          │   │   │
│  │    │  └─────────┘                                          │   │   │
│  │    │       │                                               │   │   │
│  │    │       ▼                                               │   │   │
│  │    │  diff[seed] = Fitness+ - Fitness-                     │   │   │
│  │    └───────────────────────────────────────────────────────┘   │   │
│  │                                                                │   │
│  │    ┌───────────────────────────────────────────────────────┐   │   │
│  │    │  ES Gradient Update:                                  │   │   │
│  │    │  w += (lr / N) * Σ (diff[seed] * ε[seed]) / (2σ)    │   │   │
│  │    │                                                       │   │   │
│  │    │  Noise annealing: σ *= 0.998                          │   │   │
│  │    └───────────────────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  OUTPUT:                                                               │
│    evolved_lora.safetensors (PEFT-compatible format for inference)     │
└────────────────────────────────────────────────────────────────────────┘
```

## AR Rollout Detail (fitness.py)

```
Frame 0            Frame 1            Frame 2            Frame 3
┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│ Encoder  │       │ Encoder  │       │ Encoder  │       │ Encoder  │
│ (32 blk) │       │ (32 blk) │       │ (32 blk) │       │ (32 blk) │
│          │       │          │       │          │       │          │
│ input:   │       │ input:   │       │ input:   │       │ input:   │
│ zeros    │       │ gen[0]   │       │ gen[1]   │       │ gen[2]   │
│ (σ=0)    │       │ (σ=0)    │       │ (σ=0)    │       │ (σ=0)    │
└────┬─────┘       └────┬─────┘       └────┬─────┘       └────┬─────┘
     │ enc_feat[0]      │ enc_feat[1]      │ enc_feat[2]      │ enc_feat[3]
     │                  │                  │                  │
     │ shift-by-1       │                  │                  │
     ▼                  ▼                  ▼                  ▼
┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│ Decoder  │       │ Decoder  │       │ Decoder  │       │ Decoder  │
│ (16 blk) │       │ (16 blk) │       │ (16 blk) │       │ (16 blk) │
│          │       │          │       │          │       │          │
│ context: │       │ context: │       │ context: │       │ context: │
│ zeros    │       │enc[0]    │       │enc[1]    │       │enc[2]    │
│          │       │          │       │          │       │          │
│ denoise: │       │ denoise: │       │ denoise: │       │ denoise: │
│ 8 steps  │       │ 8 steps  │       │ 8 steps  │       │ 8 steps  │
│ (distil) │       │ (distil) │       │ (distil) │       │ (distil) │
└────┬─────┘       └────┬─────┘       └────┬─────┘       └────┬─────┘
     │ gen[0]           │ gen[1]           │ gen[2]           │ gen[3]
     │                  │                  │                  │
     ▼                  ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FITNESS EVALUATION                              │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │ FM Velocity │  │ Latent      │  │ Temporal    │  │ Pixel    │  │
│  │ MSE (0.35)  │  │ Recon MSE   │  │ Coherence  │  │ LPIPS    │  │
│  │             │  │ (0.25)      │  │ Gap (0.15) │  │ (0.20)   │  │
│  │ v_pred vs   │  │ gen[f] vs   │  │ cos_sim    │  │ VAE      │  │
│  │ v_true at   │  │ GT[f]       │  │ gen-gen vs │  │ decode   │  │
│  │ random σ    │  │             │  │ GT-GT gap  │  │ cuda:1   │  │
│  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  └────┬─────┘  │
│         └────────────────┴───────────────┴───────────────┘        │
│                              │                                     │
│                              ▼                                     │
│                  total = -Σ(wᵢ × metricᵢ) + w_ssim × ssim        │
│                  (negative because lower loss = higher fitness)    │
└─────────────────────────────────────────────────────────────────────┘
```

## Dual-GPU Memory Layout

```
┌─────────────────────────────────────┐  ┌──────────────────────────────┐
│         cuda:0 (RTX 5090 32GB)      │  │   cuda:1 (PRO 4000 24GB)    │
│                                     │  │                              │
│  ┌───────────────────────────────┐  │  │  ┌────────────────────────┐  │
│  │ LTX-2 Transformer (int8)     │  │  │  │ VAE Decoder (bf16)     │  │
│  │ ~14GB quantized              │  │  │  │ ~8GB                   │  │
│  │                              │  │  │  │                        │  │
│  │ ┌──────────────────────────┐ │  │  │  │ Decode latent→pixel    │  │
│  │ │ Encoder (blocks 0-31)   │ │  │  │  │ for LPIPS + SSIM       │  │
│  │ │ QLinear base weights    │ │  │  │  └────────────────────────┘  │
│  │ │ + ManualLoRA (bf16)     │ │  │  │                              │
│  │ │ (loaded but NOT evolved)│ │  │  │  ┌────────────────────────┐  │
│  │ └──────────────────────────┘ │  │  │  │ LPIPS (alex) ~0.2GB   │  │
│  │ ┌──────────────────────────┐ │  │  │  └────────────────────────┘  │
│  │ │ Decoder (blocks 32-47)  │ │  │  │                              │
│  │ │ QLinear base weights    │ │  │  │  Free: ~16GB                 │
│  │ │ + ManualLoRA (bf16)     │ │  │  │                              │
│  │ │ ★ EVOLVED by ES ★       │ │  │  └──────────────────────────────┘
│  │ └──────────────────────────┘ │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ Evolution overhead ~2GB      │  │
│  │ (perturbation, samples)      │  │
│  └───────────────────────────────┘  │
│                                     │
│  Free: ~16GB                        │
└─────────────────────────────────────┘
```

## ManualLoRA Module Structure

```
Before ManualLoRA injection:
  transformer_blocks[32].attn.to_q = QLinear(4096, 4096)  # int8 quantized

After ManualLoRA injection:
  transformer_blocks[32].attn.to_q = ManualLoRA(
      base = QLinear(4096, 4096),    # int8 quantized (frozen)
      lora_A = Parameter[32, 4096],  # bf16 (evolved)
      lora_B = Parameter[4096, 32],  # bf16 (evolved)
      scaling = 1.0                  # alpha/rank = 32/32
  )

  forward(x):
      return base(x) + linear(linear(x, lora_A), lora_B) * scaling
      │                 └─────── bf16 LoRA path ────────┘
      └─ int8 quantized path
```

## Key Insight: Why Evolution Needs Its Own LoRA Strategy

The SCD trainer (Phase 1) uses PEFT + quanto successfully because:
- Modern PEFT (v0.14+) learned to handle QLinear targets
- Gradients flow through the LoRA adapter weights normally
- The trainer controls the full init sequence internally

The evolution engine (Phase 2) **cannot use PEFT** because:
1. It needs inference-only mode (no grad graph overhead)
2. It perturbs parameters in-place with hash-based noise
3. PEFT's adapter management layer adds complexity with no benefit
4. The PEFT+quanto interaction has edge cases when quantizing inside wrappers

ManualLoRA gives us:
- Direct parameter access for perturbation (`.lora_A.data`, `.lora_B.data`)
- Clean separation: quantized base (frozen) + bf16 adapter (evolved)
- PEFT-compatible checkpoint format for inference compatibility
- Zero dependency on PEFT internals
