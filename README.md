# OmniTransfer: Unified Spatio-Temporal Video Transfer

Implementation of **OmniTransfer** ([arXiv:2601.14250v1](https://arxiv.org/abs/2601.14250)) for LTX-2, enabling unified video-to-video transfer across 5 task types.

---

## What is OmniTransfer?

OmniTransfer is a unified framework for spatio-temporal video transfer that handles multiple tasks with a single model:

| Task Type | Description | Input | Output |
|-----------|-------------|-------|--------|
| **Effect** | Transfer visual effects (fire, smoke, particles) | Reference video + target image | Animated image with effect |
| **Motion** | Transfer movement patterns | Reference video + target image | Target animated with reference motion |
| **Camera** | Transfer camera movements | Reference video + target image | Target with camera motion applied |
| **ID** | Preserve identity across scenes | Reference video + text prompt | New video preserving identity |
| **Style** | Apply artistic styles | Reference video + text prompt | Stylized video |

### VAE Sanity Check: All 5 Task Modes

Generate verification images locally to confirm VAE decoding works for each task:

```bash
python scripts/sanity_check_vae_modes.py \
    --data-root /path/to/processed \
    --model-path /path/to/ltx-2.safetensors \
    --output-dir ./outputs/vae_sanity_check
```

This creates comparison grids showing Reference (top) → Target (bottom) for all 5 tasks.

---

## Key Components (Paper Section 4)

### 1. Task-aware Positional Bias (TPB) - Section 4.2

```
"We add an offset Δ along the spatial/temporal dimension to distinguish
reference tokens from target tokens during attention computation."
```

TPB applies RoPE position offsets to separate reference and target in attention space:
- **Temporal tasks** (motion, camera, effect): Large temporal offset, small spatial offset
- **Appearance tasks** (id, style): Large spatial offset, small temporal offset

### 2. Reference-decoupled Causal Learning (RCL) - Section 4.3

```
"The reference branch adopts a fixed t=0, meaning it remains noise-free
throughout the diffusion process... loss is computed only on target tokens."
```

RCL enables efficient training by:
- Keeping reference latents at `t=0` (noise-free)
- Only adding noise to target latents
- Computing loss only on target predictions

### 3. Task-adaptive Multimodal Alignment (TMA) - Section 4.4

Optional MLLM integration (MetaQuery) for semantic guidance. Disabled in Stage 1 training.

---

## Training Stages (Paper Section 5.1)

```
"The training process is divided into three sequential stages with distinct
optimization objectives."
```

| Stage | Steps | Components | Description |
|-------|-------|------------|-------------|
| **Stage 1** | 10,000 | TPB + RCL | Train DiT blocks with positional bias and causal learning |
| **Stage 2** | 2,000 | TMA only | Freeze DiT, train TMA connector |
| **Stage 3** | 5,000 | All | Joint fine-tuning of all components |

---

## Core Classes

### Strategy & Configuration

| Class | File | Description |
|-------|------|-------------|
| [`OmniTransferStrategy`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/strategy.py#L585) | strategy.py | Main training strategy orchestrating all components |
| [`OmniTransferConfig`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/strategy.py#L146) | strategy.py | Pydantic configuration for all training options |
| [`OmniTransferModelInputs`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/strategy.py#L552) | strategy.py | Dataclass for model inputs with metadata |
| [`OmniTransferStage`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/strategy.py#L96) | strategy.py | Enum for training stages (IN_CONTEXT, CONNECTOR, JOINT) |

### Core Components (Paper Section 4)

| Class | File | Description |
|-------|------|-------------|
| [`TaskAwarePositionalBias`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/components.py#L115) | components.py | **TPB** - RoPE offsets for ref/target separation |
| [`ReferenceDecoupledCausalLearning`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/components.py#L522) | components.py | **RCL** - Separate attention branches |
| [`TaskAdaptiveMultimodalAlignment`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/components.py#L740) | components.py | **TMA** - MLLM semantic guidance |
| [`MetaQueryBank`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/components.py#L679) | components.py | Learnable query tokens for TMA |
| [`OmniTransferTask`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/components.py#L27) | components.py | Enum for task types (MOTION, STYLE, ID, etc.) |

### Latent Construction

| Class | File | Description |
|-------|------|-------------|
| [`ReferenceLatentConstructor`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/latent_constructor.py#L58) | latent_constructor.py | Constructs ref+target latent pairs |
| [`ConstructedLatents`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/latent_constructor.py#L24) | latent_constructor.py | Dataclass holding constructed latents |

### MLLM Integration (Optional)

| Class | File | Description |
|-------|------|-------------|
| [`MetaQueryTMA`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/metaquery_tma.py#L269) | metaquery_tma.py | MetaQuery MLLM integration |
| [`QwenVLTMAIntegration`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/qwen_vl_integration.py#L370) | qwen_vl_integration.py | Qwen2.5-VL integration for TMA |
| [`QwenVLFeatureExtractor`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/qwen_vl_integration.py#L122) | qwen_vl_integration.py | Extract features from Qwen-VL |

### Multi-Concept (Movie Weaver)

| Class | File | Description |
|-------|------|-------------|
| [`ConceptEmbedding`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/components.py#L335) | components.py | Dynamic Identity Anchoring embeddings |
| [`ConceptEmbeddingConfig`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/components.py#L317) | components.py | Config for concept embeddings |

### Visualization & Callbacks

| Class | File | Description |
|-------|------|-------------|
| [`OmniTransferVisualizer`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/visualization.py#L97) | visualization.py | Create reconstruction grids |
| [`OmniTransferWandBCallback`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/visualization.py#L486) | visualization.py | W&B logging integration |
| [`OmniTransferTrainingCallback`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/training_callback.py#L46) | training_callback.py | Training loop hooks |
| [`ReconstructionSample`](packages/ltx-trainer/src/ltx_trainer/omnitransfer/visualization.py#L37) | visualization.py | Dataclass for viz samples |

### Module Structure

```
packages/ltx-trainer/src/ltx_trainer/omnitransfer/
├── __init__.py              # Module exports
├── components.py            # TPB, RCL, TMA, ConceptEmbedding
├── strategy.py              # OmniTransferStrategy, Config
├── latent_constructor.py    # Reference latent construction
├── visualization.py         # W&B visualization
├── training_callback.py     # Training hooks
├── metaquery_tma.py         # MetaQuery MLLM integration
└── qwen_vl_integration.py   # Qwen-VL integration
```

---

## Loss Functions

The implementation includes multiple loss components based on Grok recommendations for faster convergence:

### Core: Flow Matching MSE Loss
```python
# Velocity prediction: v = noise - clean
mse_loss = (target_pred - (noise - target_latents)).pow(2)
```

### Min-SNR Gamma Weighting (Commit `561d666`)
Improves gradient flow at low timesteps by clipping signal-to-noise ratio:
```python
snr = ((1 - sigma) / sigma).pow(2)
snr_weight = min(SNR, gamma) / SNR  # gamma=5.0 default
loss = mse_loss * snr_weight
```

### LPIPS Perceptual Loss (Commit `e0e4bbb`)
**Critical insight**: VGG expects RGB images, not latent vectors!
```python
# WRONG: Computing LPIPS on latents (mathematically meaningless)
# RIGHT: Decode latents to pixels first
pred_pixels = vae_decoder(predicted_latents)  # [B, 3, H, W]
target_pixels = vae_decoder(target_latents)
lpips_loss = lpips_model(pred_pixels, target_pixels)
```

### Gram Matrix Style Loss (Commit `47a0fdc`)
For style transfer tasks, compares feature correlations:
```python
# Extract multi-layer VGG features
features = vgg19.features(decoded_pixels)  # relu1_2, relu2_2, relu3_3, relu4_3

# Gram matrix captures style (texture correlations)
def gram_matrix(features):
    b, c, h, w = features.shape
    F = features.view(b, c, h * w)
    return torch.bmm(F, F.transpose(1, 2)) / (c * h * w)

style_loss = MSE(gram_matrix(pred_features), gram_matrix(ref_features))
```

### Identity Loss with CLIP/SigLIP (Commit `e0e4bbb`)
For identity preservation, uses semantic features:
```python
# SigLIP recommended for Qwen2.5-VL compatibility
clip_features_pred = siglip_model.encode_image(pred_pixels)
clip_features_ref = siglip_model.encode_image(ref_pixels)
identity_loss = 1 - cosine_similarity(clip_features_pred, clip_features_ref)
```

---

## Git Commit History

Key commits implementing OmniTransfer:

| Commit | Description |
|--------|-------------|
| `e09e3b0` | Initial OmniTransfer implementation (TPB, RCL, latent constructor) |
| `561d666` | Add min-SNR gamma, LPIPS, identity loss |
| `31021fc` | Add MetaQuery MLLM integration for TMA |
| `ae75705` | Add multi-task training (unified 5-task mode) |
| `47a0fdc` | Add Gram matrix style loss for style transfer |
| `e0e4bbb` | Pixel-space losses (Grok recommendation: decode before LPIPS/style) |
| `f5b3972` | Memory-efficient workflows for RTX 5090 (32GB VRAM) |

---

## Quick Start

### 1. Prepare Dataset

```bash
# Download demo data from OmniTransfer website
python scripts/download_omnitransfer_demos.py \
    --output-dir /path/to/raw_data

# Encode to latents (VAE only, ~8GB VRAM)
python scripts/encode_website_demos.py \
    --input-dir /path/to/raw_data \
    --output-dir /path/to/processed \
    --skip-text-encoding

# Compute text embeddings separately (~28GB VRAM)
python scripts/compute_text_embeddings.py \
    --output-dir /path/to/processed \
    --model-path /path/to/ltx-2.safetensors \
    --text-encoder-path /path/to/gemma
```

### 2. Train Stage 1 (Local GPU)

```bash
# RTX 5090 / RTX 4090 (24-32GB VRAM)
uv run python scripts/train.py configs/ltx2_omnitransfer_unified_5task.yaml
```

### 3. Sanity Check VAE

```bash
# Verify VAE decoding works for all task modes
python scripts/sanity_check_vae_modes.py \
    --data-root /path/to/processed \
    --model-path /path/to/ltx-2.safetensors \
    --output-dir ./outputs/vae_sanity_check
```

---

## Cloud Training (Vast.ai)

For faster training on A100 80GB GPUs, use the Terraform setup:

### Prerequisites

```bash
# Install Vast.ai CLI
pip install vastai
vastai set api-key YOUR_API_KEY

# Install Terraform
brew install terraform  # macOS
# or: sudo apt-get install terraform  # Linux

# AWS CLI for S3
pip install awscli
aws configure
```

### Deploy Training Instance

```bash
cd tools/vast-cloud-training

# Create terraform.tfvars with your credentials
cat > terraform.tfvars << 'EOF'
vast_api_key         = "your-vast-api-key"
aws_access_key_id    = "your-aws-access-key"
aws_secret_access_key = "your-aws-secret-key"
wandb_api_key        = "your-wandb-key"
s3_bucket            = "your-bucket-name"
wandb_project        = "omnitransfer-unified"
EOF

# Upload training data to S3 first
aws s3 sync /path/to/processed s3://your-bucket/processed/omnitransfer_unified_5task/

# Deploy instance
terraform init
terraform apply
```

### On the Vast.ai Instance

```bash
# SSH into instance
vastai ssh-url <instance_id>

# Run training script
cd /workspace/ltx2-omnitransfer
bash tools/vast-cloud-training/scripts/train_omnitransfer.sh
```

### Training Script Features

The cloud training script (`train_omnitransfer.sh`) includes:

- **Auto-shutdown**: Configurable max runtime (default 24h)
- **Checkpoint sync**: Uploads to S3 every 30 minutes
- **Resume support**: Automatically resumes from latest checkpoint
- **tmux session**: Training runs in detachable session

### Cost Estimates

| GPU | $/hr | Time for 10k steps | Total Cost |
|-----|------|-------------------|------------|
| A100 80GB | ~$1.50-2.50 | ~8-12 hours | ~$15-30 |
| H100 80GB | ~$2.50-4.00 | ~4-6 hours | ~$15-25 |

---

## Configuration Reference

### Key Config Options

```yaml
training_strategy:
  name: omnitransfer

  # Multi-task unified training
  multi_task_mode: true
  task_types: [effect, motion, camera, id, style]
  task_sampling: uniform  # or: weighted, round_robin

  # I2V mode for temporal tasks
  i2v_mode: true
  first_frame_latents_dir: target_image_latents
  reference_latents_dir: reference_latents

  # Stage 1 components
  enable_tpb: true   # Task-aware Positional Bias
  enable_rcl: true   # Reference-decoupled Causal Learning
  enable_tma: false  # Disabled in Stage 1

  # Loss configuration
  target_loss_weight: 1.0
  min_snr_gamma: 5.0
  lpips_weight: 0.0      # Enable: 0.1 (requires VAE decoder)
  style_loss_weight: 0.0  # Enable: 0.5 for style transfer

  # Grok-recommended pixel-space losses
  use_decoded_pixels_for_lpips: true
  use_decoded_pixels_for_style: true
  use_vgg_style_features: true
  vgg_style_layers: ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
```

### VRAM Requirements

| Config | VRAM | Notes |
|--------|------|-------|
| Full training | 80GB+ | A100/H100 |
| LoRA + grad checkpoint | 48GB+ | A6000 |
| LoRA + INT8 quant | 24-32GB | RTX 4090/5090 |

---

## W&B Visualization

Training logs to Weights & Biases with:

- **Loss curves**: MSE, LPIPS, style, identity losses
- **Reconstruction grids**: Reference | Target | Prediction
- **Multi-task comparison**: All 5 tasks side-by-side
- **Video comparisons**: Animated at configurable intervals

Enable in config:
```yaml
training_strategy:
  log_reconstructions: true
  reconstruction_log_interval: 500
  log_multi_task_comparison: true
  multi_task_log_interval: 500
  log_video_comparisons: true
  video_log_interval: 2000

wandb:
  enabled: true
  project: omnitransfer-unified
  tags: ["stage1", "unified", "5-task"]
```

---

## Dataset Structure

```
/path/to/processed/
├── latents/                    # Target video latents [128, F, H, W]
│   ├── 000.pt
│   ├── 001.pt
│   └── ...
├── conditions/                 # Text embeddings (precomputed)
│   ├── 000.pt                  # {prompt_embeds, prompt_attention_mask}
│   └── ...
├── reference_latents/          # Reference video latents
│   └── ...
├── target_image_latents/       # First frame for I2V mode
│   └── ...
└── metadata.json               # Task types per sample
```

### Metadata Format

```json
{
  "pairs": [
    {"id": 0, "task_type": "effect", "prompt": "A person with fire effects"},
    {"id": 1, "task_type": "motion", "prompt": "A person dancing"},
    ...
  ]
}
```

---

## Latent Shapes

Understanding the dimensional transformations:

```
Raw Video:      [B, 3, 65, 448, 832]     # 65 frames, 448x832 pixels, RGB
                        ↓ VAE Encode
Latent Space:   [B, 128, 9, 14, 26]      # 9 temporal, 14x26 spatial, 128 channels
                        ↓ Patchify
Sequence:       [B, 3276, 128]           # 9*14*26 = 3276 tokens
                        ↓ Transformer
Prediction:     [B, 3276, 128]           # Velocity prediction
                        ↓ Unpatchify
Latent:         [B, 128, 9, 14, 26]
                        ↓ VAE Decode
Output Video:   [B, 3, 65, 448, 832]
```

Compression ratios:
- **Temporal**: 65 frames → 9 latent frames (~7.2x)
- **Spatial**: 448×832 → 14×26 (~32x per dimension)
- **Channel**: 3 RGB → 128 latent channels

---

## Inference Speedup Stack

This project implements **10 complementary inference acceleration technologies** that stack multiplicatively. The unified inference script (`scripts/scd_inference.py`) supports all of them via CLI flags.

### Technology Overview

| # | Technology | Paper | Type | Speedup | Training Required? |
|---|-----------|-------|------|---------|-------------------|
| 1 | [SCD](#1-scd-separable-causal-diffusion) | [arXiv:2602.10095](https://arxiv.org/abs/2602.10095) | Architecture | ~3× | Yes (LoRA) |
| 2 | [DDiT](#2-ddit-dynamic-diffusion-transformer) | [arXiv:2602.16968](https://arxiv.org/abs/2602.16968) | Token reduction | 1.25-1.48× | Yes (adapter) |
| 3 | [Spectrum](#3-spectrum-chebyshev-velocity-forecasting) | [arXiv:2603.01623](https://arxiv.org/abs/2603.01623) | Step forecasting | 2.0× | No |
| 4 | [TeaCache](#4-teacache-temporal-adaptive-cache) | CVPR 2025 | Step caching | 1.5-2× | No |
| 5 | [BézierFlow](#5-bézierflow-learned-sigma-schedule) | [arXiv:2512.13255](https://arxiv.org/abs/2512.13255) | Learned schedule | 1.5-2× | Yes (~10 min) |
| 6 | [BSplineFlow](#6-bsplineflow-local-learned-schedule) | — | Learned schedule | 1.5-2× | Yes (~10 min) |
| 7 | [Distilled Model](#7-distilled-model-8-step) | — | Model distillation | 3.75× | Pre-trained |
| 8 | [Quantization](#8-quantization) | — | Weight compression | 1.1-1.5× | No |
| 9 | [Evolution](#9-evolution-gradient-free-lora-tuning) | — | AR quality tuning | Quality improvement | Yes |
| 10 | [Split-GPU](#10-split-gpu-inference) | — | Hardware parallelism | 1.0-1.48× | No |

### Stacking Compatibility

All technologies are orthogonal **except** TeaCache and Spectrum (mutually exclusive — both skip decoder steps). If both are enabled, Spectrum takes priority.

```
┌─────────────────────────────────────────────────────────────┐
│                    SCD (Architecture)                        │
│  Encoder (32 layers, 1× per frame) → Decoder (16 layers)   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Decoder Speedup Stack                   │    │
│  │                                                     │    │
│  │  Distilled (8 steps)     ← fewer denoising steps    │    │
│  │  + BézierFlow/BSpline    ← optimal sigma placement  │    │
│  │  + Spectrum OR TeaCache  ← skip redundant steps      │    │
│  │  + DDiT (4× fewer tokens)← spatial token merging     │    │
│  │  + Quantization (int8)   ← reduced memory bandwidth  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Benchmarks (RTX 5090, 768×448, 30s video, int8-quanto)

| Configuration | Steps | s/frame | Gen Time | Decoder | Decoder Speedup |
|---------------|-------|---------|----------|---------|-----------------|
| SCD baseline | 30 | 3.3 | 11.0 min | 237s | 1.0× |
| SCD + TeaCache (0.10) | 30 | 1.6 | 6.2 min | 118s | 2.0× |
| SCD + **Spectrum** | 30 | 1.7 | 5.2 min | 99s | 2.4× |
| SCD + DDiT 2× (dynamic) | 30 | 1.3 | 2.5 min | 93s | 2.5× |
| **Distilled** | 8 | 1.0 | 4.7 min | 78s | 3.0× |
| Distilled + TeaCache | 8 | 0.5 | 2.8 min | 39s | 6.1× |
| **Spectrum + DDiT** | 30 | 0.9 | 2.4 min | 26s | 9.2× |
| Distilled + DDiT + TeaCache | 8 | — | ~1.8 min | — | ~13× |

> **Key finding:** Spectrum (4th-order polynomial forecast) outperforms TeaCache (0th-order reuse) at the same skip rate, with 63% forecast rate vs 52% cache hit rate.

### 1. SCD (Separable Causal Diffusion)

Splits the 48-layer DiT into **encoder** (32 layers, causal, KV-cached) + **decoder** (16 layers, denoised per frame). The encoder runs **once per frame** at σ=0 (noise-free), accumulating KV-cache across the temporal sequence. The decoder iterates N denoising steps per frame.

```
Frame N: encoder(frame N-1, σ=0) → KV-cache → decoder(noisy_N, steps=30) → clean frame N
```

**Files:** `ltx-core/.../scd_model.py`, `ltx-trainer/scripts/scd_inference.py`

```bash
python scripts/scd_inference.py \
    --cached-embedding /path/to/conditions_final/000.pt \
    --num-seconds 30 --quantization int8-quanto \
    --decoder-combine token_concat \
    --output /media/2TB/omnitransfer/inference/scd_30s.mp4
```

### 2. DDiT (Dynamic Diffusion Transformer)

Merges 2×2 spatial patches → **4× fewer tokens** per decoder step. A lightweight 4.2M-param adapter learns to project between native (336 tokens) and merged (84 tokens) representations. The **dynamic scheduler** analyzes 3rd-order trajectory finite differences to pick the optimal scale per step.

**Files:** `ltx-core/.../ddit.py`, pre-trained adapter at `sparse-causal-diffusion/outputs/ddit_scd_v2/`

```bash
python scripts/scd_inference.py \
    --ddit-adapter /path/to/ddit_scd_adapter_final.safetensors \
    --ddit-scale 2 \
    # Dynamic scheduling is default; --ddit-fixed-schedule for old head/tail behavior
```

### 3. Spectrum (Chebyshev Velocity Forecasting)

Fits **Chebyshev T-polynomials** (degree 4) to the denoising velocity trajectory via ridge regression, then **forecasts** velocity at non-critical steps instead of running the decoder. Blends Chebyshev prediction with Newton forward-difference Taylor extrapolation (w=0.5). Adaptive scheduling grows the skip window over time (smooth later steps need fewer evaluations).

**Paper:** [arXiv:2603.01623](https://arxiv.org/abs/2603.01623) (CVPR 2026)
**Files:** `ltx-trainer/src/ltx_trainer/spectrum/forecaster.py`

```bash
python scripts/scd_inference.py \
    --spectrum \
    --spectrum-degree 4 --spectrum-warmup 5 \
    --spectrum-window 2.0 --spectrum-flex 0.75 \
    --spectrum-weight 0.5 --spectrum-lam 0.1
```

**Scheduling pattern (30 steps):** Steps 0-4 always computed (warmup), then adaptive — computed steps: [0,1,2,3,4,6,8,11,15,20,25], forecasted: 19/30 = **63% forecast rate**.

### 4. TeaCache (Temporal Adaptive Cache)

Tracks relative L1 distance between consecutive denoising states. When accumulated distance < threshold, **reuses the previous velocity** (0th-order). Simpler than Spectrum but lower quality at high skip rates.

```bash
python scripts/scd_inference.py \
    --teacache-thresh 0.10
```

### 5. BézierFlow (Learned Sigma Schedule)

Learns an **optimal sigma schedule** via monotonic Bézier curve with 32 control points (cumulative softmax). Trains in ~10 minutes by distilling the 30-step teacher trajectory into 4 or 8 optimal steps. The learned schedule front-loads structure steps and spaces detail steps optimally.

**Paper:** [arXiv:2512.13255](https://arxiv.org/abs/2512.13255) (ICLR 2026)
**Files:** `ltx-trainer/src/ltx_trainer/bezierflow/scheduler.py`, `scripts/train_bezierflow.py`

```bash
# Train (one-time, ~10 min)
python scripts/train_bezierflow.py --output /path/to/schedule.pt

# Inference with learned schedule
python scripts/scd_inference.py \
    --bezier-schedule /path/to/schedule.pt \
    --num-inference-steps 8
```

### 6. BSplineFlow (Local Learned Schedule)

Variant of BézierFlow using **cubic B-spline** basis (local support) instead of global Bernstein basis. Each control point affects only its 4 neighboring knot spans, enabling **independent tuning** of early-step (structure) vs late-step (detail) phases. Same 32-parameter footprint.

**Files:** `ltx-trainer/src/ltx_trainer/bsplineflow/scheduler.py`

### 7. Distilled Model (8-Step)

Pre-distilled checkpoint (`ltx-2-19b-distilled.safetensors`) with a fixed **non-uniform 8-step sigma schedule**:

```
σ = [1.0, 0.994, 0.988, 0.981, 0.975, 0.909, 0.725, 0.422, 0.0]
```

First 4 steps: tiny deltas (structure). Last 4 steps: large jumps (refinement). Matches 30-step teacher quality at 3.75× speed.

```bash
python scripts/scd_inference.py --distilled --num-inference-steps 8
```

### 8. Quantization

Runtime weight quantization via `optimum-quanto`. Reduces VRAM and (in memory-bound regimes) speeds up inference.

| Format | VRAM Saved | Best For |
|--------|-----------|----------|
| `int8-quanto` | ~50% | General use (most stable) |
| `fp8-quanto` | ~40% | Maximum throughput (JIT warmup ~20 min) |

```bash
python scripts/scd_inference.py --quantization int8-quanto
```

### 9. Evolution (Gradient-Free LoRA Tuning)

Evolutionary strategy (ES) optimization of the decoder LoRA weights for improved **autoregressive quality**. Uses antithetic perturbation pairs (+ε, -ε) with multi-metric fitness (flow matching MSE, latent reconstruction, temporal coherence, LPIPS, SSIM). Not a direct speedup, but enables fewer-step inference (4-8 steps) with acceptable quality.

**Files:** `ltx-trainer/src/ltx_trainer/evolution/`, `scripts/evolve_scd.py`

```bash
python scripts/evolve_scd.py \
    --lora-path /path/to/scd_lora.safetensors \
    --dataset /path/to/data --distilled --guidance-scale 4.0
```

### 10. Split-GPU Inference

Distributes encoder → GPU 0, decoder → GPU 1. Forces bf16 (no quantization). Makes the decoder **compute-bound** instead of memory-bound, which is where DDiT's 4× token reduction has real impact (1.48× vs 1.25× speedup).

```bash
python scripts/scd_inference.py --split-gpus  # encoder→cuda:0, decoder→cuda:1
```

> **Note:** Only beneficial with symmetric GPUs (e.g., 2× RTX 5090). Asymmetric setups (RTX 5090 + PRO 4000) are slower in absolute terms despite higher DDiT multipliers.

### Quick Reference: CLI Flags

```bash
python scripts/scd_inference.py \
    # Required
    --cached-embedding /path/to/conditions_final/000.pt \
    --output /path/to/output.mp4 \
    \
    # SCD config
    --encoder-layers 32 \
    --decoder-combine token_concat \
    \
    # Model selection
    --distilled \                          # 8-step distilled model
    --num-inference-steps 8 \              # Steps (30 default, 8 distilled)
    --bezier-schedule /path/to/sched.pt \  # Learned sigma schedule
    \
    # Step skipping (pick ONE)
    --spectrum \                           # Chebyshev forecasting (recommended)
    --teacache-thresh 0.10 \               # OR TeaCache L1 threshold
    \
    # Token reduction
    --ddit-adapter /path/to/adapter.safetensors \
    --ddit-scale 2 \
    \
    # Hardware
    --quantization int8-quanto \           # Weight quantization
    --split-gpus \                         # Dual-GPU mode (bf16 only)
    \
    # Output
    --num-seconds 30 \
    --height 448 --width 768
```

---

## Troubleshooting

### OOM during training
- Enable `quantization: int8-quanto`
- Enable `enable_gradient_checkpointing: true`
- Reduce `batch_size` to 1
- Disable pixel-space losses (`lpips_weight: 0.0`)

### OOM during text encoding
Never load text encoder and VAE simultaneously on 32GB GPUs. Use the staged pipeline:
1. `encode_website_demos.py --skip-text-encoding` (VAE only)
2. `compute_text_embeddings.py` (text encoder only)

### Model not learning
- Verify reference ≠ target (check sanity_check_vae_modes.py output)
- Ensure `min_snr_gamma: 5.0` is set
- Check W&B reconstructions for proper input/output pairs

### Style transfer not working
- Enable `style_loss_weight: 0.5`
- Set `use_decoded_pixels_for_style: true`
- Ensure VAE decoder is available

---

## References

**Training:**
- [OmniTransfer](https://arxiv.org/abs/2601.14250) - arXiv:2601.14250v1 — Unified spatio-temporal video transfer
- [LTX-2 Model](https://huggingface.co/Lightricks/LTX-Video-2B) - HuggingFace
- [Movie Weaver](https://arxiv.org/abs/2501.xxxxx) - CVPR 2025 (multi-concept)

**Inference Speedup:**
- [SCD](https://arxiv.org/abs/2602.10095) - arXiv:2602.10095 — Separable Causal Diffusion
- [DDiT](https://arxiv.org/abs/2602.16968) - arXiv:2602.16968 — Dynamic Diffusion Transformer
- [Spectrum](https://arxiv.org/abs/2603.01623) - arXiv:2603.01623 (CVPR 2026) — Chebyshev velocity forecasting
- [BézierFlow](https://arxiv.org/abs/2512.13255) - arXiv:2512.13255 (ICLR 2026) — Learned sigma schedules
- [TeaCache](https://arxiv.org/abs/2411.xxxxx) - CVPR 2025 — Temporal adaptive caching

---

## Support This Project

Training video models requires significant GPU compute. If you find this work useful, please consider donating [Vast.ai](https://vast.ai) credits to help continue development.

**Send Vast.ai credits to:** `jp@bellgeorge.com`

```bash
vastai transfer credit jp@bellgeorge.com <AMOUNT>
```

| Tier | Suggested Amount | What It Helps With |
|------|------------------|-------------------|
| **Buy Me a Coffee** | $5-10 | Quick experiments, bug fixes |
| **Mates Rates** | $25-50 | A few hours of A100 training |
| **Supporter** | $100-250 | Full training run (10k steps) |
| **Enterprise** | $500+ | Multi-stage training, new features |

Every contribution helps push this research forward. Thank you!

---

## See Also

- [packages/ltx-trainer/ltx-2.md](packages/ltx-trainer/ltx-2.md) - Original LTX-2 trainer documentation
- [docs/training-modes.md](docs/training-modes.md) - All training modes
- [docs/configuration-reference.md](docs/configuration-reference.md) - Full config options
- [CLAUDE.md](CLAUDE.md) - AI assistant guidelines
