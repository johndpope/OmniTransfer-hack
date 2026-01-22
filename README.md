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

- [OmniTransfer Paper](https://arxiv.org/abs/2601.14250) - arXiv:2601.14250v1
- [LTX-2 Model](https://huggingface.co/Lightricks/LTX-Video-2B) - HuggingFace
- [Movie Weaver](https://arxiv.org/abs/2501.xxxxx) - CVPR 2025 (multi-concept)

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
