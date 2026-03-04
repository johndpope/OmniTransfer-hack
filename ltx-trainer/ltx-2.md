# LTX-2 Trainer

This package provides tools and scripts for training and fine-tuning
Lightricks' **LTX-2** audio-video generation model. It enables LoRA training, full
fine-tuning, video-to-video transformations (IC-LoRA), and **OmniTransfer** unified
spatio-temporal video transfer on custom datasets.

---

## 📖 Documentation

All detailed guides and technical documentation are in the [docs](./docs/) directory:

- [⚡ Quick Start Guide](docs/quick-start.md)
- [🎬 Dataset Preparation](docs/dataset-preparation.md)
- [🛠️ Training Modes](docs/training-modes.md)
- [⚙️ Configuration Reference](docs/configuration-reference.md)
- [🚀 Training Guide](docs/training-guide.md)
- [🧪 Inference Guide](../ltx-pipelines/README.md)
- [🔧 Utility Scripts](docs/utility-scripts.md)
- [📚 LTX-Core Documentation](../ltx-core/README.md)
- [🛡️ Troubleshooting Guide](docs/troubleshooting.md)
- [🔄 OmniTransfer Guide](#-omnitransfer-unified-video-transfer)

---

## 🔧 Requirements

- **LTX-2 Model Checkpoint** - Local `.safetensors` file
- **Gemma Text Encoder** - Local Gemma model directory (required for LTX-2)
- **Linux with CUDA** - CUDA 13+ recommended for optimal performance
- **Nvidia GPU with 80GB+ VRAM** - Recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
  use the [low VRAM config](configs/ltx2_av_lora_low_vram.yaml) which enables INT8 quantization and other
  memory optimizations

---

## 🔄 OmniTransfer: Unified Video Transfer

OmniTransfer enables unified spatio-temporal video transfer based on [arXiv:2601.14250v1](https://arxiv.org/abs/2601.14250). Train models for:

- **Identity Preservation** - Maintain subject identity across scenes/motions
- **Style Transfer** - Apply artistic styles while preserving content
- **Motion Transfer** - Transfer motion patterns between videos
- **Pose Reenactment** - Drive target subjects with reference poses

### Key Components

| Component | Description |
|-----------|-------------|
| **TPB** (Task-aware Positional Bias) | RoPE offsets to distinguish reference vs target tokens |
| **RCL** (Reference-decoupled Causal Learning) | Separate attention branches for efficiency |
| **TMA** (Task-adaptive Multimodal Alignment) | MLLM with MetaQueries for semantic guidance |

### VRAM Requirements

| Configuration | VRAM | GPU Examples |
|--------------|------|--------------|
| Full precision | 80GB+ | A100, H100 |
| LoRA + gradient checkpointing | 48GB+ | A6000, RTX 6000 Ada |
| LoRA + INT8 quantization | 24GB+ | RTX 4090, RTX 3090 |
| LoRA + INT8 + batch=1 | 16GB+ | RTX 4080 (experimental) |

### Quick Start (24GB+ GPU)

```bash
# 1. Prepare dataset with reference/target video pairs
python scripts/prepare_omnitransfer_dataset.py \
    --input-dir /path/to/videos \
    --output-dir /path/to/processed \
    --task-type identity_preservation

# 2. Train Stage 1 (TPB + RCL, 10k steps)
uv run python scripts/train.py configs/ltx2_omnitransfer_lora.yaml

# 3. (Optional) Train Stage 2 - TMA connector (2k steps)
uv run python scripts/train.py configs/ltx2_omnitransfer_stage2.yaml

# 4. (Optional) Train Stage 3 - Joint fine-tuning (5k steps)
uv run python scripts/train.py configs/ltx2_omnitransfer_stage3.yaml
```

### Low VRAM Config (24GB)

Create or modify your config YAML:

```yaml
model:
  model_path: /path/to/ltx2_model.safetensors
  text_encoder_path: /path/to/gemma-3-12b-it
  training_mode: lora

lora:
  rank: 32          # Lower rank = less memory
  alpha: 32
  dropout: 0.0

training_strategy:
  name: omnitransfer
  task_type: identity_preservation
  enable_tpb: true
  enable_rcl: true
  enable_tma: false  # Disable TMA for Stage 1

optimization:
  learning_rate: 1.0e-5
  batch_size: 1
  gradient_accumulation_steps: 16  # Effective batch = 16
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: bf16
  load_text_encoder_in_8bit: true  # Critical for 24GB
```

### Generate Synthetic Training Data

Use the included scripts to generate identity-consistent video pairs:

```bash
# Preview prompts without generation
python scripts/test_msi_generation.py --dry-run --num-videos 5

# Generate with LTX-2 Text-to-Video
python scripts/test_msi_generation.py --backend ltx2 --num-videos 10

# Generate with LTX-2 Image-to-Video (requires reference image)
python scripts/test_msi_generation.py --backend ltx2_i2v \
    --reference-image /path/to/identity_ref.png \
    --num-videos 10
```

### W&B Visualization

Training automatically logs reconstruction comparisons to Weights & Biases:

- **Frame grids**: Reference → Target → Prediction side-by-side
- **Video comparisons**: Animated comparisons at configurable intervals
- **Metrics**: Loss, PSNR, learning rate, noise statistics

Enable in config:
```yaml
training_strategy:
  log_reconstructions: true
  reconstruction_log_interval: 500
  log_video_comparisons: true
  video_log_interval: 2000

wandb:
  enabled: true
  project: ltx2-omnitransfer
```

### Inference

```bash
python scripts/omnitransfer_inference.py \
    --checkpoint /path/to/checkpoint \
    --reference-video /path/to/reference.mp4 \
    --prompt "A person walking in a park" \
    --task-type identity_preservation \
    --output output.mp4
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Share Your Work**: If you've trained interesting LoRAs or achieved cool results, please share them with the
  community.
- **Report Issues**: Found a bug or have a suggestion? Open an issue on GitHub.
- **Submit PRs**: Help improve the codebase with bug fixes or general improvements.
- **Feature Requests**: Have ideas for new features? Let us know through GitHub issues.

---

## 💬 Join the Community

Have questions, want to share your results, or need real-time help?

Join our [community Discord server](https://discord.gg/ltxplatform) to connect with other users and the development
team!

- Get troubleshooting help
- Share your training results and workflows
- Collaborate on new ideas and features
- Stay up to date with announcements and updates

We look forward to seeing you there!

---

Happy training! 🎉
