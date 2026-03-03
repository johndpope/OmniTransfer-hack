# AGENTS.md

This file provides guidance to AI coding assistants (Claude, Cursor, etc.) when working with code in this repository.

## ⚠️ CRITICAL: Always Use Muon Optimizer

> **All training configs MUST use the Muon optimizer — never use AdamW or other optimizers.**
>
> Muon (Momentum + Newton-Schulz orthogonalization) normalizes weight updates to be spectrally
> orthogonal, giving ~1.3-2x faster convergence than AdamW on LoRA training. All LoRA params
> are 2D matrices, so pure Muon applies (no AdamW fallback needed for non-matrix params).
>
> **Required config:**
> ```yaml
> optimization:
>   optimizer_type: "muon"
>   learning_rate: 0.02      # Muon needs ~10-20x higher LR than AdamW
>   scheduler_type: "cosine"
>   scheduler_params:
>     eta_min: 1.0e-4
>   weight_decay: 0.01
> ```
>
> **Do NOT use** `optimizer_type: "adamw"` or `optimizer_type: "adam"` unless explicitly asked.
> When resuming from an AdamW checkpoint, switch to Muon — the LoRA weights transfer fine,
> only the optimizer state resets (which is expected on resume anyway).

---

## Project Overview

**LTX-2 Trainer** is a training toolkit for fine-tuning the Lightricks LTX-2 audio-video generation model. It supports:

- **LoRA training** - Efficient fine-tuning with adapters
- **Full fine-tuning** - Complete model training
- **Audio-video training** - Joint audio and video generation
- **IC-LoRA training** - In-context control adapters for video-to-video transformations
- **OmniTransfer training** - Unified spatio-temporal video transfer (identity preservation, style transfer, motion transfer)

**Key Dependencies:**

- **[`ltx-core`](../ltx-core/)** - Core model implementations (transformer, VAE, text encoder)
- **[`ltx-pipelines`](../ltx-pipelines/)** - Inference pipeline components

> **Important:** This trainer only supports **LTX-2** (the audio-video model). The older LTXV models are not supported.

## Architecture Overview

### Package Structure

```
packages/ltx-trainer/
├── src/ltx_trainer/           # Main training module
│   ├── config.py              # Pydantic configuration models
│   ├── trainer.py             # Main training orchestration with Accelerate
│   ├── model_loader.py        # Model loading using ltx-core
│   ├── validation_sampler.py  # Inference for validation samples
│   ├── datasets.py            # PrecomputedDataset for latent-based training
│   ├── training_strategies/   # Strategy pattern for different training modes
│   │   ├── __init__.py        # Factory function: get_training_strategy()
│   │   ├── base_strategy.py   # TrainingStrategy ABC, ModelInputs, TrainingStrategyConfigBase
│   │   ├── text_to_video.py   # TextToVideoStrategy, TextToVideoConfig
│   │   └── video_to_video.py  # VideoToVideoStrategy, VideoToVideoConfig
│   ├── omnitransfer/          # OmniTransfer implementation (arXiv:2601.14250v1)
│   │   ├── __init__.py        # Module exports
│   │   ├── components.py      # TPB, RCL, TMA core components
│   │   ├── latent_constructor.py  # Reference latent construction
│   │   ├── strategy.py        # OmniTransferStrategy, OmniTransferConfig
│   │   ├── visualization.py   # W&B reconstruction logging
│   │   └── training_callback.py   # Training callback for auto-logging
│   ├── timestep_samplers.py   # Flow matching timestep sampling
│   ├── captioning.py          # Video captioning utilities
│   ├── video_utils.py         # Video processing utilities
│   └── hf_hub_utils.py        # HuggingFace Hub integration
├── scripts/                   # User-facing CLI tools
│   ├── train.py               # Main training script
│   ├── process_dataset.py     # Dataset preprocessing
│   ├── process_videos.py      # Video latent encoding
│   ├── process_captions.py    # Text embedding computation
│   ├── caption_videos.py      # Automatic video captioning
│   ├── decode_latents.py      # Latent decoding for debugging
│   ├── inference.py           # Inference with trained models
│   ├── compute_reference.py   # Generate IC-LoRA reference videos
│   ├── split_scenes.py        # Scene detection and splitting
│   ├── prepare_omnitransfer_dataset.py  # OmniTransfer dataset preparation
│   ├── generate_omnitransfer_dataset.py # Generate synthetic training data
│   ├── omnitransfer_inference.py        # OmniTransfer inference
│   └── test_msi_generation.py           # Test video generation via Temporal
├── configs/                   # Example training configurations
│   ├── ltx2_av_lora.yaml      # Audio-video LoRA training
│   ├── ltx2_v2v_ic_lora.yaml  # IC-LoRA video-to-video
│   └── accelerate/            # Accelerate configs for distributed training
└── docs/                      # Documentation
```

### Key Architectural Patterns

**Model Loading:**

- `ltx_trainer.model_loader` provides component loaders using `ltx-core`
- Individual loaders: `load_transformer()`, `load_video_vae_encoder()`, `load_video_vae_decoder()`, `load_text_encoder()`, etc.
- Combined loader: `load_model()` returns `LtxModelComponents` dataclass
- Uses `SingleGPUModelBuilder` from ltx-core internally

**Training Flow:**

1. Configuration loaded via Pydantic models in `config.py`
2. `Trainer` class orchestrates the training loop
3. Training strategies (`TextToVideoStrategy`, `VideoToVideoStrategy`) prepare inputs and compute loss
4. Accelerate handles distributed training and device placement
5. Data flows as precomputed latents through `PrecomputedDataset`

**Model Interface (Modality-based):**

```python
from ltx_core.model.transformer.modality import Modality

# Create modality objects for video and audio
video = Modality(
    enabled=True,
    latent=video_latents,      # [B, seq_len, 128]
    timesteps=video_timesteps,  # [B, seq_len] per-token
    positions=video_positions,  # [B, 3, seq_len, 2]
    context=video_embeds,
    context_mask=None,
)
audio = Modality(
    enabled=True,
    latent=audio_latents,
    timesteps=audio_timesteps,
    positions=audio_positions,  # [B, 1, seq_len, 2]
    context=audio_embeds,
    context_mask=None,
)

# Forward pass returns predictions for both modalities
video_pred, audio_pred = model(video=video, audio=audio, perturbations=None)
```

> **Note:** `Modality` is immutable (frozen dataclass). Use `dataclasses.replace()` to modify.

**Configuration System:**

- All config in `src/ltx_trainer/config.py`
- Main class: `LtxTrainerConfig`
- Training strategy configs: `TextToVideoConfig`, `VideoToVideoConfig`
- Uses Pydantic field validators and model validators
- Config files in `configs/` directory

## Development Commands

### Setup and Installation

```bash
# From the repository root
uv sync
cd packages/ltx-trainer
```

### Code Quality

```bash
# Run ruff linting and formatting
uv run ruff check .
uv run ruff format .

# Run pre-commit checks
uv run pre-commit run --all-files
```

### Running Tests

```bash
cd packages/ltx-trainer
uv run pytest
```

### Running Training

```bash
# Single GPU
uv run python scripts/train.py configs/ltx2_av_lora.yaml

# Multi-GPU with Accelerate
uv run accelerate launch scripts/train.py configs/ltx2_av_lora.yaml
```

## Code Standards

### Type Hints

- **Always use type hints** for all function arguments and return values
- Use Python 3.10+ syntax: `list[str]` not `List[str]`, `str | Path` not `Union[str, Path]`
- Use `pathlib.Path` for file operations

### Class Methods

- Mark methods as `@staticmethod` if they don't access instance or class state
- Use `@classmethod` for alternative constructors

### AI/ML Specific

- Use `@torch.inference_mode()` for inference (prefer over `@torch.no_grad()`)
- Use `accelerator.device` for distributed compatibility
- Support mixed precision (`bfloat16` via dtype parameters)
- Use gradient checkpointing for memory-intensive training

### Logging

- Use `from ltx_trainer import logger` for all messages
- Avoid print statements in production code

## Important Files & Modules

### Configuration (CRITICAL)

**`src/ltx_trainer/config.py`** - Master config definitions

Key classes:
- `LtxTrainerConfig` - Main configuration container
- `ModelConfig` - Model paths and training mode
- `TrainingStrategyConfig` - Union of `TextToVideoConfig` | `VideoToVideoConfig`
- `LoraConfig` - LoRA hyperparameters
- `OptimizationConfig` - Learning rate, batch size, etc.
- `ValidationConfig` - Validation settings
- `WandbConfig` - W&B logging settings

**⚠️ When modifying config.py:**
1. Update ALL config files in `configs/`
2. Update `docs/configuration-reference.md`
3. Test that all configs remain valid

### Training Core

**`src/ltx_trainer/trainer.py`** - Main training loop

- Implements distributed training with Accelerate
- Handles mixed precision, gradient accumulation, checkpointing
- Uses training strategies for mode-specific logic

**`src/ltx_trainer/training_strategies/`** - Strategy pattern

- `base_strategy.py`: `TrainingStrategy` ABC, `ModelInputs` dataclass
- `text_to_video.py`: Standard text-to-video (with optional audio)
- `video_to_video.py`: IC-LoRA video-to-video transformations
- `omnitransfer/strategy.py`: OmniTransfer unified spatio-temporal transfer

Key methods each strategy implements:
- `get_data_sources()` - Required data directories
- `prepare_training_inputs()` - Convert batch to `ModelInputs`
- `compute_loss()` - Calculate training loss
- `requires_audio` property - Whether audio components needed

**`src/ltx_trainer/model_loader.py`** - Model loading

Component loaders:
- `load_transformer()` → `LTXModel`
- `load_video_vae_encoder()` → `VideoVAEEncoder`
- `load_video_vae_decoder()` → `VideoVAEDecoder`
- `load_audio_vae_decoder()` → `AudioVAEDecoder`
- `load_vocoder()` → `Vocoder`
- `load_text_encoder()` → `AVGemmaTextEncoderModel`
- `load_model()` → `LtxModelComponents` (convenience wrapper)

**`src/ltx_trainer/validation_sampler.py`** - Inference for validation

Uses ltx-core components for denoising:
- `LTX2Scheduler` for sigma scheduling
- `EulerDiffusionStep` for diffusion steps
- `CFGGuider` for classifier-free guidance

### Data

**`src/ltx_trainer/datasets.py`** - Dataset handling

- `PrecomputedDataset` loads pre-computed VAE latents
- Supports video latents, audio latents, text embeddings, reference latents

## Common Development Tasks

### Adding a New Configuration Parameter

1. Add field to appropriate config class in `src/ltx_trainer/config.py`
2. Add validator if needed
3. Update ALL config files in `configs/`
4. Update `docs/configuration-reference.md`

### Implementing a New Training Strategy

1. Create new file in `src/ltx_trainer/training_strategies/`
2. Create config class inheriting `TrainingStrategyConfigBase`
3. Create strategy class inheriting `TrainingStrategy`
4. Implement: `get_data_sources()`, `prepare_training_inputs()`, `compute_loss()`
5. Add to `__init__.py`: import, add to `TrainingStrategyConfig` union, update factory
6. Add discriminator tag to config.py's `TrainingStrategyConfig`
7. Create example config file in `configs/`

### Working with Modalities

```python
from dataclasses import replace
from ltx_core.model.transformer.modality import Modality

# Create modality
video = Modality(
    enabled=True,
    latent=latents,
    timesteps=timesteps,
    positions=positions,
    context=context,
    context_mask=None,
)

# Update (immutable - must use replace)
video = replace(video, latent=new_latent, timesteps=new_timesteps)

# Disable a modality
audio = replace(audio, enabled=False)
```

## Debugging Tips

**Training Issues:**

- Check logs first (rich logger provides context)
- GPU memory: Look for OOM errors, enable `enable_gradient_checkpointing: true`
- Distributed training: Check `accelerator.state` and device placement

**Model Loading:**

- Ensure `model_path` points to a local `.safetensors` file
- Ensure `text_encoder_path` points to a Gemma model directory
- URLs are NOT supported for model paths

**Configuration:**

- Validation errors: Check validators in `config.py`
- Unknown fields: Config uses `extra="forbid"` - all fields must be defined
- Strategy validation: IC-LoRA requires `reference_videos` in validation config

## OmniTransfer Training

OmniTransfer implements unified spatio-temporal video transfer based on arXiv:2601.14250v1. It enables:
- **Identity preservation** - Maintain subject identity across different scenes/motions
- **Style transfer** - Apply artistic styles while preserving content
- **Motion transfer** - Transfer motion patterns between videos
- **Pose reenactment** - Drive target with reference poses

### CRITICAL: Training Data Requirements

> **⚠️ NEVER USE THE SAME VIDEO FOR BOTH REFERENCE AND TARGET!**
>
> This is the #1 mistake that will waste training time and produce useless results.
>
> **Correct Setup:**
> - **Reference video**: Source of motion/style/effect to TRANSFER FROM
> - **Target video**: Content to apply the transfer TO (MUST BE DIFFERENT!)
> - **Ground truth**: The expected output (what target should look like after transfer)
>
> **Examples of CORRECT pairings:**
> - Style transfer: Reference=artistic video, Target=realistic video, GT=stylized version of target
> - Motion transfer: Reference=dancing video, Target=standing person, GT=that person dancing
> - I2V animation: Reference=motion source video, Target=static image, GT=animated image with that motion
>
> **WRONG (will train identity mapping, learns nothing):**
> - Reference=Video A, Target=Video A, GT=Video A ❌
> - Using first frame of same video as target with full video as reference ❌
>
> For cross-video training, always pair DIFFERENT videos that share the property you want to transfer.

### OmniTransfer Components

1. **Task-aware Positional Bias (TPB)** - RoPE offsets distinguish reference vs target
2. **Reference-decoupled Causal Learning (RCL)** - Separate attention branches for efficiency
3. **Task-adaptive Multimodal Alignment (TMA)** - MLLM with MetaQueries for semantic guidance

### OmniTransfer Training (Local GPU)

**VRAM Requirements:**
| Configuration | VRAM Required | Notes |
|--------------|---------------|-------|
| Full (no quantization) | 80GB+ | A100/H100 recommended |
| LoRA + gradient checkpointing | 48GB+ | A6000, RTX 6000 Ada |
| LoRA + INT8 + grad checkpoint | 24GB+ | RTX 4090, RTX 3090 |
| LoRA + INT8 + low batch | 16GB+ | RTX 4080, experimental |

**Quick Start (24GB+ GPU):**

```bash
# 1. Prepare dataset with reference/target video pairs
python scripts/prepare_omnitransfer_dataset.py \
    --input-dir /path/to/videos \
    --output-dir /path/to/processed \
    --task-type identity_preservation

# 2. Train Stage 1 (TPB + RCL, 10k steps)
uv run python scripts/train.py configs/ltx2_omnitransfer_lora.yaml

# 3. Train Stage 2 (TMA connector, 2k steps) - optional
uv run python scripts/train.py configs/ltx2_omnitransfer_stage2.yaml

# 4. Train Stage 3 (joint fine-tuning, 5k steps) - optional
uv run python scripts/train.py configs/ltx2_omnitransfer_stage3.yaml
```

**Low VRAM Configuration (24GB):**

Edit your config YAML:
```yaml
model:
  training_mode: lora

lora:
  rank: 32  # Lower rank saves memory
  alpha: 32

optimization:
  batch_size: 1
  gradient_accumulation_steps: 16  # Effective batch = 16
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: bf16
  load_text_encoder_in_8bit: true  # INT8 quantization
```

**Generate Synthetic Training Data:**

```bash
# Dry run to preview prompts
python scripts/test_msi_generation.py --dry-run --num-videos 5

# Generate with LTX-2 T2V
python scripts/test_msi_generation.py --backend ltx2 --num-videos 10

# Generate with LTX-2 I2V (needs reference image)
python scripts/test_msi_generation.py --backend ltx2_i2v \
    --reference-image /path/to/identity_ref.png --num-videos 10
```

### OmniTransfer W&B Visualization

The training automatically logs to Weights & Biases:
- **Reconstruction grids**: Reference → Target → Prediction comparisons
- **Video comparisons**: Side-by-side videos at configurable intervals
- **Metrics**: Loss, PSNR, learning rate, sigma statistics

Configure in YAML:
```yaml
training_strategy:
  log_reconstructions: true
  reconstruction_log_interval: 500
  num_frames_to_visualize: 8
  log_video_comparisons: true
  video_log_interval: 2000
```

## Memory-Efficient Workflows (RTX 5090 / 32GB VRAM)

The RTX 5090 has 32GB VRAM which is tight for LTX-2 training. The key insight is that **models cannot be loaded simultaneously** - you must precompute data in stages.

### VRAM Budget Breakdown

| Component | VRAM Usage | Notes |
|-----------|------------|-------|
| Gemma Text Encoder | ~28GB | Cannot run alongside other models |
| Video VAE Encoder | ~8GB | Used for latent encoding |
| Video VAE Decoder | ~8GB | Used for visualization only |
| LTX-2 Transformer (INT8) | ~12GB | Main training model |
| LTX-2 Transformer (bf16) | ~20GB | Without quantization |
| Training overhead | ~4-8GB | Gradients, optimizer states |

### Required Precomputation Pipeline

**CRITICAL: Never load text encoder and VAE simultaneously!**

```bash
# Step 1: Encode videos/images to latents (VAE only, ~8GB)
python scripts/encode_website_demos.py \
    --input-dir /path/to/raw_data \
    --output-dir /path/to/processed \
    --skip-text-encoding  # Don't load text encoder yet!

# Step 2: Compute text embeddings SEPARATELY (Gemma only, ~28GB)
python scripts/compute_text_embeddings.py \
    --output-dir /path/to/processed \
    --model-path /path/to/ltx-2.safetensors \
    --text-encoder-path /path/to/gemma

# Step 3: NOW train (transformer + pre-computed data)
python scripts/train.py configs/your_config.yaml
```

### Dataset Directory Structure

The trainer expects this structure with **precomputed** data:

```
/path/to/dataset/
├── latents/                    # Video latents [C, F, H, W]
│   ├── 0.pt                    # Sample naming must match across dirs
│   ├── 1.pt
│   └── ...
├── conditions/                 # Text embeddings (PRECOMPUTED!)
│   ├── 0.pt                    # {prompt_embeds: [1024, 3840], prompt_attention_mask: [1024]}
│   ├── 1.pt
│   └── ...
├── reference_latents/          # Reference video/image latents
│   ├── 0.pt
│   └── ...
├── target_image_latents/       # Optional: first frame latents for I2V
│   └── ...
└── metadata.json               # Optional: task types, captions
```

### Text Embedding Format

Each `conditions/*.pt` file must contain:
```python
{
    'prompt_embeds': torch.Tensor,        # Shape: [1024, 3840], dtype: bfloat16
    'prompt_attention_mask': torch.Tensor, # Shape: [1024], dtype: int64
}
```

### Memory-Safe Training Config (32GB)

```yaml
model:
  training_mode: lora

lora:
  rank: 32  # Lower rank = less memory
  alpha: 32

optimization:
  batch_size: 1
  gradient_accumulation_steps: 8
  enable_gradient_checkpointing: true  # CRITICAL for 32GB

acceleration:
  mixed_precision_mode: bf16
  quantization: int8-quanto  # Quantize transformer to INT8
  load_text_encoder_in_8bit: false  # Text encoder not loaded during training!
```

### Common OOM Scenarios and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| OOM during text encoding | Gemma + VAE loaded together | Use `--skip-text-encoding`, run separately |
| OOM during training | Batch too large | Reduce `batch_size`, increase `gradient_accumulation_steps` |
| OOM during validation | VAE decoder loaded | Set `validation.interval: null` or increase interval |
| OOM with video logging | Decoding full videos | Reduce `num_frames_to_visualize` |

### Movie Weaver Multi-Concept Training

Movie Weaver (CVPR 2025) uses multiple reference images per sample with concept embeddings:

```
/media/2TB/movie_weaver_training/
├── latents/                    # Ground truth videos
├── conditions/                 # Text embeddings with [R1], [R2] anchors
├── reference_latents_R1/       # Face references
├── reference_latents_R2/       # Body references
├── reference_latents_R3/       # Third concept (pet, etc.)
├── reference_latents_R4/       # Fourth concept
├── multi_concept_refs/         # Combined refs with concept_assignments
└── manifest.json               # Sample metadata with concept mappings
```

Concept assignments example:
```python
{
    "refs": {"R1": "man face", "R2": "man body", "R3": "woman face", "R4": "woman body"},
    "concept_assignments": [0, 0, 1, 1],  # R1,R2=Person0, R3,R4=Person1
}
```

## Key Constraints

### LTX-2 Frame Requirements

Frames must satisfy `frames % 8 == 1`:
- ✅ Valid: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- ❌ Invalid: 24, 32, 48, 64, 100

### Resolution Requirements

Width and height must be divisible by 32.

### Model Paths

- Must be local paths (URLs not supported)
- `model_path`: Path to `.safetensors` checkpoint
- `text_encoder_path`: Path to Gemma model directory

### Platform Requirements

- Linux required (uses `triton` which is Linux-only)
- CUDA GPU with 24GB+ VRAM recommended

### CUDA Environment Setup

Before running training or inference, verify CUDA is properly configured:

```bash
# Check CUDA in zshrc - ensure these are set:
# In ~/.zshrc:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify nvcc is available
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**RTX 5090 (Blackwell/sm_120) Requirements:**
- CUDA 13.0 or later
- PyTorch 2.10.0+cu130 or later
- Install with: `pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130`

**Common CUDA Issues:**
1. **Error 804 (forward compatibility)**: PyTorch CUDA version doesn't match driver
   - Solution: Install PyTorch with matching CUDA version (e.g., cu130 for CUDA 13.0)
2. **NCCL undefined symbol**: Version mismatch
   - Solution: `pip install nvidia-nccl-cu13==2.28.9` (match PyTorch requirements)
3. **No kernel image available**: GPU architecture not supported
   - Solution: Ensure PyTorch includes sm_XX for your GPU (sm_120 for RTX 5090)

## Reference: ltx-core Key Components

```
packages/ltx-core/src/ltx_core/
├── model/
│   ├── transformer/
│   │   ├── model.py              # LTXModel
│   │   ├── modality.py           # Modality dataclass
│   │   └── transformer.py        # BasicAVTransformerBlock
│   ├── video_vae/
│   │   └── video_vae.py          # Encoder, Decoder
│   ├── audio_vae/
│   │   ├── audio_vae.py          # Decoder
│   │   └── vocoder.py            # Vocoder
│   └── clip/gemma/
│       └── encoders/av_encoder.py  # AVGemmaTextEncoderModel
├── pipeline/
│   ├── components/
│   │   ├── schedulers.py         # LTX2Scheduler
│   │   ├── diffusion_steps.py    # EulerDiffusionStep
│   │   ├── guiders.py            # CFGGuider
│   │   └── patchifiers.py        # VideoLatentPatchifier, AudioPatchifier
│   └── conditioning/             # VideoLatentTools, AudioLatentTools
└── loader/
    ├── single_gpu_model_builder.py  # SingleGPUModelBuilder
    └── sd_ops.py                    # Key remapping (SDOps)
```

## Grok MCP Integration

When asking Grok for help with training issues, always provide comprehensive context:

### Required Context for Training Issues

1. **Visual Context** (always include when available):
   - Debug images showing: Source (style/reference) | Target | Prediction
   - Loss curves from W&B
   - Any error screenshots

2. **Configuration Context**:
   - Task type (identity_preservation, style_transfer, motion_transfer)
   - Training strategy settings (TPB, RCL, TMA enabled/disabled)
   - Dataset size and composition
   - Latent shapes and dimensions

3. **Training State**:
   - Current step number
   - Loss values and trends
   - GPU memory usage
   - Any error messages

### Example Grok Query Format

```
I'm training OmniTransfer for [TASK_TYPE] with:
- Dataset: [N] pairs ([description])
- Latent shape: [C, F, H, W]
- Components: TPB=[yes/no], RCL=[yes/no], TMA=[yes/no]
- Current step: [N], Loss: [value]
- GPU: [usage]GB

[Include debug image showing: Reference | Target | Prediction]

Question: [specific question about training behavior]
```

### Debug Image Format for Style Transfer

The debug visualization should show 3 images side-by-side:
1. **Reference (Style Source)**: The style video to transfer FROM
2. **Target (Content)**: The video to apply style TO
3. **Prediction**: Model output showing style applied to target

This helps diagnose:
- Style extraction quality
- Identity/content preservation
- Artifacts or mode collapse
