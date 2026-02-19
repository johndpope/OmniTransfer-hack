# AGENTS.md

This file provides guidance to AI coding assistants (Claude, Cursor, etc.) when working with code in this repository.

## ⚠️ Killing Stale Processes

> To kill all stale Python processes (e.g., competing training runs), use:
> ```bash
> ~/killer.sh python
> ```
> Run this before launching GPU-intensive training to ensure no competing processes on the GPU.

## ⚠️ CRITICAL: Private Branch Policy

> **NEVER push `private/*` branches to the public repository!**
>
> This project uses a dual-remote setup for sensitive code:
>
> | Remote | URL | Purpose |
> |--------|-----|---------|
> | `origin` | https://github.com/johndpope/ltx2-omnitransfer | Public repository |
> | `private` | https://github.com/johndpope/omnitransfer-private | Private/proprietary code |
>
> **Branch naming convention:**
> - `main`, `feat/*`, `fix/*` → Push to `origin` (public)
> - `private/*` → Push ONLY to `private` remote (omnitransfer-private)
>
> **Examples of CORRECT usage:**
> ```bash
> # Public branches → origin
> git push origin main
> git push origin feat/new-feature
>
> # Private branches → private remote ONLY
> git push private private/audio
> git push private private/proprietary-model
> ```
>
> **NEVER DO THIS:**
> ```bash
> git push origin private/audio  # ❌ NEVER push private/* to public!
> git push private/audio         # ❌ Default remote might be origin!
> ```
>
> **Before pushing any `private/*` branch, ALWAYS verify:**
> 1. You're pushing to the `private` remote explicitly
> 2. The branch name starts with `private/`
> 3. Run `git remote -v` to confirm remote URLs if unsure

---

## ⚠️ CRITICAL: Never Store Important Output on /tmp

> **`/tmp` is ephemeral — it gets wiped on reboot!** Never store:
> - Training checkpoints
> - Inference output images/videos
> - Training configs
> - Scripts
> - Any artifact you'd be upset to lose
>
> **Use persistent storage instead:**
>
> | What | Store On | Example |
> |------|----------|---------|
> | Training output (checkpoints, logs) | `/media/2TB/training_output/<experiment>/` | `/media/2TB/training_output/isometric_phase3/` |
> | Inference output (images, videos) | `/media/2TB/inference_output/` | `/media/2TB/inference_output/blade_runner.png` |
> | Training configs | In the repo: `packages/ltx-trainer/configs/` | `configs/ltx2_isometric_phase3.yaml` |
> | Scripts | In the repo: `packages/ltx-trainer/scripts/` | `scripts/omnitransfer_inference.py` |
> | Training data (latents, embeddings) | `/media/2TB/` | `/media/2TB/diorama_training/` |
>
> **When creating training configs at runtime**, save them to `configs/` in the repo so they survive.
> If you must use `/tmp` for scratch work, always copy the final result to persistent storage.
>
> **Example — WRONG:**
> ```bash
> output_dir: "/tmp/progressive_overfit/phase3/output"  # ❌ Gone after reboot!
> ```
>
> **Example — CORRECT:**
> ```bash
> output_dir: "/media/2TB/training_output/isometric_phase3/output"  # ✅ Persistent
> ```

---

## ⚠️ CRITICAL: Model Quantization for Training

> **Pre-quantized FP8 checkpoints (`ltx-2-19b-dev-fp8.safetensors`) DO NOT work for LoRA training!**
>
> PyTorch autograd doesn't support `float8_e4m3fn` tensors — backward pass fails with
> `NotImplementedError: "ufunc_add_CUDA" not implemented for 'Float8_e4m3fn'`.
> The FP8 file is only useful for **inference** (no gradients needed).
>
> **For training, use the bf16 model + quanto runtime quantization:**
>
> | Model File | Size | Use For |
> |-----------|------|---------|
> | `ltx-2-19b-dev.safetensors` | 43 GB | **Training** — bf16 weights, quantized at runtime by quanto |
> | `ltx-2-19b-dev-fp8.safetensors` | 26 GB | **Inference ONLY** — no autograd support |
> | `ltx-2-19b-dev-fp4.safetensors` | 20 GB | **DO NOT USE** — no LoRA fusion kernel in ltx-core |
>
> **Config for training:**
> ```yaml
> model:
>   model_path: "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
> acceleration:
>   quantization: "fp8-quanto"  # Runtime quantization (~20 min first time)
> ```
>
> **Note:** quanto quantization takes ~20 minutes on first load. A future improvement would
> be to cache the quantized state dict so subsequent loads are instant.

---

## Project Overview

**LTX-2 Trainer** is a training toolkit for fine-tuning the Lightricks LTX-2 audio-video generation model. It supports:

- **LoRA training** - Efficient fine-tuning with adapters
- **Full fine-tuning** - Complete model training
- **Audio-video training** - Joint audio and video generation
- **IC-LoRA training** - In-context control adapters for video-to-video transformations
- **OmniTransfer training** - Unified spatio-temporal video transfer (identity preservation, style transfer, motion transfer)

**Key Dependencies:**

- **[`ltx-core`](packages/ltx-core/)** - Core model implementations (transformer, VAE, text encoder)
- **[`ltx-pipelines`](packages/ltx-pipelines/)** - Inference pipeline components

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

## OmniTransfer Paper Insights (arXiv:2601.14250v1)

> **Key takeaway: OmniTransfer is a VIDEO-reference method. All tasks use video references, not images.**

### How OmniTransfer Actually Works (from the Paper)

The paper (ByteDance, Jan 2026) builds on **Wan2.1 I2V 14B** — a video-to-video DiT model. The critical insight is that OmniTransfer exploits **multi-view, multi-frame information** from video references, which single images cannot provide.

**Table 1 from the Paper — Task Input Formats:**

| Task | Reference Input | Conditioning | Training Mode |
|------|----------------|--------------|---------------|
| Identity Preservation | V_ref (video) | Text prompt | T2V |
| Style Transfer | V_ref (video) | Text prompt | T2V |
| Motion Transfer | V_ref (video) | First frame I | I2V |
| Camera Movement | V_ref (video) | First frame I | I2V |
| Video Effect | V_ref (video) | First frame I | I2V |

**Appearance tasks** (ID, Style) → Reference video + text prompt (style/identity comes from reference, NOT from prompt)
**Temporal tasks** (Motion, Camera, Effect) → Reference video + first frame image

### Training Strategy: 3 Stages (from Section 5.1)

| Stage | Trainable Components | Steps | LR | Batch | Purpose |
|-------|---------------------|-------|-----|-------|---------|
| Stage 1 | DiT (LoRA) + TPB + RCL | 10,000 | 1e-5 | 16 | Learn reference conditioning |
| Stage 2 | TMA connector only | 2,000 | 1e-5 | 16 | Align MLLM features |
| Stage 3 | All components jointly | 5,000 | 1e-5 | 16 | End-to-end refinement |

### CRITICAL: Prompt Design for Style Transfer

> **Do NOT leak the target style into the text prompt!**
>
> In OmniTransfer, style/identity comes from the **reference video**, not the text prompt.
> The text prompt should describe the **content/scene**, not the style.
>
> **WRONG** (leaks style into text, model ignores reference):
> ```
> "isometric 3D view of this scene, photorealistic miniature diorama"  # ❌
> ```
>
> **CORRECT** (neutral prompt, style learned from reference):
> ```
> "a scene from the movie"  # ✅ — style comes from reference latent
> "a cinematic scene"       # ✅ — generic content description
> ```
>
> If the prompt describes the style, the model learns to use text for styling instead
> of learning to extract style from the reference video — defeating the whole purpose.

### Why Video References Matter

- **Multiple viewpoints**: A 6-second video provides ~145 frames = 145 different views of the same style/subject
- **Temporal consistency**: Model learns style is consistent across time, not a per-frame artifact
- **TPB (Task-aware Positional Bias)**: RoPE offset Δ=(f, 0, 0) for appearance tasks — exploits temporal dimension to separate ref from target. With num_frames=1, this offset is meaningless.
- **RCL (Reference-decoupled Causal Learning)**: Reference at fixed t=0 (noise-free) while target is denoised. Requires temporal extent to work properly.

## Dataset Inventory

### Raw Data Sources

#### 1. Movie Diorama Pairs (`/media/2TB/movie_dioramas/`)
- **82 movie scenes**, each containing:
  - `scene.jpg` / `scene.png` — Original movie scene screenshot
  - `diorama.png` — Isometric diorama version (Grok-generated)
  - `diorama_2.png` — Alternate diorama version (some scenes)
- Examples: `matrix_lobby/`, `blade_runner_rooftop/`, `inception_hotel/`, `pulp_fiction_diner/`
- **Use**: scene = reference, diorama = target (for style transfer training)

#### 2. Grok Isometric Images (`/media/12TB/isometric_3d/r2_native_dataset/images/`)
- **592 images** (784x1168) with matching `.txt` prompt files
- Mix of sources: `harvested_*`, `r2_iso_*`, `r2_untag_*`, `frame_*`
- Grok-generated high-quality isometric 3D renders

#### 3. Grok Isometric Videos (`/media/12TB/isometric_3d/r2_native_dataset/new_grok_videos/`)
- **8 videos** (784x1168, 145 frames, 24fps, ~6 seconds each)
- With matching `.txt` prompt files
- High-quality Grok-generated isometric 3D animations
- **This is the ideal reference format for OmniTransfer** — multi-frame style source

#### 4. Extracted Frames
- `/media/12TB/isometric_3d/r2_native_dataset/new_grok_frames/` — 384 frames from 8 Grok videos
- `/media/12TB/isometric_3d/r2_native_dataset/new_video_frames/` — 216 entries (.jpg + .pt + .txt)

#### 5. Combined Dataset Metadata
- `/media/12TB/isometric_3d/r2_native_dataset/dataset_combined.json` — 3,128 entries total:
  - 2,608 frames from 108 source videos
  - 439 harvested images
  - 65 r2_iso images
  - 16 r2_untag images

### Precomputed Training Data

#### Active Training Set (`/media/2TB/diorama_training/`)
- **157 pairs** (82 movies x ~2 variants each) — currently used for training
- Structure:
  ```
  /media/2TB/diorama_training/
  ├── latents/              # 157 target (diorama) latents [128, 1, 14, 26]
  ├── reference_latents/    # 157 reference (scene) latents [128, 1, 14, 26]
  ├── conditions_final/     # 157 text embeddings [1024, 3840]
  ├── qwen_vl_features/     # 157 Qwen2.5-VL features [seq_len, 3584]
  └── metadata.json
  ```

#### Preprocessed Image Sets (on 12TB drive)
- `/media/12TB/.../preprocessed_large/` — 592 latents + 592 conditions (from Grok images)
- `/media/12TB/.../preprocessed_grok/` — 8 latents + 8 conditions (from Grok videos)

### Training Output (`/media/2TB/training_output/`)

| Directory | Stage | Status | Key Details |
|-----------|-------|--------|-------------|
| `diorama_phase1/` | Stage 1 (DiT+TPB+RCL+CE) | Complete | 1000 steps, loss 81→31.4, 57.6 min |
| `diorama_phase2/` | Stage 2 (TMA connector) | Complete | 1000 steps, loss →24.3, 58.5 min |
| `diorama_phase3/` | Stage 3 (joint fine-tune) | Not started | Config at `configs/ltx2_diorama_phase3.yaml` |

### Training Configs (in repo)

| Config | Stage | Output Dir |
|--------|-------|------------|
| `configs/ltx2_diorama_phase1.yaml` | Stage 1 | `/media/2TB/training_output/diorama_phase1` |
| `configs/ltx2_diorama_phase2.yaml` | Stage 2 | `/media/2TB/training_output/diorama_phase2` |
| `configs/ltx2_diorama_phase3.yaml` | Stage 3 | `/media/2TB/training_output/diorama_phase3` |

### Inference Script

- `packages/ltx-trainer/scripts/omnitransfer_inference.py` — Full inference with LoRA + ConceptEmbedding + TMA
- Output: `/media/2TB/inference_output/`
- Supports `--lora`, `--no-strategy`, dual GPU, int8-quanto quantization

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

> **⚠️ ALWAYS use the highest resolution that fits in VRAM!**
>
> Low-resolution training (e.g., 512×768) produces noticeably poor quality output.
> Always encode training data and run inference at near-native resolution.
>
> | Source Resolution | Training Resolution | Quality |
> |------------------|-------------------|---------|
> | 784×1168 (Grok) | 512×768 | ❌ Poor — blurry, loses detail |
> | 784×1168 (Grok) | **768×1152** | ✅ Good — near-native quality |
> | 784×1168 (Grok) | 800×1184 | ✅ Best — closest to native |
>
> **Config example:**
> ```yaml
> validation:
>   # video_dims: [512, 768, 25]    # Low-res — faster, lower quality
>   video_dims: [768, 1152, 25]    # Hi-res — near-native Grok 784×1168
> ```
>
> **VRAM impact**: Higher resolution uses more tokens (864 vs 384 per frame at 768×1152 vs 512×768).
> With int8-quanto + gradient checkpointing, 768×1152 fits on 32GB GPU (~24GB used).

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
