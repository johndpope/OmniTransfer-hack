1. Plan Node Default
•Enter plan mode for any non-trivial task (three or more steps, or involving architectural decisions).
•If something goes wrong, stop and re-plan immediately rather than continuing blindly.
•Use plan mode for verification steps, not just implementation.
•Write detailed specifications upfront to reduce ambiguity.

2. Subagent Strategy
•Use subagents liberally to keep the main context window clean.
•Offload research, exploration, and parallel analysis to subagents.
•For complex problems, allocate more compute via subagents.
•Assign one task per subagent to ensure focused execution.

3. Self-Improvement Loop
•After any correction from the user, update tasks/lessons.md with the relevant pattern.
•Create rules for yourself that prevent repeating the same mistake.
•Iterate on these lessons rigorously until the mistake rate declines.
•Review lessons at the start of each session when relevant to the project.

4. Verification Before Done
•Never mark a task complete without proving it works.
•Diff behavior between main and your changes when relevant.
•Ask: “Would a staff engineer approve this?”
•Run tests, check logs, and demonstrate correctness.

5. Demand Elegance (Balanced)
•For non-trivial changes, pause and ask whether there is a more elegant solution.
•If a fix feels hacky, implement the solution you would choose knowing everything you now know.
•Do not over-engineer simple or obvious fixes.
•Critically evaluate your own work before presenting it.

6. Autonomous Bug Fixing
•When given a bug report, fix it without asking for unnecessary guidance.
•Review logs, errors, and failing tests, then resolve them.
•Avoid requiring context switching from the user.
•Fix failing CI tests proactively.

Task Management
1.Plan First: Write the plan to tasks/todo.md with checkable items.
2.Verify Plan: Review before starting implementation.
3.Track Progress: Mark items complete as you go.
4.Explain Changes: Provide a high-level summary at each step.
5.Document Results: Add a review section to tasks/todo.md.
6.Capture Lessons: Update tasks/lessons.md after corrections.

Core Principles
•Simplicity First: Make every change as simple as possible. Minimize code impact.
•No Laziness: Identify root causes. Avoid temporary fixes. Apply senior developer standards.
•Minimal Impact: Touch only what is necessary. Avoid introducing new bugs.

7. Checkpoint Resume Policy
•When launching a new training run on a dataset that already has checkpoints, ALWAYS ask the user whether to resume from the last checkpoint or start fresh.
•Never silently start from scratch when checkpoints exist — that wastes hours of prior training.
•Never silently resume without asking — the user may want a fresh start (e.g., after changing config, dataset, or strategy).
•To resume, set `model.load_checkpoint` in the YAML config to the checkpoint path (e.g., `/media/2TB/omnitransfer/output/<experiment>/checkpoints/lora_weights_step_03000.safetensors`).
•When resuming, consider whether `optimization.steps` needs increasing (the step counter resets, so to train 3000 MORE steps, keep steps at 3000).

8. Resolution Matching for Inference
•ALWAYS match output video resolution to the source training data resolution and orientation.
•Portrait input data (e.g., 768×1152 Grok isometric) → portrait output MP4 (`--height 1152 --width 768`).
•Landscape input data → landscape output MP4.
•NEVER use the default 448×768 landscape when the training data is portrait — it produces aspect-ratio-distorted, low-quality results.
•Check source latent shape to determine orientation: latent `[C, F, H, W]` where `H > W` = portrait.

9. Training Monitoring & NaN Prevention
•When launching training, ALWAYS inspect the first 5-10 steps via W&B or log before walking away.
•Check for NaN loss or exploding gradients (loss > 1e6 or sudden jumps >100x). If NaN appears, kill the run immediately.
•Common NaN fixes (try in order):
  1. Reduce learning_rate (e.g., 1e-4 → 3e-5)
  2. Enable/check max_grad_norm (default 1.0)
  3. Disable bf16 mixed precision (set mixed_precision_mode: "no")
  4. Ensure `enable_gradient_checkpointing: true` is set
  5. Check data for NaN/inf values in latents or conditions
•After any fix, resume from last good checkpoint (not scratch) to save time.



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

## ⚠️ CRITICAL: Always Enable W&B and Reconstruction Logging

> **Every training config MUST have W&B enabled and reconstruction images turned on.**
>
> When creating or modifying training configs, always include:
> ```yaml
> wandb:
>   enabled: true
>   project: "<descriptive-project-name>"
>   log_validation_videos: true
>
> training_strategy:
>   log_reconstructions: true
>   reconstruction_log_interval: 200   # Log debug images every 200 steps
>   num_frames_to_visualize: 1         # At least 1 frame
>   max_samples_per_log: 1
> ```
>
> **Do NOT use `WANDB_MODE=disabled`** unless explicitly asked. Reconstruction images
> (Reference | Target | Prediction grids) are essential for diagnosing training quality.
> Without them, you're training blind.

---

## ⚠️ CRITICAL: Never Store Important Output on /tmp

> **`/tmp` is ephemeral — it gets wiped on reboot!** Never store:
> - Training checkpoints
> - Inference output images/videos
> - Training configs
> - Scripts
> - Any artifact you'd be upset to lose
>
> **Use persistent storage under ONE parent folder: `/media/2TB/omnitransfer/`**
>
> All project data lives under a single root to keep the HDD tidy:
>
> ```
> /media/2TB/omnitransfer/
> ├── data/                          # Precomputed training datasets
> │   ├── diorama_training/          # 157 diorama pairs (latents, ref_latents, conditions_final)
> │   ├── isometric_i2v/             # 128 isometric I2V clips
> │   ├── isometric_identity/        # 192 cross-clip identity pairs
> │   └── movie_dioramas/            # Raw 82 movie scene→diorama image pairs
> ├── output/                        # Training checkpoints & logs
> │   ├── diorama_phase1/
> │   ├── diorama_phase2/
> │   ├── isometric_i2v/
> │   └── isometric_identity/
> └── inference/                     # Inference output (images, videos)
> ```
>
> | What | Store On | Example |
> |------|----------|---------|
> | Training datasets (latents, embeddings) | `/media/2TB/omnitransfer/data/<dataset>/` | `/media/2TB/omnitransfer/data/isometric_i2v/` |
> | Training output (checkpoints, logs) | `/media/2TB/omnitransfer/output/<experiment>/` | `/media/2TB/omnitransfer/output/isometric_identity/` |
> | Inference output (images, videos) | `/media/2TB/omnitransfer/inference/` | `/media/2TB/omnitransfer/inference/blade_runner.png` |
> | Training configs | In the repo: `packages/ltx-trainer/configs/` | `configs/ltx2_isometric_identity.yaml` |
> | Scripts | In the repo: `packages/ltx-trainer/scripts/` | `scripts/omnitransfer_inference.py` |
>
> **When creating training configs at runtime**, save them to `configs/` in the repo so they survive.
> If you must use `/tmp` for scratch work, always copy the final result to persistent storage.
>
> **Example — WRONG:**
> ```bash
> output_dir: "/tmp/progressive_overfit/phase3/output"         # ❌ Gone after reboot!
> preprocessed_data_root: "/media/2TB/isometric_i2v_training"  # ❌ Scattered on HDD root!
> ```
>
> **Example — CORRECT:**
> ```bash
> output_dir: "/media/2TB/omnitransfer/output/isometric_identity"     # ✅ Organized
> preprocessed_data_root: "/media/2TB/omnitransfer/data/isometric_i2v" # ✅ Under single root
> ```

---

## ⚠️ CRITICAL: NEVER USE LTX 2.0 19B

> **NEVER use `ltx-2-19b-dev.safetensors`. Always use LTX 2.3 (`ltx-2.3-22b-distilled.safetensors`).**
>
> The 19b model is deprecated. All training must use LTX 2.3 at
> `/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors`.
>
> If a config references `ltx2/ltx-2-19b-dev.safetensors`, change it immediately.
> The ltx-core and ltx-trainer packages now exclusively target LTX 2.3.

## ⚠️ CRITICAL: Model Quantization for Training

> **Pre-quantized FP8 checkpoints (`ltx-2-19b-dev-fp8.safetensors`) DO NOT work for LoRA training!**
>
> PyTorch autograd doesn't support `float8_e4m3fn` tensors — backward pass fails with
> `NotImplementedError: "ufunc_add_CUDA" not implemented for 'Float8_e4m3fn'`.
> The FP8 file is only useful for **inference** (no gradients needed).
>
> **For training, use the bf16 LTX 2.3 model + quanto runtime quantization:**
>
> | Model File | Size | Use For |
> |-----------|------|---------|
> | ~~`ltx-2-19b-dev.safetensors`~~ | 43 GB | **DO NOT USE** — deprecated |
> | **`ltx-2.3-22b-distilled.safetensors`** | **43 GB** | **Training** — bf16 weights, quantized at runtime by quanto |
> | `ltx-2-19b-dev-fp8.safetensors` | 26 GB | **Inference ONLY** — no autograd support |
>
> **Config for training:**
> ```yaml
> model:
>   model_path: "/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors"
> acceleration:
>   quantization: "int8-quanto"  # Runtime quantization
> ```

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

## SCD + DDiT: Efficient Autoregressive Video Generation

### Overview

SCD (Separable Causal Diffusion) and DDiT (Dynamic Patch Scheduling) are complementary inference optimizations for LTX-2 that enable long-form video generation (30s+) on consumer GPUs.

- **SCD** splits the 48-layer DiT into encoder (32 layers) + decoder (16 layers). The encoder runs **once** per frame (causal, σ=0, KV-cached); the decoder runs **N steps** per frame. Zero new parameters — pure architectural wrapper.
- **DDiT** (arXiv:2602.16968) varies spatial patch sizes per denoising step. Early steps use merged 2×2 patches (4× fewer tokens → ~16× less attention compute); late steps use native resolution for fine detail. Adds a 4.2M-param adapter.

Together they attack different bottleneck dimensions: SCD reduces layers-per-step (48→16), DDiT reduces tokens-per-step (336→84 at scale=2).

### Architecture Diagram

```
Frame N generation with SCD + DDiT:

ENCODER (32 layers, runs ONCE per frame, causal with KV-cache):
  Frame N-1 (clean, σ=0) ──→ [32 transformer blocks] ──→ encoder_features
                                   KV-cache accumulates across frames

DECODER (16 layers, runs N_steps times per frame):
  Step 1-2:  Native resolution (336 tokens)  ← structure establishment
  Step 3-27: DDiT merged    (84 tokens)     ← 4× fewer tokens, 16× less attention
  Step 28-30: Native resolution (336 tokens)  ← fine detail refinement

DDiT merge/unmerge per step:
  [1, 336, 128] ──merge 2×2──→ [1, 84, 512] ──patchify──→ [1, 84, 4096]
                                  16 decoder blocks at reduced resolution
  [1, 84, 4096] ──proj_out──→ [1, 84, 512] ──unmerge──→ [1, 336, 128]
```

### Key Source Files

| File | Location | Purpose |
|------|----------|---------|
| `scd_model.py` | `ltx-core/src/ltx_core/model/transformer/scd_model.py` | `LTXSCDModel` wrapper — splits encoder/decoder, KV-cache |
| `ddit.py` | `ltx-core/src/ltx_core/model/transformer/ddit.py` | `DDiTAdapter`, `DDiTMergeLayer`, `DDiTPatchScheduler` (4.2M params) |
| `scd_inference.py` | `ltx-trainer/scripts/scd_inference.py` | Combined SCD + DDiT inference pipeline |
| `scd_strategy.py` | `ltx-trainer/src/ltx_trainer/training_strategies/scd_strategy.py` | SCD LoRA training strategy |

### Pre-trained Checkpoints

| Checkpoint | Path | Details |
|-----------|------|---------|
| SCD LoRA (Ditto-1M) | `/media/2TB/omnitransfer/output/scd_ditto_subset/checkpoints/lora_weights_step_02000.safetensors` | 500 pairs, 2000 steps, rank=32, loss=0.231 |
| DDiT adapter v2 | `sparse-causal-diffusion/outputs/ddit_scd_v2/ddit_scd_adapter_final.safetensors` | 8.4MB, scale=2, residual_weight=0.1 |
| DDiT decoder LoRA | `sparse-causal-diffusion/outputs/ddit_scd_v2/ddit_scd_lora_final.safetensors` | 16.8MB, hook-based rank=16 on to_q/k/v/to_out.0 |

### Training Recommendations

#### Phase 1: SCD LoRA (required)

Train the SCD LoRA first — this teaches the encoder/decoder split behavior:

```yaml
# configs/ltx2_scd_lora.yaml
training_strategy:
  name: scd
  encoder_layers: 32
  decoder_input_combine: add     # Use "add" — token_concat doubles sequence and OOMs at 720p
  with_audio: false

model:
  model_path: "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
  training_mode: lora

lora:
  rank: 32
  alpha: 32

optimization:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  max_train_steps: 2000

acceleration:
  quantization: fp8-quanto       # Runtime quantization for training
```

**Training data**: Any video dataset with precomputed latents + conditions works. Tested with:
- Ditto-1M subset (500 pairs): loss 0.386 → 0.231 over 2000 steps
- Isometric videos (128 clips): good convergence but teacher forcing → rapid quality collapse in autoregressive inference

**Key insight**: SCD training uses **teacher forcing** (clean frames as encoder input), but inference is **autoregressive** (model's own noisy predictions feed back). Diverse training data (like Ditto-1M) produces much more robust inference than small same-domain datasets.

#### Phase 2: DDiT Adapter (optional, for decoder speedup)

> **⚠️ The DDiT adapter MUST be trained against the specific SCD LoRA it will be used with.**
> The current DDiT v2 adapter was trained against the old isometric SCD LoRA — not the Ditto-1M LoRA.
> Retraining against the current SCD LoRA would improve DDiT quality and speedup.

DDiT adapter training lives in the `sparse-causal-diffusion` repo:
```bash
cd /home/johndpope/Documents/GitHub/sparse-causal-diffusion
python -m scd.train_ddit \
    --scd-checkpoint /path/to/scd_lora.safetensors \
    --output-dir outputs/ddit_scd_v3 \
    --scale 2 \
    --steps 3000
```

The DDiT adapter (4.2M params) trains in ~45 minutes on RTX 5090. It learns:
- `DDiTMergeLayer.patchify_proj`: projects merged tokens (128×s²) → inner_dim (4096)
- `DDiTMergeLayer.proj_out`: projects inner_dim → merged token space
- `DDiTMergeLayer.patch_id`: learned positional embedding for merged patches
- `DDiTMergeLayer.residual_block`: optional residual refinement
- Hook-based decoder LoRA (rank-16 on attn layers) to adapt decoder for merged tokens

### Inference Quick Reference

```bash
# Basic SCD inference (30 seconds)
python scripts/scd_inference.py \
    --cached-embedding /media/2TB/omnitransfer/data/isometric_i2v/conditions_final/000.pt \
    --num-seconds 30 \
    --quantization int8-quanto \
    --output /media/2TB/omnitransfer/inference/scd_30s.mp4

# SCD + DDiT (faster decoder)
python scripts/scd_inference.py \
    --cached-embedding /media/2TB/omnitransfer/data/isometric_i2v/conditions_final/000.pt \
    --num-seconds 30 \
    --quantization int8-quanto \
    --ddit-adapter /home/johndpope/Documents/GitHub/sparse-causal-diffusion/outputs/ddit_scd_v2/ddit_scd_adapter_final.safetensors \
    --ddit-scale 2 \
    --ddit-native-head 2 --ddit-native-tail 3 \
    --output /media/2TB/omnitransfer/inference/scd_ddit_30s.mp4

# With live text prompt (loads/unloads Gemma automatically)
python scripts/scd_inference.py \
    --prompt "A mountain landscape with flowing rivers" \
    --num-seconds 10 \
    --output /media/2TB/omnitransfer/inference/scd_prompt_10s.mp4
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | ltx-2-19b-dev.safetensors | Base model |
| `--lora-path` | Ditto SCD LoRA | SCD LoRA checkpoint |
| `--cached-embedding` | — | Precomputed text embedding .pt |
| `--prompt` | — | Live text (loads Gemma, ~28GB) |
| `--num-seconds` | 5.0 | Video duration |
| `--height/--width` | 448/768 | Resolution (must be ÷32) |
| `--num-inference-steps` | 30 | Denoising steps per frame |
| `--encoder-layers` | 32 | Encoder/decoder split point |
| `--decoder-combine` | add | Encoder-decoder coupling: `add` or `token_concat` |
| `--quantization` | fp8-quanto | `fp8-quanto`, `int8-quanto`, or `none` |
| `--ddit-adapter` | None | DDiT adapter .safetensors path (enables DDiT) |
| `--ddit-scale` | 2 | Spatial merge factor (2=4× fewer tokens, 4=16×) |
| `--ddit-native-head` | 2 | Initial steps at native resolution |
| `--ddit-native-tail` | 3 | Final steps at native resolution |
| `--split-gpus` | false | Distribute encoder→cuda:0, decoder→cuda:1 in bf16 (no quant) |

### Performance Benchmarks (RTX 5090 + RTX PRO 4000, 768×448)

Measured on this exact hardware:

**Single-GPU (int8-quanto on RTX 5090):**

| Duration | Method | Gen Time | Encoder | Decoder | s/frame | Total | Dec Speedup | Total Speedup |
|----------|--------|----------|---------|---------|---------|-------|-------------|---------------|
| 30s (76 frames) | SCD baseline | 2.1 min | — | 118.7s | 1.7 | 2.9 min | 1.0× | 1.0× |
| 30s (76 frames) | SCD + DDiT 2× (fixed) | 1.7 min | — | 94.8s | 1.4 | 2.6 min | 1.25× | 1.12× |
| 30s (76 frames) | SCD + DDiT 2× (**dynamic**) | 1.7 min | 6.8s | **93.3s** | **1.3** | **2.5 min** | **1.27×** | 1.16× |
| 60s (151 frames) | SCD baseline | 4.2 min | 14.6s | 237.2s | 1.7 | 5.2 min | 1.0× | 1.0× |
| 60s (151 frames) | SCD + DDiT 2× | 3.4 min | 14.4s | 189.5s | 1.4 | 4.4 min | 1.25× | 1.18× |
| 120s (301 frames) | SCD baseline | 8.4 min | 28.8s | 472.6s | 1.7 | 9.9 min | 1.0× | 1.0× |
| 120s (301 frames) | SCD + DDiT 2× | 6.7 min | 28.1s | 373.3s | 1.3 | 8.3 min | 1.27× | 1.19× |

**Split-GPU (bf16, encoder→RTX 5090, decoder→RTX PRO 4000):**

| Duration | Method | Gen Time | Encoder | Decoder | s/frame | Total | Dec Speedup |
|----------|--------|----------|---------|---------|---------|-------|-------------|
| 30s (76 frames) | Split baseline | 3.4 min | 6.6s | 198.1s | 2.7 | 4.3 min | 1.0× |
| 30s (76 frames) | Split + DDiT 2× | 2.4 min | 6.3s | 133.7s | 1.9 | 3.2 min | **1.48×** |

> **Key finding:** DDiT speedup is **1.48× in bf16** vs **1.25× in int8-quanto**, confirming that bf16 makes the decoder compute-bound where DDiT's 4× token reduction has real impact. However, because the PRO 4000 is slower than the 5090, single-GPU int8-quanto is still faster in absolute terms (1.4s vs 1.9s/frame). Split-GPU would win with two equally fast GPUs.

**Scaling Characteristics:**
- **Encoder is O(1) per frame** via KV-cache: ~14.5s per 150 frames, ~6-7% of total time
- **Decoder scales linearly** with frame count: ~1.7s/frame baseline, ~1.3-1.4s/frame with DDiT
- **DDiT speedup improves slightly at longer durations**: 1.25× → 1.27× decoder speedup (better cache utilization with fewer tokens)
- **Dynamic scheduler uses 90% merged steps** vs fixed schedule's 83% — adapts per-prompt via 3rd-order trajectory analysis
- **Total speedup improves at scale**: 1.12× at 30s → 1.19× at 120s (decoder fraction grows)

**Theoretical vs Actual Speedup:**

| Metric | Theoretical | int8-quanto (actual) | bf16 split (actual) | Notes |
|--------|-------------|---------------------|---------------------|-------|
| Decoder token reduction | 4× (336→84) | 4× | 4× | Merge works correctly |
| Attention compute reduction | 16× (O(N²)) | 16× | 16× | Fused attention kernel |
| Decoder wall-clock speedup | 2-3× | 1.25-1.27× | **1.48×** | bf16 is compute-bound ✓ |
| DDiT steps (of 30 total) | 25/30 | 25/30 | 25/30 | 2 head + 3 tail native |
| Best overall (s/frame) | — | **1.4s** | 1.9s | 5090 is faster than PRO 4000 |

**Why actual gains are lower than theoretical:**
1. **int8-quanto memory bandwidth bottleneck**: Quantized weights must be dequantized every step. This makes the decoder memory-bound, not compute-bound. Reducing compute (fewer tokens) helps less when you're bottlenecked on weight loading.
2. **bf16 split-GPU confirms**: Moving to bf16 (compute-bound) gives 1.48× DDiT speedup — significantly closer to theoretical 2-3×. The remaining gap is likely from non-attention operations (FFN, norm, projection).
3. **Asymmetric GPUs hurt split-GPU**: The PRO 4000 (24GB) has ~50% less memory bandwidth than the 5090 (32GB), negating the DDiT gains in absolute terms.
4. **With two identical fast GPUs** (e.g., 2× RTX 5090 or 2× A100): Expected ~1.5× total speedup over single-GPU int8-quanto.

### Sanity Checks

Before running SCD + DDiT inference, verify:

1. **Coupling mode match**: The DDiT adapter's `coupling` field in `ddit_scd_config.json` should ideally match `--decoder-combine`. Current adapter uses `token_concat` but inference defaults to `add` — this works but is suboptimal.

2. **Decoder block count**: `ddit_scd_config.json:scd_decoder_blocks` must equal `48 - --encoder-layers`. Default: 16 decoder blocks (48 - 32 encoder).

3. **Scale compatibility**: Only use scales the adapter was trained for (check `ddit_scd_config.json:scales`). Current adapter: `[2]` only.

4. **Native head/tail**: At least 2-3 steps at native resolution for fine detail. If output looks blocky, increase `--ddit-native-tail`.

5. **Memory check**: SCD + DDiT adds ~25MB for the adapter — negligible. The DDiT decoder LoRA hooks add ~17MB.

6. **Quality validation**: Compare SCD-only vs SCD+DDiT output side-by-side. DDiT should produce similar quality with fewer artifacts. If quality degrades significantly, the adapter needs retraining against the current SCD LoRA.

### Known Limitations

- **fp8-quanto + DDiT**: Marlin kernel JIT compilation may fail (ninja build error). Use `int8-quanto` as fallback.
- **token_concat coupling**: Doubles the decoder sequence length, causing OOM at high resolutions. Use `add` coupling for 720p+ inference.
- **DDiT scale=4**: Not yet trained. Would give 16× fewer tokens but requires separate adapter training.
- **No DDiT for encoder**: DDiT only accelerates the decoder. Encoder is already efficient with KV-cache (~6% of total time).
- **Autoregressive quality drift**: SCD uses teacher forcing during training but autoregressive rollout at inference. Quality degrades after ~10-15 seconds with small training datasets. Use diverse, large datasets (Ditto-1M) for robust long-form generation.

---

## DDiT Implementation Rules (CRITICAL — Read Before Modifying)

> **Reference:** [arXiv:2602.16968](https://arxiv.org/abs/2602.16968) — Dynamic Diffusion Transformer
> **Detailed doc:** `ltx-trainer/docs/ddit-implementation.md`

### What DDiT IS

DDiT dynamically changes spatial resolution per denoising step. Early steps use coarse patches (fewer tokens, faster attention), late steps use fine patches (more tokens, better detail). The **dynamic scheduler** (3rd-order finite differences of the denoising trajectory) picks the optimal scale per step per prompt.

### Rules for DDiT Development

> **1. ALWAYS use `DDiTPatchScheduler` for scale selection — NEVER hardcode head/tail**
>
> The paper's core contribution is the **adaptive per-step scheduling**. A fixed "skip first 2 and last 3 steps" schedule defeats the purpose. The scheduler:
> - Analyzes the 3rd-order finite difference of the denoising trajectory
> - Picks the coarsest scale where spatial variance < threshold (τ=0.001)
> - Adapts per-prompt — different content gets different schedules
> - Code: `DDiTPatchScheduler` in `ltx-core/.../transformer/ddit.py`
>
> ```python
> # CORRECT — dynamic scheduling
> scheduler = ddit_adapter.scheduler
> scheduler.reset()
> for step in range(num_steps):
>     scheduler.record(z)
>     scale = scheduler.compute_schedule(z, step, nf, h, w)
>     if scale > 1:
>         velocity = ddit_decode(noisy, enc_ctx, sigma, positions, scale)
>     else:
>         velocity = native_decode(noisy, enc_ctx, sigma, positions)
>
> # WRONG — fixed schedule (loses adaptive behavior)
> if step >= 2 and step < num_steps - 3:
>     velocity = ddit_decode(...)
> ```

> **2. DDiT MUST work for ALL LTX-2 modalities, not just SCD**
>
> The paper targets T2I (FLUX) and T2V (Wan-2.1). Our implementation must support:
>
> | Mode | Blocks | DDiT Tokens Saved | Adapter Type |
> |---|---|---|---|
> | SCD T2V (autoregressive) | 16 decoder | 4× per frame per step | SCD adapter |
> | Standard T2V | 48 full model | 4× × all frames simultaneously | Full adapter |
> | I2V / T2I / I2I | 48 full model | 4× × all frames | Full adapter |
>
> **Standard T2V is where DDiT shines most** — 97 frames × 336 tokens = 32,592 tokens. Scale=2 reduces to 8,148 → **16× less attention compute**. SCD's single-frame decoder only saves 336→84 tokens.
>
> Two adapter types are needed:
> - `train_ddit_scd.py` → SCD-specific (16 decoder blocks)
> - `train_ddit_adapter.py` → Full model (48 blocks, any modality)

> **3. LoRA targets must match the paper for each mode**
>
> | Mode | Paper Targets | Our Current | Status |
> |---|---|---|---|
> | Full model (T2V/I2V/T2I) | FFN: `net.0.proj, net.2` | FFN + attn | ✅ Superset |
> | SCD decoder | Attention: `to_q,k,v,out` | Attention only | ⚠️ Should add FFN |

> **4. Test DDiT quality with distillation metrics**
>
> After training or modifying DDiT:
> - Compare teacher (native) vs student (merged) MSE at multiple sigma values
> - Check cosine similarity between outputs (should be > 0.95)
> - Visual comparison: side-by-side at σ=0.05 (fine detail), σ=0.5 (structure), σ=0.9 (noise)
> - If quality degrades at low sigma, increase `--ddit-native-tail` or retrain

> **5. Split-GPU mode puts DDiT adapter on the decoder GPU**
>
> In `--split-gpus` mode (encoder→cuda:0, decoder→cuda:1):
> - DDiT adapter MUST be on `cuda:1` (decoder device)
> - DDiT decoder LoRA hooks MUST target decoder blocks on `cuda:1`
> - Prompt embeddings for decoder must also be on `cuda:1`

### DDiT Hyperparameter Reference

| Parameter | Paper Default | Our Default | Config Key |
|---|---|---|---|
| Scheduler threshold | τ=0.001 | 0.001 | `DDiTConfig.threshold` |
| Scheduler percentile | ρ=0.4 | 0.4 | `DDiTConfig.percentile` |
| Warmup steps | 3 | 3 | `DDiTConfig.warmup_steps` |
| Supported scales | (1, 2, 4) | (1, 2) | `DDiTConfig.supported_scales` |
| Residual weight | — | 0.1 | `DDiTConfig.residual_weight` |
| LoRA rank (full) | 32 | 32 | `ddit_config.json:lora_rank` |
| LoRA rank (SCD decoder) | — | 16 | `ddit_config.json:ddit_lora_rank` |
| Distillation LR | 1e-4 | 1e-4 | Training script arg |
| Sigma curriculum | — | cosine [0.3→0.9] | `ddit_config.json:sigma_curriculum` |

### Cross-Repository Reference

The DDiT training code lives in a separate repository:

```
/home/johndpope/Documents/GitHub/sparse-causal-diffusion/
├── scd/
│   ├── ddit_inference.py         # DDiTInferenceWrapper class
│   ├── train_ddit.py             # DDiT adapter training
│   └── scd_model.py              # SCD model (duplicated in ltx-core)
├── inference/
│   └── run_scd_ddit_inference.py # Full SCD+DDiT inference pipeline
└── outputs/
    └── ddit_scd_v2/              # Pre-trained DDiT adapter + decoder LoRA
        ├── ddit_scd_adapter_final.safetensors  (8.4MB)
        ├── ddit_scd_lora_final.safetensors     (16.8MB)
        └── ddit_scd_config.json
```

The integrated inference script in THIS repo (`ltx-trainer/scripts/scd_inference.py`) uses `DDiTAdapter` from ltx-core directly (editable install) rather than importing from sparse-causal-diffusion.

---

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

### Precomputed Training Data (`/media/2TB/omnitransfer/data/`)

All precomputed datasets live under a single root. Each dataset follows the same structure:
```
<dataset>/
├── latents/              # Target video latents [C, F, H, W]
├── reference_latents/    # Reference latents [C, F, H, W]
├── conditions_final/     # Text embeddings [1024, 3840]
├── qwen_vl_features/     # (optional) Qwen2.5-VL features [seq_len, 3584]
└── metadata.json
```

| Dataset | Pairs | Resolution | Description |
|---------|-------|------------|-------------|
| `data/diorama_training/` | 157 | 832x448 | Movie scene→diorama style transfer |
| `data/isometric_i2v/` | 128 | 768x1152 | Grok video clips for I2V animation |
| `data/isometric_identity/` | 192 | 768x1152 | Cross-clip identity preservation |
| `data/movie_dioramas/` | 82 scenes | Raw images | Source scene+diorama image pairs |

#### Preprocessed Image Sets (on 12TB drive)
- `/media/12TB/.../preprocessed_large/` — 592 latents + 592 conditions (from Grok images)
- `/media/12TB/.../preprocessed_grok/` — 8 latents + 8 conditions (from Grok videos)

### Training Output (`/media/2TB/omnitransfer/output/`)

| Directory | Strategy | Status | Key Details |
|-----------|----------|--------|-------------|
| `output/diorama_phase1/` | OmniTransfer Stage 1 | Complete | 1000 steps, loss 81→31.4, 57.6 min |
| `output/diorama_phase2/` | OmniTransfer Stage 2 | Complete | 1000 steps, loss →24.3, 58.5 min |
| `output/diorama_phase3/` | OmniTransfer Stage 3 | Not started | Config: `ltx2_diorama_phase3.yaml` |
| `output/isometric_i2v/` | TextToVideo (I2V) | Complete | 500 steps, loss 0.0813, 74.4 min |
| `output/isometric_identity/` | OmniTransfer Stage 1 | In progress | TPB+RCL+CE, 192 pairs |

### Training Configs (in repo)

| Config | Strategy | Data Dir |
|--------|----------|----------|
| `configs/ltx2_diorama_phase1.yaml` | OmniTransfer Stage 1 | `data/diorama_training` |
| `configs/ltx2_diorama_phase2.yaml` | OmniTransfer Stage 2 | `data/diorama_training` |
| `configs/ltx2_diorama_phase3.yaml` | OmniTransfer Stage 3 | `data/diorama_training` |
| `configs/ltx2_isometric_i2v.yaml` | TextToVideo (I2V) | `data/isometric_i2v` |
| `configs/ltx2_isometric_identity.yaml` | OmniTransfer Stage 1 | `data/isometric_identity` |

### Inference Script

- `packages/ltx-trainer/scripts/omnitransfer_inference.py` — Full inference with LoRA + ConceptEmbedding + TMA
- Output: `/media/2TB/omnitransfer/inference/`
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
