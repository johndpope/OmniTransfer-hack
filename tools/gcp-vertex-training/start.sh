#!/bin/bash
# =============================================================================
# OmniTransfer Training Startup Script for Vertex AI
# =============================================================================
# This script runs inside the training container
# =============================================================================

set -e

echo "=== OmniTransfer Training on Vertex AI ==="
echo "Started at: $(date)"
echo ""

# Configuration from environment
GCS_BUCKET="${GCS_BUCKET:-}"
GCS_DATA_PREFIX="${GCS_DATA_PREFIX:-omnitransfer/processed}"
GCS_CHECKPOINT_PREFIX="${GCS_CHECKPOINT_PREFIX:-omnitransfer/checkpoints}"
TRAINING_STEPS="${TRAINING_STEPS:-10000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"

WORKSPACE="/workspace/ltx2-omnitransfer"
TRAINER_PATH="$WORKSPACE/packages/ltx-trainer"
DATA_DIR="/data/omnitransfer_training"
MODEL_DIR="/data/models"
OUTPUT_DIR="$TRAINER_PATH/outputs/omnitransfer_vertex"

# =============================================================================
# Configure W&B
# =============================================================================
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "Configuring Weights & Biases..."
    wandb login --relogin "$WANDB_API_KEY"
fi

# =============================================================================
# Download Training Data from GCS
# =============================================================================
if [ ! -z "$GCS_BUCKET" ]; then
    echo ""
    echo "=== Downloading Training Data from GCS ==="
    mkdir -p "$DATA_DIR"

    gsutil -m cp -r "gs://$GCS_BUCKET/$GCS_DATA_PREFIX/*" "$DATA_DIR/"

    echo "Training data contents:"
    for dir in "$DATA_DIR"/*/; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -type f -name "*.pt" 2>/dev/null | wc -l)
            echo "  $(basename "$dir"): $count files"
        fi
    done
fi

# =============================================================================
# Download Model Weights
# =============================================================================
echo ""
echo "=== Downloading Model Weights ==="
mkdir -p "$MODEL_DIR"

LTX2_MODEL="$MODEL_DIR/ltx-2-19b-dev.safetensors"
GEMMA_DIR="$MODEL_DIR/gemma"

if [ ! -f "$LTX2_MODEL" ]; then
    # Try GCS first
    gsutil cp "gs://$GCS_BUCKET/models/ltx-2-19b-dev.safetensors" "$LTX2_MODEL" 2>/dev/null || {
        echo "Downloading from HuggingFace..."
        if [ ! -z "$HF_TOKEN" ]; then
            huggingface-cli login --token "$HF_TOKEN"
        fi
        huggingface-cli download Lightricks/LTX-Video-2B --include "*.safetensors" --local-dir "$MODEL_DIR"
    }
fi

if [ ! -d "$GEMMA_DIR" ]; then
    gsutil -m cp -r "gs://$GCS_BUCKET/models/gemma/" "$GEMMA_DIR/" 2>/dev/null || {
        echo "Downloading Gemma from HuggingFace..."
        huggingface-cli download google/gemma-2-2b --local-dir "$GEMMA_DIR"
    }
fi

# =============================================================================
# Create Training Config
# =============================================================================
echo ""
echo "=== Creating Training Config ==="

CLOUD_CONFIG="$TRAINER_PATH/configs/vertex_training.yaml"
cat > "$CLOUD_CONFIG" << EOF
# OmniTransfer Vertex AI Training Config - Auto-generated
model:
  model_path: $LTX2_MODEL
  text_encoder_path: $GEMMA_DIR
  training_mode: lora

lora:
  rank: 64
  alpha: 64
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
  dropout: 0.05

data:
  preprocessed_data_root: $DATA_DIR
  num_dataloader_workers: 4

training_strategy:
  name: omnitransfer
  multi_task_mode: true
  task_types: [effect, motion, camera, id, style]
  task_sampling: uniform
  task_type: effect
  i2v_mode: true
  first_frame_latents_dir: target_image_latents
  reference_latents_dir: reference_latents
  enable_tpb: true
  enable_rcl: true
  enable_tma: false
  tpb_max_pos: [20, 2048, 2048]
  tpb_theta: 10000.0
  rcl_ref_timestep: 0.0
  first_frame_conditioning_p: 0.1
  target_loss_weight: 1.0
  min_snr_gamma: 5.0
  log_reconstructions: true
  reconstruction_log_interval: 500

optimization:
  steps: $TRAINING_STEPS
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  scheduler_type: cosine
  optimizer_type: adamw
  weight_decay: 0.001
  enable_gradient_checkpointing: true
  max_grad_norm: 1.0

acceleration:
  mixed_precision_mode: bf16
  quantization: int8-quanto

checkpoints:
  interval: $CHECKPOINT_INTERVAL
  keep_last_n: 5

flow_matching:
  timestep_sampling_mode: shifted_logit_normal

validation:
  prompts: []
  interval: null

wandb:
  enabled: true
  project: ${WANDB_PROJECT:-omnitransfer-vertex}
  tags: ["vertex-ai", "gcp", "unified", "5-task"]

seed: 42
output_dir: $OUTPUT_DIR
EOF

echo "Config written to: $CLOUD_CONFIG"

# =============================================================================
# Setup Checkpoint Upload (background process)
# =============================================================================
if [ ! -z "$GCS_BUCKET" ]; then
    (
        while true; do
            sleep 1800  # 30 minutes
            if [ -d "$OUTPUT_DIR" ]; then
                echo "[$(date)] Syncing checkpoints to GCS..."
                gsutil -m rsync -r "$OUTPUT_DIR" "gs://$GCS_BUCKET/$GCS_CHECKPOINT_PREFIX/" || true
            fi
        done
    ) &
    echo "Checkpoint sync process started (every 30 min)"
fi

# =============================================================================
# Start Training
# =============================================================================
echo ""
echo "=== Starting OmniTransfer Training ==="
echo "Config: $CLOUD_CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$TRAINER_PATH"
mkdir -p "$OUTPUT_DIR"

# Run training
uv run python scripts/train.py "$CLOUD_CONFIG" 2>&1 | tee "$OUTPUT_DIR/training.log"

# Final checkpoint upload
if [ ! -z "$GCS_BUCKET" ]; then
    echo ""
    echo "=== Uploading Final Checkpoints ==="
    gsutil -m rsync -r "$OUTPUT_DIR" "gs://$GCS_BUCKET/$GCS_CHECKPOINT_PREFIX/"
fi

echo ""
echo "=== Training Complete ==="
echo "Finished at: $(date)"
