#!/bin/bash
# =============================================================================
# OmniTransfer Training on Vast.ai
# =============================================================================
# Run this on the Vast.ai instance after setup completes
# Usage: bash train_omnitransfer.sh [--resume /path/to/checkpoint]
# =============================================================================

set -e

echo "=== OmniTransfer Training on Vast.ai ==="
echo "Started at: $(date)"
echo ""

# Configuration from environment
S3_BUCKET=${S3_BUCKET:-"omnitransfer-training"}
S3_REGION=${S3_REGION:-"us-east-1"}
S3_DATA_PREFIX=${S3_DATA_PREFIX:-"processed/omnitransfer_unified_5task"}
TRAINING_CONFIG=${TRAINING_CONFIG:-"ltx2_omnitransfer_unified_5task.yaml"}
TRAINING_STEPS=${TRAINING_STEPS:-10000}
MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS:-24}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}

WORKSPACE="/workspace"
REPO_PATH="$WORKSPACE/OmniTransfer-hack"
TRAINER_PATH="$REPO_PATH/packages/ltx-trainer"
DATA_DIR="/data/omnitransfer_training"
MODEL_DIR="/data/models"
OUTPUT_DIR="$TRAINER_PATH/outputs/omnitransfer_cloud"
CHECKPOINT_S3_PREFIX="checkpoints/omnitransfer"

# Parse arguments
RESUME_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# =============================================================================
# Setup auto-shutdown watchdog + checkpoint upload
# =============================================================================
if [ "$MAX_RUNTIME_HOURS" != "0" ]; then
    MAX_SECONDS=$((MAX_RUNTIME_HOURS * 3600))
    echo "Auto-shutdown in $MAX_RUNTIME_HOURS hours"
    (
        sleep $MAX_SECONDS
        echo "=== MAX RUNTIME REACHED ==="
        # Upload final checkpoints
        if [ -d "$OUTPUT_DIR" ]; then
            aws s3 sync $OUTPUT_DIR s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/ --region $S3_REGION || true
        fi
        poweroff || exit 0
    ) &
fi

# =============================================================================
# Setup periodic checkpoint upload (every 30 minutes)
# =============================================================================
(
    while true; do
        sleep 1800  # 30 minutes
        if [ -d "$OUTPUT_DIR" ]; then
            echo "[$(date)] Syncing checkpoints to S3..."
            aws s3 sync $OUTPUT_DIR s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/ \
                --region $S3_REGION \
                --exclude "*.log" \
                --include "*.safetensors" \
                --include "*.json" \
                --include "*.yaml" \
                || true
        fi
    done
) &
CHECKPOINT_SYNC_PID=$!
echo "Checkpoint sync process: $CHECKPOINT_SYNC_PID"

# =============================================================================
# Configure AWS
# =============================================================================
echo ""
echo "=== Configuring AWS ==="
aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set region $S3_REGION

# Test S3 access
aws s3 ls s3://$S3_BUCKET/ --region $S3_REGION > /dev/null 2>&1 || {
    echo "Creating S3 bucket..."
    aws s3 mb s3://$S3_BUCKET --region $S3_REGION || true
}
echo "S3 bucket ready: s3://$S3_BUCKET"

# =============================================================================
# Download Training Data from S3
# =============================================================================
echo ""
echo "=== Downloading Training Data ==="
mkdir -p $DATA_DIR

echo "Downloading from s3://$S3_BUCKET/$S3_DATA_PREFIX/ ..."
aws s3 sync s3://$S3_BUCKET/$S3_DATA_PREFIX/ $DATA_DIR/ --region $S3_REGION

# Check what we downloaded
echo "Training data contents:"
for dir in $DATA_DIR/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f -name "*.pt" 2>/dev/null | wc -l)
        echo "  $(basename "$dir"): $count files"
    fi
done

# =============================================================================
# Download LTX-2 Model Weights
# =============================================================================
echo ""
echo "=== Checking Model Weights ==="
mkdir -p $MODEL_DIR

LTX2_MODEL="$MODEL_DIR/ltx-2-19b-dev.safetensors"
GEMMA_DIR="$MODEL_DIR/gemma"

# Check if model exists locally or download from HF
if [ ! -f "$LTX2_MODEL" ]; then
    echo "Downloading LTX-2 19B model..."
    if [ ! -z "$HF_TOKEN" ]; then
        huggingface-cli login --token $HF_TOKEN
    fi

    # Try to download from S3 first (faster if you've cached it)
    aws s3 cp s3://$S3_BUCKET/models/ltx-2-19b-dev.safetensors $LTX2_MODEL --region $S3_REGION 2>/dev/null || {
        echo "Downloading from HuggingFace..."
        pip install huggingface_hub -q
        huggingface-cli download Lightricks/LTX-Video-2B --include "*.safetensors" --local-dir $MODEL_DIR
    }
fi

if [ ! -d "$GEMMA_DIR" ]; then
    echo "Downloading Gemma text encoder..."
    aws s3 sync s3://$S3_BUCKET/models/gemma/ $GEMMA_DIR/ --region $S3_REGION 2>/dev/null || {
        echo "Downloading from HuggingFace..."
        huggingface-cli download google/gemma-2-2b --local-dir $GEMMA_DIR
    }
fi

echo "Model weights ready"

# =============================================================================
# Create Cloud Training Config
# =============================================================================
echo ""
echo "=== Creating Training Config ==="
CONFIG_PATH="$TRAINER_PATH/configs/$TRAINING_CONFIG"

# Create a modified config for cloud training with correct paths
CLOUD_CONFIG="$TRAINER_PATH/configs/cloud_training.yaml"
cat > $CLOUD_CONFIG << EOF
# OmniTransfer Cloud Training Config - Auto-generated
# Based on: $TRAINING_CONFIG

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
  task_types:
    - effect
    - motion
    - camera
    - id
    - style
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
  ref_preservation_loss_weight: 0.0
  min_snr_gamma: 5.0
  lpips_weight: 0.0
  style_loss_weight: 0.0
  log_reconstructions: true
  reconstruction_log_interval: 500
  num_frames_to_visualize: 8
  max_samples_per_log: 2
  log_video_comparisons: true
  video_log_interval: 2000

optimization:
  steps: $TRAINING_STEPS
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  scheduler_type: cosine
  scheduler_params: {}
  optimizer_type: adamw
  weight_decay: 0.001
  enable_gradient_checkpointing: true
  max_grad_norm: 1.0

acceleration:
  mixed_precision_mode: bf16
  quantization: int8-quanto
  load_text_encoder_in_8bit: false

checkpoints:
  interval: $CHECKPOINT_INTERVAL
  keep_last_n: 5

flow_matching:
  timestep_sampling_mode: shifted_logit_normal
  timestep_sampling_params: {}

validation:
  prompts: []
  negative_prompt: "worst quality, blurry, distorted"
  video_dims: [832, 448, 65]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: null
  videos_per_prompt: 1
  guidance_scale: 4.0
  generate_audio: false

wandb:
  enabled: true
  project: ${WANDB_PROJECT:-omnitransfer-cloud}
  tags: ["cloud", "vast-ai", "unified", "5-task", "tpb", "rcl"]

seed: 42
output_dir: $OUTPUT_DIR
EOF

echo "Config written to: $CLOUD_CONFIG"

# =============================================================================
# Configure W&B
# =============================================================================
if [ ! -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "=== Configuring Weights & Biases ==="
    wandb login --relogin $WANDB_API_KEY
    echo "W&B configured for project: ${WANDB_PROJECT:-omnitransfer-cloud}"
fi

# =============================================================================
# Check for Resume
# =============================================================================
RESUME_FLAG=""
if [ ! -z "$RESUME_PATH" ]; then
    echo ""
    echo "=== Resuming from checkpoint ==="
    echo "Checkpoint: $RESUME_PATH"
    RESUME_FLAG="--resume $RESUME_PATH"
elif [ -d "$OUTPUT_DIR" ]; then
    # Check for latest checkpoint
    LATEST=$(ls -td $OUTPUT_DIR/checkpoint-* 2>/dev/null | head -1)
    if [ ! -z "$LATEST" ]; then
        echo ""
        echo "=== Found existing checkpoint ==="
        echo "Latest: $LATEST"
        read -p "Resume from this checkpoint? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            RESUME_FLAG="--resume $LATEST"
        fi
    fi
fi

# =============================================================================
# Start Training
# =============================================================================
echo ""
echo "=== Starting OmniTransfer Training ==="
echo "Config: $CLOUD_CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Steps: $TRAINING_STEPS"
echo ""

cd $TRAINER_PATH
mkdir -p $OUTPUT_DIR

# Run training in tmux for persistence
tmux new-session -d -s training "cd $TRAINER_PATH && uv run python scripts/train.py $CLOUD_CONFIG $RESUME_FLAG 2>&1 | tee $OUTPUT_DIR/training.log"

echo "Training started in tmux session 'training'"
echo ""
echo "=== Monitoring ==="
echo "  Attach to training: tmux attach -t training"
echo "  View logs: tail -f $OUTPUT_DIR/training.log"
echo "  GPU usage: watch -n1 nvidia-smi"
echo ""
echo "=== Checkpoints ==="
echo "  Local: $OUTPUT_DIR"
echo "  S3: s3://$S3_BUCKET/$CHECKPOINT_S3_PREFIX/"
echo ""

# Attach to the session
echo "Attaching to training session (Ctrl+B, D to detach)..."
sleep 2
tmux attach -t training
