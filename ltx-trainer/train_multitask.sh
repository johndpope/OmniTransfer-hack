#!/bin/bash
# Train OmniTransfer on multiple task types sequentially
# Each task will generate reconstruction images for visualization

set -e

PROCESSED_ROOT="/media/2TB/omnitransfer_multitask_processed"
OUTPUT_BASE="outputs/ltx2_omnitransfer_multitask"
CONFIG_TEMPLATE="configs/ltx2_omnitransfer_lora.yaml"

# Task types to train
TASK_TYPES=("identity_preservation" "style_transfer" "motion_transfer")

for task in "${TASK_TYPES[@]}"; do
    echo "=============================================="
    echo "Training task: $task"
    echo "=============================================="

    DATA_ROOT="$PROCESSED_ROOT/$task"
    OUTPUT_DIR="$OUTPUT_BASE/${task}"

    # Check if data exists
    if [ ! -d "$DATA_ROOT" ]; then
        echo "Skipping $task - data not found at $DATA_ROOT"
        continue
    fi

    # Run training with overrides
    uv run python scripts/train.py "$CONFIG_TEMPLATE" \
        --data.preprocessed_data_root "$DATA_ROOT" \
        --training_strategy.task_type "$task" \
        --output_dir "$OUTPUT_DIR" \
        --wandb.tags "omnitransfer" "ltx2" "lora" "$task" "multitask" \
        --optimization.steps 500 \
        --training_strategy.reconstruction_log_interval 50 \
        --checkpoints.interval 100

    echo "Completed: $task"
    echo ""
done

echo "=============================================="
echo "All task types trained!"
echo "=============================================="
