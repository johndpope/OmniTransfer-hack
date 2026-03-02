#!/bin/bash
# Automated pipeline: wait for training → preprocess → evolve
# Usage: bash scripts/run_evolution_pipeline.sh <training_pid>

set -e
TRAIN_PID=${1:?Usage: run_evolution_pipeline.sh <training_pid>}
LORA_PATH="/media/2TB/omnitransfer/output/scd_distilled_perframe/checkpoints/lora_weights_step_02000.safetensors"
EVOLUTION_CONFIG="configs/ltx2_scd_evolution_distilled.yaml"

export TORCH_CUDA_ARCH_LIST="12.0"
export PYTHONUNBUFFERED=1

echo "============================================="
echo "SCD Evolution Pipeline"
echo "============================================="
echo "Waiting for training PID $TRAIN_PID to complete..."

# ── Phase 0: Wait for training ──
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    STEP=$(cat /media/2TB/omnitransfer/output/scd_distilled_perframe/debug_info.txt 2>/dev/null | head -1 | awk '{print $2}')
    echo "  [$(date +%H:%M)] Training step: ${STEP:-?}/2000"
    sleep 300
done

echo ""
echo "Training process finished at $(date)"

# Verify checkpoint exists
if [ ! -f "$LORA_PATH" ]; then
    echo "ERROR: Expected checkpoint not found: $LORA_PATH"
    echo "Checking available checkpoints..."
    ls -la /media/2TB/omnitransfer/output/scd_distilled_perframe/checkpoints/
    # Use latest available checkpoint
    LATEST=$(ls -t /media/2TB/omnitransfer/output/scd_distilled_perframe/checkpoints/lora_weights_step_*.safetensors 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo "No checkpoints found at all. Aborting."
        exit 1
    fi
    echo "Using latest checkpoint: $LATEST"
    LORA_PATH="$LATEST"
fi

echo "Using LoRA checkpoint: $LORA_PATH"
echo ""

# ── Phase 1: Preprocess scrya-downloads ──
echo "============================================="
echo "Phase 1: Preprocessing scrya-downloads data"
echo "============================================="
cd /home/johndpope/Documents/GitHub/ltx2-omnitransfer/ltx-trainer

python3 -u scripts/preprocess_scrya_evolution.py 2>&1 | tee /tmp/scrya_preprocess.log

# Check if merged dataset exists
if [ ! -d "/media/2TB/omnitransfer/data/evolution_merged/latents" ]; then
    echo "WARNING: Merged dataset not created. Falling back to ditto_subset."
    EVOLUTION_CONFIG="configs/ltx2_scd_evolution_distilled.yaml"
    # Override data root via CLI
    DATA_ROOT_FLAG="--data-root /media/2TB/omnitransfer/data/ditto_subset"
else
    MERGED_COUNT=$(ls /media/2TB/omnitransfer/data/evolution_merged/latents/*.pt 2>/dev/null | wc -l)
    echo "Merged dataset ready: $MERGED_COUNT samples"
    DATA_ROOT_FLAG=""
fi

echo ""

# ── Phase 2: Run evolution ──
echo "============================================="
echo "Phase 2: SCD Evolution (gradient-free)"
echo "============================================="
echo "Config: $EVOLUTION_CONFIG"
echo "LoRA:   $LORA_PATH"
echo "Start:  $(date)"

python3 -u scripts/evolve_scd.py \
    --config "$EVOLUTION_CONFIG" \
    --lora-path "$LORA_PATH" \
    $DATA_ROOT_FLAG \
    2>&1 | tee /tmp/scd_evolution.log

echo ""
echo "============================================="
echo "Pipeline complete at $(date)"
echo "============================================="
echo "Evolved LoRA: /media/2TB/omnitransfer/output/scd_evolution_distilled/checkpoints/lora_evolved_final.safetensors"
echo ""
echo "To verify quality:"
echo "  python scripts/scd_inference.py \\"
echo "    --lora-path /media/2TB/omnitransfer/output/scd_evolution_distilled/checkpoints/lora_evolved_final.safetensors \\"
echo "    --distilled --num-seconds 30 \\"
echo "    --cached-embedding /media/2TB/omnitransfer/data/ditto_subset/conditions_final/000000.pt \\"
echo "    --output /media/2TB/omnitransfer/inference/scd_evolved_30s.mp4"
