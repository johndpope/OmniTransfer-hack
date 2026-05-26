#!/bin/bash
# Persistent training launcher
cd /home/johndpope/Documents/GitHub/OmniTransfer-hack/ltx-trainer
export PYTHONPATH="/home/johndpope/Documents/GitHub/OmniTransfer-hack/packages/ltx-core/src:/home/johndpope/Documents/GitHub/OmniTransfer-hack/packages/ltx-pipelines/src:/home/johndpope/Documents/GitHub/OmniTransfer-hack/ltx-trainer/src:$PYTHONPATH"
LOG=/home/johndpope/Documents/GitHub/OmniTransfer-hack/facebook_reels/mashup_training/training.log

echo "Starting training at $(date)" >> "$LOG"
nohup /home/johndpope/miniconda3/bin/python -u scripts/train.py \
  /home/johndpope/Documents/GitHub/OmniTransfer-hack/facebook_reels/mashup_training/train_config.yaml \
  >> "$LOG" 2>&1 &
PID=$!
echo "PID: $PID" >> "$LOG"
echo "PID: $PID"
echo "W&B: https://wandb.ai/snoozie/omnitransfer-mashup-style"
