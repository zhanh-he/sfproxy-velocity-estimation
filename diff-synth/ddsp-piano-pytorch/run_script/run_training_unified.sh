#!/bin/bash
set -euo pipefail
cd ..

SEGMENT_SECONDS=${1:-10}
WORKSPACE_DIR=${2:-"../../202601_midisemi_data/ddsp-piano-pytorch/workspaces_unified_${SEGMENT_SECONDS}s"}
CACHE_DIR="$WORKSPACE_DIR/data_cache"
EXP_DIR="$WORKSPACE_DIR/models"
WANDB_PROJECT=${WANDB_PROJECT:-ddsp-piano-unified}
NUM_WORKERS=${NUM_WORKERS:-8}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-2000}

python3 train.py \
  --batch_size 6 \
  --epochs 7 \
  --lr 0.001 \
  --phase 1 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration "$SEGMENT_SECONDS" \
  --save_interval "$CHECKPOINT_INTERVAL" \
  --num_workers "$NUM_WORKERS" \
  "$CACHE_DIR" \
  "$EXP_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "phase1_${SEGMENT_SECONDS}s"
