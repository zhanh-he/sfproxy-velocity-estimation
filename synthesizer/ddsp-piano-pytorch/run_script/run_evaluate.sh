#!/bin/bash
cd .. # Run from repo root for Python imports

# ===== Paths =====
WORKSPACE_DIR="../../202601_midisemi_data/ddsp-piano-pytorch/workspaces"
maestro_cache_path="$WORKSPACE_DIR/data_cache"

# ===== Basic options =====
phase=2                 # 1/2/3
split="validation"      # train/validation/test
ckpt_order="step"      # epoch/step
wandb_project='ddsp-piano'
wandb_run_name="eval_phase${phase}_${split}_in_${ckpt_order}"

# Evaluate a directory of ckpts (recommended) or a single file
checkpoint="$WORKSPACE_DIR/models/phase_${phase}/ckpts"
# checkpoint="$WORKSPACE_DIR/models/phase_${phase}/ckpts/ddsp-piano_epoch_2_params.pt"

# Output dir
output_dir="$WORKSPACE_DIR/models/eval_phase_${phase}_${split}"

# Runtime
batch_size=6
num_workers=8
use_cuda=1

echo "Evaluating: phase=$phase, split=$split"
echo "Checkpoint path: $checkpoint"
echo "Order: $ckpt_order, output: $output_dir"


python3 evaluate.py \
  --checkpoint "$checkpoint" \
  --output_dir "$output_dir" \
  --phase "$phase" \
  --split "$split" \
  --ckpt_order "$ckpt_order" \
  --batch_size "$batch_size" \
  --num_workers "$num_workers" \
  --cuda "$use_cuda" \
  --wandb_project "$wandb_project" --wandb_run_name "$wandb_run_name" \
  $save_audio_flag \
  $debug_flag \
  "$maestro_cache_path" \
  --debug_mode        # Debug mode: limit evaluation to 20 batches when set
  # --save_audio      # Optional toggles (leave empty to disable)