#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <segment_seconds> <prepared_data_dir> <output_dir>"
  exit 1
fi

SEGMENT_SECONDS="$1"
PREPARED_DIR="$2"
OUTPUT_DIR="$3"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-1e-3}"
SCHEDULER_STEP_SIZE="${SCHEDULER_STEP_SIZE:-300}"
SCHEDULER_GAMMA="${SCHEDULER_GAMMA:-1.0}"

SEG_TAG="${SEGMENT_SECONDS%.*}"
TRAIN_DATASET="${PREPARED_DIR}/train_flgd_midi_${SEG_TAG}s.npz"
VAL_DATASET="${PREPARED_DIR}/val_flgd_midi_${SEG_TAG}s.npz"

python train_midi_synth_unified.py \
  --train_dataset_path "${TRAIN_DATASET}" \
  --val_dataset_path "${VAL_DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds "${SEGMENT_SECONDS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --scheduler_step_size "${SCHEDULER_STEP_SIZE}" \
  --scheduler_gamma "${SCHEDULER_GAMMA}" \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
