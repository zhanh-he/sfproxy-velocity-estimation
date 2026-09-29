#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <francoisleduc|gaps> <segment_seconds> <dataset_path> <output_dir>"
  exit 1
fi

DATASET_TYPE="$1"
SEGMENT_SECONDS="$2"
DATASET_PATH="$3"
OUTPUT_DIR="$4"

case "${DATASET_TYPE}" in
  francoisleduc)
    python prepare_flgd_unified.py \
      --francoisleduc_path "${DATASET_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --segment_seconds "${SEGMENT_SECONDS}" \
      --sample_rate 22050 \
      --frame_rate 100
    ;;
  gaps)
    python prepare_gaps_unified.py \
      --gaps_path "${DATASET_PATH}" \
      --output_dir "${OUTPUT_DIR}" \
      --segment_seconds "${SEGMENT_SECONDS}" \
      --sample_rate 22050 \
      --frame_rate 100
    ;;
  *)
    echo "Unsupported dataset '${DATASET_TYPE}'. Expected 'francoisleduc' or 'gaps'." >&2
    exit 1
    ;;
esac
