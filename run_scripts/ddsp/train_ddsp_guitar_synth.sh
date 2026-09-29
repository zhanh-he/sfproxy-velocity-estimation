#!/usr/bin/env bash
set -euo pipefail

# Manual config.
DEFAULT_MACHINE="3090"            # 3090 or 5090
DEFAULT_SEGMENT_SECONDS="2"       # 2 / 5 / 10
DEFAULT_GUITAR_DATASET="francoisleduc"  # francoisleduc or gaps

SEGMENT_SECONDS="${1:-${DEFAULT_SEGMENT_SECONDS}}"
SEGMENT_TAG="${SEGMENT_SECONDS%.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${ROOT_DIR}/synthesizer/ddsp-guitar-synth"
MACHINE="${MACHINE:-${DEFAULT_MACHINE}}"
GUITAR_DATASET="${GUITAR_DATASET:-${DEFAULT_GUITAR_DATASET}}"

case "${MACHINE}" in
  3090)
    DEFAULT_FRANCOISLEDUC_DIR="/media/datadisk/home/22828187/zhanh/Dataset/FrancoisLeducGuitarDataset"
    DEFAULT_GAPS_DIR="/media/datadisk/home/22828187/zhanh/Dataset/GAPS"
    DEFAULT_WORKSPACE_BASE="/media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-guitar-synth"
    ;;
  5090)
    DEFAULT_FRANCOISLEDUC_DIR="/media/mengh/SharedData/zhanh/Dataset/FrancoisLeducGuitarDataset"
    DEFAULT_GAPS_DIR="/media/mengh/SharedData/zhanh/Dataset/GAPS"
    DEFAULT_WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data/ddsp-guitar-synth"
    ;;
  *)
    echo "Unsupported MACHINE='${MACHINE}'. Expected '3090' or '5090'."
    exit 1
    ;;
esac

case "${GUITAR_DATASET}" in
  francoisleduc)
    DEFAULT_GUITAR_DATASET_DIR="${DEFAULT_FRANCOISLEDUC_DIR}"
    TRAIN_DATASET_NAME="train_flgd_midi_${SEGMENT_TAG}s.npz"
    VAL_DATASET_NAME="val_flgd_midi_${SEGMENT_TAG}s.npz"
    METADATA_RELATIVE_PATH="metadata.csv"
    ;;
  gaps)
    DEFAULT_GUITAR_DATASET_DIR="${DEFAULT_GAPS_DIR}"
    TRAIN_DATASET_NAME="train_gaps_midi_${SEGMENT_TAG}s.npz"
    VAL_DATASET_NAME="val_gaps_midi_${SEGMENT_TAG}s.npz"
    METADATA_RELATIVE_PATH="gaps_metadata_with_splits.csv"
    ;;
  *)
    echo "Unsupported GUITAR_DATASET='${GUITAR_DATASET}'. Expected 'francoisleduc' or 'gaps'." >&2
    exit 1
    ;;
esac

GUITAR_DATASET_DIR="${GUITAR_DATASET_DIR:-${DEFAULT_GUITAR_DATASET_DIR}}"
WORKSPACE_BASE="${WORKSPACE_BASE:-${DEFAULT_WORKSPACE_BASE}}"
RUN_DIR="${WORKSPACE_BASE}/${GUITAR_DATASET}_${SEGMENT_TAG}s"
DATA_DIR="${RUN_DIR}/data"
OUTPUT_DIR="${RUN_DIR}/output"
TRAIN_DATASET="${DATA_DIR}/${TRAIN_DATASET_NAME}"
VAL_DATASET="${DATA_DIR}/${VAL_DATASET_NAME}"

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

METADATA_PATH="${GUITAR_DATASET_DIR}/${METADATA_RELATIVE_PATH}"

if [ ! -f "${METADATA_PATH}" ]; then
  echo "Dataset metadata not found under: ${GUITAR_DATASET_DIR}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Please activate the correct environment first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [ ! -f "${TRAIN_DATASET}" ] || [ ! -f "${VAL_DATASET}" ]; then
  echo "Prepared datasets not found. Running data preparation first."
  case "${GUITAR_DATASET}" in
    francoisleduc)
      python prepare_flgd_unified.py \
        --francoisleduc_path "${GUITAR_DATASET_DIR}" \
        --output_dir "${DATA_DIR}" \
        --segment_seconds "${SEGMENT_SECONDS}" \
        --sample_rate 22050 \
        --frame_rate 100
      ;;
    gaps)
      python prepare_gaps_unified.py \
        --gaps_path "${GUITAR_DATASET_DIR}" \
        --output_dir "${DATA_DIR}" \
        --segment_seconds "${SEGMENT_SECONDS}" \
        --sample_rate 22050 \
        --frame_rate 100
      ;;
  esac
else
  echo "Prepared datasets found. Skipping data preparation."
fi

python train_midi_synth_unified.py \
  --train_dataset_path "${TRAIN_DATASET}" \
  --val_dataset_path "${VAL_DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds "${SEGMENT_SECONDS}" \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
