#!/usr/bin/env bash
set -euo pipefail

# Manual config.
DEFAULT_MACHINE="5090"            # 3090 or 5090
DEFAULT_SEGMENT_SECONDS="2"       # 2 / 5 / 10

SEGMENT_SECONDS="${1:-${DEFAULT_SEGMENT_SECONDS}}"
SEGMENT_TAG="${SEGMENT_SECONDS%.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${ROOT_DIR}/synthesizer/ddsp-piano-pytorch"
MACHINE="${MACHINE:-${DEFAULT_MACHINE}}"

case "${MACHINE}" in
  3090)
    DEFAULT_MAESTRO_DIR="/media/datadisk/home/22828187/zhanh/Dataset/maestro-v3.0.0"
    DEFAULT_WORKSPACE_BASE="/media/datadisk/home/22828187/zhanh/202601_midisemi_data/ddsp-piano-pytorch"
    ;;
  5090)
    DEFAULT_MAESTRO_DIR="/media/mengh/SharedData/zhanh/Dataset/maestro-v3.0.0"
    DEFAULT_WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data/ddsp-piano-pytorch"
    ;;
  *)
    echo "Unsupported MACHINE='${MACHINE}'. Expected '3090' or '5090'."
    exit 1
    ;;
esac

MAESTRO_DIR="${MAESTRO_DIR:-${DEFAULT_MAESTRO_DIR}}"
WORKSPACE_BASE="${WORKSPACE_BASE:-${DEFAULT_WORKSPACE_BASE}}"
WORKSPACE_DIR="${WORKSPACE_BASE}/workspaces_unified_${SEGMENT_TAG}s"
CACHE_DIR="${WORKSPACE_DIR}/data_cache"
EXP_DIR="${WORKSPACE_DIR}/models"

mkdir -p "${CACHE_DIR}" "${EXP_DIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -f "${MAESTRO_DIR}/maestro-v3.0.0.csv" ]; then
  echo "MAESTRO metadata file not found under: ${MAESTRO_DIR}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Please activate the correct environment first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if [ ! -f "${CACHE_DIR}/cache_config.json" ] || [ ! -d "${CACHE_DIR}/train" ] || [ ! -d "${CACHE_DIR}/validation" ] || [ ! -d "${CACHE_DIR}/test" ]; then
  echo "Prepared piano cache not found. Running preprocessing first."
  python preprocess.py \
    --splits train validation test \
    --segment_duration "${SEGMENT_SECONDS}" \
    --sample_rate 22050 \
    --frame_rate 100 \
    --max_polyphony 16 \
    "${MAESTRO_DIR}" \
    "${CACHE_DIR}"
else
  echo "Prepared piano cache found. Skipping preprocessing."
fi

python train.py \
  --phase 1 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration "${SEGMENT_SECONDS}" \
  "${CACHE_DIR}" \
  "${EXP_DIR}"
