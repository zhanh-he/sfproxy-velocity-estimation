#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/sfproxy_profile.sh"

WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data"
DATA_ROOT="${WORKSPACE_BASE}/synth-proxy/data"

INSTRUMENT="${INSTRUMENT:-piano}"
PIANO_DATASET="${PIANO_DATASET:-maestro}"
GUITAR_DATASET="${GUITAR_DATASET:-francoisleduc}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-2}"
TRAIN_PRESET="${TRAIN_PRESET:-mixed_v2}"

EXTRA_TRAIN_OVERRIDES="${EXTRA_TRAIN_OVERRIDES:-}"
BOUNDARY_MODE="default"

segment_tag() {
  local value="$1"
  if [[ "${value}" == *.* ]]; then
    value="${value%0}"
    value="${value%.}"
    value="${value//./p}"
  fi
  echo "${value}s"
}

case "${TRAIN_PRESET}" in
  boundary_v2|coverage_v2|realism_v2|stress_v2|mixed_v2)
    ;;
  *)
    echo "Unsupported TRAIN_PRESET='${TRAIN_PRESET}'." >&2
    exit 1
    ;;
esac

sfproxy_set_profile "${INSTRUMENT}" "${PIANO_DATASET}" "${GUITAR_DATASET}"

INSTRUMENT="${INSTRUMENT}" \
PIANO_DATASET="${PIANO_DATASET}" \
GUITAR_DATASET="${GUITAR_DATASET}" \
MIX_WEIGHTS="${MIX_WEIGHTS:-}" \
SEGMENT_SECONDS="${SEGMENT_SECONDS}" \
"${SCRIPT_DIR}/preprocess_sfproxy_data.sh"

cd "${ROOT_DIR}"
read -r -a extra_train_args <<< "${EXTRA_TRAIN_OVERRIDES}"
mix_train_args=()
seg_tag="$(segment_tag "${SEGMENT_SECONDS}")"
train_data_dir="${DATA_ROOT}/${INSTRUMENT_NAME}/${TRAIN_PRESET}_${seg_tag}_${BOUNDARY_MODE}/train"
val_data_dir="${DATA_ROOT}/${INSTRUMENT_NAME}/${TRAIN_PRESET}_${seg_tag}_${BOUNDARY_MODE}/val"

if [[ ! -f "${train_data_dir}/configs.pkl" ]]; then
  echo "Missing training dataset: ${train_data_dir}" >&2
  exit 1
fi

if [[ ! -f "${val_data_dir}/configs.pkl" ]]; then
  echo "Missing validation dataset: ${val_data_dir}" >&2
  exit 1
fi

if [[ -n "${MIX_WEIGHTS:-}" ]]; then
  read -r -a mix_weights <<< "${MIX_WEIGHTS//,/ }"
  if [[ "${#mix_weights[@]}" -ne 4 ]]; then
    echo "MIX_WEIGHTS must provide 4 values: boundary coverage realism stress" >&2
    exit 1
  fi
  mix_train_args=(
    "sampler_mix.boundary=${mix_weights[0]}"
    "sampler_mix.coverage=${mix_weights[1]}"
    "sampler_mix.realism=${mix_weights[2]}"
    "sampler_mix.stress=${mix_weights[3]}"
  )
fi

echo "train ${TRAIN_PRESET} ${INSTRUMENT_NAME} ${SEGMENT_SECONDS}s ${BOUNDARY_MODE}"

python "${ROOT_DIR}/synth-proxy/src/train.py" \
  --config-name train \
  "paths.repo_root=${ROOT_DIR}" \
  "sampler_preset=${TRAIN_PRESET}" \
  "segment_seconds=${SEGMENT_SECONDS}" \
  "boundary_mode=${BOUNDARY_MODE}" \
  "dataset.name=${DATASET_NAME}" \
  "dataset.train.path=${train_data_dir}" \
  "dataset.val.path=${val_data_dir}" \
  "dataset.train.instrument_name=${INSTRUMENT_NAME}" \
  "dataset.val.instrument_name=${INSTRUMENT_NAME}" \
  "reset_output_dir=true" \
  "${mix_train_args[@]}" \
  "${extra_train_args[@]}"
