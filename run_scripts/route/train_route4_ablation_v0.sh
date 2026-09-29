#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/score_hpt_profile.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="${ROOT_DIR}/score_hpt"
PROJECT_ROOT="${ROOT_DIR}/synth-proxy"

# frontend
TRAIN_SET="${TRAIN_SET:-maestro}"
MODEL_TYPE="${MODEL_TYPE:-hpt}"
SCORE_METHOD="${SCORE_METHOD:-note_editor}"
FRONTEND_PRETRAINED="${FRONTEND_PRETRAINED:-}"

# backend
SAMPLERS="${SAMPLERS:-coverage mixed realism}"
LOSS_TYPES="${LOSS_TYPES:-smooth_l1 l1 mse}"
PROXY_CKPT="${PROXY_CKPT:-${1:-}}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
BACKEND_WEIGHT="${BACKEND_WEIGHT:-1.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"

# evaluation / misc
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

score_hpt_init_context
score_hpt_set_dataset_profile "${TRAIN_SET}"

if [ "${DATASET_KIND}" = "guitar" ]; then
  DEFAULT_SEGMENT_LIST="2 5"
else
  DEFAULT_SEGMENT_LIST="2 5 10"
fi

# backend defaults derived from dataset profile
SEGMENT_LIST="${SEGMENT_LIST:-${DEFAULT_SEGMENT_LIST}}"
BOUNDARY_MODE="default"
INSTRUMENT_NAME="${SFPROXY_INSTRUMENT_NAME_DEFAULT}"
SFPROXY_DATASET_NAME="${SFPROXY_DATASET_NAME_DEFAULT}"
SFPROXY_CKPT_ROOT="${DATA_ROOT}/synth-proxy/proxy/checkpoints/${INSTRUMENT_NAME}"

sampler_preset() {
  case "$1" in
    coverage|coverage_v2) echo "coverage_v2" ;;
    mixed|mixed_v2) echo "mixed_v2" ;;
    realism|realism_v2) echo "realism_v2" ;;
    stress|stress_v2) echo "stress_v2" ;;
    *) echo "$1" ;;
  esac
}

resolve_sfproxy_ckpt() {
  local sampler="$1"
  local segment="$2"
  local seg_tag
  local preset
  local dir
  local latest

  seg_tag="$(score_hpt_segment_tag "${segment}")"
  preset="$(sampler_preset "${sampler}")"

  for dir in "${SFPROXY_CKPT_ROOT}"/"${SFPROXY_DATASET_NAME}"_"${INSTRUMENT_NAME}"_"${preset}"*_"${seg_tag}"_"${BOUNDARY_MODE}"; do
    [ -d "${dir}" ] || continue
    latest="$(find "${dir}" -maxdepth 1 -type f -name '*last*.ckpt' | sort -V | tail -n 1)"
    [ -n "${latest}" ] && printf '%s\n' "${latest}" && return 0
    latest="$(find "${dir}" -maxdepth 1 -type f -name '*loss*.ckpt' | sort -V | tail -n 1)"
    [ -n "${latest}" ] && printf '%s\n' "${latest}" && return 0
    latest="$(find "${dir}" -maxdepth 1 -type f -name '*.ckpt' | sort -V | tail -n 1)"
    [ -n "${latest}" ] && printf '%s\n' "${latest}" && return 0
  done
}

mkdir -p "${WORKSPACE_DIR}"
read -r -a extra_args <<< "${EXTRA_OVERRIDES}"

cd "${PROJECT_DIR}"

required_datasets=("${TRAIN_SET}" "${DEFAULT_TEST_SET}")
while IFS= read -r dataset_name; do
  required_datasets+=("${dataset_name}")
done < <(score_hpt_collect_eval_datasets "${DEFAULT_EVAL_SETS}")

score_hpt_prepare_required_datasets "${PYTHON_BIN}" "${required_datasets[@]}"
score_hpt_set_dataset_profile "${TRAIN_SET}"

run_one() {
  local segment="$1"
  local sampler="$2"
  local ckpt="$3"
  local loss="$4"
  local score_method
  local model_input2

  score_method="${SCORE_METHOD}"
  model_input2="null"
  if [ "${MODEL_TYPE}" = "hpt" ] && [ "${score_method}" = "note_editor" ]; then
    model_input2="onset"
  fi
  local pretrained_args=()
  if [ -n "${FRONTEND_PRETRAINED}" ]; then
    pretrained_args+=("model.frontend_pretrained_mode=route2_piano_specific")
    pretrained_args+=("model.frontend_pretrained=${FRONTEND_PRETRAINED}")
  fi

  echo "============================================================"
  echo "Route IV ablation"
  echo "Train set         : ${TRAIN_SET}"
  echo "Test set          : ${DEFAULT_TEST_SET}"
  echo "Model             : ${MODEL_TYPE}"
  echo "Score method      : ${score_method}"
  echo "Backend checkpoint: ${ckpt}"
  echo "Backend seg (s)   : ${segment}"
  echo "Sampler           : ${sampler}"
  echo "Backend loss      : ${loss}"
  echo "Instrument name   : ${INSTRUMENT_NAME}"
  echo "============================================================"

  "${PYTHON_BIN}" pytorch/train_proxy.py \
    "exp.workspace=${WORKSPACE_DIR}" \
    "dataset.train_set=${TRAIN_SET}" \
    "dataset.test_set=${DEFAULT_TEST_SET}" \
    "dataset.eval_sets=${DEFAULT_EVAL_SETS}" \
    "model.type=${MODEL_TYPE}" \
    "model.input2=${model_input2}" \
    "score_informed.method=${score_method}" \
    "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
    "loss.backend_weight=${BACKEND_WEIGHT}" \
    "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
    "backend.enabled=true" \
    "backend.type=diffproxy" \
    "backend.project_root=${PROJECT_ROOT}" \
    "backend.checkpoint=${ckpt}" \
    "backend.backend_segment_seconds=${segment}" \
    "backend.diffproxy.instrument_name=${INSTRUMENT_NAME}" \
    "backend.diffproxy.loss_type=${loss}" \
    "${pretrained_args[@]}" \
    "${extra_args[@]}"
}

if [ -n "${PROXY_CKPT}" ]; then
  for segment in ${SEGMENT_LIST}; do
    for loss in ${LOSS_TYPES}; do
      run_one "${segment}" "manual" "${PROXY_CKPT}" "${loss}"
    done
  done
  exit 0
fi

for segment in ${SEGMENT_LIST}; do
  for sampler in ${SAMPLERS}; do
    ckpt="$(resolve_sfproxy_ckpt "${sampler}" "${segment}")"
    for loss in ${LOSS_TYPES}; do
      run_one "${segment}" "${sampler}" "${ckpt}" "${loss}"
    done
  done
done
