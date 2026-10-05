#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
. "${SCRIPT_DIR}/score_hpt_profile.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="${ROOT_DIR}/score_hpt"
PROJECT_ROOT="${ROOT_DIR}/diff-sfproxy"

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
SATURATION_WEIGHT="${SATURATION_WEIGHT:-0.0}"
# New local recipes: sup,backend,prior,saturation. Set WEIGHT_RECIPES="" to fall back to the single weights above.
WEIGHT_RECIPES="${WEIGHT_RECIPES:-0.8,0.1,0.0,0.01 0.7,0.2,0.0,0.01 0.8,0.1,0.01,0.01}"
BACKEND_WARMUP_ITERATIONS="${BACKEND_WARMUP_ITERATIONS:-}"
USE_ONSET_WINDOW_POOLING="${USE_ONSET_WINDOW_POOLING:-}"
ONSET_WINDOW_LEFT="${ONSET_WINDOW_LEFT:-1}"
ONSET_WINDOW_RIGHT="${ONSET_WINDOW_RIGHT:-2}"
ONSET_WINDOW_REDUCTION="${ONSET_WINDOW_REDUCTION:-max}"
SFPROXY_FEATURE_WEIGHTS="${SFPROXY_FEATURE_WEIGHTS:-}"

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
  local supervised_weight="$5"
  local backend_weight="$6"
  local prior_weight="$7"
  local saturation_weight="$8"
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

  local run_args=()
  if [ -n "${BACKEND_WARMUP_ITERATIONS}" ]; then
    run_args+=("backend.warmup_iterations=${BACKEND_WARMUP_ITERATIONS}")
  fi
  if [ -n "${USE_ONSET_WINDOW_POOLING}" ]; then
    run_args+=("backend.diffproxy.use_onset_window_pooling=${USE_ONSET_WINDOW_POOLING}")
    run_args+=("backend.diffproxy.onset_window_left=${ONSET_WINDOW_LEFT}")
    run_args+=("backend.diffproxy.onset_window_right=${ONSET_WINDOW_RIGHT}")
    run_args+=("backend.diffproxy.onset_window_reduction=${ONSET_WINDOW_REDUCTION}")
  fi
  if [ -n "${SFPROXY_FEATURE_WEIGHTS}" ]; then
    run_args+=("backend.diffproxy.feature_weights=${SFPROXY_FEATURE_WEIGHTS}")
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
  echo "Weights (sup/backend/prior/sat): ${supervised_weight}/${backend_weight}/${prior_weight}/${saturation_weight}"
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
    "loss.supervised_weight=${supervised_weight}" \
    "loss.backend_weight=${backend_weight}" \
    "loss.velocity_prior_weight=${prior_weight}" \
    "loss.velocity_saturation_weight=${saturation_weight}" \
    "backend.enabled=true" \
    "backend.type=diffproxy" \
    "backend.project_root=${PROJECT_ROOT}" \
    "backend.checkpoint=${ckpt}" \
    "backend.backend_segment_seconds=${segment}" \
    "backend.diffproxy.instrument_name=${INSTRUMENT_NAME}" \
    "backend.diffproxy.loss_type=${loss}" \
    "${pretrained_args[@]}" \
    "${run_args[@]}" \
    "${extra_args[@]}"
}

run_recipe_grid() {
  local segment="$1"
  local sampler="$2"
  local ckpt="$3"
  local loss="$4"
  if [ -n "${WEIGHT_RECIPES}" ]; then
    local recipe
    local sup_w
    local backend_w
    local prior_w
    local sat_w
    for recipe in ${WEIGHT_RECIPES}; do
      IFS=, read -r sup_w backend_w prior_w sat_w <<< "${recipe}"
      prior_w="${prior_w:-0.0}"
      sat_w="${sat_w:-0.0}"
      run_one "${segment}" "${sampler}" "${ckpt}" "${loss}" "${sup_w}" "${backend_w}" "${prior_w}" "${sat_w}"
    done
  else
    run_one "${segment}" "${sampler}" "${ckpt}" "${loss}" "${SUPERVISED_WEIGHT}" "${BACKEND_WEIGHT}" "${PRIOR_WEIGHT}" "${SATURATION_WEIGHT}"
  fi
}

if [ -n "${PROXY_CKPT}" ]; then
  for segment in ${SEGMENT_LIST}; do
    for loss in ${LOSS_TYPES}; do
      run_recipe_grid "${segment}" "manual" "${PROXY_CKPT}" "${loss}"
    done
  done
  exit 0
fi

for segment in ${SEGMENT_LIST}; do
  for sampler in ${SAMPLERS}; do
    ckpt="$(resolve_sfproxy_ckpt "${sampler}" "${segment}")"
    for loss in ${LOSS_TYPES}; do
      run_recipe_grid "${segment}" "${sampler}" "${ckpt}" "${loss}"
    done
  done
done
