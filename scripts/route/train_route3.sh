#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
. "${SCRIPT_DIR}/score_hpt_profile.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="${ROOT_DIR}/score_hpt"

# frontend
TRAIN_SET="${TRAIN_SET:-maestro}"
MODEL_TYPE="${MODEL_TYPE:-hpt}"
SCORE_METHOD="${SCORE_METHOD:-note_editor}"
FRONTEND_PRETRAINED="${FRONTEND_PRETRAINED:-}"

# backend
SEGMENT_SECONDS="${SEGMENT_SECONDS:-5}"
DDSP_CKPT="${DDSP_CKPT:-}"
SUPERVISED_WEIGHT="${SUPERVISED_WEIGHT:-0.0}"
BACKEND_WEIGHT="${BACKEND_WEIGHT:-1.0}"
PRIOR_WEIGHT="${PRIOR_WEIGHT:-0.0}"

# evaluation / misc
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

score_hpt_init_context
score_hpt_set_dataset_profile "${TRAIN_SET}"

DDSP_CKPT="${DDSP_CKPT:-$(score_hpt_resolve_ddsp_ckpt "${TRAIN_SET}" "${SEGMENT_SECONDS}")}"

mkdir -p "${WORKSPACE_DIR}"
read -r -a extra_args <<< "${EXTRA_OVERRIDES}"

cd "${PROJECT_DIR}"

required_datasets=("${TRAIN_SET}" "${DEFAULT_TEST_SET}")
while IFS= read -r dataset_name; do
  required_datasets+=("${dataset_name}")
done < <(score_hpt_collect_eval_datasets "${DEFAULT_EVAL_SETS}")

score_hpt_prepare_required_datasets "${PYTHON_BIN}" "${required_datasets[@]}"
score_hpt_set_dataset_profile "${TRAIN_SET}"

RUN_SCORE_METHOD="${SCORE_METHOD}"
MODEL_INPUT2="null"
if [ "${MODEL_TYPE}" = "hpt" ] && [ "${RUN_SCORE_METHOD}" = "note_editor" ]; then
  MODEL_INPUT2="onset"
fi

pretrained_args=()
if [ -n "${FRONTEND_PRETRAINED}" ]; then
  pretrained_args+=("model.frontend_pretrained_mode=route2_piano_specific")
  pretrained_args+=("model.frontend_pretrained=${FRONTEND_PRETRAINED}")
fi

"${PYTHON_BIN}" pytorch/train_ddsp.py \
  "exp.workspace=${WORKSPACE_DIR}" \
  "dataset.train_set=${TRAIN_SET}" \
  "dataset.test_set=${DEFAULT_TEST_SET}" \
  "dataset.eval_sets=${DEFAULT_EVAL_SETS}" \
  "model.type=${MODEL_TYPE}" \
  "model.input2=${MODEL_INPUT2}" \
  "score_informed.method=${RUN_SCORE_METHOD}" \
  "loss.supervised_weight=${SUPERVISED_WEIGHT}" \
  "loss.backend_weight=${BACKEND_WEIGHT}" \
  "loss.velocity_prior_weight=${PRIOR_WEIGHT}" \
  "backend.enabled=true" \
  "backend.type=${DIFFSYNTH_PROXY_TYPE}" \
  "backend.project_root=${DDSP_PROJECT_ROOT_DEFAULT}" \
  "backend.checkpoint=${DDSP_CKPT}" \
  "backend.backend_segment_seconds=${SEGMENT_SECONDS}" \
  "${pretrained_args[@]}" \
  "${extra_args[@]}"
