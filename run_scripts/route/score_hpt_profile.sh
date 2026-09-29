#!/usr/bin/env bash

# naming / path helpers
score_hpt_segment_tag() {
  printf '%ss\n' "${1%.0}"
}

score_hpt_has_hdf5() {
  find "$1" -type f -name '*.h5' -print -quit 2>/dev/null | grep -q .
}

score_hpt_config_value() {
  local section="$1"
  local key="$2"
  local value

  value="$(
    awk -v section="${section}" -v key="${key}" '
      /^[^[:space:]#][^:]*:/ {
        name = $1
        sub(/:.*/, "", name)
        in_section = (name == section)
      }
      in_section && $0 ~ "^[[:space:]]*" key ":[[:space:]]*" {
        sub("^[[:space:]]*" key ":[[:space:]]*", "", $0)
        print
        exit
      }
    ' "${CONFIG_PATH}"
  )"

  value="${value%%#*}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  printf '%s\n' "${value}"
}

score_hpt_abspath() {
  local base="$1"
  local path="$2"

  case "${path}" in
    /*) printf '%s\n' "${path}" ;;
    *)
      if command -v realpath >/dev/null 2>&1; then
        realpath -m "${base}/${path}"
      else
        (cd "$(dirname "${base}/${path}")" 2>/dev/null && printf '%s/%s\n' "$PWD" "$(basename "${path}")") ||
          printf '%s/%s\n' "${base}" "${path}"
      fi
      ;;
  esac
}

score_hpt_workspace_data_root() {
  local path="${1}"
  [ -e "${path}" ] && command -v realpath >/dev/null 2>&1 && path="$(realpath "${path}")"
  dirname "$(dirname "${path}")"
}

# shared runtime context
score_hpt_init_context() {
  PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}/score_hpt}"
  CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/pytorch/config/config.yaml}"

  WORKSPACE_DIR="$(score_hpt_abspath "${PROJECT_DIR}" "${WORKSPACE_DIR:-$(score_hpt_config_value exp workspace)}")"
  SCORE_HPT_SAMPLE_RATE="${SCORE_HPT_SAMPLE_RATE:-$(score_hpt_config_value feature sample_rate)}"
  DATA_ROOT="${DATA_ROOT:-$(score_hpt_workspace_data_root "${WORKSPACE_DIR}")}"
}

# dataset / backend profile selection
score_hpt_set_dataset_profile() {
  local dataset_name="$1"
  local dataset_env=""
  local dataset_dir_key="${dataset_name}_dir"

  case "${dataset_name}" in
    maestro|smd)
      DATASET_KIND="piano"
      DIFFSYNTH_PROXY_TYPE="diffsynth_piano"
      DDSP_PROJECT_ROOT_DEFAULT="${ROOT_DIR}/synthesizer/ddsp-piano-pytorch"
      DDSP_CKPT_ROOT_DEFAULT="${DATA_ROOT}/ddsp-piano-pytorch"
      ;;
    francoisleduc|gaps)
      DATASET_KIND="guitar"
      DIFFSYNTH_PROXY_TYPE="diffsynth_guitar"
      DDSP_PROJECT_ROOT_DEFAULT="${ROOT_DIR}/synthesizer/ddsp-guitar-synth"
      DDSP_CKPT_ROOT_DEFAULT="${DATA_ROOT}/ddsp-guitar-synth"
      ;;
    *)
      DATASET_KIND="piano"
      DIFFSYNTH_PROXY_TYPE="diffsynth_piano"
      DDSP_PROJECT_ROOT_DEFAULT="${ROOT_DIR}/synthesizer/ddsp-piano-pytorch"
      DDSP_CKPT_ROOT_DEFAULT="${DATA_ROOT}/ddsp-piano-pytorch"
      ;;
  esac

  DATASET_NAME="${dataset_name}"
  PACK_MODE="pack_${dataset_name}_dataset_to_hdf5"
  PACK_OVERRIDE_KEY="dataset.${dataset_dir_key}"
  DEFAULT_TEST_SET="${dataset_name}"
  DEFAULT_EVAL_SETS="[train,${dataset_name}]"
  DATASET_REQUIRED_PATH=""

  case "${dataset_name}" in
    maestro)
      DATASET_REQUIRED_PATH="maestro-v3.0.0.csv"
      DEFAULT_EVAL_SETS="[train,maestro,smd]"
      SFPROXY_DATASET_NAME_DEFAULT="piano"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="salamander_piano"
      dataset_env="${MAESTRO_DIR:-}"
      ;;
    smd)
      SFPROXY_DATASET_NAME_DEFAULT="piano_smd"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="salamander_piano_smd"
      dataset_env="${SMD_DIR:-}"
      ;;
    francoisleduc)
      DATASET_REQUIRED_PATH="metadata.csv"
      SFPROXY_DATASET_NAME_DEFAULT="guitar"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="guitar"
      dataset_env="${FRANCOISLEDUC_DIR:-${GUITAR_DATASET_DIR:-}}"
      ;;
    gaps)
      DATASET_REQUIRED_PATH="gaps_metadata_with_splits.csv"
      SFPROXY_DATASET_NAME_DEFAULT="guitar_gaps"
      SFPROXY_INSTRUMENT_NAME_DEFAULT="guitar_gaps"
      dataset_env="${GAPS_DIR:-${GUITAR_DATASET_DIR:-}}"
      ;;
  esac

  DATASET_DIR="$(score_hpt_abspath "${PROJECT_DIR}" "${dataset_env:-$(score_hpt_config_value dataset "${dataset_dir_key}")}")"
  HDF5_DIR="${WORKSPACE_DIR}/hdf5s/${dataset_name}_sr${SCORE_HPT_SAMPLE_RATE}"
}

# dataset preparation
score_hpt_assert_dataset_dir() {
  :
}

score_hpt_ensure_dataset_hdf5() {
  local python_bin="$1"

  if [ ! -d "${HDF5_DIR}" ] || ! score_hpt_has_hdf5 "${HDF5_DIR}"; then
    echo "Packed ${DATASET_NAME} HDF5 dataset not found. Running preprocessing first."
    "${python_bin}" pytorch/data_generator.py "${PACK_MODE}" \
      "exp.workspace=${WORKSPACE_DIR}" \
      "feature.sample_rate=${SCORE_HPT_SAMPLE_RATE}" \
      "${PACK_OVERRIDE_KEY}=${DATASET_DIR}"
  fi
}

score_hpt_collect_eval_datasets() {
  local eval_sets="${1//[\[\]\"]/}"
  local item

  eval_sets="${eval_sets//\'}"

  eval_sets="${eval_sets//,/ }"
  for item in ${eval_sets}; do
    [ -n "${item}" ] && [ "${item}" != "train" ] && printf '%s\n' "${item}"
  done
}

# local checkpoint resolution
score_hpt_prepare_required_datasets() {
  local python_bin="$1"
  local dataset_name
  local seen=" "
  shift

  for dataset_name in "$@"; do
    [ -n "${dataset_name}" ] || continue
    case " ${seen} " in
      *" ${dataset_name} "*) continue ;;
    esac
    seen="${seen}${dataset_name} "
    score_hpt_set_dataset_profile "${dataset_name}"
    score_hpt_assert_dataset_dir
    score_hpt_ensure_dataset_hdf5 "${python_bin}"
  done
}

score_hpt_resolve_ddsp_ckpt() {
  local dataset_name="$1"
  local segment="$2"
  local ckpt_root
  local path

  score_hpt_set_dataset_profile "${dataset_name}"
  ckpt_root="${DDSP_CKPT_ROOT:-${DDSP_CKPT_ROOT_DEFAULT}}"

  case "${DATASET_KIND}" in
    piano)
      path="${ckpt_root}/workspaces_unified_$(score_hpt_segment_tag "${segment}")/models/phase_${DDSP_PHASE:-1}/ckpts/ddsp-piano_epoch_${CKPT_EPOCH:-7}_params.pt"
      ;;
    guitar)
      path="${ckpt_root}/${dataset_name}_$(score_hpt_segment_tag "${segment}")/output/latest_model_checkpoint.pt"
      ;;
  esac

  printf '%s\n' "${path}"
}
