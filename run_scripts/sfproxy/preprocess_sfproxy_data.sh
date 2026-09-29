#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/sfproxy_profile.sh"

WORKSPACE_BASE="/media/mengh/SharedData/zhanh/202601_midisemi_data"
ANALYSIS_DIR="${ROOT_DIR}/data_analysis"
DATA_DIR="${WORKSPACE_BASE}/synth-proxy/data"

DEFAULT_INSTRUMENT="piano"
DEFAULT_PIANO_DATASET="maestro"
DEFAULT_GUITAR_DATASET="francoisleduc"

INSTRUMENT="${INSTRUMENT:-${DEFAULT_INSTRUMENT}}"
PIANO_DATASET="${PIANO_DATASET:-${DEFAULT_PIANO_DATASET}}"
GUITAR_DATASET="${GUITAR_DATASET:-${DEFAULT_GUITAR_DATASET}}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-}"
SEGMENT_LIST="${SEGMENT_LIST:-}"
MIX_WEIGHTS="${MIX_WEIGHTS:-}"

segment_tag() {
  local value="$1"
  if [[ "${value}" == *.* ]]; then
    value="${value%0}"
    value="${value%.}"
    value="${value//./p}"
  fi
  echo "${value}s"
}

sfproxy_set_profile "${INSTRUMENT}" "${PIANO_DATASET}" "${GUITAR_DATASET}"
REALISM_STATS_JSON="${ANALYSIS_DIR}/stats/midi_sampler/${MIDI_DATASET}_sampler.json"
BOUNDARY_MODE="default"

segment_input="${SEGMENT_SECONDS:-${SEGMENT_LIST:-${DEFAULT_SEGMENTS}}}"
read -r -a SEGMENTS <<< "${segment_input//,/ }"
PRESETS=(boundary_v2 coverage_v2 realism_v2 stress_v2)

mix_weight_args=()
if [[ -n "${MIX_WEIGHTS}" ]]; then
  read -r -a mix_weights <<< "${MIX_WEIGHTS//,/ }"
  if [[ "${#mix_weights[@]}" -ne 4 ]]; then
    echo "MIX_WEIGHTS must provide 4 values: boundary coverage realism stress" >&2
    exit 1
  fi
  mix_weight_args=(
    --names boundary coverage realism stress
    --weights "${mix_weights[@]}"
  )
fi

mkdir -p "${DATA_DIR}"
cd "${ROOT_DIR}"

for segment_seconds in "${SEGMENTS[@]}"; do
  tag="$(segment_tag "${segment_seconds}")"

  for preset in "${PRESETS[@]}"; do
    case "${preset}" in
      realism_v2|mixed_v2)
        if [[ ! -f "${REALISM_STATS_JSON}" ]]; then
          echo "Missing realism stats: ${REALISM_STATS_JSON}" >&2
          exit 1
        fi
        ;;
    esac

    echo "${preset} ${segment_seconds}s"

    for split in train val; do
      out_dir="${DATA_DIR}/${INSTRUMENT_NAME}/${preset}_${tag}_${BOUNDARY_MODE}/${split}"
      if [[ -f "${out_dir}/configs.pkl" ]] \
        && [[ -f "${out_dir}/inputs_pitch.pkl" ]] \
        && [[ -f "${out_dir}/inputs_cont.pkl" ]] \
        && [[ -f "${out_dir}/inputs_mask.pkl" ]] \
        && [[ -f "${out_dir}/targets_note.pkl" ]]; then
        continue
      fi

      python "${ROOT_DIR}/synth-proxy/src/export_dataset_pkl.py" \
        --config-name "data_${INSTRUMENT}" \
        "paths.repo_root=${ROOT_DIR}" \
        "paths.workspace_dir=${WORKSPACE_BASE}" \
        "paths.analysis_dir=${ANALYSIS_DIR}" \
        "midi_dataset=${MIDI_DATASET}" \
        "instrument.name=${INSTRUMENT_NAME}" \
        "instrument.path=${INSTRUMENT_PATH}" \
        "instrument.seg_len_s=${segment_seconds}" \
        "sampler_preset=${preset}" \
        "boundary_mode=${BOUNDARY_MODE}" \
        "split=${split}" \
        "reset_output_dir=true"
    done
  done

  echo "mixed_v2 ${segment_seconds}s"
  for split in train val; do
    out_dir="${DATA_DIR}/${INSTRUMENT_NAME}/mixed_v2_${tag}_${BOUNDARY_MODE}/${split}"
    if [[ -n "${MIX_WEIGHTS}" ]]; then
      rm -rf "${out_dir}"
    fi
    if [[ -f "${out_dir}/configs.pkl" ]] \
      && [[ -f "${out_dir}/inputs_pitch.pkl" ]] \
      && [[ -f "${out_dir}/inputs_cont.pkl" ]] \
      && [[ -f "${out_dir}/inputs_mask.pkl" ]] \
      && [[ -f "${out_dir}/targets_note.pkl" ]]; then
      continue
    fi

    python "${ROOT_DIR}/synth-proxy/src/tools/compose_exported_dataset.py" \
      --input-dirs \
        "${DATA_DIR}/${INSTRUMENT_NAME}/boundary_v2_${tag}_${BOUNDARY_MODE}/${split}" \
        "${DATA_DIR}/${INSTRUMENT_NAME}/coverage_v2_${tag}_${BOUNDARY_MODE}/${split}" \
        "${DATA_DIR}/${INSTRUMENT_NAME}/realism_v2_${tag}_${BOUNDARY_MODE}/${split}" \
        "${DATA_DIR}/${INSTRUMENT_NAME}/stress_v2_${tag}_${BOUNDARY_MODE}/${split}" \
      --config-file "${ROOT_DIR}/synth-proxy/configs/data_${INSTRUMENT}.yaml" \
      --preset mixed_v2 \
      --preset-name mixed_v2 \
      "${mix_weight_args[@]}" \
      --out-dir "${out_dir}"
  done
done

echo "Done: ${DATA_DIR}/${INSTRUMENT_NAME}"
