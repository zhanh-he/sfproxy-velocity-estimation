#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INSTRUMENT="${INSTRUMENT:-piano}"
PIANO_DATASET="${PIANO_DATASET:-maestro}"
GUITAR_DATASET="${GUITAR_DATASET:-francoisleduc}"
SEGMENT_LIST="${SEGMENT_LIST:-2}"
TRAIN_PRESETS="${TRAIN_PRESETS:-coverage_v2 realism_v2 stress_v2 mixed_v2}"

read -r -a SEGMENTS <<< "${SEGMENT_LIST//,/ }"
read -r -a PRESETS <<< "${TRAIN_PRESETS//,/ }"

for preset in "${PRESETS[@]}"; do
  case "${preset}" in
    boundary_v2|coverage_v2|realism_v2|stress_v2|mixed_v2)
      ;;
    *)
      echo "Unsupported TRAIN_PRESET='${preset}'." >&2
      exit 1
      ;;
  esac
done

for segment_seconds in "${SEGMENTS[@]}"; do
  for preset in "${PRESETS[@]}"; do
    INSTRUMENT="${INSTRUMENT}" \
    PIANO_DATASET="${PIANO_DATASET}" \
    GUITAR_DATASET="${GUITAR_DATASET}" \
    SEGMENT_SECONDS="${segment_seconds}" \
    TRAIN_PRESET="${preset}" \
    "${SCRIPT_DIR}/train_sfproxy.sh"
  done
done
