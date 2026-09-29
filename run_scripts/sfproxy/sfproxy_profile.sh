#!/usr/bin/env bash

sfproxy_set_profile() {
  local instrument="$1"
  local piano_dataset="$2"
  local guitar_dataset="$3"
  local dataset

  case "${instrument}" in
    piano)
      dataset="${piano_dataset}"
      BOUNDARY_NAME="salamander_piano"
      INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz"
      PITCH_MIN=21
      PITCH_MAX=108
      PITCH_STEP=6
      REGISTER_SPLITS=(48 72)
      DEFAULT_SEGMENTS="2 5 10"
      ;;
    guitar)
      dataset="${guitar_dataset}"
      BOUNDARY_NAME="guitar"
      INSTRUMENT_PATH="/media/mengh/SharedData/zhanh/202601_midisemi_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz"
      PITCH_MIN=42
      PITCH_MAX=72
      PITCH_STEP=3
      REGISTER_SPLITS=(52 64)
      DEFAULT_SEGMENTS="2 5"
      ;;
    *)
      echo "Unsupported INSTRUMENT='${instrument}'." >&2
      exit 1
      ;;
  esac

  case "${instrument}:${dataset}" in
    piano:maestro)
      DATASET_NAME="piano"
      INSTRUMENT_NAME="salamander_piano"
      MIDI_DATASET="MAESTRO_v3"
      ;;
    piano:smd)
      DATASET_NAME="piano_smd"
      INSTRUMENT_NAME="salamander_piano_smd"
      MIDI_DATASET="SMD"
      ;;
    guitar:francoisleduc)
      DATASET_NAME="guitar"
      INSTRUMENT_NAME="guitar"
      MIDI_DATASET="FrancoisLeducGuitarDataset"
      ;;
    guitar:gaps)
      DATASET_NAME="guitar_gaps"
      INSTRUMENT_NAME="guitar_gaps"
      MIDI_DATASET="GAPS"
      ;;
    piano:*)
      echo "Unsupported PIANO_DATASET='${dataset}'." >&2
      exit 1
      ;;
    guitar:*)
      echo "Unsupported GUITAR_DATASET='${dataset}'." >&2
      exit 1
      ;;
  esac
}
