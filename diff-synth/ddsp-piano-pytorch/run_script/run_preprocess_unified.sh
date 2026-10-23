#!/bin/bash
set -euo pipefail
cd ..

SEGMENT_SECONDS=${1:-10}
MAESTRO_PATH=${2:-"../../Dataset/maestro-v3.0.0"}
WORKSPACE_DIR=${3:-"../../202601_midisemi_data/ddsp-piano-pytorch/workspaces_unified_${SEGMENT_SECONDS}s"}
CACHE_DIR="$WORKSPACE_DIR/data_cache"

python3 preprocess.py \
  --splits train validation test \
  --segment_duration "$SEGMENT_SECONDS" \
  --sample_rate 22050 \
  --frame_rate 100 \
  --max_polyphony 16 \
  "$MAESTRO_PATH" \
  "$CACHE_DIR"
