#!/bin/bash
cd ..
maestro_path="../../Dataset/maestro-v3.0.0"
WORKSPACE_DIR="../../202601_midisemi_data/ddsp-piano-pytorch/workspaces"
maestro_cache_path="$WORKSPACE_DIR/data_cache"

# Preprocess train/validation/test splits once
python3 preprocess.py \
	--splits train validation test \
	--max_polyphony 16 \
	$maestro_path \
	$maestro_cache_path
