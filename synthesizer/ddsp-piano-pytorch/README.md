## DDSP-Piano in this repo

This directory is used to train `DDSP-Piano` checkpoints for later use inside `score_hpt`.

## Preferred repo entry point

For routine runs, prefer the repo-level launcher:

- `run_scripts/train_ddsp_piano.sh`

That script exposes a manual config block at the top for:

- machine: `3090` or `5090`
- segment length: `2`, `5`, or `10`
- total epochs

It also handles the common workflow automatically:

- if the cache directory is empty, it runs preprocessing first
- if cached data already exists, it skips preprocessing and starts training
- training continues with `wandb`

## Current repo default contract

The repo now defaults to:

- sample rate = `22050`
- segment duration = `10 s`
- frame rate = `100 fps`

Historical upstream-style defaults were `16000 / 3 s / 250 fps`.
Those are still useful as reference, but they are no longer the default contract in this repo.

## Dataset usage

This code uses the full `MAESTRO v3.0.0` dataset under the official split definitions in `maestro-v3.0.0.csv`.

- preprocessing filters rows by `split == train | validation | test`
- training uses the cached `train` split
- validation uses the cached `validation` split
- evaluation uses the cached `test` split

## Preprocess

### 10 s

```bash
cd synthesizer/ddsp-piano-pytorch
python preprocess.py \
  --splits train validation test \
  --segment_duration 10 \
  --sample_rate 22050 \
  --frame_rate 100 \
  /path/to/maestro-v3.0.0 \
  /path/to/workspaces_unified_10s/data_cache
```

### 5 s

```bash
cd synthesizer/ddsp-piano-pytorch
python preprocess.py \
  --splits train validation test \
  --segment_duration 5 \
  --sample_rate 22050 \
  --frame_rate 100 \
  /path/to/maestro-v3.0.0 \
  /path/to/workspaces_unified_5s/data_cache
```

### 2 s

```bash
cd synthesizer/ddsp-piano-pytorch
python preprocess.py \
  --splits train validation test \
  --segment_duration 2 \
  --sample_rate 22050 \
  --frame_rate 100 \
  /path/to/maestro-v3.0.0 \
  /path/to/workspaces_unified_2s/data_cache
```

Each cache directory now stores a `cache_config.json`.
`train.py` validates that the requested model contract matches that cache contract.

## Train

### 10 s

```bash
cd synthesizer/ddsp-piano-pytorch
python train.py \
  --phase 1 \
  --batch_size 6 \
  --epochs 7 \
  --lr 1e-3 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration 10 \
  /path/to/workspaces_unified_10s/data_cache \
  /path/to/workspaces_unified_10s/models
```

### 5 s

```bash
cd synthesizer/ddsp-piano-pytorch
python train.py \
  --phase 1 \
  --batch_size 6 \
  --epochs 7 \
  --lr 1e-3 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration 5 \
  /path/to/workspaces_unified_5s/data_cache \
  /path/to/workspaces_unified_5s/models
```

### 2 s

```bash
cd synthesizer/ddsp-piano-pytorch
python train.py \
  --phase 1 \
  --batch_size 6 \
  --epochs 7 \
  --lr 1e-3 \
  --sample_rate 22050 \
  --frame_rate 100 \
  --duration 2 \
  /path/to/workspaces_unified_2s/data_cache \
  /path/to/workspaces_unified_2s/models
```

## Project-local helper scripts

- `run_script/run_preprocess_unified.sh <segment_seconds> <maestro_path> <workspace_dir>`
- `run_script/run_training_unified.sh <segment_seconds> <workspace_dir>`

Example:

```bash
bash run_script/run_preprocess_unified.sh 10 /path/to/maestro-v3.0.0 /path/to/workspaces_unified_10s
bash run_script/run_training_unified.sh 10 /path/to/workspaces_unified_10s
```

Use these project-local helper scripts when you explicitly want separate preprocess and train steps.
