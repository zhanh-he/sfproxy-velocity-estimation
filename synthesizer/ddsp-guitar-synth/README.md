# DDSP-Guitar-Synth in this repo

This directory is the default guitar DiffSynth backend for `score_hpt`.

## Preferred repo entry point

For day-to-day runs, prefer the repo-level launcher:

- `run_scripts/train_ddsp_guitar_synth.sh`

That script exposes a manual config block at the top for:

- machine: `3090` or `5090`
- segment length: `2`, `5`, or `10`
- total epochs

It also handles the common workflow automatically:

- if prepared `.npz` files are missing, it runs data preparation first
- if prepared `.npz` files already exist, it skips preparation and starts training
- training continues with `wandb`

## Repo default contract

New runs in this repo should use:

- sample rate = `22050`
- target frame rate = `100 fps`
- hop length = `221` samples
- effective frame rate = `22050 / 221 = 99.7738 fps`
- `n_fft = 2048`
- default segment length = `10 s`
- default training epochs = `500`
- default multi-scale spectral loss FFT sizes = `[128, 256, 512, 1024, 2048]`

The slight mismatch between target `100 fps` and effective `99.7738 fps` is unavoidable because hop length must be an integer.
The training scripts derive the hop automatically.

## Original project defaults for reference

The upstream `ddsp-guitar-synth` codebase used:

- sample rate = `16000`
- hop length = `64`
- effective frame rate = `250 fps`
- `n_fft = 2048`
- default segment length = `3 s`
- original multi-scale spectral loss FFT sizes = `[64, 128, 256, 512, 1024, 2048]`

Those values are kept only as reference.
They are **not** the new repo defaults.

## Data arrangement

This project was originally built around **GuitarSet** through `mirdata`.
The new helper scripts keep that arrangement.

Default split in the helper script:

- train = all players except `00`
- validation = player `00`

You can change the held-out player with `--test_player_id`.
You can also filter by `--mode solo` or `--mode comp`.

## 1. Prepare GuitarSet datasets

### 10 s

```bash
cd synthesizer/ddsp-guitar-synth
python prepare_guitarset_unified.py \
  --guitarset_path /path/to/GuitarSet \
  --output_dir /path/to/ddsp_guitar_synth_10s/data \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 10
```

### 5 s

```bash
cd synthesizer/ddsp-guitar-synth
python prepare_guitarset_unified.py \
  --guitarset_path /path/to/GuitarSet \
  --output_dir /path/to/ddsp_guitar_synth_5s/data \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 5
```

### 2 s

```bash
cd synthesizer/ddsp-guitar-synth
python prepare_guitarset_unified.py \
  --guitarset_path /path/to/GuitarSet \
  --output_dir /path/to/ddsp_guitar_synth_2s/data \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 2
```

Each prepare run writes:

- `train_gset_midi_<seg>s.npz`
- `val_gset_midi_<seg>s.npz`
- `prepare_metadata.json`

## 2. Train DDSP-Guitar-Synth

### 10 s

```bash
cd synthesizer/ddsp-guitar-synth
python train_midi_synth_unified.py \
  --train_dataset_path /path/to/ddsp_guitar_synth_10s/data/train_gset_midi_10s.npz \
  --val_dataset_path /path/to/ddsp_guitar_synth_10s/data/val_gset_midi_10s.npz \
  --output_dir /path/to/ddsp_guitar_synth_10s/output \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 10 \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
```

### 5 s

```bash
cd synthesizer/ddsp-guitar-synth
python train_midi_synth_unified.py \
  --train_dataset_path /path/to/ddsp_guitar_synth_5s/data/train_gset_midi_5s.npz \
  --val_dataset_path /path/to/ddsp_guitar_synth_5s/data/val_gset_midi_5s.npz \
  --output_dir /path/to/ddsp_guitar_synth_5s/output \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 5 \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
```

### 2 s

```bash
cd synthesizer/ddsp-guitar-synth
python train_midi_synth_unified.py \
  --train_dataset_path /path/to/ddsp_guitar_synth_2s/data/train_gset_midi_2s.npz \
  --val_dataset_path /path/to/ddsp_guitar_synth_2s/data/val_gset_midi_2s.npz \
  --output_dir /path/to/ddsp_guitar_synth_2s/output \
  --sample_rate 22050 \
  --frame_rate 100 \
  --segment_seconds 2 \
  --n_fft 2048 \
  --loss_fft_sizes 128,256,512,1024,2048
```

## Project-local helper scripts

### Prepare

```bash
bash run_prepare_unified.sh francoisleduc 10 /path/to/FrancoisLeducGuitarDataset /path/to/ddsp_guitar_synth_10s/data
bash run_prepare_unified.sh gaps 5 /path/to/GAPS /path/to/ddsp_guitar_synth_5s/data
bash run_prepare_unified.sh gaps 2 /path/to/GAPS /path/to/ddsp_guitar_synth_2s/data
```

### Train

```bash
bash run_training_unified.sh 10 /path/to/ddsp_guitar_synth_10s/data /path/to/ddsp_guitar_synth_10s/output
bash run_training_unified.sh 5 /path/to/ddsp_guitar_synth_5s/data /path/to/ddsp_guitar_synth_5s/output
bash run_training_unified.sh 2 /path/to/ddsp_guitar_synth_2s/data /path/to/ddsp_guitar_synth_2s/output
```

Use these project-local helper scripts when you explicitly want separate prepare and train steps.

## Checkpoints for `score_hpt`

`score_hpt` expects a direct path to a checkpoint file.
Use either:

- `.../latest_model_checkpoint.pt`
- or `.../checkpoints/<N epochs>/model_checkpoint.pt`

Those checkpoints include a `renderer_config` block with:

- sample rate
- target frame rate
- effective frame rate
- hop length
- `n_fft`
- segment length

That metadata is used by the new `score_hpt` wrapper when available.

## `score_hpt` config

```yaml
proxy:
  type: diffsynth_guitar
  checkpoint: /path/to/ddsp_guitar_synth/latest_model_checkpoint.pt
  ddsp:
    project_root: /abs/path/to/202604_midiproxy/synthesizer/ddsp-guitar-synth
  ddsp_guitar:
    implementation: ddsp_guitar_synth
```
