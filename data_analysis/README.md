# data_analysis

This folder collects the analysis, rendering, and evaluation utilities used around the `202604_midiproxy` project.

The code here is mainly used for three jobs:

1. Dataset inspection
   Analyze MIDI distributions and audio loudness distributions.
2. SoundFont probing
   Render a SoundFont over pitch and velocity and visualize its response.
3. Velocity/render evaluation
   Render the same MIDI in multiple ways and compare them against real recordings, while also keeping the synthetic `gt_vs_pred` baseline inside the same run.

This README is the top-level guide for the whole `data_analysis` folder, including the notebooks in this directory and the Python modules under `src/data_analysis/`.

## Folder Layout

Top-level notebooks:

- `dataset_download.ipynb`
- `dataset_bssl_stats.ipynb`
- `dataset_midi_stats.ipynb`
- `soundfont_eval.ipynb`
- `soundfont_visual.ipynb`
- `visualize_midi_audio.ipynb`

Source tree:

- `src/data_analysis/analysis/`
  Dataset statistics and MIDI/audio visualization helpers.
- `src/data_analysis/rendering/`
  MIDI rendering, flat-velocity MIDI creation, SoundFont probing, and SoundFont response plots.
- `src/data_analysis/evaluation/`
  BSSL / NTOT / note-level evaluation code.
- `src/data_analysis/cli/`
  Script entrypoints for the common workflows.

## Main Concepts

### 1. `gt` vs `pred`

In this project, `pred` inside the render-and-eval scripts is currently not a learned model prediction.

For the current evaluation pipeline:

- `gt`
  Render the original MIDI with its original note velocities.
- `pred`
  Render a flattened-velocity baseline MIDI where all `note_on` velocities are replaced by a constant value, default `64`.

That flattened MIDI is created in:

- `src/data_analysis/rendering/midi_velocity.py`

The paired rendering is done in:

- `src/data_analysis/rendering/render_pair.py`

So when you see:

- `*.gt.wav`
- `*.flat64.wav`

the second file is the current baseline "prediction".

### 2. Main Dataset Evaluation Script

The main dataset-level evaluation script is now `adv_render_and_eval_dataset`.

It covers all three comparisons in one run:

- `real_vs_gt`
- `real_vs_pred`
- `gt_vs_pred`

So the older standalone synthetic-vs-synthetic dataset script is no longer part of the intended workflow.

### 3. Metrics Used Here

Current metric naming in `data_analysis` is standardized to:

- `pearson`
- `cosine_sim`
- `mae`

The main feature groups are:

- `BSSL`
  Bark-scale specific loudness representation
- `NTOT`
  Bark-scale total loudness
- `Note Harmonic Energy`
  note-wise harmonic-energy observable
- `Note Onset Flux`
  note-wise onset-flux observable

## Dependencies

Core Python packages commonly needed here:

- `numpy`
- `pandas`
- `matplotlib`
- `librosa`
- `torch`
- `pretty_midi`
- `mido`
- `tqdm`

Rendering backends:

- `.sfz` instruments
  require `sfizz_render`
- `.sf2` instruments
  require `pyfluidsynth` / `fluidsynth`

Dataset-specific helpers seen in notebooks:

- `mirdata`
- `datasets`, `huggingface_hub`
- `git-lfs` for large dataset download workflows

## Notebook Guide

### `dataset_download.ipynb`

Purpose:

- quick notes for downloading datasets used by this project
- currently includes a Hugging Face / Git LFS workflow for GAPS / GOAT-style audio data

Use this notebook when:

- you need to reproduce a dataset download manually
- you want the exact shell commands used during setup

### `dataset_bssl_stats.ipynb`

Purpose:

- compute dataset-level BSSL / NTOT loudness statistics
- compare loudness distributions across datasets such as MAESTRO, GOAT, GuitarSet, MID-FiLD

Backed by:

- `src/data_analysis/analysis/dataset_bssl_stats.py`

What it produces:

- per-file loudness statistics
- dataset-level histogram counts
- JSON exports
- histogram plots with count and percentage views

Typical usage:

- check whether one dataset is systematically louder / quieter
- inspect missing or under-populated loudness ranges

### `dataset_midi_stats.ipynb`

Purpose:

- compute dataset-level MIDI statistics
- inspect pitch, velocity, duration, IOI, chord size, polyphony

Backed by:

- `src/data_analysis/analysis/dataset_midi_stats.py`

What it produces:

- pitch histograms
- velocity histograms
- duration / IOI histograms
- chord-size statistics
- JSON summaries for sampler design or dataset diagnostics

The notebook also contains exploratory `mirdata` usage for GuitarSet-style note access.

### `soundfont_eval.ipynb`

Purpose:

- evaluate SoundFonts or renderers on MAESTRO / SMD / MAPS-style data
- run the current main benchmark workflow

This notebook is the most important one for the current soundfont work.

It includes examples for:

- checking backend installation
- rendering a paired `gt` / flat-velocity baseline
- running `adv_render_and_eval_dataset`
- running single-file evaluation

Current recommended path:

- use `adv_render_and_eval_dataset` for dataset-level evaluation against real audio

### `soundfont_visual.ipynb`

Purpose:

- probe a SoundFont across pitch and velocity
- visualize how the SoundFont responds over the keyboard

Backed by:

- `src/data_analysis/rendering/soundfont_probe.py`
- `src/data_analysis/rendering/soundfont_visualization.py`

What it does:

- renders one pitch sweep per velocity
- extracts summary metrics such as Bark dB and NTOT peak
- writes CSV
- plots response curves over pitch

Use this notebook when:

- you want to understand a SoundFont itself, not just benchmark it on a dataset

### `visualize_midi_audio.ipynb`

Purpose:

- visualize MIDI note structure and compare it with audio loudness curves

Backed by:

- `src/data_analysis/analysis/visualize_midi_audio.py`

Typical usage:

- piano roll plotting
- MIDI velocity curve plotting
- BSSL total loudness plotting
- MIDI-vs-audio correlation inspection
- cross-comparison between two MIDI files or two audio files

## Source Modules

### `analysis/`

`dataset_bssl_stats.py`

- scan a dataset root for audio files
- compute BSSL / NTOT distributions
- plot histograms
- export JSON summaries

`dataset_midi_stats.py`

- scan a dataset root for MIDI files
- compute pitch / velocity / duration / IOI / chord-size statistics
- export JSON summaries and figures

`visualize_midi_audio.py`

- load MIDI
- build piano rolls
- compute BSSL total loudness from audio
- compare MIDI velocity curves and loudness curves
- compare two MIDI files or two audio files

### `rendering/`

`midi_velocity.py`

- create flat-velocity copies of MIDI files

`render_pair.py`

- render one MIDI twice:
  - original velocity render
  - flat-velocity baseline render
- choose backend automatically from `.sfz` vs `.sf2`

`sfizz.py`

- render `.sfz` instruments through `sfizz_render`

`fluidsynth.py`

- render `.sf2` instruments through FluidSynth

`soundfont_probe.py`

- systematic SoundFont probing over pitch / velocity

`soundfont_visualization.py`

- load probe CSV
- create response figures

`feature_extractor.py`

- Bark / sone / NTOT feature extraction
- also includes log-mel extraction utilities

### `evaluation/`

`bssl_eval.py`

- compare two audio files using BSSL / NTOT features

`note_dynamics.py`

- compare note-wise observables using MIDI note events plus two audio files

`adv_eval.py`

- real-audio-referenced evaluation
- computes:
  - real vs gt synthetic
  - real vs pred synthetic
  - gt vs pred synthetic

### `cli/`

`render_and_eval.py`

- single-file synthetic-vs-synthetic evaluation

`adv_render_and_eval.py`

- single-file real-referenced evaluation

`adv_render_and_eval_dataset.py`

- dataset-level real-referenced evaluation
- also includes the synthetic `gt_vs_pred` comparison
- main current benchmark script for soundfont evaluation

## Recommended Workflows

### A. SoundFont benchmark against real recordings

Use:

```bash
PYTHONPATH=src python -m data_analysis.cli.adv_render_and_eval_dataset \
  --dataset_type smd \
  --dataset_dir /path/to/SMD \
  --instrument /path/to/instrument.sfz \
  --out_dir /path/to/output \
  --render_sr 44100 \
  --eval_sr 22050 \
  --flat_velocity 64
```

Interpretation:

- `gt`
  original MIDI velocities rendered through the chosen instrument
- `pred`
  flat-velocity baseline rendered through the same instrument
- `real_audio`
  actual recording from the dataset

This produces:

- per-file JSONs under `per_file_results/`
- final batch summary JSON
- terminal summary grouped by:
  - `real_vs_gt`
  - `real_vs_pred`
  - `gt_vs_pred`
  - `BSSL / NTOT / Note Harmonic Energy / Note Onset Flux`

### B. Probe a SoundFont directly

Use `soundfont_visual.ipynb` when you want:

- response over pitch
- response over velocity
- CSV-based re-plotting

This is complementary to dataset evaluation: it tells you how the instrument behaves by itself.

### C. Inspect a dataset before training or evaluation

Use:

- `dataset_midi_stats.ipynb`
- `dataset_bssl_stats.ipynb`

This is useful for:

- understanding pitch/velocity coverage
- understanding loudness distribution
- spotting mismatches between datasets

## Notes on Output Files

Typical render outputs created by paired rendering:

- `*.gt.wav`
- `*.flat64.mid`
- `*.flat64.wav`

Typical batch evaluation outputs:

- `per_file_results/<item>.json`
- `<dataset>_batch_eval.json`
- `<dataset>_adv_batch_eval.json`

Optional plot outputs:

- NTOT comparison PNGs
- SoundFont probe figures
- dataset histogram figures

## Current Practical Recommendation

For the current state of this project:

- treat `adv_render_and_eval_dataset` as the main dataset evaluation tool
- remember that current `pred` means flat-velocity baseline, not a learned predictor

If you later replace the flat-velocity baseline with a real velocity model, update the naming in both the scripts and this README so `pred` does not become misleading.
