# Beyond Piano: Diff-SFProxy

Code and research demo for **“Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies”** (ISMIR 2026), by Zhanhong He, Hanyu Meng, David (Defeng) Huang, and Roberto Togneri.

**[Live demo and audio](https://zhanh-he.github.io/sfproxy-velocity-estimation/)** · **[Camera-ready paper](paper/2026_ISMIR_Velo_Beyond_Piano_Camera_Ready.pdf)** · **[Paper result data](analysis/README.md)**

This `camera-ready-release` branch is a reviewable release candidate. The implementation comes from the team's `202604_midiproxy` code on lab5090, with later Kaya launch scripts. Large datasets, SoundFonts, and training checkpoints stay outside Git. See [release inventory](RELEASE_INVENTORY.md) for what was found and what still needs recovery or redistribution.

## What the model does

VeloEst takes performance audio and aligned MIDI note events and predicts note velocities. The piano-pretrained front end can be adapted to a target instrument without velocity labels:

1. **Diff-Synth** renders predicted velocities with a frozen differentiable synthesizer and compares the waveform with the recording.
2. **Diff-SFProxy** sends predicted velocities through a frozen Transformer proxy for a SoundFont renderer. Its two targets are pitch-conditioned harmonic energy (PHE) and onset-window spectral flux (OSF).

Only VeloEst is updated during adaptation. In the camera-ready results, Diff-SFProxy gives the strongest guitar loudness-correlation scores: `r_BSSL = 0.794` on GAPS and `0.777` on François Leduc. Guitar note-velocity MAE cannot be reported because those datasets lack ground-truth velocity labels.

## Repository map

| Path | Contents |
| --- | --- |
| [`score_hpt/pytorch`](score_hpt/pytorch) | VeloEst training, Diff-Synth / Diff-SFProxy adaptation, inference, evaluation, configs, tests |
| [`synth-proxy`](synth-proxy) | SoundFont teacher-data export, proxy training and recovery evaluation |
| [`synthesizer`](synthesizer) | DDSP piano and guitar backends used by Diff-Synth |
| [`data_analysis`](data_analysis) | Loudness metrics, MIDI/audio evaluation utilities, dataset statistics |
| [`run_scripts`](run_scripts) | Local research launchers |
| [`kaya_scripts`](kaya_scripts) | Kaya SLURM launchers retained for provenance |
| [`analysis`](analysis) | Paper tables as CSV and reproducible SVG figure generator |
| [`docs`](docs) | Static demo site, matched audio, velocity-colored MIDI example |

## Environment and data

The code was developed for Linux with CUDA, Python 3.11, PyTorch, Hydra, and external `sfizz_render` or FluidSynth renderers. The original environment files are retained in [`score_hpt/environment.yml`](score_hpt/environment.yml) and [`synth-proxy/environment.yml`](synth-proxy/environment.yml). They include machine-specific pinned packages; adapt the CUDA/PyTorch builds to your host.

The core data inputs are aligned audio/MIDI from MAESTRO v3, SMD, GAPS, and François Leduc. Training also requires the relevant SoundFonts (Salamander Grand Piano and Spanish Classical Guitar) and preprocessed HDF5 files. Paths live in [`score_hpt/pytorch/config/config.yaml`](score_hpt/pytorch/config/config.yaml) and the proxy YAMLs under [`synth-proxy/configs`](synth-proxy/configs). Override local paths before training; the inherited defaults point to the original research machines.

The proxy package can be installed from its directory:

```bash
cd synth-proxy
python -m pip install -r requirements.txt
```

`score_hpt` uses its research environment and runs from the `score_hpt` directory. See [`run_scripts/train_route.md`](run_scripts/train_route.md), [`score_hpt/README_scoreinf_proxy.md`](score_hpt/README_scoreinf_proxy.md), and [`synth-proxy/README.md`](synth-proxy/README.md) for entry points. Key commands are:

```bash
# SoundFont proxy teacher data and proxy training (from synth-proxy/)
python src/export_dataset_pkl.py --config-name data_piano paths.workspace_dir=/path/to/workspace paths.analysis_dir=../data_analysis instrument.path=/path/to/SalamanderGrandPianoV3.sfz
python src/train.py --config-name train dataset.train.path=/path/to/train dataset.val.path=/path/to/val
python src/eval.py ckpt_path=/path/to/proxy.ckpt device=cuda

# Velocity adaptation (from score_hpt/; override dataset/backend paths in config)
python pytorch/train_ddsp.py backend.checkpoint=/path/to/ddsp.pt backend.backend_segment_seconds=5
python pytorch/train_proxy.py backend.checkpoint=/path/to/proxy.ckpt backend.backend_segment_seconds=5
```

The camera-ready study uses a 10 s VeloEst front end and 5 s backend crops for its main runs (2 s is an ablation), 22,050 Hz mono audio, 2048-point FFT, and about 100 frames/s. Some inherited YAML defaults still say 2 s, so set `backend.backend_segment_seconds=5` explicitly for the main comparison. The model launch scripts expose the method variants, but this branch does not claim end-to-end reproducibility without the external data and selected checkpoints.

## Evaluation and figures

The guitar primary metric is Pearson `r` between Bark-scale specific loudness (BSSL) contours of the real recording and SoundFont resynthesis. Bark-scale total loudness (BSTL) is the companion metric. The MAESTRO/SMD piano results also include per-note velocity MAE.

```bash
python3 analysis/build_figures.py
```

This regenerates the presentation-ready SVGs and the site's result JSON from the transcribed camera-ready tables. The [analysis notes](analysis/README.md) explain the distinction between paper aggregates and the single audio example.

## Audio demo

Open the [live demo](https://zhanh-he.github.io/sfproxy-velocity-estimation/) or serve [`docs/index.html`](docs/index.html) locally. From the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

The demo compares a 20 s MAESTRO test excerpt (60–80 s of one 2009 performance): human reference audio, Flat 64, a saved 5 s Diff-Synth run, and a saved 5 s Diff-SFProxy run. The three rendered examples use the same Salamander SoundFont and render settings. Each method has a velocity-colored piano roll, audio playback, and a downloadable MIDI excerpt. The [export script](scripts/export_maestro_demo.py) documents how to regenerate those files from the saved prediction MIDI and renders. The score badges are full-recording velocity MAE for that one piece, not the paper's dataset averages.

The MAESTRO excerpt is attributed to Google LLC under CC BY-NC-SA 4.0. Salamander Grand Piano V3 samples are attributed to Alexander Holm / FreePats under CC BY 3.0. The demo is intended for research and non-commercial presentation. Dataset recordings and SoundFont sample libraries are not included beyond the short attributed demo audio.

## Citation

```bibtex
@inproceedings{he2026beyond,
  author = {Zhanhong He and Hanyu Meng and David Defeng Huang and Roberto Togneri},
  title = {Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference},
  year = {2026}
}
```

The `score_hpt` component retains its existing MIT license. Review the license terms for the whole release and upstream DDSP components before merging this branch to `main`.
