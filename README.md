# SFProxy Velocity Estimation

Official implementation of **“Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies” [PDF](demo/paper/2026_ISMIR_Velo_Beyond_Piano_Camera_Ready.pdf)**, accepted at **ISMIR 2026**. This repo contains:

- **[DEMO Audio & MIDI](https://zhanh-he.github.io/sfproxy-velocity-estimation/)**
- **[Checkpoints and Inference Instructions](score_hpt/checkpoints/README.md)**

## Overview
Many music datasets provide aligned audio and MIDI note events but lack reliable velocity labels, particularly out-of-the-piano. This repository studies cross-instrument MIDI velocity estimation in this label-scarce setting.

Starting from a velocity estimator pretrained on piano, we adapt it to target instruments using real performance audio. The goal is to predict velocities whose rendering matches the dynamics of the recording. We compare two adaptation strategies:

- **Diff-Synth**: waveform-domain supervision through a differentiable synthesiser.
- **Diff-SFProxy**: note-wise supervision through a differentiable proxy of a non-differentiable SoundFont renderer.

Diff-SFProxy predicts two loudness-related acoustic parameters:

- **Pitch-conditioned harmonic energy (PHE)**
- **Onset-window spectral flux (OSF)**

This focuses the adaptation signal on velocity-dependent intensity and attack behaviour rather than full waveform reconstruction.


## Repository layout

| Directory | Purpose |
| --- | --- |
| [`demo/`](demo/README.md) | Listening website, matched MIDI/audio, paper PDF and LaTeX source, result tables, analysis code, and slide-ready SVG figures |
| [`diff-sfproxy/`](diff-sfproxy/README.md) | SoundFont teacher-data export, note-wise proxy training and recovery evaluation; includes SFZ/`sfizz_render` setup |
| [`score_hpt/`](score_hpt/README.md) | Onset-assisted Score-HPT/VeloEst frontend and Diff-Synth/Diff-SFProxy adaptation, inference, evaluation, and tests |
| [`diff-synth/`](diff-synth/README.md) | Adapted piano and guitar differentiable synthesizer backends, with original-project acknowledgements |
| [`scripts/`](scripts/README.md) | Experiment launchers for local machines and Kaya |

The code starts from the team's previous [cross-machine research repository](https://github.com/zhanh-he/202604_midiproxy). This layout separates the paper and presentation evidence from the training implementations. The two VeloEst inference checkpoints are included; large datasets, SoundFont sample libraries, and backend training checkpoints remain external. The [provenance inventory](demo/analysis/PROVENANCE.md) records recovered files and unresolved gaps.

## Typical order of work

1. Prepare MAESTRO/SMD/GAPS/François Leduc data and a local SoundFont. See [`diff-sfproxy/README.md`](diff-sfproxy/README.md) for the SFZ renderer and paths.
2. Train or supply a SoundFont proxy checkpoint with `diff-sfproxy/` and `scripts/sfproxy/`.
3. For immediate inference, run [`scripts/infer_compare.py`](scripts/infer_compare.py) with aligned piano audio/MIDI; its two VeloEst checkpoints are bundled. For retraining and the Diff-Synth comparison, supply the relevant backend checkpoints. See `score_hpt/` and `diff-synth/`.
4. Run the method launchers under `scripts/route/` or `scripts/kaya/`.
5. Rebuild paper figures with `python3 demo/analysis/build_figures.py`. Serve the demo locally with `python3 -m http.server 8000 --directory demo/docs`.

The main paper comparison uses a 10 s HPT input window and a 5 s backend crop. Some inherited configuration defaults remain at 2 s; set `backend.backend_segment_seconds=5` for the main comparison. The released piano inference path is self-contained apart from Python dependencies; end-to-end retraining still requires external data and backend checkpoints.

## Citation and release status

```bibtex
@inproceedings{he2026beyond,
  author = {Zhanhong He and Hanyu Meng and David Defeng Huang and Roberto Togneri},
  title = {Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference},
  year = {2026}
}
```

See each component README for upstream attribution. The final 5 s guitar prediction files and selected checkpoint list have not been recovered from the inspected machines; the website labels paper aggregates separately from its single MAESTRO audio example.
