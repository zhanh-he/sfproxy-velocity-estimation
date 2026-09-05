# SFProxy Velocity Estimation

Official implementation of **“Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies”**, accepted at **ISMIR 2026**.

- Too busy in recent ... will cleanup code and provide DEMO here ~ mid Septermber :)
- Codes before cleanup are ready in https://github.com/zhanh-he/202604_midiproxy

## Overview

Many music datasets provide aligned audio and MIDI note events but lack reliable note-level velocity labels, particularly outside the piano domain. This repository studies cross-instrument MIDI velocity estimation in this label-scarce setting.

Starting from a velocity estimator trained on piano, we adapt it to target instruments using real performance audio. The goal is to predict velocities whose rendering matches the dynamics of the recording.

We compare two adaptation strategies:

- **Diff-Synth**: waveform-domain supervision through a differentiable synthesiser.
- **Diff-SFProxy**: note-wise supervision through a differentiable proxy of a non-differentiable SoundFont renderer.

Diff-SFProxy predicts two loudness-related acoustic parameters:

- **Pitch-conditioned harmonic energy (PHE)**
- **Onset-window spectral flux (OSF)**

This focuses the adaptation signal on velocity-dependent intensity and attack behaviour rather than full waveform reconstruction.

## Paper

**Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies**  
ISMIR 2026

Paper and supplementary materials will be added here when available.

## Method

The framework contains three main components:

1. **VeloEst**  
   A score-informed MIDI velocity estimator pretrained with piano velocity labels.

2. **Diff-Synth**  
   A frozen differentiable waveform synthesiser used to adapt VeloEst through an audio reconstruction objective.

3. **Diff-SFProxy**  
   A frozen differentiable proxy trained to reproduce the PHE and OSF responses of a selected SoundFont renderer.

During target-instrument adaptation, only VeloEst is updated.

## Experiments

The paper evaluates the framework on piano and guitar recordings using:

- **MAESTRO**
- **Saarland Music Data (SMD)**
- **GAPS**
- **François Leduc guitar dataset**

The results show that Diff-SFProxy improves guitar velocity estimation over zero-shot transfer, whereas waveform-domain Diff-Synth adaptation can degrade performance under timbral mismatch.

## Repository Status

The camera-ready implementation and documentation are being prepared.

Planned release contents:

- Training and adaptation code
- Diff-SFProxy training scripts
- Pretrained checkpoints
- Evaluation scripts
- Configuration files
- Audio examples

## Installation

Installation instructions will be added with the code release.

## Usage

Training, adaptation, inference, and evaluation commands will be added with the code release.

## Citation

Please cite the paper if you use this repository:

```bibtex
@inproceedings{he2026beyond,
  title     = {Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference},
  year      = {2026}
}
```

The BibTeX entry will be updated with the final author list and proceedings metadata.

## Licence

The licence will be specified with the public code release.

## Contact

For questions, please open a GitHub issue.
