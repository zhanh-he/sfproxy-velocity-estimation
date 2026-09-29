# Diff-Synth baselines

This directory contains the two differentiable synthesizer backends used for the waveform-reconstruction comparison. They are trained separately and then loaded **frozen** by `score_hpt/pytorch/train_ddsp.py`; only VeloEst is updated during velocity adaptation.

| Subdirectory | Provenance | Used for |
| --- | --- | --- |
| [`ddsp-piano-pytorch/`](ddsp-piano-pytorch/README.md) | Adapted from [ytsrt66589/ddsp-piano-pytorch](https://github.com/ytsrt66589/ddsp-piano-pytorch), a PyTorch implementation of the DDSP-Piano model | MAESTRO / SMD piano backend |
| [`ddsp-guitar-synth/`](ddsp-guitar-synth/README.md) | Adapted from [andywiggins/ddsp-guitar-synth](https://github.com/andywiggins/ddsp-guitar-synth), *A Differentiable Acoustic Guitar Model for String-Specific Polyphonic Synthesis* | GAPS / François Leduc guitar backend |

The upstream projects supplied the synthesizer architectures. This research snapshot adds dataset-preparation and training wrappers, 22,050 Hz / approximately 100 fps support, and the checkpoint contract used by the Score-HPT backend adapters. The two subdirectory READMEs retain their detailed training commands and parameter notes. Please cite the corresponding original DDSP work as well as the *Beyond Piano* paper when using these baselines.

## Train the backends

Run from the repository root after placing the datasets on disk and setting the environment variables in the launchers:

```bash
bash scripts/ddsp/train_ddsp_piano.sh 5
bash scripts/ddsp/train_ddsp_guitar_synth.sh 5
```

The piano launcher prepares MAESTRO caches; the guitar launcher prepares GuitarSet-style data. Their defaults are research-machine paths, so set the dataset and workspace paths before running. The subdirectory READMEs give direct `preprocess.py` / `train.py` and guitar `prepare_*_unified.py` / `train_midi_synth_unified.py` commands for independent use.

## Load a frozen backend for adaptation

From `score_hpt/`, pass the matching checkpoint and a 5 s backend crop:

```bash
python pytorch/train_ddsp.py \
  backend.checkpoint=/path/to/ddsp_checkpoint.pt \
  backend.backend_segment_seconds=5
```

`score_hpt/pytorch/proxy/ddsp_piano.py` and `ddsp_guitar_synth.py` resolve these two source directories by default. Saved backend weights and training data are not included in Git; recovered locations are recorded in [`../demo/analysis/PROVENANCE.md`](../demo/analysis/PROVENANCE.md).
