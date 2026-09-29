# Score-HPT / VeloEst

This is the velocity estimator and adaptation code used by the paper. Its starting point is our earlier [score-informed-amt](https://github.com/zhanh-he/score-informed-amt) project, *Score-Informed Transformer for Refining MIDI Velocity in Automatic Music Transcription* (SMC 2026). That project provides the Score-HPT architecture, `note_editor` score conditioning, multiple frontend and input options, and an example pretrained checkpoint. The inherited MIT license is in [`LICENSE`](LICENSE). We acknowledge that model and code here; the Diff-Synth and Diff-SFProxy adaptation routes are the additions for this ISMIR paper.

## Which Score-HPT option is used here?

The main configuration is **HPT with onset-assisted note editing**:

```text
model.type=hpt
score_informed.method=note_editor
model.input2=onset
model.input3=null
```

The aligned MIDI supplies note identities and timing, the audio supplies performance evidence, and the frontend predicts note velocities. The original [score-informed-amt README](https://github.com/zhanh-he/score-informed-amt) describes other branches (`hppnet`, `dynest`, direct estimation) and `note_editor` inputs (`frame` / `exframe`). For this paper we selected the **onset-only** piano option. Its 120k checkpoint differs slightly from the upstream repository's public onset+frame 100k example; both projects share authors. We thank the earlier Score-Informed AMT work for the model and release the two paper-era piano weights in [`checkpoints/`](checkpoints/README.md).

## Use the released checkpoints

From the repository root, with a performance recording and its aligned piano score:

```bash
python scripts/infer_compare.py \
  --audio /path/to/performance.wav \
  --midi /path/to/aligned_score.mid \
  --out /path/to/comparison
```

This writes `veloest.mid`, `diffsfproxy.mid`, `flat64.mid`, `notes.csv`, `comparison.svg`, and `summary.json`. Add `--sfz /path/to/SalamanderGrandPianoV3.sfz` to render WAV and MP3 files with `sfizz_render`. If the input MIDI contains true performance velocities, add `--reference-velocities` to report MAE; otherwise the input velocities are treated as score placeholders. See the [checkpoint details](checkpoints/README.md) and [inference script](../scripts/infer_compare.py).

## Code map

- `pytorch/config/config.yaml`: datasets, HPT inputs, loss weights, and backend paths.
- `pytorch/train.py`: supervised piano frontend training.
- `pytorch/train_ddsp.py`: Route III adaptation through frozen DDSP audio reconstruction.
- `pytorch/train_proxy.py`: Route IV adaptation through frozen Diff-SFProxy PHE/OSF features.
- `pytorch/direct_invension/`: prediction and evaluation jobs for the route variants.
- `pytorch/proxy/`: frozen backend adapters and losses.
- `pytorch/tests/`: frontend, evaluation, and proxy-gradient tests.
- `checkpoints/`: the released VeloEst and VeloEst+Diff-SFProxy piano weights.
- [`README_scoreinf_proxy.md`](README_scoreinf_proxy.md): implementation notes and loss details retained from the research workspace.

## Setup and example

Use a Linux/CUDA PyTorch environment compatible with the machine; [`environment.yml`](environment.yml) records the historical setup. Edit or override the dataset roots and `exp.workspace` in `pytorch/config/config.yaml`, then pack aligned data to HDF5. See [`../scripts/README.md`](../scripts/README.md) for the 5090/Kaya launchers. Run Python commands from this `score_hpt/` directory so the relative config path resolves:

```bash
python pytorch/train_proxy.py \
  model.type=hpt score_informed.method=note_editor \
  model.input2=onset model.input3=null \
  model.frontend_pretrained_mode=route2_piano_specific \
  model.frontend_pretrained=checkpoints/veloest_onset_only_120k.pth \
  backend.checkpoint=/path/to/soundfont_proxy.ckpt \
  backend.backend_segment_seconds=5
```

For the Diff-Synth baseline, use `python pytorch/train_ddsp.py` with its matching DDSP checkpoint and the same `backend.backend_segment_seconds=5`. Default backend source paths resolve to `../diff-sfproxy/` and `../diff-synth/ddsp-*/`; explicit `backend.project_root` overrides remain available. The HPT input window is 10 s. Some inherited YAML defaults use a 2 s backend crop, so specify 5 s for the main paper setting.

## Acknowledgement

Please cite both the ISMIR 2026 *Beyond Piano* paper and the earlier [Score-Informed AMT](https://github.com/zhanh-he/score-informed-amt) work when reusing this frontend. Its public repository also acknowledges FiLM-UNet and Transkun v2 code and pretrained weights used for benchmark comparisons. Local adaptation weights and dataset renderers are separate components with their own provenance.
