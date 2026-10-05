# Recovered guitar checkpoint inference example

This runnable example uses GAPS test piece `019_Vpswc` on lab5090. `scripts/infer_compare.py` processed its complete 156 s aligned audio/MIDI pair with the piano-pretrained VeloEst checkpoint and the two recovered adapted 5 s guitar checkpoints. The 40–60 s passage in [`comparison.svg`](comparison.svg) shows Flat 64, zero-shot VeloEst, VeloEst+Diff-Synth, and VeloEst+Diff-SFProxy on the same 71 score notes. The [full-piece CSV](notes.csv) and four full-piece MIDIs contain 462 notes.

The score has no verified guitar note velocities. The figure therefore reports mean predicted velocity, not note-velocity MAE. The paper's dataset-level correlation values are in [`../paper_results.csv`](../paper_results.csv), not computed from this individual visualization. `summary.json` records checkpoint SHA-256 hashes and the inference settings. The recovered weight paths and archived evaluation evidence are in [`../GUITAR_RESULT_RECOVERY.md`](../GUITAR_RESULT_RECOVERY.md).

Regenerate with the aligned source pair, the piano checkpoint, and the separately backed-up guitar checkpoints:

```bash
python scripts/infer_compare.py \
  --audio /path/to/019_Vpswc.wav \
  --midi /path/to/019_Vpswc.mid \
  --veloest-ckpt score_hpt/checkpoints/veloest_onset_only_120k.pth \
  --diffsynth-ckpt /path/to/guitar_diffsynth_5s_60k.pth \
  --diffsfproxy-ckpt /path/to/guitar_diffsfproxy_5s_120k.pth \
  --plot-start 40 --plot-seconds 20 \
  --out /path/to/guitar_comparison
```
