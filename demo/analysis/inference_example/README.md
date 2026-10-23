# Matched checkpoint inference example

The source is one **MAESTRO v3 test performance** (2009, Scriabin, *Entragete*, Op. 63) retained in the lab5090 Score-HPT HDF5 cache. Its 22,050 Hz waveform and MIDI events were exported to a temporary aligned WAV/MIDI pair with [`../../scripts/export_pair_from_h5.py`](../../scripts/export_pair_from_h5.py). Both released piano checkpoints were run on that same pair with [`../../../scripts/infer_compare.py`](../../../scripts/infer_compare.py), using onset-only velocity picking. The full piece has **721 notes** and 164.7 s of audio. The listening website shows the 60–80 s excerpt, with **137 aligned notes** in its visual data.

| Method | Full-piece velocity MAE (0–127) | Source |
| --- | ---: | --- |
| Flat 64 | 15.401 | Constant-velocity control |
| VeloEst | 4.060 | Released piano onset-only 120k checkpoint |
| VeloEst + Diff-SFProxy | 9.738 | Released piano 5 s adaptation 120k checkpoint |

The MAE values are recomputed from [`notes.csv`](notes.csv), which includes ground-truth/input velocity and both predicted velocities for every note. The [slide-ready comparison SVG](../../docs/assets/checkpoint_comparison.svg) uses a common time and pitch scale and shows the 60–80 s excerpt. These are **single-piece inference results**, not the dataset-level paper Table 2 metrics. The site's Diff-Synth card comes from a previously saved 60k run rather than one of these two released checkpoints.

To regenerate on a machine with the external HDF5 cache and Salamander SFZ:

```bash
python demo/scripts/export_pair_from_h5.py /path/to/maestro_piece.h5 /tmp/beyond-piano/pair
python scripts/infer_compare.py \
  --audio /tmp/beyond-piano/pair.wav --midi /tmp/beyond-piano/pair.mid \
  --out /tmp/beyond-piano/comparison --reference-velocities \
  --plot-start 60 --plot-seconds 20
python demo/scripts/update_checkpoint_demo.py \
  --veloest-midi /tmp/beyond-piano/comparison/veloest.mid \
  --diffsfproxy-midi /tmp/beyond-piano/comparison/diffsfproxy.mid \
  --assets demo/docs/assets \
  --sfz /path/to/SalamanderGrandPianoV3.sfz \
  --sfizz-render /path/to/sfizz_render
```

The original HDF5 file was under `score_hpt/workspaces/hdf5s/maestro_sr22050/2009/` in the 5090 data tree and is not included in Git. Source MIDI event timestamps can shift by a few milliseconds after re-export to ticks; the demo's chart coordinates were matched back to the reference notes by pitch and nearest onset within 10 ms. The rendered audio and downloadable MIDI retain the re-exported note times. Checkpoint hashes are listed in [`../../../score_hpt/checkpoints/README.md`](../../../score_hpt/checkpoints/README.md).
