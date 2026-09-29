# Demo preparation scripts

These scripts turn selected research-workspace results into the small, reviewable assets in `demo/docs/assets/`. Run them from the repository root. The source datasets, full experiment outputs, SoundFonts, and `sfizz_render` binary live outside this repository; see [`../analysis/LISTENING_CASES.md`](../analysis/LISTENING_CASES.md) for the recovered source paths and case provenance.

| Script | Purpose |
| --- | --- |
| `export_pair_from_h5.py` | Extract an aligned WAV/MIDI pair from a Score-HPT HDF5 item. Requires `h5py`, `numpy`, and `pretty_midi`. |
| `build_listening_case.py` | Clip one pair and any available saved prediction MIDIs to **exactly 20 s**, verify note alignment, render matching SoundFont audio, and write `notes.json`. Requires `pretty_midi`, `soundfile`, `numpy`, `ffmpeg`, and `sfizz_render`. Pass `--velocity-ground-truth` only for a dataset with measured MIDI velocities. |
| `validate_listening_cases.py` | Check every manifest entry, advertised file, note alignment, and audio duration. Requires `ffprobe`. |
| `match_listening_gain.py` | Apply one constant gain to each published MP3 and write the gain report. Requires `ffmpeg`; it does not alter the MIDI or within-clip dynamics. |
| `export_maestro_demo.py` | Original MAESTRO example exporter using the saved evaluation manifest and renders. |
| `update_checkpoint_demo.py` | Replace the original MAESTRO example's two checkpoint-derived MIDI/audio conditions. |
| `inventory_eval_summaries.py` | Index saved run summaries from a Score-HPT workspace for analysis. It does not identify the paper's selected runs automatically. |

The website reads [`../docs/assets/cases.json`](../docs/assets/cases.json). To add a case, first recover its real aligned source pair and the selected final prediction files, build the assets, then add a manifest entry with only the methods actually present. Run:

```bash
python3 demo/scripts/validate_listening_cases.py
python3 -m http.server 8000 --directory demo/docs
```

Open `http://localhost:8000/` to inspect all dataset selections. The FL method slots await selected final 5 s guitar predictions. The GAPS slot intentionally has no hosted media under that dataset's publication terms; do not fill it by copying source data into this public repository without permission.
