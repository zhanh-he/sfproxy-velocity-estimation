# Demo preparation scripts

These scripts turn selected research-workspace results into the small, reviewable assets in `demo/docs/assets/`. Run them from the repository root. The source datasets, full experiment outputs, SoundFonts, and `sfizz_render` binary live outside this repository; see [`../analysis/LISTENING_CASES.md`](../analysis/LISTENING_CASES.md) for the recovered source paths and case provenance.

| Script | Purpose |
| --- | --- |
| `export_pair_from_h5.py` | Extract an aligned WAV/MIDI pair from a Score-HPT HDF5 item. Requires `h5py`, `numpy`, and `pretty_midi`. |
| `build_listening_case.py` | Clip one pair and any available saved prediction MIDIs to **exactly 20 s**, verify note alignment, render matching SoundFont audio, and write `notes.json`. Requires `pretty_midi`, `soundfile`, `numpy`, `ffmpeg`, and `sfizz_render`. Pass `--velocity-ground-truth` only for a dataset with measured MIDI velocities. |
| `validate_listening_cases.py` | Check every manifest entry, advertised file, note alignment, and audio duration. Requires `ffprobe`. |
| `match_listening_gain.py` | Apply one constant gain to published MP3s and write the gain report. Pass repeatable `--case-id` to update selected cases while keeping report rows for the rest. Requires `ffmpeg`; it does not alter MIDI or within-clip dynamics. |
| `score_listening_cases.py` | Compute each 20 s card's BSSL/BSTL Pearson correlations from the hosted MP3s and piano MAE from the aligned note JSON. `--write-manifest` updates the site's card scores and writes `analysis/listening_case_scores.csv`. Requires the research analysis environment with `torch`, `torchaudio`, and `librosa`. Run after matching listening gain. |
| `export_maestro_demo.py` | Original MAESTRO example exporter using the saved evaluation manifest and renders. |
| `update_checkpoint_demo.py` | Replace the original MAESTRO example's two checkpoint-derived MIDI/audio conditions. |
| `inventory_eval_summaries.py` | Index saved run summaries from a Score-HPT workspace for analysis. It does not identify the paper's selected runs automatically. |

The website reads [`../docs/assets/cases.json`](../docs/assets/cases.json). To add a case, first recover its real aligned source pair and the selected final prediction files, build the assets, then add a manifest entry with only the methods actually present. Run:

```bash
python3 demo/scripts/validate_listening_cases.py
python3 -m http.server 8000 --directory demo/docs
```

Open `http://localhost:8000/` to inspect all dataset selections. GAPS and FL each have two complete 20 s, four-method guitar comparisons. The recovered 5 s guitar checkpoint and prediction provenance is recorded in [`../analysis/GUITAR_RESULT_RECOVERY.md`](../analysis/GUITAR_RESULT_RECOVERY.md); the older 2 s evaluation runs are not used.
