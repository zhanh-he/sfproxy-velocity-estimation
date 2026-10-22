# Paper result data

`paper_results.csv` transcribes Table 2 (evaluation across MAESTRO, SMD, GAPS, and FL) from the supplied camera-ready manuscript. `proxy_recovery.csv` transcribes Table 1. They are paper aggregates, not a recomputation from the example audio files.

[`FL_CANDIDATES.md`](FL_CANDIDATES.md) documents the expanded François Leduc 20 s listening review, its 79-recording screen, the original example's velocity-102 anomaly, and the recovered 5 s checkpoint hashes.

The 5 s adapted guitar checkpoints and saved Diff-Synth result summaries were recovered from the second lab5090 HDD; see [`GUITAR_RESULT_RECOVERY.md`](GUITAR_RESULT_RECOVERY.md) and [`guitar_recovery/`](guitar_recovery/README.md) for their hashes and evaluation checks.

The primary guitar metric is Pearson correlation between real and SoundFont-rendered Bark-scale specific loudness (`r_BSSL`); `r_BSTL` is the companion total-loudness metric. Guitar datasets have no ground-truth velocity labels, so there is no guitar velocity MAE. The `5 s` adaptation backend is the main method; `2 s` is an ablation. MAESTRO/SMD have ground-truth velocity labels.

Run `python3 demo/analysis/build_figures.py` from the repository root to regenerate `demo/docs/assets/guitar_results.svg`, `demo/docs/assets/recovery.svg`, and `demo/docs/assets/paper_results.json`. The script uses only the Python standard library. The existing `src/data_analysis/` package, `stats/`, `tests/`, and `notebooks/` hold the underlying evaluation and exploratory analysis code; [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) documents those tools.

The available 5090 workspace includes saved prediction MIDI and renders for a matched MAESTRO test piece used in the listening demo. The two released piano checkpoints have also been rerun on this piece; [`inference_example/`](inference_example/README.md) contains the 721-note comparison CSV, measured MAE, source procedure, and a slide-ready figure. [`guitar_inference_example/`](guitar_inference_example/README.md) contains a four-approach, slide-ready GAPS figure and full-piece MIDI/CSV from the recovered 5 s adapted models. [The listening-case inventory](LISTENING_CASES.md) records the MAESTRO, SMD, FL, and GAPS 20 s selections; [the card score CSV](listening_case_scores.csv) contains per-excerpt MAE (piano) and BSSL/BSTL correlations (all four datasets). Single-case values are separate from the rounded paper aggregates.

Source: *Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies*, ISMIR 2026 camera-ready manuscript, Tables 1–2.
