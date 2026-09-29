# Paper result data

`paper_results.csv` transcribes Table 2 (evaluation across MAESTRO, SMD, GAPS, and FL) from the supplied camera-ready manuscript. `proxy_recovery.csv` transcribes Table 1. They are paper aggregates, not a recomputation from the example audio files.

`5090_evaluation_inventory.csv` indexes 28 saved evaluation summaries found in the lab5090 workspace. It includes run names, checkpoint paths relative to that workspace, evaluated item counts, failures, velocity MAE (when available), and BSSL/BSTL Pearson correlations. It is an **exploratory run inventory**, not the selected paper table. The two guitar entries are 2 s proxy runs; the inventory does not contain the paper's final 5 s guitar comparison. Rebuild it on the source machine with `python3 demo/scripts/inventory_eval_summaries.py /path/to/score_hpt/workspaces demo/analysis/5090_evaluation_inventory.csv`.

The primary guitar metric is Pearson correlation between real and SoundFont-rendered Bark-scale specific loudness (`r_BSSL`); `r_BSTL` is the companion total-loudness metric. Guitar datasets have no ground-truth velocity labels, so there is no guitar velocity MAE. The `5 s` adaptation backend is the main method; `2 s` is an ablation. MAESTRO/SMD have ground-truth velocity labels.

Run `python3 demo/analysis/build_figures.py` from the repository root to regenerate `demo/docs/assets/guitar_results.svg`, `demo/docs/assets/recovery.svg`, and `demo/docs/assets/paper_results.json`. The script uses only the Python standard library. The existing `src/data_analysis/` package, `stats/`, `tests/`, and `notebooks/` hold the underlying evaluation and exploratory analysis code; [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) documents those tools.

The available 5090 workspace includes saved prediction MIDI and renders for a matched MAESTRO test piece used in the listening demo. The two released piano checkpoints have also been rerun on this piece; [`inference_example/`](inference_example/README.md) contains the 721-note comparison CSV, measured MAE, source procedure, and a slide-ready figure. [The listening-case inventory](LISTENING_CASES.md) records the MAESTRO, SMD, FL, and GAPS 20 s slots, their sources, and remaining gaps. Single-case values are separate from the rounded paper aggregates. The 3090 results are unavailable.

Source: *Beyond Piano: Cross-Instrument MIDI Velocity Estimation via Differentiable SoundFont Proxies*, ISMIR 2026 camera-ready manuscript, Tables 1–2. All decimal values should be checked once more against the final proceedings version before merging to `main`.
