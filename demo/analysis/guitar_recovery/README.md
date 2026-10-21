# Archived guitar evaluation evidence

These are copies of `result_summary.txt` and saved prediction MIDIs recovered from lab5090's `/storage/zhanh_storage/transfer1_results.tar.gz` (Diff-SFProxy) and `transfer2_results.tar.gz` (Diff-Synth) on 2026-09-29. Trailing line whitespace was removed from the text files. They identify the adapted 5 s guitar checkpoints, evaluation scope, and full-dataset metrics.

| File | Evaluation | Pearson r_BSSL | Paper rounded |
| --- | --- | ---: | ---: |
| `gaps_diffsynth_5s_result_summary.txt` | GAPS test, 30/30 pieces | 0.669420 | 0.669 |
| `fl_diffsynth_5s_result_summary.txt` | François Leduc full, 79/79 pieces | 0.645489 | 0.646 |
| `fl_diffsfproxy_5s_result_summary.txt` | François Leduc full, 79/79 pieces; recovered 1:0.2:0.4 checkpoint | 0.768812 | 0.777 |

The two Diff-Synth runs reproduce the paper's rounded BSSL values. The recovered FL Diff-SFProxy evaluation **does not** reproduce its paper value. [`fl_diffsfproxy_5s_archived_sweep.csv`](fl_diffsfproxy_5s_archived_sweep.csv) indexes all six archived 5 s FL Route IV summaries; none reaches the paper's 0.777. The demo labels the result as a checkpoint rerun and keeps the paper aggregate figure separate. `fl_diffsfproxy_5s_saved.mid`, `fl_diffsynth_5s_saved.mid`, and `gaps_diffsynth_5s_saved.mid` are the original archived predictions, not the site's complete aligned excerpts. The listening examples are individual 20 s excerpts and do not inherit dataset aggregates. The full recovery account, including the Diff-SFProxy checkpoint and MIDI reconciliation, is in [`../GUITAR_RESULT_RECOVERY.md`](../GUITAR_RESULT_RECOVERY.md).

[`fl_20s_candidate_screen.csv`](fl_20s_candidate_screen.csv) and its [metadata](fl_20s_candidate_screen.meta.json) record the later 79-recording FL excerpt screen using the recovered 5 s checkpoints at authentic input gain. See [`../FL_CANDIDATES.md`](../FL_CANDIDATES.md) for how the six new demo candidates were selected and what the velocity-102 diagnostic found.

[`fl_qdc4c_5s_saved.mid`](fl_qdc4c_5s_saved.mid) is the original archived `QDC4c` prediction from the 5 s, 120k Diff-SFProxy run, extracted on 2026-10-21. Its two published-window velocity-102 fractions are 42.1% and 35.6%, corroborating that the clustering was present in the archived experiment output. The [FL candidate review](../FL_CANDIDATES.md) records the archive member and comparison with the complete rerun.
