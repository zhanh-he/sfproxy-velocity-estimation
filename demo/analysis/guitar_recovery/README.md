# Archived guitar evaluation evidence

These are copies of `result_summary.txt` recovered from lab5090's `/storage/zhanh_storage/transfer2_results.tar.gz` on 2026-09-29, with trailing line whitespace removed. They identify the adapted 5 s Diff-Synth guitar checkpoint, evaluation scope, and full-dataset metrics. `r_BSSL` Pearson values round to the camera-ready Table 2 entries.

| File | Evaluation | Pearson r_BSSL | Paper rounded |
| --- | --- | ---: | ---: |
| `gaps_diffsynth_5s_result_summary.txt` | GAPS test, 30/30 pieces | 0.669420 | 0.669 |
| `fl_diffsynth_5s_result_summary.txt` | François Leduc full, 79/79 pieces | 0.645489 | 0.646 |

The listening examples are individual 20 s excerpts and do not inherit these aggregate metric values. The full recovery account, including the Diff-SFProxy checkpoint and the GAPS MIDI reconciliation, is in [`../GUITAR_RESULT_RECOVERY.md`](../GUITAR_RESULT_RECOVERY.md).
