# François Leduc excerpt review

The [review page](../docs/fl-review.html) presents six aligned, 20-second FL guitar excerpts from different recordings. The earlier two `QDC4c` examples were retired from the public selector; their diagnostic analysis remains below. The [main demo](../docs/index.html) still presents 00 original recording, 01 Flat 64, 02 Diff-Synth, and 03 Diff-SFProxy for every selection. The guitar score supplies pitch and time, but no verified velocity ground truth.

## What the screen found

[`screen_fl_candidates.py`](../scripts/screen_fl_candidates.py) loaded the same two recovered **5 s** guitar checkpoints used by the existing GAPS/FL demo, ran onset-assisted HPT inference on each of the 79 packed FL recordings on lab5090, and summarized 683 non-overlapping 20 s windows with at least 50 score onsets. Its complete [window CSV](guitar_recovery/fl_20s_candidate_screen.csv) and [checkpoint metadata](guitar_recovery/fl_20s_candidate_screen.meta.json) are retained. The inputs were the original HDF5 waveforms at their original gain. This is a distribution screen, not a note-velocity accuracy evaluation.

The retired `QDC4c` 40–60 s and 80–100 s MIDI files contained 40.0% and 35.2% of notes **exactly at velocity 102**. Among the 570 screened windows with at least 70 notes and source RMS of at least 1,500 int16 units, the median fraction at 102 is 6.6%. Across all 683 windows, source audio RMS and the fraction at 102 correlate at `r = 0.876`. This suggests recording level contributes to the clustering; correlation alone does not establish the complete cause.

The [original archived `QDC4c` prediction MIDI](guitar_recovery/fl_qdc4c_5s_saved.mid) from the **same 5 s, 120k Diff-SFProxy run** also has velocity-102 fractions of 42.1% (40–60 s, 126 notes) and 35.6% (80–100 s, 118 notes). Thus the cluster was already present in the saved experiment output, before the demo rerun or colorized website. The archive member is `transfer1_results/route4/francoisleduc_full_guitar/5s_auto_hpt_onset_score_note_editor_backend_diffproxy_smooth_l1_sup0_backend1_prior0p2_sat0p4_120000_iterations/pred_midis/midi/QDC4c.route2.mid` in `/storage/zhanh_storage/transfer1_results.tar.gz`, SHA-256 `6c91a88eba6c9a6f93f4e3ef13185d7656031d04f383e87fdf7b1eb527fc1dd0`. Its 2,039 notes were matched by pitch and onset/offset within 20 ms to the 2,071-note complete rerun: 1,679 integer velocities agree exactly, with mean absolute difference `0.355/127`. The missing 32 notes explain the different window denominators.

The slide-sized [scatter figure](figures/fl_velocity102_vs_audio_level.svg) highlights all six new candidates and the original `QDC4c` pair; a [PNG export](figures/fl_velocity102_vs_audio_level.png) is included for presentation software. [`plot_fl_saturation.py`](../scripts/plot_fl_saturation.py) generates both from the retained window CSV.

As a diagnostic, halving the original `QDC4c` waveform before feeding the **same checkpoint and score** reduced the screen's 40–60 s fraction at 102 from 39.7% to 14.5%, and its 80–100 s fraction from 35.0% to 8.3%. This confirms input-gain sensitivity in inference. The gain-adjusted predictions are **not** used in the public demo or paper comparisons; all retained demo MIDI uses the authentic-gain prediction. Post-rendering listening gain is one constant scalar per MP3 and cannot change its MIDI colors.

## Excerpts available for visual choice

The six new candidates were chosen before computing the excerpt BSSL/BSTL scores, spanning low, typical, and higher velocity-102 clustering. All use the original FL audio/score, the same FreePats guitar SFZ, and the same recovered checkpoints. Card scores compare each rendered 20 s MP3 against its original recording after listening-gain matching; they are not the paper's Table 2 aggregates.

| Case | Notes | Diff-SFProxy at 102 | Diff-Synth BSSL | Diff-SFProxy BSSL | Diff-SFProxy BSTL |
| --- | ---: | ---: | ---: | ---: | ---: |
| [`5xh4c`, 140–160 s](../docs/index.html?case=fl-5xh4c-140#listen) | 197 | 6.1% | 0.557 | 0.675 | 0.619 |
| [`mwC4c`, 0–20 s](../docs/index.html?case=fl-mwc4c-0#listen) | 166 | 0.0% | 0.697 | 0.704 | 0.684 |
| [`yfC4c`, 160–180 s](../docs/index.html?case=fl-yfc4c-160#listen) | 174 | 0.0% | 0.420 | 0.532 | 0.569 |
| [`cDC4c`, 40–60 s](../docs/index.html?case=fl-cdc4c-40#listen) | 159 | 0.0% | 0.555 | 0.806 | 0.695 |
| [`LDC4c`, 80–100 s](../docs/index.html?case=fl-ldc4c-80#listen) | 148 | 6.8% | 0.695 | 0.789 | 0.586 |
| [`WWh4c`, 100–120 s](../docs/index.html?case=fl-wwh4c-100#listen) | 131 | 19.8% | 0.666 | 0.775 | 0.725 |

The published Diff-Synth checkpoint is SHA-256 `1712310c5b662685f7ed03371e52dd491a48e9c242d6557548a314f20ea860ee` (5 s, 60k), and Diff-SFProxy is `2d1e3d0dff33088689ae0a10d318b369625fa89b184e36791b1c943880e38131` (5 s, 120k). Each complete source pair was exported from its HDF5 file; [`prepare_fl_review.py`](../scripts/prepare_fl_review.py) reran [`infer_compare.py`](../../scripts/infer_compare.py) with `onset_only` velocities and built the 20 s cases. The [gain](listening_gain_report.csv) and [score](listening_case_scores.csv) reports cover the exact published MP3s. [`validate_listening_cases.py`](../scripts/validate_listening_cases.py) checks note alignment and duration.

The recovered Diff-SFProxy checkpoint matches the paper's 5 s and 1:0.2:0.4 recipe. Its archived FL dataset result is `r_BSSL = 0.768812`, while the paper reports `0.777`. The exact paper-selected checkpoint has not been identified, so these examples are labeled as checkpoint reruns rather than claimed to be its exact saved outputs. See the [guitar recovery record](GUITAR_RESULT_RECOVERY.md). No alternate 2 s model, velocity remapping, or visually selected weight has been substituted.
