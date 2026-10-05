# Listening demo cases

The public site reads [`../docs/assets/cases.json`](../docs/assets/cases.json). Every selection is **exactly 20 seconds** and presents four playable positions: 00 original recording, 01 Flat 64, 02 Diff-Synth, 03 Diff-SFProxy. The case menu can be extended by adding another manifest entry and its asset directory.

| Dataset | 20 s windows | 00 / 01 | 02 / 03 | Velocity reference |
| --- | --- | --- | --- | --- |
| MAESTRO v3 test | Scriabin, 60–80 s and 100–120 s | Ready | Ready; Diff-Synth saved 5 s 60k, Diff-SFProxy rerun from released piano 5 s 120k | Yes; MAE uses each selected 20 s window |
| SMD | Bach BWV 849, 40–60 s and 80–100 s | Ready | Ready; saved piano 5 s 60k Diff-Synth and 5 s 120k Diff-SFProxy runs | Yes; MAE uses each selected 20 s window |
| François Leduc (FL) | `QDC4c`, 40–60 s and 80–100 s | Ready | Ready; both models rerun from recovered 5 s guitar checkpoints | No; aligned score velocities are not ground truth |
| GAPS test | `019_Vpswc`, 40–60 s and 80–100 s | Ready | Ready; both methods rerun from recovered 5 s adapted checkpoints | No; aligned score velocities are not ground truth |

## Recovered source paths

The SMD audio/MIDI pair was reconstructed with [`../scripts/export_pair_from_h5.py`](../scripts/export_pair_from_h5.py) from `score_hpt/workspaces/hdf5s/smd_sr22050/Bach_BWV849-01_001_20090916-SMD.h5` on lab5090. Its saved prediction MIDIs are under:

- `score_hpt/workspaces/route3/smd_full_piano/5s_hpt_onset_score_note_editor_backend_diffsynth_piano_piano_ssm_spectral_sup0_backend1_prior0p1_sat0p2_60000_iterations/pred_midis/`
- `score_hpt/workspaces/route4/smd_full_piano/5s_hpt_onset_score_note_editor_backend_diffproxy_smooth_l1_sup0p5_backend0p5_prior0p1_sat0p2_120000_iterations/pred_midis/`

Both use the key `Bach_BWV849-01_001_20090916-SMD.route2.mid`. The SMD 40–60 s case has 113 aligned notes, with Flat 64 / Diff-Synth / Diff-SFProxy MAE of 13.460 / 11.735 / 6.735. The 80–100 s case has 108 aligned notes, with MAE 14.824 / 11.435 / 5.250. These are case values, **not** the paper's dataset aggregate, and the saved runs have not been identified as its selected Table 2 checkpoints.

The MAESTRO 100–120 s case uses the same packed Scriabin recording as the first case and contains 117 aligned notes. Its 20 s Flat 64 / Diff-Synth / Diff-SFProxy MAE is 12.658 / 16.026 / 8.487. The first 60–80 s case contains 137 notes. Both cases display excerpt MAE on their cards; the separate full-piece 721-note inference analysis is still available.

The FL reference pair was reconstructed from `score_hpt/workspaces/hdf5s/francoisleduc_sr22050/QDC4c.h5`. Its two 20 s windows contain 135 and 122 score notes. Diff-Synth and Diff-SFProxy were rerun across the full piece from the recovered 5 s guitar checkpoints in `transfer2_ckpts.tar.gz` and `transfer1_ckpts.tar.gz`, respectively. The [inference summary](guitar_recovery/fl_qdc4c_inference_summary.json) records the 2,071-note full-piece pass and checkpoint hashes. Both complete prediction MIDIs keep the same score notes as the reference. The archived FL evaluation for this Diff-SFProxy checkpoint reports `r_BSSL = 0.768812`; it is not the paper's `0.777` aggregate. All three resynthesized clips use the FreePats Spanish Classical Guitar SFZ. Guitar note velocities are not evaluated against the score.

The GAPS reference pair was reconstructed from `score_hpt/workspaces/hdf5s/gaps_sr22050_old/019_Vpswc.h5` on lab5090. Its 40–60 s and 80–100 s windows contain 71 and 72 aligned score notes. Both methods were rerun from the recovered 5 s adapted guitar checkpoints. The saved Diff-Synth MIDI for this piece omitted some score notes, so the complete rerun supplies the public example; the [recovery record](GUITAR_RESULT_RECOVERY.md) gives the overlap check and hashes. All three resynthesized clips use the FreePats Spanish Classical Guitar SFZ. The project team confirmed that its GAPS access, obtained through the dataset's Zenodo application process, permits publication of these research demo excerpts. Guitar note velocities are not evaluated against the score.

Run [`../scripts/build_listening_case.py`](../scripts/build_listening_case.py) with the external source pair, available prediction MIDIs, `--start`, the matching SFZ, and `sfizz_render` to regenerate a case directory. The builder checks note pitch/onset/duration alignment and writes clipped MIDI, 20 s MP3, note JSON, and excerpt MAE only when `--velocity-ground-truth` is passed. It refuses windows other than 20 s.

The published MP3 previews were then adjusted by [`../scripts/match_listening_gain.py`](../scripts/match_listening_gain.py) toward −24 LUFS with a −1 dBFS peak ceiling. The [gain report](listening_gain_report.csv) records one scalar gain per clip. This adjustment makes switching players more comfortable while preserving within-clip dynamics; the MIDI and MAE values are untouched. The first MAESTRO reference clip reaches the peak ceiling before the loudness target.

[`../scripts/score_listening_cases.py`](../scripts/score_listening_cases.py) evaluates the **hosted, gain-matched MP3s** against each case's original recording using the paper analysis implementation of BSSL (sone, 22,050 Hz, 50 frames/s, 1,024-point FFT) and BSTL (Stevens total loudness). It reports Pearson correlation after resampling each feature curve to 2,048 samples. For piano it also computes the mean absolute MIDI velocity difference over the same aligned notes. The [case score CSV](listening_case_scores.csv) preserves the numbers displayed in the site manifest, rounded to three decimals for correlations and one decimal for visible MAE. The 00 reference cards show self-comparison values of 1.000 and MAE 0.0. These 20 s preview scores are not the Table 2 dataset aggregates.

## Dataset use

- [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro): Google LLC, CC BY-NC-SA 4.0.
- [Saarland Music Data](https://resources.mpi-inf.mpg.de/SMD/): CC BY-NC-SA 3.0; the dataset authors ask for citation.
- [François Leduc Guitar Dataset](https://huggingface.co/datasets/xavriley/FrancoisLeducGuitarDataset): current dataset page lists MIT; cite Riley, Edwards, and Dixon (ICASSP 2024).
- [GAPS](https://aim-qmul.github.io/GAPS/): Riley, Guo, Edwards, and Dixon (ISMIR 2024). The project team confirmed permission for these 20 s research demo excerpts after applying for dataset access through Zenodo.
- [FreePats Spanish Classical Guitar](https://freepats.zenvoid.org/Guitar/acoustic-guitar.html): CC0 1.0. [Salamander Grand Piano V3](https://github.com/sfzinstruments/SalamanderGrandPiano) is credited in the site footer.

The [guitar recovery record](GUITAR_RESULT_RECOVERY.md) identifies the adapted VeloEst checkpoints and archived Diff-Synth evaluation summaries found on lab5090's second HDD. The older 2 s proxy evaluation files are **not** used in this 5 s listening comparison. The paper's aggregate result bars and individual listening examples are labeled separately.
