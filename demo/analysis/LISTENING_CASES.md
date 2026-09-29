# Listening demo cases

The public site reads [`../docs/assets/cases.json`](../docs/assets/cases.json). Every selection is **exactly 20 seconds** and always presents the same four positions: 00 original recording, 01 Flat 64, 02 Diff-Synth, 03 Diff-SFProxy. Missing runs are shown as unavailable; no substitute predictions or audio are generated for them. The case menu can be extended by adding another manifest entry and its asset directory.

| Dataset | 20 s windows | 00 / 01 | 02 / 03 | Velocity reference |
| --- | --- | --- | --- | --- |
| MAESTRO v3 test | Scriabin, 60–80 s and 100–120 s | Ready | Ready; Diff-Synth saved 5 s 60k, Diff-SFProxy rerun from released piano 5 s 120k | Yes; first case badges use full-piece 721-note MAE, second uses 20 s MAE |
| SMD | Bach BWV 849, 40–60 s and 80–100 s | Ready | Ready; saved piano 5 s 60k Diff-Synth and 5 s 120k Diff-SFProxy runs | Yes; badges use each selected 20 s window |
| François Leduc (FL) | `JGDyc`, 40–60 s and 80–100 s | Ready | Awaiting selected final 5 s guitar prediction MIDIs | No; aligned score velocities are not ground truth |
| GAPS | Reserved 20 s slot | Awaiting publication permission | Awaiting final 5 s guitar predictions | No |

## Recovered source paths

The SMD audio/MIDI pair was reconstructed with [`../scripts/export_pair_from_h5.py`](../scripts/export_pair_from_h5.py) from `score_hpt/workspaces/hdf5s/smd_sr22050/Bach_BWV849-01_001_20090916-SMD.h5` on lab5090. Its saved prediction MIDIs are under:

- `score_hpt/workspaces/route3/smd_full_piano/5s_hpt_onset_score_note_editor_backend_diffsynth_piano_piano_ssm_spectral_sup0_backend1_prior0p1_sat0p2_60000_iterations/pred_midis/`
- `score_hpt/workspaces/route4/smd_full_piano/5s_hpt_onset_score_note_editor_backend_diffproxy_smooth_l1_sup0p5_backend0p5_prior0p1_sat0p2_120000_iterations/pred_midis/`

Both use the key `Bach_BWV849-01_001_20090916-SMD.route2.mid`. The SMD 40–60 s case has 113 aligned notes, with Flat 64 / Diff-Synth / Diff-SFProxy MAE of 13.460 / 11.735 / 6.735. The 80–100 s case has 108 aligned notes, with MAE 14.824 / 11.435 / 5.250. These are case values, **not** the paper's dataset aggregate, and the saved runs have not been identified as its selected Table 2 checkpoints.

The MAESTRO 100–120 s case uses the same packed Scriabin recording as the first case and contains 117 aligned notes. Its 20 s Flat 64 / Diff-Synth / Diff-SFProxy MAE is 12.658 / 16.026 / 8.487. The first case's badges instead use full-piece metrics, which are labeled as such on the site.

The FL reference pair was reconstructed from `score_hpt/workspaces/hdf5s/francoisleduc_sr22050/JGDyc.h5`. Its two 20 s windows contain 138 and 120 score notes. Flat 64 was rendered with the FreePats Spanish Classical Guitar SFZ. Guitar note velocities are not evaluated against the score.

Run [`../scripts/build_listening_case.py`](../scripts/build_listening_case.py) with the external source pair, available prediction MIDIs, `--start`, the matching SFZ, and `sfizz_render` to regenerate a case directory. The builder checks note pitch/onset/duration alignment and writes clipped MIDI, 20 s MP3, note JSON, and excerpt MAE only when `--velocity-ground-truth` is passed. It refuses windows other than 20 s.

The published MP3 previews were then adjusted by [`../scripts/match_listening_gain.py`](../scripts/match_listening_gain.py) toward −24 LUFS with a −1 dBFS peak ceiling. The [gain report](listening_gain_report.csv) records one scalar gain per clip. This adjustment makes switching players more comfortable while preserving within-clip dynamics; the MIDI and MAE values are untouched. The first MAESTRO reference clip reaches the peak ceiling before the loudness target.

## Dataset use

- [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro): Google LLC, CC BY-NC-SA 4.0.
- [Saarland Music Data](https://resources.mpi-inf.mpg.de/SMD/): CC BY-NC-SA 3.0; the dataset authors ask for citation.
- [François Leduc Guitar Dataset](https://huggingface.co/datasets/xavriley/FrancoisLeducGuitarDataset): current dataset page lists MIT; cite Riley, Edwards, and Dixon (ICASSP 2024).
- [GAPS](https://aim-qmul.github.io/GAPS/): its public terms prohibit distributing the dataset or reproduction-enabling data to third parties without written permission. Therefore the site's GAPS case contains no hosted media, MIDI, or note JSON. Its dataset link points to the authors' site.
- [FreePats Spanish Classical Guitar](https://freepats.zenvoid.org/Guitar/acoustic-guitar.html): CC0 1.0. [Salamander Grand Piano V3](https://github.com/sfzinstruments/SalamanderGrandPiano) is credited in the site footer.

The public guitar cards remain visibly incomplete until selected final 5 s predictions are recovered. In particular, the older 2 s proxy evaluation files on 5090 are **not** presented as the paper's guitar listening result.
