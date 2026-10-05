# Released piano velocity checkpoints

These two checkpoints are provided for **inference comparison** with the onset-assisted HPT `note_editor` in this repository. Both take aligned performance audio and piano MIDI as input. The output MIDI keeps note pitch and timing while replacing velocities.

| File | Role | Training snapshot | SHA-256 |
| --- | --- | --- | --- |
| [`veloest_onset_only_120k.pth`](veloest_onset_only_120k.pth) | Piano-trained VeloEst before target-instrument adaptation | 120,000 iterations | `8cede73af4481661c51d7b409939489f32b208d6e971d833aac4a6e7eed438aa` |
| [`veloest_diffsfproxy_piano_5s_120k.pth`](veloest_diffsfproxy_piano_5s_120k.pth) | VeloEst adapted with a frozen Diff-SFProxy piano backend | 5 s backend crop; 120,000 iterations | `d22e589b2c46b6efd4fbd307c5130b5f083abedda543fdeb50f117b1bb230343` |

Both files are copies of the corresponding 5090 research-workspace checkpoints recorded in [`../../demo/analysis/PROVENANCE.md`](../../demo/analysis/PROVENANCE.md). The proxy **backend** weights used during adaptation are a separate training dependency and are not needed to run inference with the adapted VeloEst file. These released files are piano snapshots. The 5 s adapted guitar checkpoints were subsequently recovered from a separate 5090 HDD archive and are [documented for release review](../../demo/analysis/GUITAR_RESULT_RECOVERY.md).

The earlier [Score-Informed AMT project](https://github.com/zhanh-he/score-informed-amt) introduced the model and supplies other pretrained choices. Its public `hpt+onset+frame+score_note_editor/100000_iterations.pth` includes a frame input. Here we selected an **onset-only** `hpt+onset+score_note_editor/120000_iterations.pth` piano checkpoint as the starting point. Set `model.input2=onset`, `model.input3=null`, and `score_informed.method=note_editor` for these files. We thank and cite the earlier work; this slightly different checkpoint selection reflects the setup used for this paper.

Run [`../../scripts/infer_compare.py`](../../scripts/infer_compare.py) to produce the two predictions, a Flat 64 control, per-note CSV, and a comparison SVG. The script can also render the output through an SFZ instrument when `sfizz_render` is installed.
