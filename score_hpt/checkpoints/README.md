# VeloEst inference checkpoints

All four files below are bundled in this repository so a clone can run piano
or guitar inference without a separate weight download. The [inference
release](https://github.com/zhanh-he/sfproxy-velocity-estimation/releases/tag/v0.1.0-inference)
also mirrors the three most useful starting points: original piano-pretrained
VeloEst, guitar Diff-SFProxy adaptation, and guitar Diff-Synth/DDSP adaptation.
The piano-adapted Diff-SFProxy checkpoint stays in the repository for the
piano comparison but is not part of that three-file release.

| File | Inference role | Snapshot | SHA-256 |
| --- | --- | --- | --- |
| [`veloest_onset_only_120k.pth`](veloest_onset_only_120k.pth) | Piano-pretrained VeloEst; zero-shot guitar control | 120k steps | `8cede73af4481661c51d7b409939489f32b208d6e971d833aac4a6e7eed438aa` |
| [`veloest_diffsfproxy_guitar_5s_120k.pth`](veloest_diffsfproxy_guitar_5s_120k.pth) | VeloEst adapted to guitar through Diff-SFProxy | 5 s backend crop, 120k steps | `2d1e3d0dff33088689ae0a10d318b369625fa89b184e36791b1c943880e38131` |
| [`veloest_diffsynth_guitar_5s_60k.pth`](veloest_diffsynth_guitar_5s_60k.pth) | VeloEst adapted to guitar through the Diff-Synth/DDSP guitar backend | 5 s backend crop, 60k steps | `1712310c5b662685f7ed03371e52dd491a48e9c242d6557548a314f20ea860ee` |
| [`veloest_diffsfproxy_piano_5s_120k.pth`](veloest_diffsfproxy_piano_5s_120k.pth) | VeloEst adapted with a piano Diff-SFProxy backend | 5 s backend crop, 120k steps | `d22e589b2c46b6efd4fbd307c5130b5f083abedda543fdeb50f117b1bb230343` |

The guitar file hashes and original 5090 archive members are in the
[recovery record](../../demo/analysis/GUITAR_RESULT_RECOVERY.md). The adapted
files are **VeloEst inference weights**. Frozen Diff-SFProxy and DDSP backend
checkpoints were used during adaptation, but they are not needed to infer
velocities with these files. The model is onset-assisted HPT `note_editor`:
`model.input2=onset`, `model.input3=null`, and
`score_informed.method=note_editor`.

## Run a bundled guitar example

Install the [minimal inference dependencies](../../scripts/requirements-inference.txt)
using the [inference guide](../../scripts/README.md#inference-comparison), then
run from the repository root:

```bash
python scripts/infer_compare.py \
  --audio demo/docs/assets/cases/gaps-019-100/reference.mp3 \
  --midi demo/docs/assets/cases/gaps-019-100/reference.mid \
  --out /tmp/veloest-guitar --checkpoint-set guitar --device cpu
```

The guitar set loads all three mirrored checkpoints and writes `flat64.mid`,
`veloest.mid`, `diffsynth.mid`, `diffsfproxy.mid`, `notes.csv`,
`comparison.svg`, and `summary.json`. `veloest.mid` is the piano-pretrained
zero-shot control. For piano, omit `--checkpoint-set guitar`; the default
compares the piano-pretrained and piano-adapted weights. Use
`--reference-velocities` only when input MIDI velocities are verified ground
truth; the GAPS/FL aligned score velocities are not such labels. Optional
rendered audio requires an SFZ instrument, `sfizz_render`, and `ffmpeg`.

The earlier [Score-Informed AMT project](https://github.com/zhanh-he/score-informed-amt)
introduced this model and provides other pretrained choices. Its public
example uses onset plus frame input at 100k steps. Here we selected the
onset-only piano 120k starting checkpoint used in this paper and thank and
cite the earlier work. These weights support inference comparisons; the
exact run behind the paper's FL Diff-SFProxy aggregate `r_BSSL = 0.777`
remains unidentified. See the [release audit](../../demo/analysis/RELEASE_READINESS.md#3-fl-paper-result-provenance)
for that distinction.
