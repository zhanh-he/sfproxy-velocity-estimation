# Release-readiness audit (backup branch, 2026-10-22)

This records checks for the public inference-weight release. The guitar
inference weights were subsequently bundled in the repository and mirrored
in the [three-file release](https://github.com/zhanh-he/sfproxy-velocity-estimation/releases/tag/v0.1.0-inference).
No repository-wide license was designated, and no listening example is claimed
to reproduce a paper aggregate. The demo cases and paper numbers were not
changed.

## 1. Standalone inference check

The portable entry point is [`../../scripts/infer_compare.py`](../../scripts/infer_compare.py).
Its direct, pinned dependencies are in
[`../../scripts/requirements-inference.txt`](../../scripts/requirements-inference.txt),
separate from the historical training `environment.yml`. Imports used only by
training logs, HDF5 dataset evaluation, and feature plots now load inside those
functions. A fresh environment exposed one otherwise hidden dependency:
`pretty_midi==0.2.10` imports `pkg_resources`, so `setuptools` is explicit.

Verified on macOS 26.5, arm64, Apple Silicon, CPython 3.11.15, CPU. MPS was
available but the script supports CPU/CUDA selection, so this run requested
`--device cpu`. Direct package versions: torch/torchaudio 2.2.2, numpy 1.26.4,
librosa 0.10.2.post1, hydra-core 1.3.2, pretty_midi 0.2.10, nnAudio 0.3.3,
mido 1.3.2, tqdm 4.66.5, and setuptools 70.3.0. A second fresh environment
created from the requirements file resolved 46 packages and completed the
MAESTRO inference command below with identical example metrics.

Create a fresh Python 3.11 environment and run from the repository root:

```bash
python3.11 -m venv .venv-infer
. .venv-infer/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/requirements-inference.txt

python scripts/infer_compare.py \
  --audio demo/docs/assets/cases/maestro-scriabin-40/reference.mp3 \
  --midi demo/docs/assets/cases/maestro-scriabin-40/reference.mid \
  --out /tmp/sfproxy-maestro-inference --device cpu \
  --reference-velocities --plot-start 0 --plot-seconds 20
```

The 20 s MAESTRO excerpt contains 85 notes. The run wrote `flat64.mid`,
`veloest.mid`, `diffsfproxy.mid`, `notes.csv`, `comparison.svg`, and
`summary.json`. These files are generated outputs in the chosen `--out`
directory, not new repository assets. The two bundled piano checkpoint
SHA-256 values in `summary.json` matched the checkpoint README. The example
MAEs (Flat 64 15.765, VeloEst 10.612, adapted Diff-SFProxy 18.541) describe
only this excerpt and are not paper Table 2 results.

The same environment also ran the GAPS guitar 20 s excerpt (60 notes) using
the two locally recovered 5 s weights listed below:

```bash
python scripts/infer_compare.py \
  --audio demo/docs/assets/cases/gaps-019-100/reference.mp3 \
  --midi demo/docs/assets/cases/gaps-019-100/reference.mid \
  --out /tmp/sfproxy-gaps-inference --device cpu \
  --checkpoint-set guitar \
  --plot-start 0 --plot-seconds 20
```

The guitar run wrote the same six files plus `diffsynth.mid`. The bundled
`--checkpoint-set guitar` run produced MIDI files byte-identical to the
earlier explicit-checkpoint run. The input score
does not have verified performance velocities, so `--reference-velocities`
was omitted. For every output MIDI from both runs, `pretty_midi` found the
same note count, pitch, onset and offset time as the input (maximum time
difference **0 seconds**), with the same instrument program. An independent
`mido` check found identical tempo/program/control metadata events, 220 ticks
per beat and two tracks for these two examples. Note-on and note-off pitch and
tick positions also matched exactly. This is a check of these
files, not a guarantee for every possible MIDI feature. SFZ audio rendering
was outside the minimal Python inference run; it separately needs a complete
instrument, `sfizz_render` and `ffmpeg`.

## 2. Guitar weight release manifest

The recovered files remain backed up outside Git. Their repository copies
and release assets have the same SHA-256 and byte sizes as
[`GUITAR_RESULT_RECOVERY.md`](../../demo/analysis/GUITAR_RESULT_RECOVERY.md):

| Release file | Adapted inference model | Size | SHA-256 |
| --- | --- | ---: | --- |
| `veloest_diffsynth_guitar_5s_60k.pth` | Onset-assisted HPT VeloEst after 5 s Diff-Synth guitar adaptation, 60k steps | 20,761,532 bytes | `1712310c5b662685f7ed03371e52dd491a48e9c242d6557548a314f20ea860ee` |
| `veloest_diffsfproxy_guitar_5s_120k.pth` | Onset-assisted HPT VeloEst after 5 s Diff-SFProxy guitar adaptation, 120k steps | 20,761,642 bytes | `2d1e3d0dff33088689ae0a10d318b369625fa89b184e36791b1c943880e38131` |

The repository paths are under
[`../../score_hpt/checkpoints/`](../../score_hpt/checkpoints/README.md).
The [inference release](https://github.com/zhanh-he/sfproxy-velocity-estimation/releases/tag/v0.1.0-inference)
also includes `veloest_onset_only_120k.pth` (SHA-256
`8cede73af4481661c51d7b409939489f32b208d6e971d833aac4a6e7eed438aa`).
The original local backups had the older `guitar_diffsynth_5s_60k.pth` and
`guitar_diffsfproxy_5s_120k.pth` names; the weight bytes were not changed.
These are **adapted VeloEst inference checkpoints**. The frozen DDSP guitar
synthesizer and frozen Diff-SFProxy Transformer checkpoints were used to
train/adapt them but are **not inference inputs** and are not part of this
three-file release. See the recovery record for exact 5090 archive members.

## 3. FL paper-result provenance

The paper's FL Diff-SFProxy `r_BSSL = 0.777` has **no identified matching
checkpoint, configuration, prediction MIDI, and evaluation record** in the
recovered materials. The known 5 s / 120k guitar checkpoint above is tied by
its archived path to the full 79/79 François Leduc evaluation in
[`guitar_recovery/fl_diffsfproxy_5s_result_summary.txt`](../../demo/analysis/guitar_recovery/fl_diffsfproxy_5s_result_summary.txt),
which reports `r_BSSL = 0.768812` and `r_BSTL = 0.750307`. Saved prediction
MIDIs from that run are documented under [`guitar_recovery/`](../../demo/analysis/guitar_recovery).
The [six archived 5 s Route IV FL evaluations](../../demo/analysis/guitar_recovery/fl_diffsfproxy_5s_archived_sweep.csv)
range from 0.766198 to 0.772093; none rounds to 0.777. A further 5090
archive, `transfer3_ckpt.tar.gz`, contains piano Route IV checkpoints but no
`route4_guitar` / François Leduc entries. This search does not establish
which run produced the paper value. Preserve the paper number as reported,
and do not label a selected demo excerpt or the recovered 0.768812 run as its
reproduction. Author run records or a matching archive remain needed.

## 4. Paper and separate rights evidence

The repository's camera-ready PDF is
[`../paper/2026_ISMIR_Velo_Beyond_Piano_Camera_Ready.pdf`](../../demo/paper/2026_ISMIR_Velo_Beyond_Piano_Camera_Ready.pdf).
[arXiv:2608.08985](https://arxiv.org/abs/2608.08985) verifies the title,
four listed authors, 2026 submission, and ISMIR 2026 acceptance. The
repository BibTeX has those fields and a conference booktitle, but no final
proceedings page, DOI, pages or publisher has been verified. The arXiv DOI
`10.48550/arXiv.2608.08985` identifies the preprint, not an ISMIR
proceedings DOI. Keep the local PDF and arXiv link distinguishable until
final proceedings metadata is confirmed.

| Component | Evidence now available | Release decision still needed |
| --- | --- | --- |
| Score-HPT frontend code | [`../../score_hpt/LICENSE`](../../score_hpt/LICENSE) is MIT; [earlier model repository](https://github.com/zhanh-he/score-informed-amt) is credited. | Confirm scope for changes and weights produced by this project. |
| Diff-SFProxy code and project scripts | No root or component `LICENSE` file is present. | Authors choose a code license and add its scope explicitly. |
| Diff-Synth adapted code | [Piano upstream](https://github.com/ytsrt66589/ddsp-piano-pytorch) and [guitar upstream](https://github.com/andywiggins/ddsp-guitar-synth) are credited in the component README; neither upstream GitHub repository advertises a detected license in its metadata. | Verify original permissions/terms before assigning a license to the adapted copies. |
| Four VeloEst checkpoints | Two piano and two guitar files are in Git; the piano-pretrained and both guitar files are mirrored as release assets. No checkpoint-specific license is recorded. | Authors specify separate weight reuse terms. |
| MAESTRO excerpts | [Official dataset page](https://magenta.tensorflow.org/datasets/maestro) states CC BY-NC-SA 4.0. | Preserve attribution and version/source in the media manifest. |
| SMD excerpts | [Official MIDI-audio page](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi) states CC BY-NC-SA 3.0 and requests citation. | Preserve attribution and source in the media manifest. |
| GAPS excerpts | [Current dataset card](https://huggingface.co/datasets/xavriley/GAPS) lists MIT and includes audio; the [Zenodo record](https://zenodo.org/records/17152440) points to it. The project team has separately confirmed demo-publication permission for the audio obtained through its application process. | Record which source/permission applies to the exact published files and keep the permission evidence internally. |
| François Leduc excerpts | [Current dataset card](https://huggingface.co/datasets/xavriley/FrancoisLeducGuitarDataset) lists MIT and includes audio/MIDI. The team confirms permission for the audio obtained through its Zenodo application process. | Record which source/permission applies to the exact published files and keep the permission evidence internally. |
| Rendered instrument audio | [FreePats Spanish guitar](https://freepats.zenvoid.org/Guitar/acoustic-guitar.html) states CC0 1.0; the [Salamander piano license](https://github.com/sfzinstruments/SalamanderGrandPiano/blob/master/LICENSE) is CC BY 3.0. | Verify the exact SFZ/sample versions used for each rendered asset and carry their attribution/terms. |

No single license has been assigned to all code, weights, and media. The
items above are an evidence inventory, not a substitute for the authors'
component-by-component release decision.

## Remaining author decisions

1. Identify the FL run behind `r_BSSL = 0.777`, including its checkpoint,
   configuration, full-set prediction directory and evaluation summary, or
   leave the paper value explicitly unverified by the released artifacts.
2. Specify weight reuse terms and decide whether the frozen training backends
   should have a separate future release.
3. Decide code terms per component, especially the Diff-Synth adaptations
   whose upstream permissions need checking.
4. Record exact demo-media source/permission and instrument version per asset,
   including the team's GAPS/FL audio authorization, without publishing the
   private permission correspondence.
5. Confirm the final ISMIR proceedings citation/DOI when it becomes available.
