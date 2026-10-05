# Experiment launchers

Run these from the repository root unless a script states otherwise. The training launchers were recovered from the 5090/Kaya research workspaces and still require external datasets and backend checkpoints. The two released VeloEst inference checkpoints are under [`../score_hpt/checkpoints/`](../score_hpt/checkpoints/README.md).

## Inference comparison

[`infer_compare.py`](infer_compare.py) takes one aligned audio/MIDI pair and runs the two released piano checkpoints by default. It preserves the source MIDI's note timing, programs, tempo, and control changes while replacing note velocities. It writes three MIDIs (Flat 64, VeloEst, VeloEst+Diff-SFProxy), `notes.csv`, a 1600 × 860 SVG piano-roll comparison, and `summary.json`. Pass `--diffsynth-ckpt` to include a fourth MIDI, Diff-Synth column, and 1600 × 1070 figure. An optional SFZ path adds rendered WAV/MP3 audio. No dataset packing, proxy backend checkpoint, or training step is needed.

```bash
python scripts/infer_compare.py \
  --audio /path/to/performance.wav \
  --midi /path/to/aligned_score.mid \
  --out /path/to/inference_output \
  --plot-start 60 --plot-seconds 20 \
  --reference-velocities \
  --sfz /path/to/SalamanderGrandPianoV3.sfz
```

Use `--reference-velocities` only when the input MIDI's note velocities are ground truth. Otherwise the figure shows each approach's mean velocity without an accuracy claim. For a guitar pair, pass `--diffsfproxy-ckpt /path/to/guitar_diffsfproxy_5s_120k.pth` and optionally `--diffsynth-ckpt /path/to/guitar_diffsynth_5s_60k.pth` with the piano-pretrained `--veloest-ckpt` for a zero-shot control. The recovered guitar checkpoints and the GAPS/FL demo derivation are documented in [`../demo/analysis/GUITAR_RESULT_RECOVERY.md`](../demo/analysis/GUITAR_RESULT_RECOVERY.md). Guitar score velocity is not verified ground truth, so omit `--reference-velocities`. SFZ rendering needs `sfizz_render` and `ffmpeg`; see [`../diff-sfproxy/README.md`](../diff-sfproxy/README.md).

## Training and evaluation launchers

| Directory | Purpose | Main entry points |
| --- | --- | --- |
| [`sfproxy/`](sfproxy/) | Export SoundFont teacher data and train Diff-SFProxy | `preprocess_sfproxy_data.sh`, `train_sfproxy.sh`, `train_sfproxy_ablations.sh` |
| [`ddsp/`](ddsp/) | Prepare/train the piano and guitar Diff-Synth backends | `train_ddsp_piano.sh`, `train_ddsp_guitar_synth.sh` |
| [`route/`](route/) | Score-HPT Route III (Diff-Synth) and Route IV (Diff-SFProxy) local runs | `train_route3.sh`, `train_route3_ablation.sh`, `train_route4_ablation.sh` |
| [`kaya/`](kaya/README.md) | SLURM copies for the same route families | `kaya_hpt_route3_*`, `kaya_hpt_route4_*` |
| [`notes/`](notes/) | Historical run notes from the research workspace | `train_route.md`, `train_sfproxy.md` |

Typical sequence:

```bash
export SFIZZ_RENDER_BIN=/absolute/path/to/sfizz_render
export PIANO_SFZ=/absolute/path/to/SalamanderGrandPianoV3.sfz
export GUITAR_SFZ=/absolute/path/to/SpanishClassicalGuitar-20190618.sfz
export WORKSPACE_BASE=/absolute/path/to/external_workspace
bash scripts/sfproxy/preprocess_sfproxy_data.sh
bash scripts/sfproxy/train_sfproxy.sh
```

Then supply `FRONTEND_PRETRAINED` and the frozen backend checkpoint for the route you are running. For example:

```bash
FRONTEND_PRETRAINED=/path/to/onset_only_frontend.pth \
DDSP_CKPT=/path/to/ddsp_checkpoint.pt \
bash scripts/route/train_route3.sh

FRONTEND_PRETRAINED=/path/to/onset_only_frontend.pth \
bash scripts/route/train_route4_ablation.sh /path/to/proxy.ckpt
```

The `sfproxy/` launchers accept `WORKSPACE_BASE`, `PIANO_SFZ`, and `GUITAR_SFZ` overrides. The `route/` launchers derive their workspace from `score_hpt/pytorch/config/config.yaml` unless `WORKSPACE_DIR` and `DATA_ROOT` are provided. The `ddsp/` launchers still contain 5090/3090 defaults; set `MAESTRO_DIR` or the corresponding guitar dataset path and `WORKSPACE_BASE` for the current machine. `kaya/` expects a cloned repository in `$HOME/sfproxy-velocity-estimation` and an external `$MYSCRATCH/202604_midiproxy_data` tree by default.

The workspace data directories retain historical names such as `synth-proxy`; the **source** directory is now `diff-sfproxy`. Adaptation uses a 5 s backend crop in the main paper runs. The historical `notes/` files document older recipes and should be read alongside the current component READMEs.
