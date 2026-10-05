# Diff-SFProxy: SoundFont proxy

This component learns a differentiable note-wise surrogate for a SoundFont renderer. It exports synthetic note/audio pairs, trains a frozen Transformer proxy to predict pitch-conditioned harmonic energy (PHE) and onset-window spectral flux (OSF), and evaluates velocity recovery. The VeloEst adaptation code that *uses* this trained proxy is in [`../score_hpt/`](../score_hpt/README.md).

## Directory guide

- `src/`: data export, feature extraction, renderer wrappers, Transformer, training, and evaluation.
- `configs/`: Hydra configurations. `data_piano.yaml` and `data_guitar.yaml` select the instrument and sampler.
- `tests/`: focused proxy tests.
- `Sfproxy_Eval.md` / `Sfproxy_Eval_zh.md`: detailed evaluation notes.

## The extra renderer dependency: `sfizz_render`

The paper's piano and guitar instruments are **SFZ files**, not a single `.sf2` file. An SFZ file refers to samples by relative path, so keep the `.sfz` together with its complete sample directory. The Python requirements alone do **not** install the offline SFZ renderer. Install/build [`sfizz_render`](https://sfztools.github.io/sfizz/development/build/) from the sfizz project, then either place it on `PATH` or set `SFIZZ_RENDER_BIN` to its executable. The renderer wrapper checks `SFIZZ_RENDER_BIN` first, then `sfizz_render` / `sfizz-render` on `PATH`.

On the inspected 5090 machine, the working binary was:

```text
/media/mengh/SharedData/zhanh/202601_midisemi/sfizz/build/library/bin/sfizz_render
```

For a local setup, replace the path below with your own:

```bash
export SFIZZ_RENDER_BIN=/absolute/path/to/sfizz_render
"$SFIZZ_RENDER_BIN" --help
```

The instruments used in the recovered configuration are [Salamander Grand Piano V3](https://github.com/sfzinstruments/SalamanderGrandPiano) and [FreePats Spanish Classical Guitar](https://github.com/freepats/spanish-classical-guitar). Download their *full* SFZ repositories separately and set their paths:

```bash
export PIANO_SFZ=/absolute/path/to/SalamanderGrandPianoV3.sfz
export GUITAR_SFZ=/absolute/path/to/SpanishClassicalGuitar-20190618.sfz
"$SFIZZ_RENDER_BIN" --sfz "$PIANO_SFZ" --midi ../demo/docs/assets/reference.mid --wav /tmp/sfproxy-render-check.wav
```

The code also supports `.sf2` instruments through FluidSynth (`fluidsynth` / `pyfluidsynth`), but that is a different renderer and instrument contract. Do not substitute an SF2 instrument when attempting to reproduce the SFZ-based results.

## Python setup and paths

From `diff-sfproxy/`, install the Python environment (adjust the pinned PyTorch/CUDA build for your machine):

```bash
python -m pip install -r requirements.txt
```

The inherited YAML files still contain 5090 absolute paths. Override `instrument.path`, `paths.workspace_dir`, and `paths.analysis_dir` on the command line. `paths.analysis_dir` points to the velocity-boundary and dataset-statistics files in `../demo/analysis/`; `paths.workspace_dir` is an external writable location for large generated data and checkpoints.

```bash
python src/export_dataset_pkl.py --config-name data_piano \
  instrument.path="$PIANO_SFZ" \
  paths.analysis_dir=../demo/analysis \
  paths.workspace_dir=/path/to/workspace \
  dataset_size=20000
```

For guitar, create the missing `guitar_boundaries.json` from *your installed guitar SFZ* before running the `boundary_v2` sampler:

```bash
python src/tools/discover_velocity_boundaries.py \
  --instrument_path "$GUITAR_SFZ" --backend sfizz \
  --pitch_min 42 --pitch_max 72 --pitch_step 3 \
  --register_splits 52 64 \
  --out_json ../demo/analysis/stats/sfproxy_boundaries/guitar_boundaries.json
python src/export_dataset_pkl.py --config-name data_guitar \
  instrument.path="$GUITAR_SFZ" \
  paths.analysis_dir=../demo/analysis \
  paths.workspace_dir=/path/to/workspace \
  dataset_size=20000
```

Train and evaluate after both train and validation exports exist:

```bash
python src/train.py --config-name train \
  dataset.train.path=/path/to/export_train \
  dataset.val.path=/path/to/export_val \
  paths.workspace_dir=/path/to/workspace
python src/eval.py ckpt_path=/path/to/proxy.ckpt device=cuda \
  velocity_recovery.instrument.path="$PIANO_SFZ" \
  paths.analysis_dir=../demo/analysis
```

The experiment launchers in [`../scripts/sfproxy/`](../scripts/README.md) provide machine-specific batches. Their 5090 data directory retains the historical name `synth-proxy` even though this source directory is now `diff-sfproxy`.
