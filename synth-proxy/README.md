# SFProxy

This repository is trimmed to the SoundFont proxy workflow we actually use.

It covers one project only:

- export note-conditioned teacher data from `.sf2` or `.sfz` instruments
- train a neural proxy on note-wise dynamics targets (PHE, OSF)
- evaluate the proxy via synthetic velocity recovery (in-domain + stress)

## Repo layout

- `src/`: the full runtime codebase
- `configs/`: Hydra configs for export, training, and evaluation
- `tests/`: focused tests for the current proxy pipeline
- `Sfproxy_Eval_zh.md`: evaluation notes (中文)
- `eval.ipynb`: 4-checkpoint 2x2 sweep (piano vs guitar, 2 s vs 5 s)

## Install

System tools:
- `fluidsynth` for `.sf2`
- `sfizz_render` for `.sfz`

Python:

```bash
python -m pip install -r requirements.txt
```

## Typical workflow

Export teacher data:

```bash
python src/export_dataset_pkl.py \
  --config-name data_piano \
  dataset_size=20000 \
  start_index=0 end_index=20000
```

Train:

```bash
python src/train.py \
  --config-name train \
  dataset.train.path=/path/to/export_train_folder \
  dataset.val.path=/path/to/export_val_folder
```

Evaluate (single checkpoint):

```bash
python src/eval.py \
  ckpt_path=/path/to/checkpoint.ckpt \
  device=cuda
```

Evaluate (full 4-ckpt sweep with 2x2 plot): open `eval.ipynb` and Run All.

## Notes

- The default Hydra paths point at the shared workspace used in the current setup.
  Override `paths.workspace_dir`, dataset paths, or instrument paths from the CLI
  to relocate outputs.
- `configs/data_piano.yaml` and `configs/data_guitar.yaml` hold the current
  sampler presets used during training and inherited by `eval.yaml`.
- Per-run outputs land under `${paths.proxy_dir}/eval/velocity_recovery_<ts>/`
  as `velocity_recovery_results.json`.
