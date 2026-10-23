# Direct inversion evaluation jobs

This folder keeps Route I direct inversion, the flat-velocity baseline, Route II/III/IV inference helpers, and the shared rendering/evaluation backend.

## Supported job entrypoints

Run these from `score_hpt/`.

```bash
python pytorch/direct_invension/route1_eval_job.py   --dataset smd   --instrument piano   --compute_velo_mae   --eval_scope full

python pytorch/direct_invension/flat_eval_job.py   --dataset smd   --instrument piano   --compute_velo_mae   --flat_velo 64   --eval_scope full

python pytorch/direct_invension/route2_eval_job.py   --dataset smd   --instrument piano   --ckpt_path /path/to/checkpoint.pth   --compute_velo_mae   --eval_scope full

python pytorch/direct_invension/route4_eval_job.py   --dataset smd   --instrument piano   --ckpt_path /path/to/checkpoint.pth   --compute_velo_mae   --eval_scope full
```

Route I, Route II, and flat baseline evaluation are now single-job interfaces. The implementation modules `route1_infer.py`, `route2_infer.py`, and `flat_infer.py` remain importable for code reuse, but the public Route II evaluation entrypoint is `route2_eval_job.py`.

## Files

- `route1_eval_job.py`: Route I direct inversion plus evaluation.
- `flat_eval_job.py`: flat-velocity MIDI export plus evaluation.
- `route2_eval_job.py`, `route3_eval_job.py`, `route4_eval_job.py`: checkpoint inference plus evaluation for Routes II/III/IV.
- `route1_infer.py`: Route I implementation, note-wise loudness extraction, dataset-statistics velocity mapping, same-structure MIDI export.
- `flat_infer.py`: fixed-velocity MIDI export implementation.
- `route2_infer.py`, `route3_infer.py`, `route4_infer.py`: model inference helpers.
- `eval_job_common.py`: shared dataset/soundfont/workspace resolution and summary writing for job entrypoints.
- `eval_runner.py`: shared evaluation runner.
- `common.py`: shared config/path/JSON helpers and PrettyMIDI velocity utilities.
- `eval_framework.py`: shared rendering and evaluation backend.

## Config source

Defaults come from:

```text
pytorch/config/config.yaml
```

Important Route I defaults are under `route1.infer.*` and `route1.eval.*`. Important Route II defaults are under `route2.infer.*` and `route2.eval.*`. Important flat-baseline defaults are under `flat.infer.*` and `flat.eval.*`. The job entrypoints override dataset, split/scope, soundfont, velocity MAE, and output folders explicitly.

## Evaluation scope

- `--eval_scope test`: evaluate the test split.
- `--eval_scope full`: evaluate all available items; internally this resolves to split `all` where the dataset scanner requires it.
- `--eval_scope one`: run one test item for fast debugging.

## Audio reference modes

The job entrypoints use dataset-mode real-audio references by default through the shared evaluation runner. Folder-mode evaluation remains available through the lower-level Python API rather than the public CLI job interface.
