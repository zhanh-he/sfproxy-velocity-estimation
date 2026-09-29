# SFProxy Local Scripts

This note only documents the current local SFProxy path in [`run_scripts/`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts).

## Main idea

The local SFProxy flow is intentionally narrow:

1. export the component datasets
2. compose `mixed_v2` when needed
3. train SFProxy from the exported data

The script path is:

- [`preprocess_sfproxy_data.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/preprocess_sfproxy_data.sh)
- [`train_sfproxy.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_sfproxy.sh)
- [`train_sfproxy_ablations.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_sfproxy_ablations.sh)

## User-facing parameters

Only keep these in mind.

Shared:

- `INSTRUMENT=piano|guitar`
- `SEGMENT_SECONDS` or `SEGMENT_LIST`
- `MIX_WEIGHTS`

Piano or guitar dataset:

- `PIANO_DATASET=maestro|smd`
- `GUITAR_DATASET=francoisleduc|gaps`

Training:

- `TRAIN_PRESET=coverage_v2|realism_v2|stress_v2|mixed_v2`
- `TRAIN_PRESETS="..."`
- `EXTRA_TRAIN_OVERRIDES`

Everything else is treated as script internals.

## Fixed choices

These are no longer local ablation knobs:

- `boundary_mode` stays `default`
- preprocessing does not expose export batch size or export worker overrides
- local training does not expose wandb project or group overrides

## Preprocess only

Piano:

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
INSTRUMENT=piano PIANO_DATASET=maestro SEGMENT_LIST="2 5 10" bash run_scripts/preprocess_sfproxy_data.sh
```

Guitar:

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
INSTRUMENT=guitar GUITAR_DATASET=francoisleduc SEGMENT_LIST="2 5" bash run_scripts/preprocess_sfproxy_data.sh
```

Custom `mixed_v2` weights:

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
MIX_WEIGHTS="0.3 0.4 0.2 0.1" bash run_scripts/preprocess_sfproxy_data.sh
```

The four component datasets are:

- `boundary_v2`
- `coverage_v2`
- `realism_v2`
- `stress_v2`

`mixed_v2` is composed from those four components.

## Train one SFProxy model

Default main run:

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
TRAIN_PRESET=mixed_v2 SEGMENT_SECONDS=2 bash run_scripts/train_sfproxy.sh
```

Coverage-only:

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
TRAIN_PRESET=coverage_v2 SEGMENT_SECONDS=2 bash run_scripts/train_sfproxy.sh
```

Guitar:

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
INSTRUMENT=guitar GUITAR_DATASET=francoisleduc TRAIN_PRESET=mixed_v2 SEGMENT_SECONDS=2 bash run_scripts/train_sfproxy.sh
```

If you need extra Hydra overrides, use:

```bash
EXTRA_TRAIN_OVERRIDES="trainer.max_epochs=100"
```

## Run local SFProxy ablations

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
SEGMENT_LIST="2 5 10" TRAIN_PRESETS="coverage_v2 realism_v2 stress_v2 mixed_v2" bash run_scripts/train_sfproxy_ablations.sh
```

Guitar:

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy
INSTRUMENT=guitar GUITAR_DATASET=francoisleduc SEGMENT_LIST="2 5" bash run_scripts/train_sfproxy_ablations.sh
```

## Notes

- `mixed_v2` is the main preset
- `coverage_v2`, `realism_v2`, and `stress_v2` are the main component ablations
- if you want to discuss results later, the cleanest thing to paste is the checkpoint folder name plus the final validation metrics
