# Route III / IV Local Training

This file is the practical launcher guide for local Route III and Route IV runs.

## Main rules

Only two model combinations are intended in this project:

- `hpt-direct`
- `hpt-note_editor`

Script behavior:

- `MODEL_TYPE` is always `hpt`
- `SCORE_METHOD` is `direct` or `note_editor`
- `input3` stays `null`
- `FRONTEND_PRETRAINED` is optional
- if `FRONTEND_PRETRAINED` is empty, training is from scratch

## Main environment variables

Common:

- `TRAIN_SET`
- `MODEL_TYPE`
- `SCORE_METHOD`
- `FRONTEND_PRETRAINED`
- `SUPERVISED_WEIGHT`
- `BACKEND_WEIGHT`
- `PRIOR_WEIGHT`
- `EXTRA_OVERRIDES`

Route III:

- `SEGMENT_LIST`
- `LOSS_TYPES`
- `DDSP_CKPTS`

Route IV:

- `SEGMENT_LIST`
- `SAMPLERS`
- `LOSS_TYPES`
- `PROXY_CKPT`

## Batch size

Batch size is controlled by Hydra config only.

Current config value:

- [`config.yaml`](../../score_hpt/pytorch/config/config.yaml#L24): `exp.batch_size=8`

The local route scripts do not override it anymore.

## Route III

Main script:

- [`train_route3_ablation.sh`](../route/train_route3_ablation.sh)

Default sweep:
```bash
SEGMENT_LIST="2 5"
LOSS_TYPES="piano_ssm_spectral piano_ssm_spectral_plus_log_rms"
MODEL_TYPE=hpt SCORE_METHOD=note_editor
MODEL_TYPE=hpt SCORE_METHOD=direct
FRONTEND_PRETRAINED=/path/to/ckpt.pth # hpt-note_editor only for now
```

Example:
```bash
TRAIN_SET=maestro \
SEGMENT_LIST="2" \
LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
MODEL_TYPE=hpt SCORE_METHOD=note_editor \
FRONTEND_PRETRAINED=/path/to/ckpt.pth \
bash scripts/route/train_route3_ablation.sh
```


## Route IV

Main script:

- [`train_route4_ablation.sh`](../route/train_route4_ablation.sh)

Default sweep:
```bash
SEGMENT_LIST="2 5"
LOSS_TYPES="smooth_l1 l1 mse"
SAMPLERS="coverage mixed realism"
MODEL_TYPE=hpt SCORE_METHOD=note_editor
MODEL_TYPE=hpt SCORE_METHOD=direct
FRONTEND_PRETRAINED=/path/to/ckpt.pth # hpt-note_editor only for now
```

Example:

```bash
TRAIN_SET=maestro \
MODEL_TYPE=hpt SCORE_METHOD=note_editor \
SEGMENT_LIST="2" \
SAMPLERS="mixed" \
LOSS_TYPES="smooth_l1" \
bash scripts/route/train_route4_ablation.sh
```

Use one specific SFProxy checkpoint:
```bash
TRAIN_SET=maestro \
MODEL_TYPE=hpt \
SCORE_METHOD=note_editor \
SEGMENT_LIST="2" \
LOSS_TYPES="smooth_l1" \
bash scripts/route/train_route4_ablation.sh /path/to/proxy.ckpt
```

## Kaya runs

See [`../kaya/README.md`](../kaya/README.md) for the five recovered SLURM scripts. The Route III single-run helper cited in the original notes was not present in the recovered Kaya checkout.

## Local BSSL / BSTL evaluation

Default behavior:

- `train_eval.audio_metrics.enabled=false`
- `train_eval.audio_metrics.instrument_path=""`

Local soundfont root:

```text
/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont
```

Recommended instrument paths:

- piano: `/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz`
- guitar: `/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SpanishClassicalGuitar/SpanishClassicalGuitar-20190618.sfz`

Enable local BSSL for Route III:

```bash
TRAIN_SET=maestro \
MODEL_TYPE=hpt \
SCORE_METHOD=note_editor \
SEGMENT_LIST="2" \
LOSS_TYPES="piano_ssm_spectral_plus_log_rms" \
EXTRA_OVERRIDES="train_eval.audio_metrics.enabled=true train_eval.audio_metrics.instrument_path=/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz" \
bash scripts/route/train_route3_ablation.sh
```

Enable local BSSL for Route IV:

```bash
TRAIN_SET=maestro \
MODEL_TYPE=hpt \
SCORE_METHOD=note_editor \
SEGMENT_LIST="2" \
SAMPLERS="mixed" \
LOSS_TYPES="smooth_l1" \
EXTRA_OVERRIDES="train_eval.audio_metrics.enabled=true train_eval.audio_metrics.instrument_path=/media/mengh/SharedData/zhanh/202604_midiproxy_data/soundfont/SalamanderGrandPiano/SalamanderGrandPianoV3.sfz" \
bash scripts/route/train_route4_ablation.sh
```

If rendered-audio evaluation is slow, add:

- `train_eval.audio_metrics.max_segments=1`
