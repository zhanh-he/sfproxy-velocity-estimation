# Kaya Route III / IV

This file documents the current Kaya launchers for Route III and Route IV.

## Scripts

Main Kaya array launchers:

- [`kaya_hpt_route3_ablation.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/kaya_hpt_route3_ablation.sh)
- [`kaya_hpt_route3_guitar_ablation.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/kaya_hpt_route3_guitar_ablation.sh)
- [`kaya_hpt_route4_ablation.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/kaya_hpt_route4_ablation.sh)

Interactive single-run helpers:

- [`kaya_hpt_route3_single.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/kaya_hpt_route3_single.sh)
- [`kaya_hpt_route4_single.sh`](/media/mengh/SharedData/zhanh/202604_midiproxy/kaya_scripts/kaya_hpt_route4_single.sh)

## Current training combinations

Kaya follows the same narrowed model combinations as the local scripts:

- `filmunet-direct`
- `hpt-direct`
- `hpt-note_editor`

Rules:

- `MODEL_TYPES` should be `hpt` or `filmunet`
- `SCORE_METHOD` is only meaningful for `hpt`
- `filmunet` is always forced to `direct`
- `FRONTEND_PRETRAINED` is optional and overrides auto selection when set
- `USE_ROUTE2_PIANO_PRETRAINED_CKPT=1` auto-loads the latest Route II piano frontend ckpt for the current model combination
- if you set `FRONTEND_PRETRAINED`, it should usually be used with one model type at a time

## Data layout on Kaya

Default data root:

```text
$MYSCRATCH/202604_midiproxy_data
```

Required subtrees:

### Score-HPT HDF5

```text
202604_midiproxy_data/
  score_hpt/
    workspaces/
      hdf5s/
        maestro_sr22050/
        smd_sr22050/
      pretrained_checkpoints/
        hpt/
        hpt+onset+score_note_editor/
        filmunet/
```

### Route III DDSP checkpoints

```text
202604_midiproxy_data/
  ddsp-piano-pytorch/
    workspaces_unified_2s/models/phase_1/ckpts/ddsp-piano_epoch_7_params.pt
    workspaces_unified_5s/models/phase_1/ckpts/ddsp-piano_epoch_7_params.pt
  ddsp-guitar-synth/
    flgd_5s/output/ddsp_guitar_synth_sr22050_fps100_seg5s/latest_model_checkpoint.pt
```

### Route IV SFProxy checkpoints

```text
202604_midiproxy_data/
  synth-proxy/
    proxy/
      checkpoints/
        salamander_piano/
          piano_salamander_piano_coverage_v2_b0_c1_r0_s0_2s_default/
          piano_salamander_piano_coverage_v2_b0_c1_r0_s0_5s_default/
          piano_salamander_piano_coverage_v2_b0_c1_r0_s0_10s_default/
          piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_2s_default/
          piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_5s_default/
          piano_salamander_piano_mixed_v2_b0p3_c0p4_r0p2_s0p1_10s_default/
          piano_salamander_piano_realism_v2_b0_c0_r1_s0_2s_default/
          piano_salamander_piano_realism_v2_b0_c0_r1_s0_5s_default/
          piano_salamander_piano_realism_v2_b0_c0_r1_s0_10s_default/
```

## What Kaya scripts do

These scripts:

- copy the repo to scratch
- link `workspaces/hdf5s` to the scratch data root
- link `pretrained_checkpoints` to `score_hpt/workspaces/pretrained_checkpoints`
- resolve DDSP or SFProxy checkpoints from `$MYSCRATCH/202604_midiproxy_data`
- expand SLURM arrays over ablation combinations
- move `checkpoints/` and `logs/` back to `${MYGROUP}` after training

They do not require:

- backend data caches
- soundfont directories
- local render outputs

unless you explicitly turn on audio evaluation.

## Default sweeps

### Route III

- `MODEL_TYPES=("hpt" "filmunet")`
- `SEGMENTS=("2" "5")`
- `AUDIO_LOSSES=("piano_ssm_spectral" "piano_ssm_spectral_plus_log_rms" "piano_ssm_spectral_plus_diffsynth_loudness" "piano_ssm_combined_rm")`
- `SUP_BACKEND_PAIRS=("0.0,1.0" "0.5,0.5")`
- `PRIOR_WEIGHTS=("0.0" "0.01")`

### Route IV

- `MODEL_TYPES=("hpt" "filmunet")`
- `SAMPLERS=("coverage" "mixed" "realism")`
- `SEGMENTS=("2" "5" "10")`
- `PROXY_LOSSES=("smooth_l1" "l1" "mse")`
- `SUP_BACKEND_PAIRS=("0.0,1.0" "0.5,0.5")`
- `PRIOR_WEIGHTS=("0.0" "0.01")`

## Main commands

Submit the default sweeps:

```bash
sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
sbatch kaya_scripts/kaya_hpt_route3_guitar_ablation.sh
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```

Run with a pretrained checkpoint:

```bash
MODEL_TYPES="hpt" FRONTEND_PRETRAINED=/abs/path/to/hpt_checkpoint.pth \
sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
```

```bash
MODEL_TYPES="filmunet" FRONTEND_PRETRAINED=/abs/path/to/filmunet_checkpoint.pth \
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```

Run with the latest Route II piano frontend ckpt chosen automatically:

```bash
MODEL_TYPES="hpt" SCORE_METHOD=note_editor USE_ROUTE2_PIANO_PRETRAINED_CKPT=1 \
sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
```

```bash
MODEL_TYPES="filmunet" USE_ROUTE2_PIANO_PRETRAINED_CKPT=1 \
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```

Use a relative frontend checkpoint path under `score_hpt/workspaces/pretrained_checkpoints`:

```bash
MODEL_TYPES="hpt" FRONTEND_PRETRAINED=hpt/120000_iterations.pth \
sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
```

```bash
MODEL_TYPES="hpt" SCORE_METHOD=note_editor FRONTEND_PRETRAINED=hpt+onset+score_note_editor/120000_iterations.pth \
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```

Trim the sweep:

```bash
MODEL_TYPES="hpt" SUP_BACKEND_PAIRS="0.5,0.5" PRIOR_WEIGHTS="0.0" \
sbatch kaya_scripts/kaya_hpt_route3_ablation.sh
```

```bash
MODEL_TYPES="filmunet" PRIOR_WEIGHTS="0.0" \
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```

## Interactive single runs

Allocate a node:

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=32G --time=08:00:00
```

Then run:

```bash
MODEL_TYPE=hpt SCORE_METHOD=direct SEGMENT_SECONDS=5 AUDIO_LOSS=piano_ssm_spectral \
bash kaya_scripts/kaya_hpt_route3_single.sh
```

```bash
MODEL_TYPE=filmunet SEGMENT_SECONDS=5 SAMPLER=realism PROXY_LOSS=l1 \
bash kaya_scripts/kaya_hpt_route4_single.sh
```

## Optional BSSL / BSTL on Kaya

Audio evaluation stays off by default.

If you explicitly want it:

```bash
ENABLE_AUDIO_METRICS=1 \
INSTRUMENT_PATH=/abs/path/to/your.sfz \
AUDIO_METRIC_MAX_SEGMENTS=1 \
sbatch kaya_scripts/kaya_hpt_route4_ablation.sh
```

Use this only when the machine actually has the renderer environment you need.
