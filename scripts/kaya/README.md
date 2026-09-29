# Kaya SLURM launchers

These five scripts are the recovered Kaya variants. They copy the repository into job scratch, link the large data/checkpoint tree, and launch Score-HPT adaptation. The old research `kaya_scripts` README named other scripts that were **not present** in the recovered workspace; this list only describes files actually included here.

| Script | Backend and instrument |
| --- | --- |
| [`kaya_hpt_route3_piano_ablation.sh`](kaya_hpt_route3_piano_ablation.sh) | Diff-Synth piano sweep |
| [`kaya_hpt_route3_guitar_ablation.sh`](kaya_hpt_route3_guitar_ablation.sh) | Diff-Synth guitar sweep |
| [`kaya_hpt_route4_piano_ablation.sh`](kaya_hpt_route4_piano_ablation.sh) | Diff-SFProxy piano sweep |
| [`kaya_hpt_route4_guitar_ablation.sh`](kaya_hpt_route4_guitar_ablation.sh) | Diff-SFProxy guitar sweep |
| [`kaya_hpt_route4_single.sh`](kaya_hpt_route4_single.sh) | Single Route IV debug run |

The scripts default to the source checkout `$HOME/sfproxy-velocity-estimation` and the historical data tree `$MYSCRATCH/202604_midiproxy_data`. `PROJECT_NAME` changes the checkout name; `DATA_PROJECT` changes the external data-tree name. Clone or sync this reorganized repository to the expected path before `sbatch`. The old 5090/Kaya checkpoint and SoundFont paths must be checked against the actual cluster state.

Example:

```bash
sbatch --export=ALL,SEGMENT_SECONDS=5 scripts/kaya/kaya_hpt_route4_single.sh
```

The scripts call `module load` and `source activate bark_env`; update those commands if Kaya's modules or environment changed. They are retained as experiment provenance and have passed shell syntax checks, but this local layout pass does not claim a completed Kaya training run. The paper's final 5 s guitar checkpoint and predictions were not in the inspected Kaya scratch paths.
