# Experiment launchers

Run these from the repository root unless a script states otherwise. The scripts were recovered from the 5090/Kaya research workspaces; they expose experimental settings but require datasets and checkpoints outside Git.

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
