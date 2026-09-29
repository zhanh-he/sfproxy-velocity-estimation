# Guitar result recovery status — 2026-09-29

The guitar listening cards need **adapted VeloEst velocity predictions** for the selected 5 s Diff-Synth and Diff-SFProxy runs. A trained differentiable synthesizer or SoundFont proxy is only the *frozen backend* used during adaptation; it does not itself contain the final per-note velocity estimator or its prediction MIDI.

| Artifact | Checked location | Status |
| --- | --- | --- |
| GAPS / FL aligned source recordings and scores | lab5090 `score_hpt/workspaces/hdf5s/{gaps_sr22050_old,francoisleduc_sr22050}` | Present; the 20 s original and Flat 64 demos were built from these sources. |
| 5 s guitar Diff-Synth backend | lab5090 `ddsp-guitar-synth/flgd_5s/output/ddsp_guitar_synth_sr22050_fps100_seg5s/latest_model_checkpoint.pt` | Present. This is a synthesizer checkpoint, not an adapted VeloEst checkpoint. |
| 5 s guitar Diff-SFProxy backend | lab5090 `synth-proxy/proxy/checkpoints/guitar/*_5s_default/*.ckpt` | Present. These are proxy checkpoints, not adapted VeloEst checkpoints. |
| Guitar prediction MIDI and evaluation renders | lab5090 `score_hpt/workspaces/route3/{gaps_test_guitar,francoisleduc_full_guitar}` and matching `route3_eval` | Only a **2 s Diff-SFProxy** run was found in these directories. It is an ablation and is not used for the site's final 5 s method slots. |
| Selected 5 s adapted guitar VeloEst checkpoint and its prediction MIDI | lab5090 `score_hpt/workspaces/checkpoints`, `route3`, `route3_eval`, and other accessible paths under `202604_midiproxy_data` | Not found. The retained Score-HPT checkpoint folders are piano runs. |
| Kaya copies / results | `/scratch/ems011/zhe/202604_midiproxy`, `/home/zhe/202604_midiproxy`, the September scratch archive, and the logged `/group/ems011/zhe/202604_midiproxy_results` location | Scratch and archive retain code copies but no `.pth`/`.pt`/`.ckpt` or prediction MIDI. The logged results directory is absent. |
| 3090 | Machine unavailable | Cannot inspect its former storage. |

**Conclusion:** The backend models have not all been lost, but the paper's selected **5 s adapted guitar VeloEst models and prediction files have not been located** in the accessible 5090/Kaya storage. The site is not merely waiting for an inference command against a recovered final checkpoint. The old 2 s proxy MIDI cannot be presented as a 5 s Diff-SFProxy result, and no guitar Diff-Synth prediction MIDI was found.

The remaining routes are to recover the selected checkpoints/predictions from another backup, or to rerun 5 s guitar adaptation with the retained code, source data, and backend checkpoints, then infer and validate matched 20 s clips. Any rerun should be identified as a regenerated example until its recipe and metrics are checked against the camera-ready results.
