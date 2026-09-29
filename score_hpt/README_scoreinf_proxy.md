# Score-HPT + Backend Supervision

This note is the training reference for `score_hpt/` after the Route III / IV
loss cleanup. It only documents the objectives that still exist in the
codebase.

The central idea is the same for every route: a velocity-estimation front-end
(Score-HPT) predicts a per-frame, per-pitch velocity roll. The
differences are **what supervises that prediction**.

## Training entrypoints

All three entrypoints live under [`pytorch/`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch):

- [`train.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train.py) -- Route II, supervised velocity only
- [`train_ddsp.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train_ddsp.py) -- Route III, weak supervision via a frozen **Diff-Synth** (DDSP-Piano / DDSP-Guitar) renderer + audio loss
- [`train_proxy.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train_proxy.py) -- Route IV, weak supervision via a frozen **Diff-SFProxy** (SoundFont neural proxy) note-wise loss
- [`train_backend.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train_backend.py) -- the shared training loop that Route III / IV reuse

Route III and Route IV use the same loop and the same total-loss structure:

```
total = supervised_weight * supervised_loss
      + backend_weight    * backend_loss          # route-specific
      + prior_weight      * velocity_prior_loss   # optional
      + saturation_weight * velocity_saturation_loss  # optional
```

The only thing that changes between Route III and Route IV is which `backend_loss`
is used. `supervised_weight=0` is the typical setting for weakly-supervised runs.

## Model combinations

The project is intentionally narrowed to two main combinations:

- `hpt-direct`
- `hpt-note_editor`

Operational rules:

- `model.type` is always `hpt`
- `score_informed.method` selects `direct` or `note_editor`
- `model.frontend_pretrained_mode=scratch` means training from scratch
- `model.frontend_pretrained_mode=route2_piano_specific` + `model.frontend_pretrained=/path/to/ckpt` loads a Route II checkpoint

## Route II -- supervised velocity (`train.py`)

Available supervised losses (all defined in [`losses.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py)):

- `velocity_bce`  -- see [losses.py:96](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L96)
- `velocity_mse`  -- see [losses.py:102](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L102)
- `kim_bce_l1`    -- see [losses.py:108](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L108), **recommended default**

These are the only `loss.loss_type` values accepted by [`get_loss_func`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L549).

## Route III -- Diff-Synth backend loss (`train_ddsp.py`)

Route III renders predicted velocities through a frozen DDSP renderer and
supervises the front-end by matching the rendered audio against the real audio.

Supported backend audio losses (canonical names, see
[`AUDIO_LOSS_TYPES`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L481)):

- `piano_ssm_spectral`               -- multi-resolution spectral loss, Piano-SSM / DDSP style
- `piano_ssm_spectral_plus_log_rms`  -- spectral loss + small clip-level log-RMS auxiliary (**default**)

The selected audio loss is built by
[`build_audio_loss`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L515) from `cfg.backend.audio_loss.*`.

Typical Route III overrides:

```bash
backend.enabled=true
backend.type=diffsynth_piano            # or diffsynth_guitar
backend.audio_loss.type=piano_ssm_spectral_plus_log_rms
backend.checkpoint=/path/to/ddsp.ckpt
loss.supervised_weight=0.0
loss.backend_weight=1.0
```

## Route IV -- Diff-SFProxy backend loss (`train_proxy.py`)

Route IV replaces the differentiable renderer with a frozen neural proxy
(`synth-proxy`). Instead of comparing audio, it compares **per-note feature
embeddings** (harmonic energy + onset flux) between GT-aligned notes and the
proxy's prediction driven by the predicted velocities.

The whole pipeline lives in
[`proxy/sfproxy.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/proxy/sfproxy.py):

- `SFProxyObjective.compute` -- the top-level entry used by `train_backend.py` ([sfproxy.py:532](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/proxy/sfproxy.py#L532))
- `_build_note_batch` packs the GT-aligned note list with velocity values read from the predicted roll at each onset frame
- `_extract_target_features` runs the frozen `extract_note_features_padded` on the real audio to get the target per-note features
- `self.model(...)` (a frozen `NoteProxyTransformer`) predicts features from `(pitch, onset, duration, velocity)` tokens
- `_masked_loss` compares predicted vs target features with the chosen point-wise loss

### Supported Route IV losses

Only three values of `backend.diffproxy.loss_type` are accepted, all implemented
in [`SFProxyObjective._masked_loss`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/proxy/sfproxy.py#L493):

- `smooth_l1` -- `F.smooth_l1_loss(pred, target, beta=loss_beta)` (**default / recommended**)
- `l1`        -- `|pred - target|`
- `mse`       -- `(pred - target) ** 2`

Any other value raises `ValueError` at construction time.

### How the smooth-L1 supervision is actually wired

All the following happens inside
[`SFProxyObjective.compute`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/proxy/sfproxy.py#L532):

1. `_crop_inputs` -- crop audio + rolls to the backend segment (`backend.backend_segment_seconds`, e.g. 2 s or 5 s inside the 10 s Score-HPT segment).
2. `_crop_note_events_batch` -- crop the dataloader's GT aligned note list to the same window; `use_gt_aligned_note_events=true` means velocities are read from the predicted roll at each GT onset frame (no threshold-based note building).
3. `_build_note_batch` -> `_extract_note_list_from_events` -> `_read_note_velocity` -- optional onset window pooling is controlled by `use_onset_window_pooling` / `onset_window_left|right|reduction`.
4. `_extract_target_features` -- computes target features from the real audio via the frozen `DynamicsFeatureConfig` extractor (no gradients).
5. `self.model(pitch, cont_norm, mask)` -- frozen transformer predicts features from the note tokens (velocity tokens carry gradients back into `vel_pred`).
6. `_masked_loss(pred, target, mask)` -- masked, feature-weighted point-wise loss. **This is where the `smooth_l1` is computed**, at
   [sfproxy.py:501-511](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/proxy/sfproxy.py#L501-L511).

The training loop in `train_backend.py` then does, at
[train_backend.py:633-643](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/train_backend.py#L633-L643):

```python
proxy_stats = proxy_objective.compute(batch_data_dict, audio, vel_pred, iteration)
total_loss = total_loss + backend_weight * proxy_stats["proxy_loss"]
```

So the **smooth-L1 loss of Route IV is `proxy_stats["proxy_loss"]` returned by
`SFProxyObjective._masked_loss`**, and the gradient flows:

```
smooth_l1(pred_note_feat, target_note_feat)
   -> pred_note_feat = frozen_NoteProxyTransformer(pitch, onset, dur, velocity)
   -> velocity = vel_pred[onset_frame, pitch_idx]   (optional onset-window pooling)
   -> vel_pred = ScoreInfWrapper(audio, cond, ...)
```

Only `vel_pred` receives gradients; the transformer and the feature extractor
are frozen (`.eval()` / `param.requires_grad = False`).

### Typical Route IV overrides

```bash
backend.enabled=true
backend.type=diffproxy
backend.diffproxy.loss_type=smooth_l1          # or l1 | mse
backend.diffproxy.loss_beta=1.0
backend.diffproxy.feature_weights=[1.0, 0.25]  # [harmonic_energy, onset_flux]
backend.diffproxy.use_gt_aligned_note_events=true
backend.checkpoint=/path/to/sfproxy/last.ckpt
loss.supervised_weight=0.0
loss.backend_weight=1.0
```

## Prior and saturation losses (shared, kept for both routes)

Two auxiliary regularizers in [`losses.py`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py) complement the backend loss:

- `velocity_prior_loss` -- weak anti-collapse prior
  `L = (mean(v) - mu)^2 + relu(var_min - var(v))`, on onset positions by default.
  Enabled with `loss.velocity_prior_weight > 0` (and `loss.velocity_prior_mean`, `loss.velocity_prior_min_var`). See [losses.py:121](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L121).
- `velocity_saturation_loss` -- penalises predictions saturating near the top.
  Enabled with `loss.velocity_saturation_weight > 0` (and `loss.velocity_saturation_threshold`). See [losses.py:152](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L152).

These are the **core** regularizers kept on top of the Route III / IV backend
losses. They are intentionally not ablated away.

## Wandb logging

### Run names

Route III / IV runs encode the experiment identity in `wandb.name`:

- `route3-maestro-2s-hpt-note_editor-...`
- `route3-maestro-2s-hpt-direct-...`
- `route4-maestro-5s-hpt-note_editor-...`
- `route4-maestro-5s-hpt-direct-...`

The backend segment length is part of the run name.

### Common train metrics

- `train_total_loss`
- `train_supervised_loss` / `train_supervised_weighted`
- `train_backend_loss`    / `train_backend_weighted`
- `train_prior_loss`      / `train_prior_weighted`
- `train_saturation_loss` / `train_saturation_weighted`

### Common eval metrics

- `train_stat.frame_max_error`, `train_stat.frame_max_std`
- `train_stat.onset_masked_error`, `train_stat.onset_masked_std`
- `valid_maestro_stat.*`, `valid_smd_stat.*`

### Route III extra backend charts

With `piano_ssm_spectral_plus_log_rms`, each sub-term is logged separately by
[`CompositeAudioLoss`](/media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt/pytorch/losses.py#L334):

- `train_backend_spectral_raw`, `train_backend_spectral_weighted`
- `train_backend_log_rms_raw`, `train_backend_log_rms_weighted`
- `train_backend_log_rms_log_rms_loss`
- `train_backend_log_rms_log_rms_pred`, `train_backend_log_rms_log_rms_target`
- `train_backend_*audio_rms_*`

With plain `piano_ssm_spectral`, only `train_backend_loss` and the audio RMS
statistics are logged.

### Route IV extra backend charts

- `train_backend_note_mae`  (per-batch MAE on the feature tensor)
- `train_backend_note_count`

### Summary-only constants

Kept in `wandb.summary` and `training.log` rather than charts:

- `backend_loss_sample_rate`, `backend_loss_frame_rate`
- `backend_proxy_frames`, `backend_proxy_polyphony`
- `backend_renderer_sample_rate`, `backend_renderer_frame_rate`
- `backend_renderer_segment_seconds`, `backend_native_segment_seconds`
- `backend_renderer_hop_length`, `backend_renderer_n_fft`

## Local log files

Each Route III / IV checkpoint directory writes:

- `training_stats.txt` -- fixed experiment header (loss type, weights, params count, ...)
- `training.log`       -- per-eval-iteration snapshot of train aggregates, eval aggregates, and summary constants

`training.log` is the most useful file for after-the-fact discussion.

## BSSL / BSTL evaluation (optional)

Rendered-audio Pearson evaluation stays off by default:

- `train_eval.audio_metrics.enabled=false`
- `train_eval.audio_metrics.instrument_path=""`

Enable it only on machines with an `sfz` / `sf2` renderer installed. When
enabled, wandb additionally logs:

- `real_pred_bssl_pearson_correlation`
- `real_pred_bstl_pearson_correlation`

## Core loss set

- Route II: `velocity_bce`, `velocity_mse`, `kim_bce_l1` (default)
- Route III: `piano_ssm_spectral`, `piano_ssm_spectral_plus_log_rms` (default)
- Route IV: `smooth_l1` (default), `l1`, `mse`
- Shared regularizers: `velocity_prior_loss`, `velocity_saturation_loss`
