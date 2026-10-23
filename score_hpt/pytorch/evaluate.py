import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
from pytorch_utils import move_data_to_device, append_to_dict
from calculate_scores import frame_max_metrics_from_list, onset_pick_metrics_from_list
from utilities import pick_velocity_from_roll, write_events_to_midi

def _segments_from_output(output_dict):
    """Convert batched output/target rolls to per-segment dicts used by Kim metrics."""
    velocity = output_dict.get('velocity_output')
    if velocity is None:
        return [], []
    vel_roll = output_dict.get('velocity_roll')
    frame_roll = output_dict.get('frame_roll')
    onset_roll = output_dict.get('onset_roll')
    pedal_roll = output_dict.get('pedal_frame_roll')

    segments = []
    targets = []
    segs = velocity.shape[0]
    for idx in range(segs):
        pred = velocity[idx]
        gt_vel = vel_roll[idx]
        frames = min(pred.shape[0], gt_vel.shape[0])
        seg_pred = {'velocity_output': pred[:frames]}
        segments.append(seg_pred)

        pedal = pedal_roll[idx] if pedal_roll is not None else np.zeros(frames)
        if pedal.ndim > 1:
            pedal = np.squeeze(pedal, axis=-1)
        target_entry = {
            'velocity_roll': gt_vel[:frames],
            'frame_roll': frame_roll[idx][:frames],
            'onset_roll': onset_roll[idx][:frames],
            'pedal_frame_roll': pedal[:frames],
        }
        targets.append(target_entry)
    return segments, targets


def _kim_metrics_from_segments(output_dict_list, target_dict_list):
    """Run the same Kim-style metrics used in calculate_scores."""
    if not output_dict_list or not target_dict_list:
        return {}
    frame_max_err, frame_max_std = frame_max_metrics_from_list(output_dict_list, target_dict_list)
    onset_masked_error, onset_masked_std = onset_pick_metrics_from_list(output_dict_list, target_dict_list)
    stats = {
        'frame_max_error': round(frame_max_err, 4),
        'frame_max_std': round(frame_max_std, 4),
        'onset_masked_error': round(onset_masked_error, 4),
        'onset_masked_std': round(onset_masked_std, 4),
    }
    return stats


def _clean_optional_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return ""
    return text


class _AudioMetricEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        train_eval_cfg = getattr(cfg, "train_eval", None)
        audio_cfg = getattr(train_eval_cfg, "audio_metrics", None) if train_eval_cfg is not None else None
        self.enabled = bool(getattr(audio_cfg, "enabled", False)) if audio_cfg is not None else False
        self.include_train = bool(getattr(audio_cfg, "include_train", False)) if audio_cfg is not None else False
        self.max_segments = int(getattr(audio_cfg, "max_segments", 0) or 0) if audio_cfg is not None else 0
        self.instrument_path = _clean_optional_text(getattr(audio_cfg, "instrument_path", "")) if audio_cfg is not None else ""
        self.render_sr = int(getattr(audio_cfg, "render_sr", getattr(cfg.feature, "sample_rate", 22050))) if audio_cfg is not None else int(cfg.feature.sample_rate)
        self.eval_sr = int(getattr(audio_cfg, "eval_sr", getattr(cfg.feature, "sample_rate", 22050))) if audio_cfg is not None else int(cfg.feature.sample_rate)
        self.frames_per_second = float(getattr(audio_cfg, "frames_per_second", getattr(cfg.feature, "frames_per_second", 100))) if audio_cfg is not None else float(cfg.feature.frames_per_second)
        self.fft_size = int(getattr(audio_cfg, "fft_size", getattr(cfg.feature, "fft_size", 2048))) if audio_cfg is not None else int(cfg.feature.fft_size)
        self.bssl_mode = str(getattr(audio_cfg, "bssl_mode", "sone")) if audio_cfg is not None else "sone"
        self.num_samples = int(getattr(audio_cfg, "num_samples", 2048)) if audio_cfg is not None else 2048
        self.normalization = str(getattr(audio_cfg, "normalization", "zscore")) if audio_cfg is not None else "zscore"
        self.backend = str(getattr(audio_cfg, "backend", "auto")) if audio_cfg is not None else "auto"
        self.device = _clean_optional_text(getattr(audio_cfg, "device", "")) if audio_cfg is not None else ""
        default_velocity_method = getattr(getattr(cfg, "route2", None), "infer", None)
        default_velocity_method = getattr(default_velocity_method, "velocity_method", "onset_only")
        self.velocity_method = str(getattr(audio_cfg, "velocity_method", default_velocity_method)) if audio_cfg is not None else str(default_velocity_method)
        self.waveform_sample_rate = int(cfg.feature.sample_rate)
        self._deps_loaded = False

    def _ensure_ready(self):
        if not self.enabled:
            return
        if self.max_segments <= 0:
            raise ValueError("train_eval.audio_metrics.max_segments must be > 0 when audio metrics are enabled.")
        if not self.instrument_path:
            raise ValueError("train_eval.audio_metrics.instrument_path must be set when audio metrics are enabled.")
        instrument_path = Path(self.instrument_path).expanduser().resolve()
        if not instrument_path.exists():
            raise FileNotFoundError(f"train_eval.audio_metrics.instrument_path does not exist: {instrument_path}")
        self.instrument_path = str(instrument_path)
        if self._deps_loaded:
            return

        from direct_invension.eval_framework import _extract_pair_metrics, _import_bssl_eval, render_midi_to_audio
        import soundfile as sf

        self._extract_pair_metrics = _extract_pair_metrics
        self._evaluate_bssl_pair = _import_bssl_eval()
        self._render_midi_to_audio = render_midi_to_audio
        self._soundfile = sf
        self._deps_loaded = True

    def should_run(self, eval_name: str) -> bool:
        if not self.enabled:
            return False
        if eval_name == "train" and not self.include_train:
            return False
        return True

    def evaluate_batch(
        self,
        batch_data_dict,
        velocity_pred: torch.Tensor,
        eval_name: str,
        tmp_dir: Path,
        remaining: int,
    ) -> Dict[str, list]:
        if remaining <= 0:
            return {}
        self._ensure_ready()

        note_events_batch = batch_data_dict.get("aligned_note_events")
        waveform_batch = batch_data_dict.get("waveform")
        if note_events_batch is None or waveform_batch is None:
            return {}

        pred_np = velocity_pred.detach().cpu().numpy()
        metrics = {
            "real_pred_bssl_pearson_correlation": [],
            "real_pred_bstl_pearson_correlation": [],
        }

        batch_size = min(len(note_events_batch), pred_np.shape[0], remaining)
        for idx in range(batch_size):
            aligned_note_events = [dict(event) for event in note_events_batch[idx]]
            if not aligned_note_events:
                continue

            pred_roll = pred_np[idx]
            pick_velocity_from_roll(
                aligned_note_events,
                pred_roll,
                self.cfg,
                strategy=self.velocity_method,
            )

            stem = f"{eval_name}_{idx:03d}"
            pred_midi_path = tmp_dir / f"{stem}.pred.mid"
            pred_wav_path = tmp_dir / f"{stem}.pred.wav"
            real_wav_path = tmp_dir / f"{stem}.real.wav"

            write_events_to_midi(0.0, aligned_note_events, None, str(pred_midi_path))

            waveform = np.asarray(waveform_batch[idx], dtype=np.float32)
            waveform = np.clip(waveform, -1.0, 1.0)
            self._soundfile.write(str(real_wav_path), waveform, self.waveform_sample_rate)

            self._render_midi_to_audio(
                midi_path=pred_midi_path,
                instrument_path=self.instrument_path,
                wav_path=pred_wav_path,
                sample_rate=int(self.render_sr),
                backend=self.backend,
                skip_existing=False,
            )

            eval_payload = self._evaluate_bssl_pair(
                pred_wav=pred_wav_path,
                gt_wav=real_wav_path,
                sample_rate=int(self.eval_sr),
                frames_per_second=float(self.frames_per_second),
                fft_size=int(self.fft_size),
                bssl_mode=self.bssl_mode,
                num_samples=int(self.num_samples),
                normalization=self.normalization,
                device=self.device or None,
            )
            pair_metrics = self._extract_pair_metrics(eval_payload)
            if "bssl_pearson_correlation" in pair_metrics:
                metrics["real_pred_bssl_pearson_correlation"].append(
                    float(pair_metrics["bssl_pearson_correlation"])
                )
            if "bstl_pearson_correlation" in pair_metrics:
                metrics["real_pred_bstl_pearson_correlation"].append(
                    float(pair_metrics["bstl_pearson_correlation"])
                )

        return metrics


class SegmentEvaluator(object):
    def __init__(self, model, cfg):
        """Evaluate segment-wise metrics.
        Args:
            model: nn.Module
            cfg: OmegaConf config
        """
        self.model = model
        self.cfg = cfg
        self.input2 = cfg.model.input2
        self.input3 = cfg.model.input3
        score_cfg = getattr(cfg, "score_informed", None)
        self.score_method = str(getattr(score_cfg, "method", "direct") if score_cfg is not None else "direct").strip()
        self.score_cond_keys = self._resolve_score_cond_keys()
        self.audio_metric_evaluator = _AudioMetricEvaluator(cfg)

    def _resolve_score_cond_keys(self):
        cond_selected = []
        for key in [self.input2, self.input3]:
            if key and key not in cond_selected:
                cond_selected.append(key)

        if self.score_method == "direct":
            return []
        if self.score_method == "note_editor":
            return ["onset"] + ([self.input3] if self.input3 else [])
        return cond_selected

    def _forward_score_inf(self, batch_data_dict, device):
        audio = move_data_to_device(batch_data_dict["waveform"], device)
        cond = {k: move_data_to_device(batch_data_dict[f"{k}_roll"], device) for k in self.score_cond_keys}

        with torch.no_grad():
            self.model.eval()
            out = self.model(audio, cond)

        if "velocity_output" not in out and "vel_corr" in out:
            out = dict(out)
            out["velocity_output"] = out["vel_corr"]
        return out

    def evaluate(
        self,
        dataloader,
        loss_fn=None,
        target_rolls=None,
        eval_name: str = "",
    ):
        """Evaluate over dataloader and compute metrics, optional loss, and optional audio Pearson."""
        output_dict = {}
        device = next(self.model.parameters()).device
        required_target_keys = ("velocity_roll", "frame_roll", "onset_roll", "pedal_frame_roll")
        has_all_targets = True
        loss_values = []
        target_rolls = list(target_rolls or [])
        audio_metric_values = {
            "real_pred_bssl_pearson_correlation": [],
            "real_pred_bstl_pearson_correlation": [],
        }
        remaining_audio_segments = self.audio_metric_evaluator.max_segments
        audio_enabled = self.audio_metric_evaluator.should_run(eval_name)

        with tempfile.TemporaryDirectory(prefix=f"train_eval_{eval_name or 'split'}_") as tmp_root:
            tmp_root_path = Path(tmp_root)
            for batch_data_dict in dataloader:
                out = self._forward_score_inf(batch_data_dict, device)
                pred = out.get("velocity_output")
                if torch.is_tensor(pred):
                    append_to_dict(output_dict, "velocity_output", pred.data.cpu().numpy())

                missing_targets = [key for key in required_target_keys if key not in batch_data_dict]
                if missing_targets:
                    has_all_targets = False
                    continue

                for key in required_target_keys:
                    append_to_dict(output_dict, key, batch_data_dict[key])

                if loss_fn is not None and target_rolls:
                    cond_torch = {
                        k: move_data_to_device(batch_data_dict[f"{k}_roll"], device)
                        for k in self.score_cond_keys
                    }
                    target_torch = {
                        k: move_data_to_device(batch_data_dict[k], device)
                        for k in target_rolls
                    }
                    with torch.no_grad():
                        loss = loss_fn(self.cfg, out, target_torch, cond_dict=cond_torch)
                    loss_values.append(float(loss.detach().item()))

                if audio_enabled and torch.is_tensor(pred) and remaining_audio_segments > 0:
                    batch_audio_metrics = self.audio_metric_evaluator.evaluate_batch(
                        batch_data_dict=batch_data_dict,
                        velocity_pred=pred,
                        eval_name=eval_name or "eval",
                        tmp_dir=tmp_root_path,
                        remaining=remaining_audio_segments,
                    )
                    used_count = 0
                    for key, values in batch_audio_metrics.items():
                        audio_metric_values[key].extend(values)
                        used_count = max(used_count, len(values))
                    remaining_audio_segments = max(0, remaining_audio_segments - used_count)

        stats = {}
        if not has_all_targets:
            if loss_values:
                stats["avg_loss"] = round(float(np.mean(loss_values)), 6)
            for key, values in audio_metric_values.items():
                if values:
                    stats[key] = round(float(np.mean(values)), 4)
            return stats

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        if 'velocity_output' in output_dict:
            segments, targets = _segments_from_output(output_dict)
            stats.update(_kim_metrics_from_segments(segments, targets))

        if loss_values:
            stats["avg_loss"] = round(float(np.mean(loss_values)), 6)
        for key, values in audio_metric_values.items():
            if values:
                stats[key] = round(float(np.mean(values)), 4)
        return stats
