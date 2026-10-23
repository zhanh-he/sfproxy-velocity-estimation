import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from inference import VeloTranscription, resolve_checkpoint
from utilities import (
    OnsetsFramesPostProcessor,
    TargetProcessor,
    create_folder,
    get_model_name,
    int16_to_float32,
    resolve_hdf5_dir,
    traverse_folder,
)


def _score_method(method) -> str:
    return str(method or "direct").strip()


def _is_direct(method) -> bool:
    return _score_method(method) == "direct"


def _clean_name(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return ""
    return text


def _cond_tag(cfg, method: str) -> str:
    if _is_direct(method):
        return ""
    conds = []
    for value in (cfg.model.input2, cfg.model.input3):
        text = _clean_name(value)
        if text and text not in conds:
            conds.append(text)
    return f"+{'_'.join(conds)}" if conds else ""


def _wandb_name(cfg) -> str:
    explicit_name = _clean_name(getattr(cfg.wandb, "name", ""))
    if explicit_name:
        return explicit_name
    method = _score_method(cfg.score_informed.method)
    return f"eval-{cfg.dataset.test_set}-{cfg.model.type}-{method}{_cond_tag(cfg, method)}"


def _mean_dict(stats_dict: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(np.mean(values))
        for key, values in stats_dict.items()
        if values
    }


def _consume_flag(args: Sequence[str], flag: str) -> Tuple[bool, List[str]]:
    enabled = False
    remaining: List[str] = []
    for arg in args:
        if arg == flag:
            enabled = True
        else:
            remaining.append(arg)
    return enabled, remaining


def _midi_to_hz(notes: np.ndarray) -> np.ndarray:
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))


def mir_eval_metrics(
    cfg,
    output_dict: Dict[str, np.ndarray],
    ref_note_events: Sequence[Dict[str, float]],
) -> Tuple[float, float, float]:
    import mir_eval

    if not ref_note_events:
        return 0.0, 0.0, 0.0

    post_processor = OnsetsFramesPostProcessor(cfg)
    detected_notes = post_processor.output_dict_to_detected_notes(output_dict)
    if len(detected_notes) == 0:
        return 0.0, 0.0, 0.0

    detected_notes = np.asarray(detected_notes, dtype=np.float32)
    est_intervals = detected_notes[:, 0:2].copy()
    invalid = est_intervals[:, 1] <= est_intervals[:, 0]
    est_intervals[invalid, 1] = est_intervals[invalid, 0] + 1e-6

    ref_intervals = np.asarray(
        [[e["onset_time"], e["offset_time"]] for e in ref_note_events],
        dtype=np.float32,
    )
    ref_pitches = _midi_to_hz(
        np.asarray([e["midi_note"] for e in ref_note_events], dtype=np.float32)
    )
    ref_velocities = np.asarray(
        [e["velocity"] for e in ref_note_events],
        dtype=np.float32,
    )
    est_pitches = _midi_to_hz(detected_notes[:, 2])
    est_velocities = detected_notes[:, 3] * float(cfg.feature.velocity_scale)

    precision, recall, f1, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        ref_intervals=ref_intervals,
        ref_pitches=ref_pitches,
        ref_velocities=ref_velocities,
        est_intervals=est_intervals,
        est_pitches=est_pitches,
        est_velocities=est_velocities,
    )
    return float(precision), float(recall), float(f1)


def _iterations(cfg) -> List[int]:
    start = int(getattr(cfg.exp, "eval_start_iteration", 0))
    end = int(getattr(cfg.exp, "eval_end_iteration", cfg.exp.total_iteration))
    step = int(
        getattr(
            cfg.exp,
            "eval_step_iteration",
            getattr(cfg.exp, "save_iteration", cfg.exp.eval_iteration),
        )
    )
    return list(range(start, end + 1, step))


def _onset_l1(
    output_segment: Dict[str, np.ndarray],
    target_segment: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Kim et al. note-level L1 error for a single segment."""
    error_rows: List[np.ndarray] = []
    num_notes = 0
    frames = min(
        output_segment["velocity_output"].shape[0],
        target_segment["velocity_roll"].shape[0],
    )
    for nth_frame in range(frames):
        gt_onset_frame = target_segment["onset_roll"][nth_frame]
        if np.count_nonzero(gt_onset_frame) == 0:
            continue
        pred_frame = output_segment["velocity_output"][nth_frame]
        gt_frame = target_segment["velocity_roll"][nth_frame]
        pred_onset = np.multiply(pred_frame, gt_onset_frame) * 128.0
        gt_onset = np.multiply(gt_frame, gt_onset_frame)
        note_error = np.abs(pred_onset - gt_onset)
        num_notes += int(np.count_nonzero(gt_onset_frame))
        error_rows.append(note_error[np.newaxis, :])
    if error_rows:
        segment_error = np.concatenate(error_rows, axis=0)
    else:
        segment_error = np.empty((0, 88), dtype=float)
    return segment_error, num_notes


def _note_spans(midi_vel_roll: np.ndarray) -> List[Dict[str, np.ndarray]]:
    sound_profile: List[Dict[str, np.ndarray]] = []
    for pitch, key in enumerate(midi_vel_roll):
        iszero = np.concatenate(([0], np.equal(key, 0).astype(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0]
        if ranges.size <= 2:
            continue
        temp = np.delete(ranges, [0, -1])
        sound_durations = temp.reshape(-1, 2)
        for duration in sound_durations:
            vel = midi_vel_roll[pitch, duration[0]]
            sound_profile.append({"pitch": pitch, "velocity": vel, "duration": duration})
    return sound_profile


get_midi_sound_profile = _note_spans


def _stack_rolls(
    output_dict_list: Sequence[Dict[str, np.ndarray]],
    target_list: Sequence[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    score_rows: List[np.ndarray] = []
    estimation_rows: List[np.ndarray] = []

    for target_segment, output_segment in zip(target_list, output_dict_list):
        frames = target_segment["velocity_roll"].shape[0]
        for nth_frame in range(frames):
            score_rows.append(target_segment["velocity_roll"][nth_frame][np.newaxis, :])
            estimation_rows.append(output_segment["velocity_output"][nth_frame][np.newaxis, :])

    if not score_rows:
        empty_2d = np.empty((0, 88), dtype=float)
        return empty_2d, empty_2d

    score = np.concatenate(score_rows, axis=0)
    estimation = np.concatenate(estimation_rows, axis=0)
    return score, estimation


def frame_max_metrics_from_list(output_dict_list: Sequence[Dict[str, np.ndarray]], target_list: Sequence[Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """Compute only frame-max MAE/STD (lightweight path for train-time evaluation)."""
    score, estimation = _stack_rolls(output_dict_list, target_list)
    if score.size == 0:
        return 0.0, 0.0

    score_t = np.transpose(score)
    estimation_t = np.transpose(estimation)
    score_sound_profile = _note_spans(score_t)
    accum_error: List[float] = []

    for note_profile in score_sound_profile:
        start, end = note_profile["duration"]
        vel_est = estimation_t[note_profile["pitch"]][start:end].copy()
        vel_est[vel_est <= 0.0001] = 0
        max_estimation = float(np.max(vel_est) * 128.0) if vel_est.size else 0.0
        notelevel_error = abs(max_estimation - float(note_profile["velocity"]))
        accum_error.append(notelevel_error)

    frame_max_error = float(np.mean(accum_error)) if accum_error else 0.0
    std_max_error = float(np.std(accum_error)) if accum_error else 0.0
    return frame_max_error, std_max_error


def onset_pick_metrics_from_list(output_dict_list: Sequence[Dict[str, np.ndarray]], target_dict_list: Sequence[Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """Kim et al. onset-only evaluation."""
    score_error_rows: List[np.ndarray] = []
    num_note = 0
    for output_dict_segmentseconds, target_dict_segmentseconds in zip(
        output_dict_list, target_dict_list
    ):
        segment_error, num_onset = _onset_l1(
            output_dict_segmentseconds, target_dict_segmentseconds
        )
        if segment_error.size:
            score_error_rows.append(segment_error)
        num_note += num_onset
    if num_note == 0:
        return 0.0, 0.0

    score_error = np.concatenate(score_error_rows, axis=0) if score_error_rows else np.empty((0, 88))
    mean_error = float(np.sum(score_error) / num_note)
    non_zero = score_error[score_error != 0]
    std_error = float(non_zero.std()) if non_zero.size else 0.0
    return mean_error, std_error


class KimStyleEvaluator:
    """Run HPT inference and compute Kim et al. evaluation metrics."""

    CSV_FIELDS = [
        "test_h5",
        "frame_max_error",
        "frame_max_std",
        "onset_masked_error",
        "onset_masked_std",
    ]
    MIREVAL_CSV_FIELDS = [
        "mir_eval_precision",
        "mir_eval_recall",
        "mir_eval_f1",
    ]

    def __init__(
        self,
        cfg,
        enable_mireval: bool = False,
        checkpoint_path: Optional[str] = None,
        results_subdir: str = "kim_eval",
    ):
        self.cfg = cfg
        self.enable_mireval = enable_mireval
        self.csv_fields = list(self.CSV_FIELDS)
        if self.enable_mireval:
            self.csv_fields[1:1] = self.MIREVAL_CSV_FIELDS
        model_name = get_model_name(cfg)
        score_method = _score_method(cfg.score_informed.method)
        if not _is_direct(score_method):
            model_name = f"{model_name}+score_{score_method}"

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else resolve_checkpoint(cfg, explicit_path=None)
        self.ckpt_iteration = self.checkpoint_path.stem.replace("_iterations", "")
        self.model_name = self.checkpoint_path.parent.name or model_name

        self.transcriptor = VeloTranscription(str(self.checkpoint_path), cfg)
        self.params_count = int(
            sum(p.numel() for p in self.transcriptor.model.parameters())
        )
        self.params_count_k = float(self.params_count / 1e3)
        self.params_count_m = float(self.params_count / 1e6)

        hdf5_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
        _, self.hdf5_paths = traverse_folder(hdf5_dir)

        self.results_dir = (
            Path(cfg.exp.workspace)
            / results_subdir
            / cfg.dataset.test_set
            / self.model_name
            / f"{self.ckpt_iteration}_iterations"
        )

    def _prepare_inputs(self, target_dict: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        target_dict["exframe_roll"] = target_dict["frame_roll"] * (1 - target_dict["onset_roll"])
        input2 = target_dict.get(f"{self.cfg.model.input2}_roll") if self.cfg.model.input2 else None
        input3 = target_dict.get(f"{self.cfg.model.input3}_roll") if self.cfg.model.input3 else None
        return input2, input3

    def _process_file(self, hdf5_path: str) -> Optional[Dict[str, float | str]]:
        with h5py.File(hdf5_path, "r") as hf:
            if hf.attrs["split"].decode() != "test":
                return None
            audio = int16_to_float32(hf["waveform"][:])
            midi_events = [e.decode() for e in hf["midi_event"][:]]
            midi_events_time = hf["midi_event_time"][:]

        segment_seconds = len(audio) / self.cfg.feature.sample_rate
        target_processor = TargetProcessor(segment_seconds=segment_seconds, cfg=self.cfg)
        target_dict, note_events, _ = target_processor.process(
            start_time=0, midi_events_time=midi_events_time, midi_events=midi_events, extend_pedal=True
        )

        input2, input3 = self._prepare_inputs(target_dict)
        transcribed = self.transcriptor.transcribe(audio, input2, input3, midi_path=None)
        output_dict = transcribed["output_dict"]

        predicted_roll = output_dict["velocity_output"]
        align_len = min(predicted_roll.shape[0], target_dict["velocity_roll"].shape[0])

        output_entry = {
            "velocity_output": predicted_roll[:align_len],
        }
        target_entry = {
            "velocity_roll": target_dict["velocity_roll"][:align_len],
            "onset_roll": target_dict["onset_roll"][:align_len],
        }
        mireval_output_dict = {
            "velocity_output": predicted_roll[:align_len],
            "frame_output": target_dict["frame_roll"][:align_len],
            "onset_output": target_dict["onset_roll"][:align_len],
            "offset_output": target_dict["offset_roll"][:align_len],
        }

        output_dict_list = [output_entry]
        target_dict_list = [target_entry]

        frame_max_error, frame_max_std = frame_max_metrics_from_list(output_dict_list, target_dict_list)
        onset_masked_error, onset_masked_std = onset_pick_metrics_from_list(output_dict_list, target_dict_list)
        metrics: Dict[str, float | str] = {
            "audio_name": Path(hdf5_path).name,
            "frame_max_error": frame_max_error,
            "frame_max_std": frame_max_std,
            "onset_masked_error": onset_masked_error,
            "onset_masked_std": onset_masked_std,
        }

        if self.enable_mireval:
            (
                metrics["mir_eval_precision"],
                metrics["mir_eval_recall"],
                metrics["mir_eval_f1"],
            ) = mir_eval_metrics(self.cfg, mireval_output_dict, note_events)

        return metrics

    def run(self) -> Dict[str, List[float]]:
        csv_path = self.results_dir / f"{self.model_name}_{self.cfg.dataset.test_set}_kim.csv"
        create_folder(str(self.results_dir))

        aggregated: Dict[str, List[float]] = {field: [] for field in self.csv_fields if field != "test_h5"}
        processed = 0

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_fields)

            progress = tqdm(sorted(self.hdf5_paths), desc="Kim Eval", unit="file", ncols=80)
            for hdf5_path in progress:
                metrics = self._process_file(hdf5_path)
                if not metrics:
                    continue

                audio_name = metrics["audio_name"]
                row = [audio_name] + [metrics[field] for field in self.csv_fields[1:]]
                writer.writerow(row)

                for field in aggregated.keys():
                    aggregated[field].append(float(metrics[field]))

                processed += 1
                avg_frame_err = np.mean(aggregated["frame_max_error"])
                progress.set_postfix({"frame_err": f"{avg_frame_err:.2f}"}, refresh=False)

        if not processed:
            return {}

        return aggregated


def _run_single_mode(
    cfg,
    enable_mireval: bool = False,
) -> None:
    evaluator = KimStyleEvaluator(
        cfg,
        enable_mireval=enable_mireval,
    )
    print("=" * 80)
    print("Evaluation Mode : Kim et al. (single checkpoint)")
    print(f"Model Name      : {evaluator.model_name}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print(f"Checkpoint      : {evaluator.checkpoint_path}")
    print(f"Enable MirEval  : {evaluator.enable_mireval}")
    print(
        f"Params          : {evaluator.params_count} "
        f"({evaluator.params_count_k:.3f} K, {evaluator.params_count_m:.3f} M)"
    )
    print("=" * 80)

    stats_dict = evaluator.run()
    if not stats_dict:
        print("No test files processed.")
        return

    mean_stats = _mean_dict(stats_dict)
    print("\n===== Kim-style Average Metrics =====")
    for key, value in mean_stats.items():
        print(f"{key}: {value:.4f}")


def _run_multi_mode(
    cfg,
    enable_mireval: bool = False,
) -> None:
    model_name = get_model_name(cfg)
    score_method = _score_method(cfg.score_informed.method)
    if not _is_direct(score_method):
        model_name = f"{model_name}+score_{score_method}"
    iterations = _iterations(cfg)
    ckpt_root = Path(cfg.exp.workspace) / "checkpoints" / model_name

    summary_dir = Path(cfg.exp.workspace) / "kim_eval_summary" / cfg.dataset.test_set / model_name
    create_folder(str(summary_dir))
    summary_csv = summary_dir / "iter_summary.csv"

    print("=" * 80)
    print("Evaluation Mode : Kim et al. (multi-checkpoint)")
    print(f"Model Name      : {model_name}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print(f"Checkpoint Dir  : {ckpt_root}")
    print(f"Enable MirEval  : {enable_mireval}")
    print(
        f"Iterations      : {iterations[0]} -> {iterations[-1]} "
        f"(step={iterations[1] - iterations[0] if len(iterations) > 1 else 0})"
    )
    print("=" * 80)

    run_name = _wandb_name(cfg)
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )

    summary_fields = [
        "iteration",
        "checkpoint_path",
        "params_count",
        "params_count_k",
        "params_count_m",
    ]
    if enable_mireval:
        summary_fields.extend(KimStyleEvaluator.MIREVAL_CSV_FIELDS)
    summary_fields.extend(
        [
            "frame_max_error",
            "frame_max_std",
            "onset_masked_error",
            "onset_masked_std",
        ]
    )

    with open(summary_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
        writer.writeheader()

        for iteration in iterations:
            ckpt_path = ckpt_root / f"{iteration}_iterations.pth"
            evaluator = KimStyleEvaluator(
                cfg,
                enable_mireval=enable_mireval,
                checkpoint_path=str(ckpt_path),
            )
            print(
                f"[eval] iter={iteration} | params={evaluator.params_count_k:.3f} K / "
                f"{evaluator.params_count_m:.3f} M"
            )
            stats_dict = evaluator.run()

            mean_stats = _mean_dict(stats_dict)
            row = {
                "iteration": iteration,
                "checkpoint_path": str(ckpt_path),
                "params_count": evaluator.params_count,
                "params_count_k": round(evaluator.params_count_k, 6),
                "params_count_m": round(evaluator.params_count_m, 6),
            }
            row.update({k: round(v, 6) for k, v in mean_stats.items()})
            writer.writerow(row)

            payload = {
                "iteration": iteration,
                "eval/params_count": evaluator.params_count,
                "eval/params_count_k": evaluator.params_count_k,
                "eval/params_count_m": evaluator.params_count_m,
            }
            payload.update({f"eval/{k}": v for k, v in mean_stats.items()})
            wandb.log(payload)

    wandb.finish()

    print(f"\n[done] Wrote evaluation summary: {summary_csv}")


def main() -> None:
    enable_mireval, overrides = _consume_flag(sys.argv[1:], "--enable_mireval")
    initialize(config_path="./config", job_name="kim_eval", version_base=None)
    cfg = compose(config_name="config", overrides=overrides)
    mode = str(getattr(cfg.exp, "run_infer", "single")).lower()
    if mode == "multi":
        _run_multi_mode(cfg, enable_mireval=enable_mireval)
    else:
        _run_single_mode(cfg, enable_mireval=enable_mireval)


if __name__ == "__main__":
    main()
