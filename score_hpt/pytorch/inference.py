import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from hydra import compose, initialize
from tqdm import tqdm

# Ensure project root is importable when running `python pytorch/*.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pytorch_utils import forward, forward_velo, move_data_to_device
from velo_model import build_adapter
from velo_model.pretrained_utils import get_frontend_pretrained_value
from velo_model.model_registry import build_model
from score_inf import build_score_inf, ScoreInfWrapper
from feature_extractor import PsychoFeatureExtractor
from utilities import (
    OnsetsFramesPostProcessor,
    RegressionPostProcessor,
    TargetProcessor,
    check_duration_alignment,
    collect_audio_midi_pairs,
    create_folder,
    get_filename,
    get_model_name,
    int16_to_float32,
    iteration_label_from_path,
    load_mono_audio,
    note_events_to_velocity_roll,
    original_score_events,
    pick_velocity_from_roll,
    prepare_aux_rolls,
    read_midi,
    resolve_audio_midi_pair,
    resolve_hdf5_dir,
    select_condition_roll,
    traverse_folder,
    write_events_to_midi,
)


VELO_SCORE_FIELDS = [
    "test_h5",
    "frame_max_error",
    "frame_max_std",
    "f1_score",
    "precision",
    "recall",
    "frame_mask_f1",
    "frame_mask_precision",
    "frame_mask_recall",
    "onset_masked_error",
    "onset_masked_std",
]


AUDIO_SCORE_FIELDS = [
    "test_h5",
    "bark_band_cosine",
    "bark_overall_cosine",
    "ntot_cosine",
]

def _normalized_cond_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = str(name)
    if s == "null":
        return None
    return s


def _resolve_score_inf_build(cfg) -> Tuple[str, Dict, List[str]]:
    method = str(cfg.score_informed.method)
    score_params = dict(cfg.score_informed.params)
    cond_selected = [
        key
        for key in [_normalized_cond_name(cfg.model.input2), _normalized_cond_name(cfg.model.input3)]
        if key
    ]

    if method == "note_editor":
        note_extra = _normalized_cond_name(cfg.model.input3)
        use_cond_feats = [note_extra] if note_extra else []
        score_params["use_cond_feats"] = use_cond_feats
        return method, score_params, ["onset"] + use_cond_feats

    if method in ("bilstm", "scrr", "dual_gated"):
        score_params["cond_keys"] = cond_selected
        return method, score_params, cond_selected

    return method, score_params, []


class TranscriptionBase:
    def __init__(self, checkpoint_path, cfg):
        self.cfg = cfg
        self.device = (
            torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")
        )
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        self.segment_frames = int(round(cfg.feature.frames_per_second * cfg.feature.segment_seconds)) + 1

        self.model_type = str(cfg.model.type)
        self.is_score_wrapper = False
        self.model = None
        self.score_cond_keys: List[str] = []
        self.input2_key = _normalized_cond_name(cfg.model.input2)
        self.input3_key = _normalized_cond_name(cfg.model.input3)
        if os.path.getsize(checkpoint_path) == 0:
            raise ValueError(f"Checkpoint file for inference is empty: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

        def _strip_prefix(state, prefix):
            keys = list(state.keys())
            if keys and all(k.startswith(prefix) for k in keys):
                return {k[len(prefix) :]: v for k, v in state.items()}
            return state

        state_dict = _strip_prefix(state_dict, "module.")

        self.is_score_wrapper = any(k.startswith("base_adapter.") for k in state_dict.keys())
        if self.is_score_wrapper:
            model_cfg = {"type": self.model_type, "params": cfg.model.params}
            adapter = build_adapter(model_cfg, model=None, cfg=cfg)
            score_method, score_params, score_cond_keys = _resolve_score_inf_build(cfg)
            post = build_score_inf(score_method, score_params)
            self.model = ScoreInfWrapper(adapter, post, freeze_base=False)
            self.score_cond_keys = score_cond_keys
        else:
            self.model = build_model(cfg)

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

    def enframe(self, x, is_audio=True):
        segment_length = self.segment_samples if is_audio else self.segment_frames
        assert x.shape[1 if is_audio else 0] % segment_length == 0

        batch = [
            x[:, pointer : pointer + segment_length] if is_audio else x[pointer : pointer + segment_length, :]
            for pointer in range(0, (x.shape[1] if is_audio else x.shape[0]) - segment_length + 1, segment_length // 2)
        ]
        return np.concatenate(batch, axis=0) if is_audio else np.stack(batch, axis=0)

    def deframe(self, x):
        if x.shape[0] == 1:
            return x[0]

        x = x[:, :-1, :]
        segment_samples = x.shape[1]
        quarter = max(1, segment_samples // 4)
        three_quarters = segment_samples - quarter
        if three_quarters <= quarter:
            return np.concatenate(x, axis=0)

        y = [
            x[0, :three_quarters],
            *[x[i, quarter:three_quarters] for i in range(1, x.shape[0] - 1)],
            x[-1, quarter:],
        ]
        return np.concatenate(y, axis=0)

    def _forward_score_wrapper(
        self,
        audio_segments: np.ndarray,
        input2_segments: Optional[np.ndarray],
        input3_segments: Optional[np.ndarray],
        batch_size: int = 1,
    ) -> Dict[str, np.ndarray]:
        output_dict = {"velocity_output": []}
        source_rolls: Dict[str, np.ndarray] = {}
        if self.input2_key and input2_segments is not None:
            source_rolls[self.input2_key] = input2_segments
        if self.input3_key and input3_segments is not None:
            source_rolls[self.input3_key] = input3_segments

        pointer = 0
        while pointer < len(audio_segments):
            batch_audio = move_data_to_device(audio_segments[pointer : pointer + batch_size], self.device)
            cond_batch = {
                key: move_data_to_device(source_rolls[key][pointer : pointer + batch_size], self.device)
                for key in self.score_cond_keys
            }
            pointer += batch_size

            with torch.no_grad():
                self.model.eval()
                batch_output_dict = self.model(batch_audio, cond_batch)

            output_dict["velocity_output"].append(batch_output_dict["vel_corr"].data.cpu().numpy())

        output_dict["velocity_output"] = np.concatenate(output_dict["velocity_output"], axis=0)
        return output_dict


class VeloTranscription(TranscriptionBase):
    def transcribe(self, audio, input2=None, input3=None, midi_path=None):
        audio = audio[None, :]
        audio_len = audio.shape[1]
        audio_segments_num = int(np.ceil(audio_len / self.segment_samples))
        pad_audio_len = audio_segments_num * self.segment_samples - audio_len
        pad_audio = np.pad(audio, ((0, 0), (0, pad_audio_len)), mode="constant")
        audio_segments = self.enframe(pad_audio, is_audio=True)

        def process_extra_input(aux_input, audio_segments_num):
            if aux_input is None:
                return None
            aux_len = aux_input.shape[0]
            aux_segments_num = int(np.ceil(aux_len / self.segment_frames))
            if aux_segments_num != audio_segments_num:
                aux_segments_num = audio_segments_num
            pad_aux_len = aux_segments_num * self.segment_frames - aux_len
            pad_aux = np.pad(aux_input, ((0, pad_aux_len), (0, 0)), mode="constant")
            aux_segments = self.enframe(pad_aux, is_audio=False)
            return aux_segments

        input2_segments = process_extra_input(input2, audio_segments_num)
        input3_segments = process_extra_input(input3, audio_segments_num)

        if self.is_score_wrapper:
            output_dict = self._forward_score_wrapper(
                audio_segments,
                input2_segments,
                input3_segments,
                batch_size=1,
            )
        else:
            output_dict = forward_velo(self.model, audio_segments, input2_segments, input3_segments, batch_size=1)
        output_dict["velocity_output"] = self.deframe(output_dict["velocity_output"])[0:audio_len]
        return {
            "output_dict": output_dict,
        }


class PianoTranscription(TranscriptionBase):
    def transcribe(self, audio, midi_path):
        audio = audio[None, :]
        audio_len = audio.shape[1]
        segments_num = int(np.ceil(audio_len / self.segment_samples))
        pad_len = segments_num * self.segment_samples - audio_len
        pad_audio = np.pad(audio, ((0, 0), (0, pad_len)), mode="constant")
        segments = self.enframe(pad_audio, is_audio=True)

        output_dict = forward(self.model, segments, batch_size=1)
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0:audio_len]

        if self.cfg.post.post_processor_type == "regression":
            post_processor = RegressionPostProcessor(self.cfg)
        elif self.cfg.post.post_processor_type == "onsets_frames":
            post_processor = OnsetsFramesPostProcessor(self.cfg)
        else:
            raise ValueError(f"Unknown post processor: {self.cfg.post.post_processor_type}")

        est_note_events, est_pedal_events = post_processor.output_dict_to_midi_events(output_dict)

        if midi_path:
            write_events_to_midi(0, est_note_events, est_pedal_events, midi_path)
            print(f"Write out to {midi_path}")

        return {
            "output_dict": output_dict,
            "est_note_events": est_note_events,
            "est_pedal_events": est_pedal_events,
        }


def _latest_checkpoint_under(workspace: str, model_name: str) -> Path:
    ckpt_dir = Path(workspace) / "checkpoints" / model_name
    checkpoints = []
    for path in ckpt_dir.glob("*_iterations.pth"):
        stem = path.stem.replace("_iterations", "")
        if stem.isdigit():
            checkpoints.append((int(stem), path))
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]
    return ckpt_dir / "latest_iterations.pth"


def _resolve_checkpoint(cfg, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    pretrained_path = get_frontend_pretrained_value(getattr(cfg, "model", None))
    if pretrained_path:
        return Path(pretrained_path).expanduser().resolve()

    return _latest_checkpoint_under(cfg.exp.workspace, get_model_name(cfg))


def resolve_checkpoint(cfg, explicit_path: Optional[str] = None) -> Path:
    return _resolve_checkpoint(cfg, explicit_path)


def _make_roll_adapter(cfg, velocity_method: str):
    def adapter(output_dict, target_dict, context):
        midi_events_time = context["midi_events_time"]
        midi_events = context["midi_events"]
        duration = context["duration"]

        note_events, _ = original_score_events(cfg, midi_events_time, midi_events, duration)
        pick_velocity_from_roll(note_events, output_dict["velocity_output"], cfg, strategy=velocity_method)
        pred_roll = note_events_to_velocity_roll(
            note_events=note_events,
            frames_num=target_dict["velocity_roll"].shape[0],
            num_keys=target_dict["velocity_roll"].shape[1],
            frames_per_second=cfg.feature.frames_per_second,
            begin_note=cfg.feature.begin_note,
            velocity_scale=cfg.feature.velocity_scale,
        )
        return pred_roll

    return adapter


def _predict_velocity_from_alignment(
    cfg,
    transcriber: "VeloTranscription",
    audio: np.ndarray,
    midi_events_time: np.ndarray,
    midi_events: List[str],
    duration: float,
    velocity_method: str,
):
    target_dict, _, _ = prepare_aux_rolls(cfg, midi_events_time, midi_events, duration)
    input2 = select_condition_roll(target_dict, cfg.model.input2)
    input3 = select_condition_roll(target_dict, cfg.model.input3)

    transcribed = transcriber.transcribe(audio, input2=input2, input3=input3)
    velocity_roll = transcribed["output_dict"]["velocity_output"]

    note_events, pedal_events = original_score_events(cfg, midi_events_time, midi_events, duration)
    gt_note_events = [dict(event) for event in note_events]
    gt_pedal_events = [dict(event) for event in pedal_events]
    pick_velocity_from_roll(note_events, velocity_roll, cfg, strategy=velocity_method)

    pred_roll = note_events_to_velocity_roll(
        note_events=note_events,
        frames_num=target_dict["velocity_roll"].shape[0],
        num_keys=target_dict["velocity_roll"].shape[1],
        frames_per_second=cfg.feature.frames_per_second,
        begin_note=cfg.feature.begin_note,
        velocity_scale=cfg.feature.velocity_scale,
    )
    return note_events, pedal_events, target_dict, pred_roll, gt_note_events, gt_pedal_events


def _transcribe_pair_to_midi(
    transcriber: VeloTranscription,
    cfg,
    audio_path: Path,
    midi_path: Path,
    output_path: Path,
    midi_format: str,
    velocity_method: str,
) -> int:
    audio = load_mono_audio(audio_path, sample_rate=cfg.feature.sample_rate)
    midi_dict = read_midi(str(midi_path), dataset=midi_format)
    midi_events_time = np.asarray(midi_dict["midi_event_time"], dtype=float)
    midi_events = [
        msg.decode() if isinstance(msg, (bytes, bytearray)) else str(msg)
        for msg in midi_dict["midi_event"]
    ]

    note_events, pedal_events, _, _, _, _ = _predict_velocity_from_alignment(
        cfg,
        transcriber,
        audio,
        midi_events_time,
        midi_events,
        duration=float(len(audio) / cfg.feature.sample_rate),
        velocity_method=velocity_method,
    )

    create_folder(str(output_path.parent))
    write_events_to_midi(0.0, note_events, pedal_events, str(output_path))
    return len(note_events)


def _process_dataset_file(
    cfg,
    transcriber: "VeloTranscription",
    hdf5_path: str,
    velocity_method: str,
    midi_dir: Optional[Path] = None,
    gt_midi_dir: Optional[Path] = None,
):
    with h5py.File(hdf5_path, "r") as hf:
        if hf.attrs["split"].decode() != "test":
            return None
        audio = int16_to_float32(hf["waveform"][:])
        midi_events = [e.decode() for e in hf["midi_event"][:]]
        midi_events_time = hf["midi_event_time"][:]

    duration = len(audio) / cfg.feature.sample_rate
    (
        note_events,
        pedal_events,
        target_dict,
        pred_roll,
        gt_note_events,
        gt_pedal_events,
    ) = _predict_velocity_from_alignment(
        cfg,
        transcriber,
        audio,
        midi_events_time,
        midi_events,
        duration,
        velocity_method,
    )

    stem = Path(hdf5_path).stem
    pred_midi_path = None
    gt_midi_path = None
    if midi_dir:
        pred_midi_path = midi_dir / f"{stem}_pred.mid"
        create_folder(str(pred_midi_path.parent))
        write_events_to_midi(0.0, note_events, pedal_events, str(pred_midi_path))

    if gt_midi_dir:
        gt_midi_path = gt_midi_dir / f"{stem}_gt.mid"
        create_folder(str(gt_midi_path.parent))
        write_events_to_midi(0.0, gt_note_events, gt_pedal_events, str(gt_midi_path))

    align_len = min(pred_roll.shape[0], target_dict["velocity_roll"].shape[0])
    output_entry = {
        "velocity_output": pred_roll[:align_len],
    }
    target_entry = {
        "velocity_roll": target_dict["velocity_roll"][:align_len],
        "frame_roll": target_dict["frame_roll"][:align_len],
        "onset_roll": target_dict["onset_roll"][:align_len],
        "pedal_frame_roll": target_dict["pedal_frame_roll"][:align_len],
    }
    return Path(hdf5_path).name, output_entry, target_entry, pred_midi_path, gt_midi_path


def _export_dataset_midis(cfg, checkpoint_path: Path, iteration_label: str, velocity_method: str) -> Path:
    transcriber = VeloTranscription(checkpoint_path=str(checkpoint_path), cfg=cfg)
    hdf5s_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
    _, hdf5_paths = traverse_folder(hdf5s_dir)

    output_dir = (
        Path(cfg.exp.workspace)
        / "midi_outputs"
        / cfg.dataset.test_set
        / get_model_name(cfg)
        / f"{iteration_label}_iterations"
    )
    create_folder(str(output_dir))

    progress = tqdm(sorted(hdf5_paths), desc=f"Proc {iteration_label} Ckpt", unit="file", ncols=80)
    for hdf5_path in progress:
        _process_dataset_file(cfg, transcriber, hdf5_path, velocity_method, midi_dir=output_dir)
    return output_dir


def _render_midi_to_audio(midi_path: Path, soundfont_path: str, sample_rate: int) -> np.ndarray:
    try:
        import fluidsynth
    except ImportError as exc:
        raise ImportError(
            "pyfluidsynth is required for audio-based evaluation. Install it with `pip install pyfluidsynth`."
        ) from exc

    fs = fluidsynth.Synth(samplerate=sample_rate)
    fs.start()
    sfid = fs.sfload(soundfont_path)
    fs.program_select(0, sfid, 0, 0)

    player = fluidsynth.Player(fs)
    player.add(str(midi_path))
    player.play()

    audio = []
    block = 2048
    while player.get_status() == fluidsynth.FLUID_PLAYER_PLAYING:
        samples = fs.get_samples(block)
        audio.extend(samples)
    fs.delete()

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32)
    audio = audio.reshape(-1, 2).mean(axis=1)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = vec1.reshape(-1)
    vec2 = vec2.reshape(-1)
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-9
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


def _compute_audio_metrics(
    pred_midi: Path,
    gt_midi: Path,
    soundfont_path: str,
    sample_rate: int,
    frames_per_second: int,
    fft_size: int,
    bark_extractor: PsychoFeatureExtractor,
    ntot_extractor: PsychoFeatureExtractor,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not pred_midi or not gt_midi:
        return None, None, None

    pred_audio = _render_midi_to_audio(pred_midi, soundfont_path, sample_rate)
    gt_audio = _render_midi_to_audio(gt_midi, soundfont_path, sample_rate)
    min_len = min(pred_audio.size, gt_audio.size)
    if min_len == 0:
        return None, None, None
    pred_audio = pred_audio[:min_len]
    gt_audio = gt_audio[:min_len]

    with torch.no_grad():
        pred_tensor = torch.from_numpy(pred_audio).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_audio).unsqueeze(0)
        pred_bark = bark_extractor(pred_tensor).squeeze(0).cpu().numpy()
        gt_bark = bark_extractor(gt_tensor).squeeze(0).cpu().numpy()
        bands = min(pred_bark.shape[0], gt_bark.shape[0])
        pred_bark = pred_bark[:bands]
        gt_bark = gt_bark[:bands]

        band_cos = []
        for band in range(bands):
            band_cos.append(_cosine_similarity(pred_bark[band], gt_bark[band]))
        bark_band_cosine = float(np.mean(band_cos)) if band_cos else None
        bark_overall_cosine = _cosine_similarity(pred_bark, gt_bark)

        pred_ntot = ntot_extractor(pred_tensor).squeeze(0).cpu().numpy()
        gt_ntot = ntot_extractor(gt_tensor).squeeze(0).cpu().numpy()
        length = min(pred_ntot.size, gt_ntot.size)
        ntot_cosine = _cosine_similarity(pred_ntot[:length], gt_ntot[:length])

    return bark_band_cosine, bark_overall_cosine, ntot_cosine


def run_single_pair_mode(cfg, args) -> None:
    if not args.input_path:
        raise ValueError("Single mode requires --input-path pointing to an audio or MIDI file.")
    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint_path)
    transcriber = VeloTranscription(checkpoint_path=str(checkpoint_path), cfg=cfg)

    audio_path, midi_path = resolve_audio_midi_pair(Path(args.input_path))
    output_path = Path(args.output_path) if args.output_path else audio_path.parent / f"{audio_path.stem}_pred.mid"

    note_count = _transcribe_pair_to_midi(
        transcriber=transcriber,
        cfg=cfg,
        audio_path=audio_path,
        midi_path=midi_path,
        output_path=output_path,
        midi_format=args.midi_format,
        velocity_method=args.velocity_method,
    )
    print(f"[single] Wrote {note_count} notes to {output_path}")


def run_folder_mode(cfg, args) -> None:
    if not args.input_path:
        raise ValueError("Folder mode requires --input-path pointing to a directory.")
    folder = Path(args.input_path)
    pairs = collect_audio_midi_pairs(folder)
    output_dir = Path(args.output_dir) if args.output_dir else folder / "pred_midis"
    create_folder(str(output_dir))

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint_path)
    transcriber = VeloTranscription(checkpoint_path=str(checkpoint_path), cfg=cfg)

    for audio_path, midi_path in pairs:
        out_path = output_dir / f"{audio_path.stem}.mid"
        note_count = _transcribe_pair_to_midi(
            transcriber=transcriber,
            cfg=cfg,
            audio_path=audio_path,
            midi_path=midi_path,
            output_path=out_path,
            midi_format=args.midi_format,
            velocity_method=args.velocity_method,
        )
        print(f"[folder] {audio_path.name} -> {out_path.name} ({note_count} notes)")


def run_dataset_velo_score_mode(cfg, args) -> None:
    from calculate_scores import (
        detailed_f1_metrics_from_list,
        frame_max_metrics__from_list,
        onset_pick_metrics_from_list,
    )

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint_path)
    iteration_label = iteration_label_from_path(checkpoint_path)
    transcriber = VeloTranscription(checkpoint_path=str(checkpoint_path), cfg=cfg)

    results_dir = (
        Path(cfg.exp.workspace)
        / "dataset_velo_score"
        / cfg.dataset.test_set
        / get_model_name(cfg)
        / f"{iteration_label}_iterations"
    )
    midi_dir = results_dir / "midis"
    error_dir = results_dir / "error_dict"
    create_folder(str(midi_dir))
    create_folder(str(error_dir))

    csv_path = results_dir / f"{cfg.dataset.test_set}_dataset_velo_score.csv"
    hdf5s_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
    _, hdf5_paths = traverse_folder(hdf5s_dir)

    device = torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")
    print("=" * 80)
    print("Inference Mode : DATASET_VELO_SCORE (single checkpoint)")
    print(f"Model Name     : {get_model_name(cfg)}")
    print(f"Test Set       : {cfg.dataset.test_set}")
    print(f"Using Device   : {device}")
    print(
        f"Feature Config : {cfg.feature.audio_feature} | sr={cfg.feature.sample_rate} "
        f"| fps={cfg.feature.frames_per_second} | seg={cfg.feature.segment_seconds}s"
    )
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"MIDI Output    : {midi_dir}")
    print(f"Results Dir    : {results_dir}")
    print("=" * 80)

    aggregated = {field: [] for field in VELO_SCORE_FIELDS if field != "test_h5"}
    processed = 0

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(VELO_SCORE_FIELDS)

        progress = tqdm(sorted(hdf5_paths), desc="Dataset Velo Score", unit="file", ncols=80)
        for hdf5_path in progress:
            result = _process_dataset_file(
                cfg,
                transcriber,
                hdf5_path,
                args.velocity_method,
                midi_dir=midi_dir,
            )
            if not result:
                continue

            audio_name, output_entry, target_entry, _, _ = result
            output_list = [output_entry]
            target_list = [target_entry]

            frame_max_error, frame_max_std = frame_max_metrics__from_list(output_list, target_list)
            (
                error_profile,
                f1,
                precision,
                recall,
                frame_mask_f1,
                frame_mask_precision,
                frame_mask_recall,
            ) = detailed_f1_metrics_from_list(output_list, target_list)
            onset_masked_error, onset_masked_std = onset_pick_metrics_from_list(output_list, target_list)

            error_dict_path = error_dir / f"{Path(audio_name).stem}_dataset_score.npy"
            np.save(error_dict_path, np.array(error_profile, dtype=object), allow_pickle=True)

            row = [
                audio_name,
                frame_max_error,
                frame_max_std,
                f1,
                precision,
                recall,
                frame_mask_f1,
                frame_mask_precision,
                frame_mask_recall,
                onset_masked_error,
                onset_masked_std,
            ]
            writer.writerow(row)

            aggregated["frame_max_error"].append(frame_max_error)
            aggregated["frame_max_std"].append(frame_max_std)
            aggregated["f1_score"].append(f1)
            aggregated["precision"].append(precision)
            aggregated["recall"].append(recall)
            aggregated["frame_mask_f1"].append(frame_mask_f1)
            aggregated["frame_mask_precision"].append(frame_mask_precision)
            aggregated["frame_mask_recall"].append(frame_mask_recall)
            aggregated["onset_masked_error"].append(onset_masked_error)
            aggregated["onset_masked_std"].append(onset_masked_std)

            processed += 1
            avg_frame_err = float(np.mean(aggregated["frame_max_error"]))
            progress.set_postfix({"frame_err": f"{avg_frame_err:.2f}"}, refresh=False)

    if not processed:
        print("No test files processed.")
        return

    print("\n===== Dataset Velocity Score Averages =====")
    for key, values in aggregated.items():
        mean_value = float(np.mean(values))
        print(f"{key}: {mean_value:.4f}")


def run_dataset_audio_score_mode(cfg, args) -> None:
    if not args.soundfont_path:
        raise ValueError("dataset_audio_score mode requires --soundfont-path for rendering.")

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint_path)
    iteration_label = iteration_label_from_path(checkpoint_path)
    transcriber = VeloTranscription(checkpoint_path=str(checkpoint_path), cfg=cfg)

    results_dir = (
        Path(cfg.exp.workspace)
        / "dataset_audio_score"
        / cfg.dataset.test_set
        / get_model_name(cfg)
        / f"{iteration_label}_iterations"
    )
    pred_midi_dir = results_dir / "midis"
    gt_midi_dir = results_dir / "gt_midis"
    create_folder(str(pred_midi_dir))
    create_folder(str(gt_midi_dir))

    csv_path = results_dir / f"{cfg.dataset.test_set}_dataset_audio_score.csv"
    hdf5s_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
    _, hdf5_paths = traverse_folder(hdf5s_dir)

    device = torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")
    print("=" * 80)
    print("Inference Mode : DATASET_AUDIO_SCORE (single checkpoint)")
    print(f"Model Name     : {get_model_name(cfg)}")
    print(f"Test Set       : {cfg.dataset.test_set}")
    print(f"Using Device   : {device}")
    print(
        f"Feature Config : {cfg.feature.audio_feature} | sr={cfg.feature.sample_rate} "
        f"| fps={cfg.feature.frames_per_second} | seg={cfg.feature.segment_seconds}s"
    )
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"MIDI Output    : {pred_midi_dir}")
    print(f"Results Dir    : {results_dir}")
    print("=" * 80)

    aggregated = {field: [] for field in AUDIO_SCORE_FIELDS if field != "test_h5"}
    processed = 0

    bark_extractor = PsychoFeatureExtractor(
        sample_rate=args.audio_eval_sample_rate,
        fft_size=args.audio_eval_fft_size,
        frames_per_second=args.audio_eval_frames_per_second,
        return_mode="sone",
    ).to("cpu")
    ntot_extractor = PsychoFeatureExtractor(
        sample_rate=args.audio_eval_sample_rate,
        fft_size=args.audio_eval_fft_size,
        frames_per_second=args.audio_eval_frames_per_second,
        return_mode="ntot",
    ).to("cpu")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(AUDIO_SCORE_FIELDS)

        progress = tqdm(sorted(hdf5_paths), desc="Dataset Audio Score", unit="file", ncols=80)
        for hdf5_path in progress:
            result = _process_dataset_file(
                cfg,
                transcriber,
                hdf5_path,
                args.velocity_method,
                midi_dir=pred_midi_dir,
                gt_midi_dir=gt_midi_dir,
            )
            if not result:
                continue
            audio_name, _, _, pred_midi_path, gt_midi_path = result

            bark_band_cos, bark_overall_cos, ntot_cos = _compute_audio_metrics(
                pred_midi_path,
                gt_midi_path,
                args.soundfont_path,
                args.audio_eval_sample_rate,
                args.audio_eval_frames_per_second,
                args.audio_eval_fft_size,
                bark_extractor,
                ntot_extractor,
            )

            row = [
                audio_name,
                bark_band_cos,
                bark_overall_cos,
                ntot_cos,
            ]
            writer.writerow(row)

            if bark_band_cos is not None:
                aggregated["bark_band_cosine"].append(bark_band_cos)
            if bark_overall_cos is not None:
                aggregated["bark_overall_cosine"].append(bark_overall_cos)
            if ntot_cos is not None:
                aggregated["ntot_cosine"].append(ntot_cos)

            processed += 1

    if not processed:
        print("No test files processed.")
        return

    print("\n===== Dataset Audio Score Averages =====")
    for key, values in aggregated.items():
        if not values:
            print(f"{key}: n/a")
            continue
        mean_value = float(np.mean(values))
        print(f"{key}: {mean_value:.4f}")


def run_dataset_mode(cfg, args) -> None:
    if cfg.exp.run_infer and cfg.exp.run_infer.lower() != "single":
        tqdm.write("[warn] cfg.exp.run_infer != 'single'; forcing single-checkpoint inference.")
    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint_path)
    iteration_label = iteration_label_from_path(checkpoint_path)

    device = torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")
    print("=" * 80)
    print("Inference Mode : DATASET (single checkpoint)")
    print(f"Model Name     : {get_model_name(cfg)}")
    print(f"Test Set       : {cfg.dataset.test_set}")
    print(f"Using Device   : {device}")
    print(
        f"Feature Config : {cfg.feature.audio_feature} | sr={cfg.feature.sample_rate} "
        f"| fps={cfg.feature.frames_per_second} | seg={cfg.feature.segment_seconds}s"
    )
    print(f"Checkpoint     : {checkpoint_path}")
    print("=" * 80)

    t1 = time.time()
    midi_dir = _export_dataset_midis(cfg, checkpoint_path, iteration_label, args.velocity_method)
    print(f"\n[Done] Dataset inference finished in {time.time() - t1:.2f} sec")
    print(f"[info] MIDI outputs saved to {midi_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified inference utilities.")
    parser.add_argument(
        "--mode",
        choices=["single", "folder", "dataset", "dataset_velo_score", "dataset_audio_score"],
        default="dataset",
        help=(
            "single: run on one audio+MIDI pair; folder: iterate over a folder of pairs; "
            "dataset: run dataset inference (single checkpoint); "
            "dataset_velo_score: run inference + Kim-style score; "
            "dataset_audio_score: render audio and compute Bark loudness similarity."
        ),
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Audio/MIDI file path for single mode or folder for folder mode.",
    )
    parser.add_argument("--output-path", default=None, help="Output MIDI for single mode.")
    parser.add_argument("--output-dir", default=None, help="Output directory for folder mode.")
    parser.add_argument(
        "--velocity-method",
        default="onset_only",
        choices=["max_frame", "onset_only"],
        help="Strategy when converting velocity rolls to per-note values. Defaults to onset_only (Score-HPT / VeloEst).",
    )
    parser.add_argument(
        "--midi-format",
        default="maestro",
        choices=["maestro", "hpt", "smd", "maps", "francoisleduc", "gaps"],
        help="Layout hint for parsing MIDI files when reading standalone pairs.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Explicit checkpoint path. Otherwise resolved from model.frontend_pretrained or latest checkpoint under workspace.",
    )
    parser.add_argument("--config-path", default="./config", help="Hydra config path.")
    parser.add_argument("--config-name", default="config", help="Hydra config name.")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Optional Hydra overrides, e.g. model.frontend_pretrained=/path/to/100000_iterations.pth model.type=hppnet",
    )
    parser.add_argument(
        "--soundfont-path",
        default=None,
        help="Path to an SF2 soundfont used for audio rendering (required for dataset_score).",
    )
    parser.add_argument(
        "--audio-eval-sample-rate",
        type=int,
        default=44100,
        help="Sample rate used when rendering MIDI for audio evaluation.",
    )
    parser.add_argument(
        "--audio-eval-frames-per-second",
        type=int,
        default=86,
        help="Frames per second for Bark loudness evaluation.",
    )
    parser.add_argument(
        "--audio-eval-fft-size",
        type=int,
        default=1024,
        help="FFT size for Bark loudness evaluation.",
    )
    args, remaining = parser.parse_known_args()
    hydra_overrides = list(args.overrides) + remaining

    with initialize(config_path=args.config_path, job_name="infer", version_base=None):
        cfg = compose(config_name=args.config_name, overrides=hydra_overrides)

    mode_handlers = {
        "dataset": run_dataset_mode,
        "dataset_velo_score": run_dataset_score_mode,
        "dataset_audio_score": run_dataset_audio_score_mode,
        "single": run_single_pair_mode,
        "folder": run_folder_mode,
    }
    handler = mode_handlers.get(args.mode)
    if not handler:
        raise ValueError(f"Unknown mode: {args.mode}")
    handler(cfg, args)
