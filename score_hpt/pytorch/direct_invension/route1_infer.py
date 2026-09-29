from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from direct_invension.common import (
    dump_json,
    ensure_repo_imports,
    load_json,
    repo_root,
    require_path as _require_path,
    replace_note_velocities,
    SortedMidiNote,
    extract_sorted_notes,
    resolve_dataset_split,
    write_flat_velocity_copy,
)


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class VelocityDistributionMapping:
    midi_values: np.ndarray
    counts: np.ndarray
    cdf: np.ndarray
    dataset_name: str
    observed_min: int
    observed_max: int
    boundaries_01: Tuple[float, float]
    source_json: Optional[Path] = None

    @classmethod
    def from_stats_json(cls, stats_json: str | Path) -> "VelocityDistributionMapping":
        payload = load_json(stats_json)
        if "velocities_01" not in payload:
            raise KeyError(f"'velocities_01' not found in stats JSON: {stats_json}")

        velocity_block = payload["velocities_01"]
        values_01 = np.asarray(velocity_block.get("values", []), dtype=np.float64)
        counts = np.asarray(velocity_block.get("counts", []), dtype=np.float64)
        if values_01.size == 0 or counts.size == 0:
            raise ValueError(f"Velocity histogram is empty in: {stats_json}")
        if values_01.shape != counts.shape:
            raise ValueError("Velocity histogram values/counts shape mismatch.")

        midi_values = np.clip(np.rint(values_01 * 127.0), 1.0, 127.0).astype(np.int64)
        total = float(np.sum(counts))
        if total <= 0:
            raise ValueError("Velocity histogram counts sum to zero.")
        cdf = np.cumsum(counts) / total

        boundaries = payload.get("velocity_boundaries_01", [0.33, 0.66])
        if len(boundaries) != 2:
            boundaries = [0.33, 0.66]

        observed_range = payload.get("velocity_range_observed") or [int(midi_values.min()), int(midi_values.max())]
        if len(observed_range) != 2:
            observed_range = [int(midi_values.min()), int(midi_values.max())]

        return cls(
            midi_values=midi_values,
            counts=counts,
            cdf=cdf,
            dataset_name=str(payload.get("dataset", Path(stats_json).stem)),
            observed_min=int(observed_range[0]),
            observed_max=int(observed_range[1]),
            boundaries_01=(float(boundaries[0]), float(boundaries[1])),
            source_json=Path(stats_json),
        )

    def map_percentiles(self, percentiles: ArrayLike) -> np.ndarray:
        p = np.asarray(percentiles, dtype=np.float64)
        clipped = np.clip(p, 0.0, 1.0)
        idx = np.searchsorted(self.cdf, clipped, side="left")
        idx = np.clip(idx, 0, len(self.midi_values) - 1)
        return self.midi_values[idx].astype(np.int64)

    def summary(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "observed_min": int(self.observed_min),
            "observed_max": int(self.observed_max),
            "boundaries_01": [float(self.boundaries_01[0]), float(self.boundaries_01[1])],
            "num_support_values": int(self.midi_values.size),
            "source_json": str(self.source_json) if self.source_json else None,
        }


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def percentile_rank(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    valid = arr[mask]
    if valid.size == 1:
        out[mask] = 0.5
        return out
    ranks = _rankdata_average(valid)
    out[mask] = ranks / float(valid.size - 1)
    return out


def robust_unit_scale(values: ArrayLike, *, q_low: float = 0.02, q_high: float = 0.98) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    valid = arr[mask]
    lo = float(np.quantile(valid, q_low))
    hi = float(np.quantile(valid, q_high))
    if hi <= lo + 1e-12:
        out[mask] = 0.5
        return out
    out[mask] = np.clip((valid - lo) / (hi - lo), 0.0, 1.0)
    return out


def combine_note_loudness_to_percentiles(
    harmonic_energy: ArrayLike,
    onset_flux: ArrayLike,
    *,
    mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend",
    harmonic_weight: float = 1.0,
    flux_weight: float = 0.25,
) -> np.ndarray:
    e = np.asarray(harmonic_energy, dtype=np.float64).reshape(-1)
    x = np.asarray(onset_flux, dtype=np.float64).reshape(-1)
    if e.shape != x.shape:
        raise ValueError("harmonic_energy and onset_flux must have the same shape.")
    mode = str(mode).strip().lower()

    if mode == "rank_harmonic":
        return percentile_rank(e)
    if mode == "rank_flux":
        return percentile_rank(x)
    if mode == "robust_blend":
        e_score = robust_unit_scale(e)
        x_score = robust_unit_scale(x)
    elif mode == "rank_blend":
        e_score = percentile_rank(e)
        x_score = percentile_rank(x)
    else:
        raise ValueError(f"Unsupported combination mode: {mode}")

    w_sum = float(harmonic_weight + flux_weight)
    if w_sum <= 1e-12:
        raise ValueError("At least one weight must be positive.")

    out = np.full(e.shape, np.nan, dtype=np.float64)
    mask_e = np.isfinite(e_score)
    mask_x = np.isfinite(x_score)
    both = mask_e & mask_x
    only_e = mask_e & ~mask_x
    only_x = mask_x & ~mask_e

    if np.any(both):
        out[both] = (harmonic_weight * e_score[both] + flux_weight * x_score[both]) / w_sum
    if np.any(only_e):
        out[only_e] = e_score[only_e]
    if np.any(only_x):
        out[only_x] = x_score[only_x]
    return out


def map_note_loudness_to_midi_velocity(
    harmonic_energy: ArrayLike,
    onset_flux: ArrayLike,
    mapping: VelocityDistributionMapping,
    *,
    mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend",
    harmonic_weight: float = 1.0,
    flux_weight: float = 0.25,
    fallback_velocity: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    percentiles = combine_note_loudness_to_percentiles(
        harmonic_energy,
        onset_flux,
        mode=mode,
        harmonic_weight=harmonic_weight,
        flux_weight=flux_weight,
    )
    velocities = mapping.map_percentiles(np.nan_to_num(percentiles, nan=0.5))
    if fallback_velocity is not None:
        velocities[~np.isfinite(percentiles)] = int(np.clip(round(fallback_velocity), 1, 127))
    return velocities.astype(np.int64), percentiles


@dataclass(frozen=True)
class NoteLoudnessConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop: int = 256
    win_length: Optional[int] = None
    harmonic_count: int = 5
    band_bins: int = 1
    energy_window_s: float = 0.12
    onset_pre_s: float = 0.02
    onset_post_s: float = 0.08


@dataclass(frozen=True)
class NoteFeatureResult:
    times: np.ndarray
    freqs: np.ndarray
    magnitude: np.ndarray
    harmonic_energy: np.ndarray
    onset_flux: np.ndarray
    features: np.ndarray


class _NoteLikeProtocol:
    onset: float
    offset: float
    pitch: int


def load_audio_mono(audio_path: str, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(audio_path), sr=int(sample_rate), mono=True)
    if audio.ndim != 1:
        audio = np.mean(np.asarray(audio), axis=0)
    return np.asarray(audio, dtype=np.float32)


def _hz_from_midi(pitch: int) -> float:
    return 440.0 * (2.0 ** ((int(pitch) - 69) / 12.0))


def compute_magnitude_stft(
    audio: np.ndarray,
    *,
    sample_rate: int,
    cfg: NoteLoudnessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if audio.ndim != 1:
        raise ValueError("audio must be 1-D mono waveform")
    win_length = int(cfg.win_length) if cfg.win_length else int(cfg.n_fft)
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    window = torch.hann_window(win_length)
    spec = torch.stft(
        waveform,
        n_fft=int(cfg.n_fft),
        hop_length=int(cfg.hop),
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().cpu().numpy().astype(np.float64)
    times = np.arange(mag.shape[1], dtype=np.float64) * (float(cfg.hop) / float(sample_rate))
    freqs = np.linspace(0.0, sample_rate / 2.0, mag.shape[0], dtype=np.float64)
    return times, freqs, mag


def _note_window_indices(times: np.ndarray, t0: float, t1: float) -> Tuple[int, int]:
    i0 = int(np.searchsorted(times, max(float(t0), 0.0), side="left"))
    i1 = int(np.searchsorted(times, float(t1), side="right"))
    i0 = max(0, min(i0, len(times)))
    i1 = max(0, min(i1, len(times)))
    return i0, i1


def extract_note_loudness_features(
    audio: np.ndarray,
    notes: Sequence[_NoteLikeProtocol],
    *,
    sample_rate: int,
    cfg: Optional[NoteLoudnessConfig] = None,
) -> NoteFeatureResult:
    cfg = cfg or NoteLoudnessConfig(sample_rate=int(sample_rate))
    times, freqs, mag = compute_magnitude_stft(audio, sample_rate=sample_rate, cfg=cfg)
    mag_sq = mag ** 2

    if mag.shape[1] >= 2:
        diff = mag[:, 1:] - mag[:, :-1]
        flux_per_frame = np.maximum(diff, 0.0).sum(axis=0)
        flux_per_frame = np.concatenate([np.zeros(1, dtype=np.float64), flux_per_frame], axis=0)
    else:
        flux_per_frame = np.zeros(mag.shape[1], dtype=np.float64)

    freq_res = float(sample_rate) / float(cfg.n_fft)
    harmonic_energy = np.full(len(notes), np.nan, dtype=np.float64)
    onset_flux = np.full(len(notes), np.nan, dtype=np.float64)

    for idx, note in enumerate(notes):
        onset_s = float(note.onset)
        if not np.isfinite(onset_s):
            continue

        i0, i1 = _note_window_indices(times, onset_s, onset_s + float(cfg.energy_window_s))
        if i1 <= i0:
            continue

        f0 = _hz_from_midi(int(note.pitch))
        energy = 0.0
        for h in range(1, int(cfg.harmonic_count) + 1):
            fh = f0 * float(h)
            if fh >= freqs[-1]:
                break
            center_bin = int(round(fh / max(freq_res, 1e-12)))
            if center_bin < 0 or center_bin >= mag.shape[0]:
                continue
            lo = max(0, center_bin - int(cfg.band_bins))
            hi = min(mag.shape[0], center_bin + int(cfg.band_bins) + 1)
            energy += float(np.mean(mag_sq[lo:hi, i0:i1]))
        harmonic_energy[idx] = np.log1p(max(energy, 0.0))

        j0, j1 = _note_window_indices(
            times,
            onset_s - float(cfg.onset_pre_s),
            onset_s + float(cfg.onset_post_s),
        )
        if j1 > j0:
            onset_flux[idx] = np.log1p(float(np.mean(flux_per_frame[j0:j1])))

    features = np.stack([harmonic_energy, onset_flux], axis=1)
    return NoteFeatureResult(
        times=times,
        freqs=freqs,
        magnitude=mag,
        harmonic_energy=harmonic_energy,
        onset_flux=onset_flux,
        features=features,
    )


def extract_note_loudness_from_files(
    audio_path: str,
    notes: Sequence[_NoteLikeProtocol],
    *,
    sample_rate: int,
    cfg: Optional[NoteLoudnessConfig] = None,
) -> NoteFeatureResult:
    audio = load_audio_mono(str(audio_path), int(sample_rate))
    return extract_note_loudness_features(audio, notes, sample_rate=sample_rate, cfg=cfg)


@dataclass(frozen=True)
class Route1Config:
    dataset_stats_json: Optional[str] = None
    mapping_mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend"
    harmonic_weight: float = 1.0
    flux_weight: float = 0.25
    fallback_velocity: int = 64
    flat_velocity: int = 64
    note_sample_rate: int = 22050
    note_fft_size: int = 2048
    note_hop: int = 256
    harmonic_count: int = 5
    band_bins: int = 1
    energy_window_s: float = 0.12
    onset_pre_s: float = 0.02
    onset_post_s: float = 0.08

    def note_feature_config(self) -> NoteLoudnessConfig:
        return NoteLoudnessConfig(
            sample_rate=int(self.note_sample_rate),
            n_fft=int(self.note_fft_size),
            hop=int(self.note_hop),
            harmonic_count=int(self.harmonic_count),
            band_bins=int(self.band_bins),
            energy_window_s=float(self.energy_window_s),
            onset_pre_s=float(self.onset_pre_s),
            onset_post_s=float(self.onset_post_s),
        )


def _relative_key(dataset_dir: Path, midi_path: Path) -> str:
    try:
        return str(midi_path.relative_to(dataset_dir).with_suffix(""))
    except Exception:
        return midi_path.stem


def _route1_out_paths(
    out_dir: Path,
    dataset_dir: Optional[Path],
    midi_path: Path,
    *,
    flat_velocity: int,
) -> Tuple[Path, Path, Path]:
    if dataset_dir is not None:
        try:
            rel = midi_path.relative_to(dataset_dir)
        except Exception:
            rel = Path(midi_path.name)
    else:
        rel = Path(midi_path.name)
    pred_midi = (out_dir / "pred_midis" / rel).with_suffix(".direct.mid")
    flat_midi = (out_dir / "flat_midis" / rel).with_suffix(f".flat{int(flat_velocity)}.mid")
    feature_json = (out_dir / "note_features" / rel).with_suffix(".route1.json")
    return pred_midi, flat_midi, feature_json


def _feature_summary(
    harmonic_energy: np.ndarray,
    onset_flux: np.ndarray,
    percentiles: np.ndarray,
    velocities: np.ndarray,
) -> Dict[str, object]:
    def stats(values: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "num_notes": int(len(velocities)),
        "harmonic_energy": stats(harmonic_energy),
        "onset_flux": stats(onset_flux),
        "combined_percentile": stats(percentiles),
        "pred_velocity": {
            "mean": float(np.mean(velocities)) if len(velocities) else float("nan"),
            "std": float(np.std(velocities)) if len(velocities) else float("nan"),
            "min": int(np.min(velocities)) if len(velocities) else 0,
            "max": int(np.max(velocities)) if len(velocities) else 0,
        },
    }


def predict_direct_inversion_for_pair(
    *,
    audio_path: str | Path,
    midi_path: str | Path,
    output_midi_path: str | Path,
    config: Route1Config,
    dataset_mapping: Optional[VelocityDistributionMapping] = None,
    feature_json_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    midi_path = Path(midi_path)
    audio_path = Path(audio_path)
    output_midi_path = Path(output_midi_path)

    if dataset_mapping is None:
        if not config.dataset_stats_json:
            raise ValueError("Route1Config.dataset_stats_json is required when dataset_mapping is not provided.")
        dataset_mapping = VelocityDistributionMapping.from_stats_json(config.dataset_stats_json)

    notes = extract_sorted_notes(midi_path)
    note_features = extract_note_loudness_from_files(
        str(audio_path),
        notes,
        sample_rate=int(config.note_sample_rate),
        cfg=config.note_feature_config(),
    )

    velocities, percentiles = map_note_loudness_to_midi_velocity(
        note_features.harmonic_energy,
        note_features.onset_flux,
        dataset_mapping,
        mode=config.mapping_mode,
        harmonic_weight=float(config.harmonic_weight),
        flux_weight=float(config.flux_weight),
        fallback_velocity=int(config.fallback_velocity),
    )
    replace_note_velocities(midi_path, velocities, output_midi_path)

    payload = {
        "route": "route1_direct_inversion",
        "audio_path": str(audio_path),
        "midi_path": str(midi_path),
        "output_midi_path": str(output_midi_path),
        "mapping": dataset_mapping.summary(),
        "config": asdict(config),
        "summary": _feature_summary(
            note_features.harmonic_energy,
            note_features.onset_flux,
            percentiles,
            velocities,
        ),
    }

    if feature_json_path is not None:
        feature_json_path = Path(feature_json_path)
        payload["feature_json_path"] = str(dump_json(feature_json_path, payload))
    return payload


def _build_dataset_scan_helpers():
    ensure_repo_imports()
    from data_analysis.cli._dataset_utils import load_maestro_audio_map, resolve_real_audio, scan_midis  # type: ignore

    return scan_midis, load_maestro_audio_map, resolve_real_audio


def predict_route1_dataset(
    *,
    dataset_type: str,
    dataset_dir: str | Path,
    out_dir: str | Path,
    dataset_stats_json: str | Path,
    split: str = "test",
    maps_pianos: str = "both",
    flat_velocity: int = 64,
    mapping_mode: Literal["rank_blend", "rank_harmonic", "rank_flux", "robust_blend"] = "rank_blend",
    harmonic_weight: float = 1.0,
    flux_weight: float = 0.25,
    fallback_velocity: int = 64,
    note_sample_rate: int = 22050,
    note_fft_size: int = 2048,
    note_hop: int = 256,
    harmonic_count: int = 5,
    band_bins: int = 1,
    energy_window_s: float = 0.12,
    onset_pre_s: float = 0.02,
    onset_post_s: float = 0.08,
    skip_existing: bool = True,
    max_items: Optional[int] = None,
) -> Dict[str, object]:
    scan_midis, load_maestro_audio_map, resolve_real_audio = _build_dataset_scan_helpers()

    dataset_dir = Path(dataset_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = VelocityDistributionMapping.from_stats_json(dataset_stats_json)
    cfg = Route1Config(
        dataset_stats_json=str(dataset_stats_json),
        mapping_mode=mapping_mode,
        harmonic_weight=float(harmonic_weight),
        flux_weight=float(flux_weight),
        fallback_velocity=int(fallback_velocity),
        flat_velocity=int(flat_velocity),
        note_sample_rate=int(note_sample_rate),
        note_fft_size=int(note_fft_size),
        note_hop=int(note_hop),
        harmonic_count=int(harmonic_count),
        band_bins=int(band_bins),
        energy_window_s=float(energy_window_s),
        onset_pre_s=float(onset_pre_s),
        onset_post_s=float(onset_post_s),
    )

    scan_split = resolve_dataset_split(split)
    midi_files = scan_midis(dataset_type, dataset_dir, split=scan_split, maps_pianos=maps_pianos)
    if max_items is not None:
        midi_files = midi_files[: int(max_items)]
    maestro_audio_map = load_maestro_audio_map(dataset_type, dataset_dir, split=scan_split)

    direct_items: List[Dict[str, object]] = []
    flat_items: List[Dict[str, object]] = []
    file_summaries: List[Dict[str, object]] = []

    for midi_path in tqdm(
        midi_files,
        desc=f"Route1 infer [{dataset_type}]",
        unit="file",
        dynamic_ncols=True,
    ):
        midi_path = Path(midi_path).resolve()
        real_audio = resolve_real_audio(dataset_type, dataset_dir, midi_path, maestro_audio_map)
        key = _relative_key(dataset_dir, midi_path)
        pred_midi_path, flat_midi_path, feature_json_path = _route1_out_paths(
            out_dir,
            dataset_dir,
            midi_path,
            flat_velocity=int(flat_velocity),
        )

        if skip_existing and pred_midi_path.exists():
            pair_payload: Dict[str, object] = {
                "route": "route1_direct_inversion",
                "audio_path": str(real_audio),
                "midi_path": str(midi_path),
                "output_midi_path": str(pred_midi_path),
                "mapping": mapping.summary(),
                "config": asdict(cfg),
                "summary": {"num_notes": len(extract_sorted_notes(midi_path)), "skipped_existing": True},
                "feature_json_path": str(feature_json_path) if feature_json_path.exists() else None,
            }
        else:
            pair_payload = predict_direct_inversion_for_pair(
                audio_path=real_audio,
                midi_path=midi_path,
                output_midi_path=pred_midi_path,
                config=cfg,
                dataset_mapping=mapping,
                feature_json_path=feature_json_path,
            )

        if not (skip_existing and flat_midi_path.exists()):
            write_flat_velocity_copy(midi_path, flat_midi_path, flat_velocity=int(flat_velocity))

        direct_items.append(
            {
                "key": key,
                "label": "route1_direct_inversion",
                "gt_midi": str(midi_path),
                "pred_midi": str(pred_midi_path),
                "real_audio": str(real_audio),
                "feature_json": str(feature_json_path) if feature_json_path.exists() else pair_payload.get("feature_json_path"),
            }
        )
        flat_items.append(
            {
                "key": key,
                "label": f"flat_velocity_{int(flat_velocity)}",
                "gt_midi": str(midi_path),
                "pred_midi": str(flat_midi_path),
                "real_audio": str(real_audio),
            }
        )
        file_summaries.append(pair_payload)

    direct_manifest = {
        "format_version": 1,
        "label": "route1_direct_inversion",
        "dataset_type": str(dataset_type),
        "dataset_dir": str(dataset_dir),
        "split": scan_split,
        "requested_split": split,
        "maps_pianos": maps_pianos,
        "config": asdict(cfg),
        "items": direct_items,
    }
    flat_manifest = {
        "format_version": 1,
        "label": f"flat_velocity_{int(flat_velocity)}",
        "dataset_type": str(dataset_type),
        "dataset_dir": str(dataset_dir),
        "split": scan_split,
        "requested_split": split,
        "maps_pianos": maps_pianos,
        "config": asdict(cfg),
        "items": flat_items,
    }
    direct_manifest_path = dump_json(out_dir / "route1_direct_manifest.json", direct_manifest)
    flat_manifest_path = dump_json(out_dir / f"route1_flat{int(flat_velocity)}_manifest.json", flat_manifest)
    file_summary_path = dump_json(out_dir / "route1_direct_predictions.json", {"files": file_summaries})

    return {
        "dataset_type": dataset_type,
        "dataset_dir": str(dataset_dir),
        "num_items": len(direct_items),
        "out_dir": str(out_dir),
        "direct_manifest_path": str(direct_manifest_path),
        "flat_manifest_path": str(flat_manifest_path),
        "file_summary_path": str(file_summary_path),
    }

def resolve_route1_stats_json(cfg, dataset_type: str) -> Path:
    configured = str(getattr(cfg.route1, "stats_json", "") or "").strip()
    if configured:
        return _require_path(configured, field_name="route1.stats_json")

    default_map = {
        "smd": repo_root() / "data_analysis" / "stats" / "midi_sampler" / "SMD_sampler.json",
        "maestro": repo_root() / "data_analysis" / "stats" / "midi_sampler" / "MAESTRO_v3_sampler.json",
        "maps": repo_root() / "data_analysis" / "stats" / "midi_sampler" / "MAPS_ENSTDkCl_ENSTDkAm_sampler.json",
        "francoisleduc": repo_root() / "data_analysis" / "stats" / "midi_sampler" / "FrancoisLeducGuitarDataset_sampler.json",
        "gaps": repo_root() / "data_analysis" / "stats" / "midi_sampler" / "GAPS_sampler.json",
    }
    if dataset_type not in default_map:
        raise KeyError(
            f"No default stats JSON is registered for dataset '{dataset_type}'. "
            "Set route1.stats_json=/abs/path/to/stats.json"
        )
    return default_map[dataset_type]



if __name__ == "__main__":
    raise SystemExit("route1_infer.py is an implementation module. Use route1_eval_job.py for the supported Route I evaluation job.")
