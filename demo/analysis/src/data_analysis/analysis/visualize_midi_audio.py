"""
Utilities for visualising MIDI and audio side by side:
1. MIDI piano roll.
2. MIDI velocity scatter + smoothed curve.
3. BSSL (Bark-scale specific loudness) total loudness in sones.

The BSSL implementation adapts the PsychoFeatureExtractor that powers
`data_analysis.rendering.feature_extractor` so we can compute the same
curve directly from any WAV file without relying on .h5 intermediates.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
try:
    import pretty_midi
    _PRETTY_MIDI_ERROR = None
except Exception as exc:  # noqa: BLE001
    pretty_midi = None  # type: ignore[assignment]
    _PRETTY_MIDI_ERROR = exc

from ..rendering.feature_extractor import PsychoFeatureExtractor


def _load_audio_mono_tensor(
    wav_path: Path | str,
    *,
    target_sample_rate: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    audio, sample_rate = librosa.load(str(wav_path), sr=None, mono=True)
    if target_sample_rate and int(target_sample_rate) != int(sample_rate):
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    return waveform.to(device or torch.device("cpu")), int(sample_rate)


# ---------------------------------------------------------------------------
# MIDI helpers
# ---------------------------------------------------------------------------

def load_pretty_midi(midi_path: Path | str) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file and raise a clear error if anything goes wrong."""
    if pretty_midi is None:
        raise ImportError("pretty_midi is required. Install via `pip install pretty_midi`.") from _PRETTY_MIDI_ERROR
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    return pretty_midi.PrettyMIDI(str(midi_path))


def extract_note_events(pm: pretty_midi.PrettyMIDI) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return note start times, velocities, and pitches sorted by onset."""
    starts, velocities, pitches = [], [], []
    for instrument in pm.instruments:
        for note in instrument.notes:
            starts.append(note.start)
            velocities.append(note.velocity)
            pitches.append(note.pitch)
    if not starts:
        return np.empty(0), np.empty(0), np.empty(0)
    order = np.argsort(starts)
    return np.asarray(starts)[order], np.asarray(velocities)[order], np.asarray(pitches)[order]


def build_piano_roll(pm: pretty_midi.PrettyMIDI, fs: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a dense piano roll using PrettyMIDI.

    Returns:
        times: shape (frames,) in seconds
        roll: shape (128, frames) with velocities [0, 127]
    """
    roll = pm.get_piano_roll(fs=fs)
    times = np.arange(roll.shape[1]) / float(fs)
    return times, roll


def plot_piano_roll(times: np.ndarray, roll: np.ndarray, *, figsize=(12, 4), output_path: Optional[Path] = None):
    """Visualise the piano roll as image."""
    if roll.ndim != 2 or roll.shape[0] != 128:
        raise ValueError("Expected piano roll with shape (128, frames)")
    fig, ax = plt.subplots(figsize=figsize)
    extent = [times[0], times[-1], 0, 128]
    im = ax.imshow(
        roll,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=127,
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title("MIDI Piano Roll")
    fig.colorbar(im, ax=ax, label="Velocity")
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_velocity_curve(
    times: np.ndarray,
    velocities: np.ndarray,
    *,
    window: int = 25,
    figsize=(12, 4),
    output_path: Optional[Path] = None,
):
    """Scatter MIDI velocities and overlay a moving-average curve."""
    fig, ax = plt.subplots(figsize=figsize)
    if len(times):
        ax.scatter(times, velocities, c=velocities, cmap="inferno", s=12, alpha=0.6, vmin=0, vmax=127, label="Notes")
        if len(velocities) >= 2:
            win = min(window, len(velocities))
            kernel = np.ones(win) / win
            smoothed_velo = np.convolve(velocities, kernel, mode="valid")
            smoothed_time = np.convolve(times, kernel, mode="valid")
            ax.plot(smoothed_time, smoothed_velo, color="cyan", lw=2, label=f"Moving Avg (window={win})")
    ax.set_xlim(times[0] if len(times) else 0, times[-1] if len(times) else 1)
    ax.set_ylim(0, 130)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title("MIDI Velocity Curve")
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# BSSL total loudness extractor (reuse PsychoFeatureExtractor directly)
# ---------------------------------------------------------------------------

# The heavy lifting lives in `rendering.feature_extractor.PsychoFeatureExtractor`, so we
# just instantiate it in ntot mode whenever we need the curve for plotting.


def compute_bssl_total_loudness(
    wav_path: Path | str,
    *,
    target_sample_rate: Optional[int] = None,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    db_max: float = 96.0,
    outer_ear: str = "terhardt",
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load the WAV file, compute the BSSL total loudness curve, and return (times, loudness, sample_rate).
    """
    waveform, sample_rate = _load_audio_mono_tensor(
        wav_path,
        target_sample_rate=target_sample_rate,
        device=device or torch.device("cpu"),
    )
    extractor = PsychoFeatureExtractor(
        sample_rate=sample_rate,
        fft_size=fft_size,
        frames_per_second=frames_per_second,
        db_max=db_max,
        outer_ear=outer_ear,
        return_mode="ntot",
    ).to(waveform.device)
    with torch.no_grad():
        loudness = extractor(waveform).squeeze(0).cpu().numpy()
    hop_duration = extractor.hop_size / extractor.sample_rate
    times = np.arange(loudness.shape[0]) * hop_duration
    return times, loudness, sample_rate


def plot_bssl_total_loudness(
    times: np.ndarray,
    loudness: np.ndarray,
    *,
    figsize=(12, 3.5),
    output_path: Optional[Path] = None,
):
    """Plot the loudness curve."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, loudness, color="tab:blue", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Loudness (sones)")
    ax.set_title("BSSL Total Loudness (Ntot)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Curve comparison + similarity helpers
# ---------------------------------------------------------------------------

def compute_velocity_curve_from_midi(
    midi_path: Path | str,
    *,
    frames_per_second: float = 50.0,
    smoothing_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample MIDI note velocities onto a uniform time grid.

    Returns (times, values) where len(times) == len(values).
    """
    pm = load_pretty_midi(midi_path)
    note_times, velocities, _ = extract_note_events(pm)
    duration = max(pm.get_end_time(), note_times[-1] if len(note_times) else 0.0)
    if duration <= 0:
        duration = 1.0
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    step = 1.0 / frames_per_second
    grid_times = np.arange(0.0, duration + step, step)
    velocity_curve = _resample_curve_to_times(note_times, velocities, grid_times)
    if smoothing_window and smoothing_window > 1:
        velocity_curve = _moving_average(velocity_curve, smoothing_window)
    return grid_times, velocity_curve


def compare_velocity_with_bssl(
    midi_path: Path | str,
    wav_path: Path | str,
    *,
    target_sample_rate: Optional[int] = 22050,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    velocity_smoothing: Optional[int] = 25,
    normalization: str = "zscore",
    micro_window_seconds: Optional[float] = None,
    micro_window_frames: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Align the MIDI velocity envelope with the BSSL loudness curve for the same song.

    Returns a dict containing times, the two curves, and similarity metrics
    (macro over the entire piece + micro windows if requested).
    """
    pm = load_pretty_midi(midi_path)
    note_times, velocities, _ = extract_note_events(pm)
    bssl_times, loudness, sample_rate = compute_bssl_total_loudness(
        wav_path,
        target_sample_rate=target_sample_rate,
        frames_per_second=frames_per_second,
        fft_size=fft_size,
        device=device,
    )
    velocity_curve = _resample_curve_to_times(note_times, velocities, bssl_times)
    if velocity_smoothing and velocity_smoothing > 1:
        velocity_curve = _moving_average(velocity_curve, velocity_smoothing)
    velocity_norm = _normalize_curve(velocity_curve, normalization)
    loudness_norm = _normalize_curve(loudness, normalization)
    macro_metrics = compute_curve_similarity_metrics(velocity_norm, loudness_norm)
    micro_metrics = _compute_micro_window_stats(
        velocity_norm,
        loudness_norm,
        time_axis=bssl_times,
        window_seconds=micro_window_seconds,
        window_frames=micro_window_frames,
        default_frame_step=1.0 / frames_per_second if frames_per_second > 0 else None,
    )
    return {
        "times": bssl_times,
        "velocity_curve": velocity_curve,
        "loudness_curve": loudness,
        "velocity_normalized": velocity_norm,
        "loudness_normalized": loudness_norm,
        "metrics": {
            "macro": macro_metrics,
            "micro": micro_metrics,
        },
        "sample_rate": sample_rate,
    }


def compare_midi_velocity_files(
    midi_path_a: Path | str,
    midi_path_b: Path | str,
    *,
    frames_per_second: float = 50.0,
    smoothing_window: Optional[int] = 5,
    num_samples: Optional[int] = None,
    normalization: str = "zscore",
    micro_window_seconds: Optional[float] = None,
    micro_window_frames: Optional[int] = None,
) -> dict:
    """
    Compare MIDI velocity envelopes between two pieces by aligning them on the same
    time axis (optionally downsampled) and computing macro/micro metrics.
    """
    times_a, vel_a = compute_velocity_curve_from_midi(
        midi_path_a, frames_per_second=frames_per_second, smoothing_window=smoothing_window
    )
    times_b, vel_b = compute_velocity_curve_from_midi(
        midi_path_b, frames_per_second=frames_per_second, smoothing_window=smoothing_window
    )
    max_len = min(vel_a.size, vel_b.size)
    if max_len == 0:
        raise ValueError("No overlapping velocity frames detected between the two MIDI files.")
    vel_a = vel_a[:max_len]
    vel_b = vel_b[:max_len]
    aligned_times = times_a[:max_len]
    if num_samples is not None:
        if num_samples <= 1:
            raise ValueError("num_samples must be greater than 1 when provided.")
        total_duration = float(aligned_times[-1]) if aligned_times.size else 0.0
        resample_times = np.linspace(0.0, total_duration, num_samples)
        vel_a = np.interp(resample_times, aligned_times, vel_a, left=vel_a[0], right=vel_a[-1])
        vel_b = np.interp(resample_times, aligned_times, vel_b, left=vel_b[0], right=vel_b[-1])
        aligned_times = resample_times
    norm_a = _normalize_curve(vel_a, normalization)
    norm_b = _normalize_curve(vel_b, normalization)
    macro_metrics = compute_curve_similarity_metrics(norm_a, norm_b)
    micro_metrics = _compute_micro_window_stats(
        norm_a,
        norm_b,
        time_axis=aligned_times,
        window_seconds=micro_window_seconds,
        window_frames=micro_window_frames,
        default_frame_step=1.0 / frames_per_second if frames_per_second > 0 else None,
    )
    return {
        "times": aligned_times,
        "curve_a": vel_a,
        "curve_b": vel_b,
        "normalized_a": norm_a,
        "normalized_b": norm_b,
        "metrics": {
            "macro": macro_metrics,
            "micro": micro_metrics,
        },
    }


def compare_bssl_audio_files(
    wav_path_a: Path | str,
    wav_path_b: Path | str,
    *,
    target_sample_rate: Optional[int] = 22050,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    num_samples: int = 2048,
    normalization: str = "zscore",
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compare BSSL loudness curves extracted from two audio files by aligning them on a normalized grid.
    """
    times_a, loudness_a, _ = compute_bssl_total_loudness(
        wav_path_a,
        target_sample_rate=target_sample_rate,
        frames_per_second=frames_per_second,
        fft_size=fft_size,
        device=device,
    )
    times_b, loudness_b, _ = compute_bssl_total_loudness(
        wav_path_b,
        target_sample_rate=target_sample_rate,
        frames_per_second=frames_per_second,
        fft_size=fft_size,
        device=device,
    )
    grid, resampled_a = _resample_curve_to_uniform_grid(times_a, loudness_a, num_samples)
    _, resampled_b = _resample_curve_to_uniform_grid(times_b, loudness_b, num_samples)
    norm_a = _normalize_curve(resampled_a, normalization)
    norm_b = _normalize_curve(resampled_b, normalization)
    metrics = compute_curve_similarity_metrics(norm_a, norm_b)
    return {
        "grid": grid,
        "curve_a": resampled_a,
        "curve_b": resampled_b,
        "normalized_a": norm_a,
        "normalized_b": norm_b,
        "metrics": metrics,
    }


def compute_curve_similarity_metrics(curve_a: np.ndarray, curve_b: np.ndarray) -> dict:
    """
    Return cosine similarity, Pearson/Spearman correlation, and MAE/MSE between two equal-length curves.
    """
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Curves must have the same shape for comparison.")
    if a.size == 0:
        raise ValueError("Curves must be non-empty.")
    mask = ~(np.isnan(a) | np.isnan(b))
    if not np.any(mask):
        raise ValueError("Curves only contain NaNs after masking.")
    a = a[mask]
    b = b[mask]
    metrics = {
        "cosine_sim": _cosine_sim(a, b),
        "pearson_correlation": _pearson_correlation(a, b),
        "spearman_correlation": _spearman_correlation(a, b),
        "mean_absolute_error": float(np.mean(np.abs(a - b))),
        "mean_squared_error": float(np.mean((a - b) ** 2)),
    }
    return metrics


def _resample_curve_to_times(
    source_times: np.ndarray | list,
    source_values: np.ndarray | list,
    target_times: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    source_times = np.asarray(source_times, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if source_times.size == 0 or source_values.size == 0:
        return np.full_like(target_times, fill_value, dtype=np.float64)
    return np.interp(target_times, source_times, source_values, left=fill_value, right=fill_value)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if values.size == 0:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _normalize_curve(values: np.ndarray, method: str = "zscore") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if method == "none":
        return arr
    if method == "minmax":
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        span = vmax - vmin
        if span <= 1e-8:
            return arr * 0.0
        return (arr - vmin) / span
    # Default: z-score
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 1e-8:
        return arr - mean
    return (arr - mean) / std


def _compute_micro_window_stats(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    *,
    time_axis: Optional[np.ndarray],
    window_seconds: Optional[float],
    window_frames: Optional[int],
    default_frame_step: Optional[float],
) -> dict:
    """
    Slice the curves into windows and compute per-window similarity metrics.
    """
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Curves must have identical shapes for micro metrics.")
    total_frames = a.size
    if total_frames == 0:
        return {
            "window_frames": 0,
            "window_seconds": None,
            "num_windows": 0,
            "per_window": [],
            "aggregated": {},
        }
    frame_step = _frame_step_from_time_axis(time_axis, default_frame_step)
    if window_frames is not None:
        if window_frames <= 0:
            raise ValueError("micro_window_frames must be positive.")
        chunk = int(window_frames)
    elif window_seconds is not None:
        if window_seconds <= 0:
            raise ValueError("micro_window_seconds must be positive.")
        if frame_step is None or frame_step <= 0:
            raise ValueError("Cannot derive frame duration for micro_window_seconds.")
        chunk = max(int(round(window_seconds / frame_step)), 1)
    else:
        chunk = total_frames
    actual_window_seconds = frame_step * chunk if frame_step and chunk else None
    per_window: List[dict] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk, total_frames)
        metrics = compute_curve_similarity_metrics(a[start:end], b[start:end])
        start_time, end_time = _window_time_range(start, end, time_axis, frame_step)
        per_window.append(
            {
                "index": len(per_window),
                "start_frame": start,
                "end_frame": end,
                "num_frames": end - start,
                "start_time": start_time,
                "end_time": end_time,
                "metrics": metrics,
            }
        )
        start = end
    aggregated = _aggregate_micro_metrics(per_window)
    return {
        "window_frames": chunk,
        "window_seconds": actual_window_seconds,
        "num_windows": len(per_window),
        "per_window": per_window,
        "aggregated": aggregated,
    }


def _frame_step_from_time_axis(time_axis: Optional[np.ndarray], default_frame_step: Optional[float]) -> Optional[float]:
    if time_axis is not None and time_axis.size >= 2:
        diffs = np.diff(time_axis)
        positive = diffs[diffs > 0]
        if positive.size:
            return float(np.median(positive))
    return default_frame_step


def _window_time_range(
    start_frame: int,
    end_frame: int,
    time_axis: Optional[np.ndarray],
    frame_step: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if end_frame <= start_frame:
        return None, None
    start_time = None
    end_time = None
    if time_axis is not None and time_axis.size:
        max_index = time_axis.size - 1
        if start_frame <= max_index:
            start_time = float(time_axis[start_frame])
        elif frame_step:
            start_time = time_axis[-1] + frame_step * (start_frame - max_index)
        if end_frame - 1 <= max_index:
            end_time = float(time_axis[end_frame - 1])
        elif frame_step:
            end_time = (time_axis[-1] if time_axis.size else 0.0) + frame_step * (end_frame - 1 - max_index)
        if end_time is not None and frame_step:
            end_time += frame_step
    elif frame_step:
        start_time = start_frame * frame_step
        end_time = end_frame * frame_step
    return start_time, end_time


def _aggregate_micro_metrics(per_window: List[dict]) -> dict:
    if not per_window:
        return {}
    metric_keys = per_window[0]["metrics"].keys()
    summary: dict = {}
    for key in metric_keys:
        values = [
            window["metrics"][key]
            for window in per_window
            if not np.isnan(window["metrics"][key])
        ]
        summary[key] = float(np.mean(values)) if values else float("nan")
    return summary


def _resample_curve_to_uniform_grid(
    times: np.ndarray,
    values: np.ndarray,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1")
    if times.size == 0 or values.size == 0:
        grid = np.linspace(0.0, 1.0, num_samples)
        return grid, np.zeros(num_samples, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    duration = max(times[-1], 1e-8)
    normalized_times = np.clip(times / duration, 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, num_samples)
    resampled = np.interp(grid, normalized_times, values, left=values[0], right=values[-1])
    return grid, resampled


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denom)


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    rank_a = _rankdata(a)
    rank_b = _rankdata(b)
    return _pearson_correlation(rank_a, rank_b)


def _rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(arr.size, dtype=np.float64)
    sorted_vals = arr[order]
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------


def _maybe_save(fig: plt.Figure, output_path: Optional[Path]):
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
