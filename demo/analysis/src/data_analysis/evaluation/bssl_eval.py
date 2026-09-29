from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import torch

from ..analysis.visualize_midi_audio import compute_curve_similarity_metrics
from ..rendering.feature_extractor import PsychoFeatureExtractor


_METRIC_FIELDS = (
    "pearson_correlation",
    "mean_absolute_error",
    "cosine_sim",
    "spearman_correlation",
    "mean_squared_error",
)


def _nan_metrics() -> Dict[str, float]:
    return {key: float("nan") for key in _METRIC_FIELDS}


def _nan_metrics_for(metric_fields: Optional[Sequence[str]]) -> Dict[str, float]:
    keys = tuple(metric_fields) if metric_fields is not None else _METRIC_FIELDS
    return {key: float("nan") for key in keys}


def _prepare_metric_vectors(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Curves must have the same shape for comparison.")
    if a.size == 0:
        raise ValueError("Curves must be non-empty.")
    mask = ~(np.isnan(a) | np.isnan(b))
    if not np.any(mask):
        raise ValueError("Curves only contain NaNs after masking.")
    return a[mask], b[mask]


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _safe_curve_metrics(
    a: np.ndarray,
    b: np.ndarray,
    *,
    metric_fields: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    selected_fields = tuple(metric_fields) if metric_fields is not None else _METRIC_FIELDS
    try:
        if selected_fields == _METRIC_FIELDS:
            metrics = compute_curve_similarity_metrics(a, b)
        else:
            a_vec, b_vec = _prepare_metric_vectors(a, b)
            metrics = {}
            if "pearson_correlation" in selected_fields:
                metrics["pearson_correlation"] = _pearson_correlation(a_vec, b_vec)
            if "cosine_sim" in selected_fields:
                metrics["cosine_sim"] = _cosine(a_vec, b_vec)
            if "mean_absolute_error" in selected_fields:
                metrics["mean_absolute_error"] = float(np.mean(np.abs(a_vec - b_vec)))
            if "mean_squared_error" in selected_fields:
                metrics["mean_squared_error"] = float(np.mean((a_vec - b_vec) ** 2))
            if "spearman_correlation" in selected_fields:
                metrics["spearman_correlation"] = float(
                    compute_curve_similarity_metrics(a_vec, b_vec)["spearman_correlation"]
                )
    except Exception:
        return _nan_metrics_for(selected_fields)
    out = _nan_metrics_for(selected_fields)
    for key in selected_fields:
        if key in metrics:
            out[key] = float(metrics[key])
    return out



def _load_audio_mono(
    wav_path: str | Path,
    *,
    target_sample_rate: Optional[int],
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio not found: {wav_path}")

    audio, sample_rate = librosa.load(str(wav_path), sr=None, mono=True)
    if target_sample_rate and int(target_sample_rate) != int(sample_rate):
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    return waveform.to(device), int(sample_rate)



def _compute_ntot_from_sone(sone_cf: torch.Tensor) -> torch.Tensor:
    """Compute Stevens total loudness (Ntot) from sone features.

    Args:
        sone_cf: (C, F)

    Returns:
        ntot: (F,)
    """
    max_val, _ = torch.max(sone_cf, dim=0, keepdim=True)  # (1, F)
    rest = (sone_cf.sum(dim=0, keepdim=True) - max_val)  # (1, F)
    ntot = max_val.squeeze(0) + 0.15 * rest.squeeze(0)
    return ntot



def _zero_pad_to_same_length(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]:
    """Align two waveforms by zero-padding the shorter one to max length."""
    if y_pred.ndim != 2 or y_gt.ndim != 2:
        raise ValueError(f"Expected 2D waveforms (1, T), got pred={y_pred.shape}, gt={y_gt.shape}")

    pred_len = int(y_pred.shape[-1])
    gt_len = int(y_gt.shape[-1])
    target_len = max(pred_len, gt_len)
    pred_pad = max(0, target_len - pred_len)
    gt_pad = max(0, target_len - gt_len)

    if pred_pad > 0:
        y_pred = torch.nn.functional.pad(y_pred, (0, int(pred_pad)))
    if gt_pad > 0:
        y_gt = torch.nn.functional.pad(y_gt, (0, int(gt_pad)))

    return y_pred, y_gt, pred_len, gt_len, target_len, pred_pad, gt_pad



def _resample_to_unit_grid(times: np.ndarray, values: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (times, values) onto a normalized 0..1 grid of length num_samples."""
    if num_samples <= 1:
        raise ValueError("num_samples must be > 1")

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if times.size == 0 or values.size == 0:
        grid = np.linspace(0.0, 1.0, num_samples)
        return grid, np.zeros(num_samples, dtype=np.float64)

    duration = float(max(times[-1], 1e-8))
    t_norm = np.clip(times / duration, 0.0, 1.0)
    grid = np.linspace(0.0, 1.0, num_samples)
    resampled = np.interp(grid, t_norm, values, left=float(values[0]), right=float(values[-1]))
    return grid, resampled



def _normalize(values: np.ndarray, method: str) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    method = (method or "none").lower()
    if method == "none":
        return v
    if method == "minmax":
        vmin, vmax = float(np.min(v)), float(np.max(v))
        span = vmax - vmin
        if span <= 1e-12:
            return v * 0.0
        return (v - vmin) / span
    mean = float(np.mean(v))
    std = float(np.std(v))
    if std <= 1e-12:
        return v - mean
    return (v - mean) / std



def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)



def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v * 0.0
    return v / n



def _stats_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }



def _dynamic_range_stats_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            "p5": float("nan"),
            "p95": float("nan"),
            "p95_minus_p5": float("nan"),
            "std": float("nan"),
            "peak_ratio_max_over_p5": float("nan"),
        }
    p5 = float(np.percentile(x, 5.0))
    p95 = float(np.percentile(x, 95.0))
    std = float(np.std(x))
    peak = float(np.max(x))
    peak_ratio = float("nan") if abs(p5) <= 1e-12 else float(peak / p5)
    return {
        "p5": p5,
        "p95": p95,
        "p95_minus_p5": float(p95 - p5),
        "std": std,
        "peak_ratio_max_over_p5": peak_ratio,
    }



def _save_ntot_comparison_plot(
    *,
    times_pred: np.ndarray,
    ntot_pred: np.ndarray,
    times_gt: np.ndarray,
    ntot_gt: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times_gt, ntot_gt, label="gt ntot", lw=1.5, color="tab:blue")
    ax.plot(times_pred, ntot_pred, label="pred ntot", lw=1.5, color="tab:orange", alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ntot (sone)")
    ax.set_title("Ntot Comparison: pred vs gt")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)



def _average_metric_dicts(
    metric_dicts: list[Dict[str, float]],
    *,
    metric_fields: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    selected_fields = tuple(metric_fields) if metric_fields is not None else _METRIC_FIELDS
    out = _nan_metrics_for(selected_fields)
    if not metric_dicts:
        return out
    for key in selected_fields:
        vals = [float(m[key]) for m in metric_dicts if key in m and np.isfinite(float(m[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out



def evaluate_bssl_pair(
    pred_wav: str | Path,
    gt_wav: str | Path,
    *,
    sample_rate: int,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    db_max: float = 96.0,
    outer_ear: str = "terhardt",
    bssl_mode: str = "sone",  # "sone" or "bark"
    num_samples: int = 2048,
    normalization: str = "zscore",
    device: Optional[str] = None,
    ntot_plot_path: Optional[str | Path] = None,
    pearson_only: bool = False,
) -> Dict[str, object]:
    """Extract BSSL + total loudness from two audio files and compute similarities.

    BSSL metrics are reported on a time-resampled (num_samples x num_bands) matrix flattened
    into a single vector. This keeps the earlier cosine-similarity behaviour while also exposing
    Pearson / Spearman / MAE.
    """
    bssl_mode = bssl_mode.lower()
    if bssl_mode not in ("sone", "bark"):
        raise ValueError("bssl_mode must be 'sone' or 'bark'")

    dev = torch.device(device) if device else torch.device("cpu")

    y_pred, sr_pred = _load_audio_mono(pred_wav, target_sample_rate=int(sample_rate), device=dev)
    y_gt, sr_gt = _load_audio_mono(gt_wav, target_sample_rate=int(sample_rate), device=dev)
    if sr_pred != sr_gt:
        raise RuntimeError("Internal error: sample rates diverged after resampling")
    sr = sr_pred
    y_pred, y_gt, pred_len_raw, gt_len_raw, target_len, pred_pad, gt_pad = _zero_pad_to_same_length(y_pred, y_gt)

    extractor = PsychoFeatureExtractor(
        sample_rate=sr,
        fft_size=int(fft_size),
        frames_per_second=float(frames_per_second),
        db_max=float(db_max),
        outer_ear=outer_ear,
        return_mode=bssl_mode,
    ).to(dev)

    with torch.no_grad():
        bssl_pred = extractor(y_pred).squeeze(0)  # (C, F)
        bssl_gt = extractor(y_gt).squeeze(0)

    C_pred, F_pred = int(bssl_pred.shape[0]), int(bssl_pred.shape[1])
    C_gt, F_gt = int(bssl_gt.shape[0]), int(bssl_gt.shape[1])
    if C_pred != C_gt:
        raise ValueError(f"BSSL band mismatch: pred {C_pred}, gt {C_gt}")
    C = C_pred

    hop_duration = extractor.hop_size / extractor.sample_rate
    times_pred = np.arange(F_pred, dtype=np.float64) * float(hop_duration)
    times_gt = np.arange(F_gt, dtype=np.float64) * float(hop_duration)

    bssl_pred_fc = bssl_pred.transpose(0, 1).cpu().numpy().astype(np.float64)  # (F, C)
    bssl_gt_fc = bssl_gt.transpose(0, 1).cpu().numpy().astype(np.float64)

    if bssl_mode == "sone":
        ntot_pred = _compute_ntot_from_sone(bssl_pred).cpu().numpy().astype(np.float64)
        ntot_gt = _compute_ntot_from_sone(bssl_gt).cpu().numpy().astype(np.float64)
    else:
        extractor_sone = PsychoFeatureExtractor(
            sample_rate=sr,
            fft_size=int(fft_size),
            frames_per_second=float(frames_per_second),
            db_max=float(db_max),
            outer_ear=outer_ear,
            return_mode="sone",
        ).to(dev)
        with torch.no_grad():
            sone_pred = extractor_sone(y_pred).squeeze(0)  # (C, F)
            sone_gt = extractor_sone(y_gt).squeeze(0)
        ntot_pred = _compute_ntot_from_sone(sone_pred).cpu().numpy().astype(np.float64)
        ntot_gt = _compute_ntot_from_sone(sone_gt).cpu().numpy().astype(np.float64)

    mean_pred = np.mean(bssl_pred_fc, axis=0)
    mean_gt = np.mean(bssl_gt_fc, axis=0)
    bssl_mean_cos = _cosine(_l2_normalize(mean_pred), _l2_normalize(mean_gt))

    pred_resampled = np.zeros((num_samples, C), dtype=np.float64)
    gt_resampled = np.zeros((num_samples, C), dtype=np.float64)
    for c in range(C):
        _, pred_resampled[:, c] = _resample_to_unit_grid(times_pred, bssl_pred_fc[:, c], num_samples)
        _, gt_resampled[:, c] = _resample_to_unit_grid(times_gt, bssl_gt_fc[:, c], num_samples)

    metric_fields = ("pearson_correlation",) if pearson_only else None

    pred_flat = pred_resampled.reshape(-1)
    gt_flat = gt_resampled.reshape(-1)
    pred_flat_norm = _normalize(pred_flat, normalization)
    gt_flat_norm = _normalize(gt_flat, normalization)
    bssl_flat_metrics_raw = _safe_curve_metrics(pred_flat, gt_flat, metric_fields=metric_fields)
    bssl_flat_metrics_norm = _safe_curve_metrics(pred_flat_norm, gt_flat_norm, metric_fields=metric_fields)
    bssl_flat_cos = float(bssl_flat_metrics_raw.get("cosine_sim", float("nan")))

    band_cos = [] if not pearson_only else None
    band_metrics_raw = []
    band_metrics_norm = []
    for c in range(C):
        band_pred = pred_resampled[:, c]
        band_gt = gt_resampled[:, c]
        if band_cos is not None:
            band_cos.append(_cosine(_l2_normalize(band_pred), _l2_normalize(band_gt)))
        band_metrics_raw.append(_safe_curve_metrics(band_pred, band_gt, metric_fields=metric_fields))
        band_metrics_norm.append(
            _safe_curve_metrics(
                _normalize(band_pred, normalization),
                _normalize(band_gt, normalization),
                metric_fields=metric_fields,
            )
        )
    bssl_band_cos_mean = float(np.nanmean(band_cos)) if band_cos else float("nan")

    _, ntot_pred_rs = _resample_to_unit_grid(times_pred, ntot_pred, num_samples)
    _, ntot_gt_rs = _resample_to_unit_grid(times_gt, ntot_gt, num_samples)
    ntot_pred_norm = _normalize(ntot_pred_rs, normalization)
    ntot_gt_norm = _normalize(ntot_gt_rs, normalization)
    ntot_metrics_raw = _safe_curve_metrics(ntot_pred_rs, ntot_gt_rs, metric_fields=metric_fields)
    ntot_metrics_norm = _safe_curve_metrics(ntot_pred_norm, ntot_gt_norm, metric_fields=metric_fields)

    artifacts: Dict[str, object] = {}
    if ntot_plot_path:
        plot_path = Path(ntot_plot_path)
        _save_ntot_comparison_plot(
            times_pred=times_pred,
            ntot_pred=ntot_pred,
            times_gt=times_gt,
            ntot_gt=ntot_gt,
            output_path=plot_path,
        )
        artifacts["ntot_plot_path"] = str(plot_path)

    summary = {
        "bssl_pearson_correlation": float(bssl_flat_metrics_raw["pearson_correlation"]),
        "ntot_pearson_correlation": float(ntot_metrics_raw["pearson_correlation"]),
    }
    if not pearson_only:
        summary.update(
            {
                "bssl_mean_absolute_error": float(bssl_flat_metrics_raw["mean_absolute_error"]),
                "bssl_spearman_correlation": float(bssl_flat_metrics_raw["spearman_correlation"]),
                "bssl_cosine_sim": float(bssl_flat_metrics_raw["cosine_sim"]),
                "ntot_cosine_sim": float(ntot_metrics_raw["cosine_sim"]),
                "ntot_mean_absolute_error": float(ntot_metrics_raw["mean_absolute_error"]),
                "ntot_spearman_correlation": float(ntot_metrics_raw["spearman_correlation"]),
                "ntot_cosine_sim_raw": float(ntot_metrics_raw["cosine_sim"]),
                "ntot_cosine_sim_normalized": float(ntot_metrics_norm["cosine_sim"]),
            }
        )

    result: Dict[str, object] = {
        "config": {
            "sample_rate": int(sample_rate),
            "frames_per_second": float(frames_per_second),
            "fft_size": int(fft_size),
            "db_max": float(db_max),
            "outer_ear": outer_ear,
            "bssl_mode": bssl_mode,
            "num_samples": int(num_samples),
            "normalization": normalization,
            "pearson_only": bool(pearson_only),
            "device": str(dev),
        },
        "paths": {
            "pred_wav": str(Path(pred_wav)),
            "gt_wav": str(Path(gt_wav)),
        },
        "bssl": {
            "num_bands": int(C),
            "mean_vector_pred": mean_pred.tolist(),
            "mean_vector_gt": mean_gt.tolist(),
            "flattened_raw_metrics": bssl_flat_metrics_raw,
            "flattened_normalized_metrics": bssl_flat_metrics_norm,
            "bandwise_metrics_raw_mean": _average_metric_dicts(band_metrics_raw, metric_fields=metric_fields),
            "bandwise_metrics_normalized_mean": _average_metric_dicts(band_metrics_norm, metric_fields=metric_fields),
        },
        "ntot": {
            "pred_stats": _stats_1d(ntot_pred),
            "gt_stats": _stats_1d(ntot_gt),
            "pred_dynamic_range": _dynamic_range_stats_1d(ntot_pred),
            "gt_dynamic_range": _dynamic_range_stats_1d(ntot_gt),
            "pred_minus_gt_mean": float(np.mean(ntot_pred) - np.mean(ntot_gt)) if ntot_pred.size and ntot_gt.size else float("nan"),
            "curve_metrics_raw": ntot_metrics_raw,
            "curve_metrics": ntot_metrics_norm,
            "curve_metrics_normalized": ntot_metrics_norm,
        },
        "durations": {
            "policy": "zero-pad shorter side to longer side (no trimming)",
            "pred_seconds_raw": float(pred_len_raw) / float(sr),
            "gt_seconds_raw": float(gt_len_raw) / float(sr),
            "aligned_seconds_target": float(target_len) / float(sr),
            "padding": {
                "pred_padded": bool(pred_pad > 0),
                "pred_pad_samples": int(pred_pad),
                "pred_pad_seconds": float(pred_pad) / float(sr),
                "gt_padded": bool(gt_pad > 0),
                "gt_pad_samples": int(gt_pad),
                "gt_pad_seconds": float(gt_pad) / float(sr),
                "padded_side": (
                    "pred" if pred_pad > 0 and gt_pad == 0
                    else "gt" if gt_pad > 0 and pred_pad == 0
                    else "none"
                ),
            },
            "pred_seconds": float(times_pred[-1]) if times_pred.size else 0.0,
            "gt_seconds": float(times_gt[-1]) if times_gt.size else 0.0,
        },
        "artifacts": artifacts,
        "summary": summary,
    }
    if not pearson_only:
        result["bssl"]["cosine_mean_vector"] = float(bssl_mean_cos)  # type: ignore[index]
        result["bssl"]["bssl_cosine_sim"] = float(bssl_flat_cos)  # type: ignore[index]
        result["bssl"]["bandwise_cosine"] = [float(x) for x in band_cos or []]  # type: ignore[index]
        result["bssl"]["bandwise_cosine_mean"] = float(bssl_band_cos_mean)  # type: ignore[index]

    return result



def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Extract BSSL (bark/sone) + total loudness (ntot) and compute similarities."
    )
    p.add_argument("pred_wav", help="arg1: predicted WAV (flattened velocity)")
    p.add_argument("gt_wav", help="arg2: ground-truth WAV (original velocity)")
    p.add_argument("sample_rate", type=int, help="arg3: sampling rate for feature extraction")
    p.add_argument("--fps", type=float, default=50.0, help="frames_per_second")
    p.add_argument("--fft", type=int, default=1024, help="fft_size")
    p.add_argument("--bssl_mode", type=str, default="sone", choices=["sone", "bark"], help="BSSL mode")
    p.add_argument("--num_samples", type=int, default=2048, help="resample length")
    p.add_argument("--norm", type=str, default="zscore", choices=["zscore", "minmax", "none"], help="normalization for curve metrics")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda")
    p.add_argument("--ntot_plot", type=str, default=None, help="Optional PNG path for pred-vs-gt ntot plot")
    p.add_argument("--json_out", type=str, default=None, help="Optional JSON output path")
    p.add_argument("--mute_output", action="store_true", help="Do not print JSON to stdout")
    return p



def main() -> None:
    import json

    args = build_arg_parser().parse_args()
    res = evaluate_bssl_pair(
        args.pred_wav,
        args.gt_wav,
        sample_rate=args.sample_rate,
        frames_per_second=args.fps,
        fft_size=args.fft,
        bssl_mode=args.bssl_mode,
        num_samples=args.num_samples,
        normalization=args.norm,
        device=args.device,
        ntot_plot_path=args.ntot_plot,
    )
    payload = json.dumps(res, ensure_ascii=False, indent=2)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not args.mute_output:
        print(payload)


if __name__ == "__main__":
    main()
