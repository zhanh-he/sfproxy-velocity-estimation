"""
Audio BSSL total loudness statistics utilities.

Given a collection of audio files, these helpers compute the Bark-scale total
loudness (Ntot) curves, aggregate the distributions across frames, and produce
histograms showing both total counts and frame percentages.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from .visualize_midi_audio import compute_bssl_total_loudness


@dataclass
class BsslPerFileStats:
    path: Path
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float


@dataclass
class BsslDatasetStats:
    dataset_name: str
    audio_files: List[Path]
    per_file_stats: List[BsslPerFileStats]
    hist_counts: np.ndarray
    bin_edges: np.ndarray
    errors: List[Tuple[Path, str]]

    @property
    def total_frames(self) -> int:
        return int(self.hist_counts.sum())

    @property
    def hist_percentages(self) -> np.ndarray:
        total = self.hist_counts.sum()
        if total <= 0:
            return np.zeros_like(self.hist_counts, dtype=float)
        return self.hist_counts / total * 100.0

    @property
    def zero_bins(self) -> List[Tuple[float, float]]:
        zero_idx = np.where(self.hist_counts == 0)[0]
        edges = self.bin_edges
        return [(float(edges[i]), float(edges[i + 1])) for i in zero_idx]


def discover_audio_files(
    dataset_root: Path | str,
    *,
    glob_patterns: Optional[Sequence[str]] = None,
    allowed_suffixes: Tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> List[Path]:
    """
    Return a sorted list of audio file paths within dataset_root.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    allowed = tuple(s.lower() for s in allowed_suffixes)
    results: List[Path] = []

    def maybe_add(path: Path):
        if path.is_file() and path.suffix.lower() in allowed:
            results.append(path)

    if glob_patterns:
        for pattern in glob_patterns:
            for path in root.glob(pattern):
                maybe_add(path)
    else:
        for suffix in allowed:
            results.extend(root.rglob(f"*{suffix}"))
    unique_sorted = sorted({p.resolve() for p in results})
    return unique_sorted


def analyze_bssl_audio_files(
    audio_files: Iterable[Path | str],
    *,
    dataset_name: str = "dataset",
    target_sample_rate: Optional[int] = 22050,
    frames_per_second: float = 50.0,
    fft_size: int = 1024,
    hist_range: Tuple[float, float] = (0.0, 80.0),
    hist_bin_size: float = 0.5,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> BsslDatasetStats:
    """
    Compute BSSL Ntot curves for each audio file and aggregate statistics.
    """
    audio_paths = [Path(p) for p in audio_files]
    if hist_bin_size <= 0:
        raise ValueError("hist_bin_size must be positive")
    bin_edges = np.arange(hist_range[0], hist_range[1] + hist_bin_size, hist_bin_size, dtype=np.float64)
    if bin_edges.shape[0] < 2:
        raise ValueError("Histogram range/bin size produce insufficient bins")
    hist_counts = np.zeros(bin_edges.shape[0] - 1, dtype=np.int64)
    per_file_stats: List[BsslPerFileStats] = []
    errors: List[Tuple[Path, str]] = []

    iterator: Iterable[Path]
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError:
            iterator = audio_paths
        else:
            iterator = tqdm(audio_paths, desc=f"BSSL {dataset_name}", unit="file")
    else:
        iterator = audio_paths

    for audio_path in iterator:
        try:
            times, loudness, _ = compute_bssl_total_loudness(
                audio_path,
                target_sample_rate=target_sample_rate,
                frames_per_second=frames_per_second,
                fft_size=fft_size,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append((audio_path, str(exc)))
            continue
        if loudness.size == 0:
            continue
        per_file_stats.append(
            BsslPerFileStats(
                path=audio_path,
                mean=float(np.mean(loudness)),
                median=float(np.median(loudness)),
                std=float(np.std(loudness)),
                minimum=float(np.min(loudness)),
                maximum=float(np.max(loudness)),
            )
        )
        hist_counts += np.histogram(loudness, bins=bin_edges)[0]

    return BsslDatasetStats(
        dataset_name=dataset_name,
        audio_files=audio_paths,
        per_file_stats=per_file_stats,
        hist_counts=hist_counts,
        bin_edges=bin_edges,
        errors=errors,
    )


def plot_bssl_histogram(
    hist_counts: np.ndarray,
    bin_edges: np.ndarray,
    *,
    title: str,
    x_max: Optional[float] = None,
    output_path: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """
    Render histogram of BSSL loudness with dual y-axes (counts + percentage).
    """
    if bin_edges.shape[0] != hist_counts.shape[0] + 1:
        raise ValueError("bin_edges must have length hist_counts + 1")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    total = hist_counts.sum()
    percentages = (hist_counts / total * 100.0) if total > 0 else np.zeros_like(hist_counts, dtype=float)

    fig, ax_left = plt.subplots(figsize=(6, 2))
    ax_left.bar(centers, hist_counts, width=widths, align="center", color="tab:green", alpha=0.8, label="Total frames")
    ax_left.set_xlabel("BSSL total loudness (sones)")
    ax_left.set_ylabel("Frame count")

    ax_right = ax_left.twinx()
    ax_right.plot(centers, percentages, color="tab:red", linewidth=1.5, label="Percentage")
    ax_right.set_ylabel("Percentage (%)")
    if x_max is not None:
        ax_left.set_xlim(float(bin_edges[0]), float(x_max))

    ax_left.set_title(title)
    ax_left.grid(True, axis="y", alpha=0.25)

    handles, labels = [], []
    for ax in (ax_left, ax_right):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax_left.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig


def bssl_stats_to_dict(
    stats: BsslDatasetStats,
    *,
    config: Optional[dict] = None,
) -> dict:
    """
    Convert BsslDatasetStats into a JSON-serialisable dictionary.

    Args:
        stats: Aggregated dataset statistics.
        config: Optional configuration dict to embed (e.g., FPS, sample rate).
    """
    data = {
        "dataset": stats.dataset_name,
        "audio_file_count": len(stats.audio_files),
        "total_frames": stats.total_frames,
        "hist_counts": stats.hist_counts.tolist(),
        "bin_edges": stats.bin_edges.tolist(),
        "zero_bins": stats.zero_bins,
        "per_file_stats": [
            {
                "path": str(per.path),
                "mean": per.mean,
                "median": per.median,
                "std": per.std,
                "min": per.minimum,
                "max": per.maximum,
            }
            for per in stats.per_file_stats
        ],
        "errors": [(str(path), message) for path, message in stats.errors],
    }
    if config is not None:
        data["config"] = config
    return data


def _safe_dataset_slug(name: str) -> str:
    """Build a filesystem-safe dataset name token."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "dataset"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan one audio dataset, compute BSSL statistics, and generate outputs."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name used in logs and output file names.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help=(
            "Optional glob patterns relative to --root. "
            "Example: --patterns '**/*.wav' '**/*.flac'. "
            "If omitted, recursively scans .wav/.flac/.mp3."
        ),
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=22050,
        help="Target sample rate used for BSSL extraction.",
    )
    parser.add_argument(
        "--frames-per-second",
        type=float,
        default=50.0,
        help="Analysis frame rate in Hz.",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=1024,
        help="FFT size used by BSSL extraction.",
    )
    parser.add_argument(
        "--hist-min",
        type=float,
        default=0.0,
        help="Histogram minimum (sones).",
    )
    parser.add_argument(
        "--hist-max",
        type=float,
        default=80.0,
        help="Histogram maximum (sones).",
    )
    parser.add_argument(
        "--hist-bin-size",
        type=float,
        default=0.5,
        help="Histogram bin size (sones).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional x-axis maximum for the plotted histogram.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string, e.g. 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bar when available.",
    )
    parser.add_argument(
        "--figure-output-dir",
        type=Path,
        default=Path("figures/bssl_dataset_stats"),
        help="Directory to save histogram figure.",
    )
    parser.add_argument(
        "--stats-output-dir",
        type=Path,
        default=Path("stats/bssl"),
        help="Directory to save JSON stats.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure save DPI.",
    )
    parser.add_argument(
        "--print-errors",
        action="store_true",
        help="Print parsing errors for failed audio files.",
    )
    parser.add_argument(
        "--max-error-lines",
        type=int,
        default=20,
        help="Maximum number of error lines to print when --print-errors is enabled.",
    )
    return parser


def main(args: argparse.Namespace) -> int:
    audio_files = discover_audio_files(args.root, glob_patterns=args.patterns)
    print(f"Found {len(audio_files)} audio files under {args.root}")

    hist_range = (float(args.hist_min), float(args.hist_max))
    device = torch.device(args.device)
    stats = analyze_bssl_audio_files(
        audio_files,
        dataset_name=args.dataset,
        target_sample_rate=int(args.target_sample_rate),
        frames_per_second=float(args.frames_per_second),
        fft_size=int(args.fft_size),
        hist_range=hist_range,
        hist_bin_size=float(args.hist_bin_size),
        device=device,
        show_progress=bool(args.show_progress),
    )

    print(f"Scanned {args.dataset}: total frames = {stats.total_frames:,}")
    if stats.errors:
        print(f"Failed files: {len(stats.errors)}")
    print(f"Zero-count hist bins: {stats.zero_bins[:10]}... total {len(stats.zero_bins)} bins")
    per_file_means = [s.mean for s in stats.per_file_stats]
    if per_file_means:
        avg_mean = sum(per_file_means) / len(per_file_means)
        print(
            "Per-file loudness mean -> "
            f"min {min(per_file_means):.2f}, max {max(per_file_means):.2f}, avg {avg_mean:.2f}"
        )

    slug = _safe_dataset_slug(args.dataset)

    args.figure_output_dir.mkdir(parents=True, exist_ok=True)
    fig_output_path = args.figure_output_dir / f"{slug}_bssl.png"
    fig = plot_bssl_histogram(
        stats.hist_counts,
        stats.bin_edges,
        title=f"{args.dataset} BSSL Loudness Distribution",
        x_max=args.x_max,
        output_path=fig_output_path,
        dpi=int(args.dpi),
    )
    print(f"Saved: {fig_output_path}")
    plt.show()
    plt.close(fig)

    config = {
        "target_sample_rate": int(args.target_sample_rate),
        "frames_per_second": float(args.frames_per_second),
        "fft_size": int(args.fft_size),
        "hist_range": hist_range,
        "hist_bin_size": float(args.hist_bin_size),
        "device": args.device,
    }
    payload = bssl_stats_to_dict(stats, config=config)
    args.stats_output_dir.mkdir(parents=True, exist_ok=True)
    stats_output_path = args.stats_output_dir / f"{slug}_bssl.json"
    with stats_output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved stats to: {stats_output_path}")

    if args.print_errors and stats.errors:
        max_lines = max(0, int(args.max_error_lines))
        print("\nSample parsing errors:")
        for audio_path, message in stats.errors[:max_lines]:
            print(f"- {audio_path}: {message}")
        hidden = len(stats.errors) - min(len(stats.errors), max_lines)
        if hidden > 0:
            print(f"... ({hidden} more)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(build_arg_parser().parse_args()))
