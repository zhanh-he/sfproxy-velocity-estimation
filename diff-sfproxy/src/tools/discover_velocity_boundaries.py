from __future__ import annotations

"""Discover velocity-response boundaries from rendered single-note sweeps.

This utility renders single notes for velocities 1..127, extracts the note-wise
DiffProxy target features, and identifies the largest changes in the feature-vs-
velocity curve. The resulting JSON can be consumed by the sampler through
`velocity_boundary_path`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from features.dynamics import DynamicsFeatureConfig, extract_note_features_padded
from data.datasets_online import InstrumentSpec
from renderers.base import NoteEvent
from renderers.fluidsynth_sf2 import FluidSynthConfig, FluidSynthSF2Renderer
from renderers.sfizz_sfz import SfizzConfig, SfizzSFZRenderer


def _moving_average(x: torch.Tensor, width: int) -> torch.Tensor:
    width = max(1, int(width))
    if width <= 1 or x.numel() == 0:
        return x.clone()
    pad = width // 2
    kernel = torch.ones((1, 1, width), dtype=torch.float32) / float(width)
    data = x.view(1, 1, -1)
    data = torch.nn.functional.pad(data, (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(data, kernel).view(-1)[: x.numel()]


def _pitch_bucket(pitch: int, splits: Sequence[int]) -> str:
    splits = [int(v) for v in splits]
    if len(splits) >= 2:
        if pitch < splits[0]:
            return "low"
        if pitch < splits[1]:
            return "mid"
        return "high"
    if len(splits) == 1:
        return "low" if pitch < splits[0] else "high"
    return "global"


def _select_top_boundaries(score_curve: torch.Tensor, top_k: int, min_spacing: int) -> List[float]:
    if score_curve.numel() == 0 or top_k <= 0:
        return []

    order = torch.argsort(score_curve, descending=True).tolist()
    selected: List[int] = []
    for idx in order:
        idx = int(idx)
        if score_curve[idx].item() <= 0:
            continue
        if any(abs(idx - chosen) < int(min_spacing) for chosen in selected):
            continue
        selected.append(idx)
        if len(selected) >= int(top_k):
            break

    selected = sorted(selected)
    # score_curve[i] represents the jump between MIDI velocities (i+1) and (i+2).
    # Use the midpoint as the sampled boundary and normalize into [0, 1].
    return [round(((idx + 1) + 0.5) / 127.0, 6) for idx in selected]


def _make_renderer(instrument: InstrumentSpec):
    backend = instrument.render_backend
    if backend == "fluidsynth":
        return FluidSynthSF2Renderer(
            FluidSynthConfig(
                sf2_path=str(instrument.instrument_path),
                bank=int(instrument.bank),
                program=int(instrument.program),
                polyphony=int(instrument.polyphony),
                gain_db=float(instrument.gain_db),
                disable_reverb=True,
                disable_chorus=True,
                allow_cli_fallback=True,
            )
        )
    if backend == "sfizz":
        return SfizzSFZRenderer(
            SfizzConfig(
                sfz_path=str(instrument.instrument_path),
                block_size=int(instrument.sfizz_block_size),
                polyphony=int(instrument.polyphony),
                quality=int(instrument.sfizz_quality),
                use_eot=bool(instrument.sfizz_use_eot),
            )
        )
    raise ValueError(f"Unsupported instrument backend: {backend}")


@torch.no_grad()
def _extract_single_note_curve(
    renderer,
    feature_cfg: DynamicsFeatureConfig,
    *,
    pitch: int,
    sr: int,
    seg_len_s: float,
    onset_s: float,
    dur_s: float,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    pitch_tensor = torch.tensor([int(pitch)], dtype=torch.long)
    mask_tensor = torch.tensor([True], dtype=torch.bool)

    for velocity in range(1, 128):
        velocity_01 = float(velocity / 127.0)
        note = NoteEvent(pitch=int(pitch), onset_s=float(onset_s), dur_s=float(dur_s), velocity_01=velocity_01)
        audio_np = renderer.render_segment([note], sr=int(sr), seg_len_s=float(seg_len_s))
        audio = torch.from_numpy(audio_np).to(torch.float32)
        cont = torch.tensor([[float(onset_s), float(dur_s), float(velocity_01)]], dtype=torch.float32)
        feat, _ = extract_note_features_padded(
            audio=audio,
            pitch=pitch_tensor,
            cont=cont,
            mask=mask_tensor,
            sr=int(sr),
            seg_len_s=float(seg_len_s),
            cfg=feature_cfg,
        )
        feats.append(feat[0].cpu())
    return torch.stack(feats, dim=0)  # (127, D)


@torch.no_grad()
def discover_velocity_boundaries(
    *,
    instrument: InstrumentSpec,
    bank: int,
    program: int,
    polyphony: int,
    gain_db: float,
    sr: int,
    seg_len_s: float,
    pitches: Iterable[int],
    register_splits: Sequence[int],
    onset_s: float,
    dur_s: float,
    top_k: int,
    min_spacing: int,
    smooth_width: int,
    feature_cfg: DynamicsFeatureConfig,
) -> Dict:
    renderer = _make_renderer(instrument)

    per_bucket_scores: Dict[str, List[torch.Tensor]] = {"global": []}
    per_pitch_summary: List[Dict] = []

    for pitch in [int(p) for p in pitches]:
        curve = _extract_single_note_curve(
            renderer,
            feature_cfg,
            pitch=int(pitch),
            sr=int(sr),
            seg_len_s=float(seg_len_s),
            onset_s=float(onset_s),
            dur_s=float(dur_s),
        )
        diffs = torch.linalg.norm(curve[1:] - curve[:-1], ord=2, dim=1)
        diffs = _moving_average(diffs, int(smooth_width)).cpu()
        bucket = _pitch_bucket(int(pitch), register_splits)
        per_bucket_scores.setdefault(bucket, []).append(diffs)
        per_bucket_scores["global"].append(diffs)
        per_pitch_summary.append(
            {
                "pitch": int(pitch),
                "bucket": bucket,
                "mean_adjacent_delta": float(diffs.mean().item()),
                "max_adjacent_delta": float(diffs.max().item()),
            }
        )

    out: Dict = {
        "meta": {
            "instrument_path": str(instrument.instrument_path),
            "instrument_backend": str(instrument.render_backend),
            "bank": int(bank),
            "program": int(program),
            "polyphony": int(polyphony),
            "gain_db": float(gain_db),
            "sr": int(sr),
            "seg_len_s": float(seg_len_s),
            "onset_s": float(onset_s),
            "dur_s": float(dur_s),
            "pitches": [int(p) for p in pitches],
            "register_pitch_splits": [int(v) for v in register_splits],
            "top_k": int(top_k),
            "min_spacing": int(min_spacing),
            "smooth_width": int(smooth_width),
            "feature_cfg": {
                "n_fft": int(feature_cfg.n_fft),
                "hop": int(feature_cfg.hop),
                "win_length": None if feature_cfg.win_length is None else int(feature_cfg.win_length),
                "harmonic_count": int(feature_cfg.harmonic_count),
                "band_bins": int(feature_cfg.band_bins),
                "energy_window_s": float(feature_cfg.energy_window_s),
                "onset_pre_s": float(feature_cfg.onset_pre_s),
                "onset_post_s": float(feature_cfg.onset_post_s),
                "include_f0_ratio": bool(feature_cfg.include_f0_ratio),
            },
        },
        "per_pitch_summary": per_pitch_summary,
        "global_boundaries_01": [],
        "global_scores": [],
        "register_boundaries_01": {},
        "register_scores": {},
    }

    for bucket, curves in per_bucket_scores.items():
        if not curves:
            continue
        stacked = torch.stack(curves, dim=0)
        mean_curve = stacked.mean(dim=0)
        boundaries = _select_top_boundaries(mean_curve, int(top_k), int(min_spacing))
        score_list = [round(float(v), 6) for v in mean_curve.tolist()]
        if bucket == "global":
            out["global_boundaries_01"] = boundaries
            out["global_scores"] = score_list
        else:
            out["register_boundaries_01"][bucket] = boundaries
            out["register_scores"][bucket] = score_list

    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover velocity boundaries for SFProxy sampling.")
    parser.add_argument("--instrument_path", default=None, help="Path to the source instrument file (.sf2 or .sfz)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "fluidsynth", "sfizz"])
    parser.add_argument("--instrument_name", default="instrument", help="Optional instrument name for metadata only")
    parser.add_argument("--bank", type=int, default=0)
    parser.add_argument("--program", type=int, default=0)
    parser.add_argument("--polyphony", type=int, default=64)
    parser.add_argument("--gain_db", type=float, default=-6.0)
    parser.add_argument("--sfizz_block_size", type=int, default=1024)
    parser.add_argument("--sfizz_quality", type=int, default=3)
    parser.add_argument("--sfizz_use_eot", action="store_true")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--seg_len_s", type=float, default=2.0)
    parser.add_argument("--pitch_min", type=int, default=21)
    parser.add_argument("--pitch_max", type=int, default=108)
    parser.add_argument("--pitch_step", type=int, default=6)
    parser.add_argument("--register_splits", nargs="*", type=int, default=[48, 72])
    parser.add_argument("--onset_s", type=float, default=0.20)
    parser.add_argument("--dur_s", type=float, default=0.60)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_spacing", type=int, default=6, help="Minimum distance between discovered boundaries in MIDI-velocity steps")
    parser.add_argument("--smooth_width", type=int, default=5)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=221)
    parser.add_argument("--win_length", type=int, default=None)
    parser.add_argument("--harmonic_count", type=int, default=5)
    parser.add_argument("--band_bins", type=int, default=1)
    parser.add_argument("--energy_window_s", type=float, default=0.12)
    parser.add_argument("--onset_pre_s", type=float, default=0.02)
    parser.add_argument("--onset_post_s", type=float, default=0.08)
    parser.add_argument("--include_f0_ratio", action="store_true")
    parser.add_argument("--out_json", required=True, help="Where to save the discovered boundaries JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    instrument_path = str(args.instrument_path or "").strip()
    if not instrument_path:
        raise ValueError("Set --instrument_path to an .sf2 or .sfz file")
    pitches = list(range(int(args.pitch_min), int(args.pitch_max) + 1, max(1, int(args.pitch_step))))
    feature_cfg = DynamicsFeatureConfig(
        n_fft=int(args.n_fft),
        hop=int(args.hop),
        win_length=args.win_length,
        harmonic_count=int(args.harmonic_count),
        band_bins=int(args.band_bins),
        energy_window_s=float(args.energy_window_s),
        onset_pre_s=float(args.onset_pre_s),
        onset_post_s=float(args.onset_post_s),
        include_f0_ratio=bool(args.include_f0_ratio),
    )
    instrument = InstrumentSpec(
        name=str(args.instrument_name),
        path=str(instrument_path),
        backend=str(args.backend),
        bank=int(args.bank),
        program=int(args.program),
        polyphony=int(args.polyphony),
        gain_db=float(args.gain_db),
        sfizz_block_size=int(args.sfizz_block_size),
        sfizz_quality=int(args.sfizz_quality),
        sfizz_use_eot=bool(args.sfizz_use_eot),
        sr=int(args.sr),
        seg_len_s=float(args.seg_len_s),
    )
    result = discover_velocity_boundaries(
        instrument=instrument,
        bank=int(args.bank),
        program=int(args.program),
        polyphony=int(args.polyphony),
        gain_db=float(args.gain_db),
        sr=int(args.sr),
        seg_len_s=float(args.seg_len_s),
        pitches=pitches,
        register_splits=args.register_splits,
        onset_s=float(args.onset_s),
        dur_s=float(args.dur_s),
        top_k=int(args.top_k),
        min_spacing=int(args.min_spacing),
        smooth_width=int(args.smooth_width),
        feature_cfg=feature_cfg,
    )
    result["meta"]["instrument_name"] = str(args.instrument_name)
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved discovered boundaries to {out_path}")
    print(json.dumps({
        "instrument_name": result["meta"].get("instrument_name"),
        "global_boundaries_01": result.get("global_boundaries_01", []),
        "register_boundaries_01": result.get("register_boundaries_01", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
