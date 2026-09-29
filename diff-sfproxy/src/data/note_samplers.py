from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import torch

from renderers.base import NoteEvent


class BaseNoteSampler(Protocol):
    def sample(self, rng: torch.Generator) -> list[NoteEvent]:
        ...


def _rand_uniform(rng: torch.Generator, low: float, high: float) -> float:
    return float(torch.empty((), dtype=torch.float32).uniform_(low, high, generator=rng).item())


def _rand_int(rng: torch.Generator, low: int, high_inclusive: int) -> int:
    # torch.randint high is exclusive
    return int(torch.randint(low, high_inclusive + 1, (1,), generator=rng).item())


def _randn(rng: torch.Generator) -> float:
    return float(torch.randn((1,), generator=rng).item())


def _weighted_choice(rng: torch.Generator, values: Sequence, counts: Sequence) -> float | int:
    probs = torch.tensor(counts, dtype=torch.float32)
    index = int(torch.multinomial(probs, 1, generator=rng).item())
    return values[index]


def _sample_histogram(rng: torch.Generator, bin_edges: Sequence[float], hist_counts: Sequence[float]) -> float:
    index = int(_weighted_choice(rng, range(len(hist_counts)), hist_counts))
    return _rand_uniform(rng, float(bin_edges[index]), float(bin_edges[index + 1]))


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _normalize_boundary_list(values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    normalized: list[float] = []
    for raw in values:
        try:
            value = float(raw)
        except Exception:
            continue
        if value > 1.5:
            value = value / 127.0
        if 0.0 < value < 1.0:
            normalized.append(value)
    normalized = sorted(set(normalized))
    return tuple(float(v) for v in normalized)


class VelocityBoundaryProvider:
    def __init__(
        self,
        default_boundaries: Sequence[float],
        *,
        boundary_path: str = "",
        strategy: str = "global",
        register_pitch_splits: Sequence[int] = (48, 72),
    ):
        self.default_boundaries = _normalize_boundary_list(default_boundaries)
        self.global_boundaries = self.default_boundaries
        self.register_boundaries: dict[str, tuple[float, ...]] = {}
        self.strategy = str(strategy or "global").strip().lower()
        self.register_pitch_splits = tuple(int(v) for v in register_pitch_splits)

        path = str(boundary_path or "").strip()
        if not path:
            return

        boundary_file = Path(path)
        if not boundary_file.exists():
            return

        data = json.loads(boundary_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            parsed = _normalize_boundary_list(data)
            if parsed:
                self.global_boundaries = parsed
            return

        if not isinstance(data, dict):
            return

        parsed_global = _normalize_boundary_list(
            data.get("global_boundaries_01", data.get("boundaries_01", data.get("velocity_boundaries_01")))
        )
        if parsed_global:
            self.global_boundaries = parsed_global

        register_block = data.get("register_boundaries_01", {})
        if isinstance(register_block, dict):
            for key, values in register_block.items():
                parsed = _normalize_boundary_list(values)
                if parsed:
                    self.register_boundaries[str(key)] = parsed

    def _pitch_bucket_key(self, pitch: int) -> str:
        if len(self.register_pitch_splits) >= 2:
            low_hi, mid_hi = int(self.register_pitch_splits[0]), int(self.register_pitch_splits[1])
            if pitch < low_hi:
                return "low"
            if pitch < mid_hi:
                return "mid"
            return "high"
        if len(self.register_pitch_splits) == 1:
            return "low" if pitch < int(self.register_pitch_splits[0]) else "high"
        return "global"

    def boundaries_for_pitch(self, pitch: int) -> tuple[float, ...]:
        if self.strategy == "register" and self.register_boundaries:
            bucket_key = self._pitch_bucket_key(int(pitch))
            if bucket_key in self.register_boundaries:
                return self.register_boundaries[bucket_key]
        return self.global_boundaries or self.default_boundaries

    def boundaries_for_pitches(self, pitches: Sequence[int]) -> tuple[float, ...]:
        if not pitches:
            return self.global_boundaries or self.default_boundaries
        if self.strategy == "register" and self.register_boundaries:
            median_pitch = sorted(int(p) for p in pitches)[len(pitches) // 2]
            return self.boundaries_for_pitch(median_pitch)
        return self.global_boundaries or self.default_boundaries


def _sample_velocity_stratified(
    rng: torch.Generator,
    boundaries_01: Sequence[float],
    p_uniform: float = 0.5,
    p_near_boundary: float = 0.3,
    p_extreme: float = 0.2,
    boundary_width_01: float = 0.03,
) -> float:
    """Stratified velocity sampling in [0, 1]."""

    s = p_uniform + p_near_boundary + p_extreme
    if s <= 0:
        p_uniform, p_near_boundary, p_extreme = 1.0, 0.0, 0.0
    else:
        p_uniform, p_near_boundary, p_extreme = p_uniform / s, p_near_boundary / s, p_extreme / s

    u = _rand_uniform(rng, 0.0, 1.0)
    if u < p_uniform:
        return _clip01(_rand_uniform(rng, 0.0, 1.0))

    if u < p_uniform + p_near_boundary and len(boundaries_01) > 0:
        b = float(boundaries_01[_rand_int(rng, 0, len(boundaries_01) - 1)])
        low = max(0.0, b - boundary_width_01)
        high = min(1.0, b + boundary_width_01)
        return _clip01(_rand_uniform(rng, low, high))

    if _rand_uniform(rng, 0.0, 1.0) < 0.5:
        return _clip01(_rand_uniform(rng, 0.0, boundary_width_01))
    return _clip01(_rand_uniform(rng, 1.0 - boundary_width_01, 1.0))


@dataclass
class VelocitySamplingConfig:
    velocity_boundaries_01: tuple[float, ...] = (0.33, 0.66)
    velocity_boundary_path: str = ""
    velocity_boundary_strategy: str = "global"  # global | register
    register_pitch_splits: tuple[int, int] = (48, 72)

    p_uniform: float = 0.5
    p_near_boundary: float = 0.3
    p_extreme: float = 0.2
    boundary_width_01: float = 0.03

    chord_velocity_mode: str = "shared"  # shared | independent | correlated | mixed
    p_chord_velocity_shared: float = 0.10
    p_chord_velocity_independent: float = 0.70
    p_chord_velocity_correlated: float = 0.20
    correlated_jitter_01: float = 0.05


@dataclass
class CoverageSamplerConfig(VelocitySamplingConfig):
    seg_len_s: float = 2.0
    pitch_range: tuple[int, int] = (21, 108)
    max_notes: int = 64
    polyphony_limit: int = 16

    chord_prob: float = 0.25
    max_chord_size: int = 6

    duration_range: tuple[float, float] = (0.08, 0.8)
    ioi_range: tuple[float, float] = (0.03, 0.35)


@dataclass
class RealismSamplerConfig(VelocitySamplingConfig):
    seg_len_s: float = 2.0
    max_notes: int = 64
    polyphony_limit: int = 16

    stats_path: str = ""
    use_velocity_stats: bool = False


class _VelocityAwareSampler:
    def __init__(self, cfg: VelocitySamplingConfig):
        self.cfg = cfg
        self.boundary_provider = VelocityBoundaryProvider(
            cfg.velocity_boundaries_01,
            boundary_path=getattr(cfg, "velocity_boundary_path", ""),
            strategy=getattr(cfg, "velocity_boundary_strategy", "global"),
            register_pitch_splits=getattr(cfg, "register_pitch_splits", (48, 72)),
        )

    def _sample_scalar_velocity(self, rng: torch.Generator, boundaries_01: Sequence[float]) -> float:
        return _sample_velocity_stratified(
            rng,
            boundaries_01,
            p_uniform=float(self.cfg.p_uniform),
            p_near_boundary=float(self.cfg.p_near_boundary),
            p_extreme=float(self.cfg.p_extreme),
            boundary_width_01=float(self.cfg.boundary_width_01),
        )

    def _sample_scalar_velocity_any(self, rng: torch.Generator, pitch: int | None = None) -> float:
        boundaries = self.boundary_provider.boundaries_for_pitch(int(pitch)) if pitch is not None else self.boundary_provider.global_boundaries
        return self._sample_scalar_velocity(rng, boundaries)

    def _resolve_chord_velocity_mode(self, rng: torch.Generator) -> str:
        mode = str(getattr(self.cfg, "chord_velocity_mode", "shared") or "shared").strip().lower()
        if mode != "mixed":
            return mode

        weights = [
            max(0.0, float(getattr(self.cfg, "p_chord_velocity_shared", 0.0))),
            max(0.0, float(getattr(self.cfg, "p_chord_velocity_independent", 0.0))),
            max(0.0, float(getattr(self.cfg, "p_chord_velocity_correlated", 0.0))),
        ]
        modes = ["shared", "independent", "correlated"]
        if sum(weights) <= 0:
            return "independent"
        return str(_weighted_choice(rng, modes, weights))

    def _sample_chord_velocities(self, rng: torch.Generator, chord_pitches: Sequence[int]) -> list[float]:
        if not chord_pitches:
            return []
        if len(chord_pitches) == 1:
            return [self._sample_scalar_velocity_any(rng, chord_pitches[0])]

        mode = self._resolve_chord_velocity_mode(rng)
        if mode == "shared":
            v = self._sample_scalar_velocity(rng, self.boundary_provider.boundaries_for_pitches(chord_pitches))
            return [float(v) for _ in chord_pitches]

        if mode == "correlated":
            base = self._sample_scalar_velocity(rng, self.boundary_provider.boundaries_for_pitches(chord_pitches))
            jitter = max(0.0, float(getattr(self.cfg, "correlated_jitter_01", 0.0) or 0.0))
            return [_clip01(base + _randn(rng) * jitter) for _ in chord_pitches]

        return [self._sample_scalar_velocity_any(rng, p) for p in chord_pitches]


class CoverageNoteSampler(_VelocityAwareSampler):
    def __init__(self, cfg: CoverageSamplerConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def sample(self, rng: torch.Generator) -> list[NoteEvent]:
        seg_len = float(self.cfg.seg_len_s)
        pmin, pmax = int(self.cfg.pitch_range[0]), int(self.cfg.pitch_range[1])
        max_notes = int(self.cfg.max_notes)
        poly_lim = int(self.cfg.polyphony_limit)

        t = 0.0
        notes: list[NoteEvent] = []
        active_end_times: list[float] = []

        while len(notes) < max_notes and t < seg_len:
            active_end_times = [e for e in active_end_times if e > t]
            active = len(active_end_times)

            chord = 1
            if poly_lim > 1 and _rand_uniform(rng, 0.0, 1.0) < float(self.cfg.chord_prob):
                chord = _rand_int(rng, 2, max(2, int(self.cfg.max_chord_size)))
            chord = min(chord, poly_lim - active, max_notes - len(notes))
            if chord <= 0:
                if not active_end_times:
                    break
                t = min(active_end_times)
                continue

            dur = _rand_uniform(rng, float(self.cfg.duration_range[0]), float(self.cfg.duration_range[1]))
            dur = max(0.01, min(dur, seg_len - t))
            end_t = t + dur

            chord_pitches: set[int] = set()
            while len(chord_pitches) < chord:
                chord_pitches.add(_rand_int(rng, pmin, pmax))
            ordered_pitches = sorted(chord_pitches)
            chord_velocities = self._sample_chord_velocities(rng, ordered_pitches)

            for p, vel in zip(ordered_pitches, chord_velocities):
                notes.append(NoteEvent(pitch=int(p), onset_s=float(t), dur_s=float(dur), velocity_01=float(vel)))
                active_end_times.append(end_t)

            ioi = _rand_uniform(rng, float(self.cfg.ioi_range[0]), float(self.cfg.ioi_range[1]))
            t = t + max(0.0, ioi)

        notes.sort(key=lambda n: (n.onset_s, n.pitch))
        return notes


class RealismNoteSampler(_VelocityAwareSampler):
    def __init__(self, cfg: RealismSamplerConfig):
        super().__init__(cfg)
        self.cfg = cfg
        stats = json.loads(Path(cfg.stats_path).read_text(encoding="utf-8"))
        self.pitch_values = stats["pitches"]["values"]
        self.pitch_counts = stats["pitches"]["counts"]
        self.duration_edges = stats["durations"]["bin_edges"]
        self.duration_counts = stats["durations"]["hist_counts"]
        self.ioi_edges = stats["iois"]["bin_edges"]
        self.ioi_counts = stats["iois"]["hist_counts"]
        self.chord_values = stats["chord_sizes"]["values"]
        self.chord_counts = stats["chord_sizes"]["counts"]
        self.velocity_values = stats.get("velocities_01", {}).get("values", [])
        self.velocity_counts = stats.get("velocities_01", {}).get("counts", [])

    def _sample_scalar_velocity(self, rng: torch.Generator, boundaries_01: Sequence[float]) -> float:
        if bool(self.cfg.use_velocity_stats) and self.velocity_values and self.velocity_counts:
            return _clip01(float(_weighted_choice(rng, self.velocity_values, self.velocity_counts)))
        return super()._sample_scalar_velocity(rng, boundaries_01)

    def sample(self, rng: torch.Generator) -> list[NoteEvent]:
        seg_len = float(self.cfg.seg_len_s)
        max_notes = int(self.cfg.max_notes)
        poly_lim = int(self.cfg.polyphony_limit)

        t = 0.0
        notes: list[NoteEvent] = []
        active_end_times: list[float] = []

        while len(notes) < max_notes and t < seg_len:
            active_end_times = [e for e in active_end_times if e > t]
            active = len(active_end_times)
            chord = max(1, int(_weighted_choice(rng, self.chord_values, self.chord_counts)))
            chord = min(chord, poly_lim - active, max_notes - len(notes), len(self.pitch_values))
            if chord <= 0:
                if not active_end_times:
                    break
                t = min(active_end_times)
                continue

            dur = max(0.01, min(_sample_histogram(rng, self.duration_edges, self.duration_counts), seg_len - t))
            chord_pitches: set[int] = set()
            while len(chord_pitches) < chord:
                chord_pitches.add(int(_weighted_choice(rng, self.pitch_values, self.pitch_counts)))
            ordered_pitches = sorted(chord_pitches)
            chord_velocities = self._sample_chord_velocities(rng, ordered_pitches)

            end_t = t + dur
            for p, vel in zip(ordered_pitches, chord_velocities):
                notes.append(NoteEvent(pitch=int(p), onset_s=float(t), dur_s=float(dur), velocity_01=float(vel)))
                active_end_times.append(end_t)

            t = t + max(0.0, _sample_histogram(rng, self.ioi_edges, self.ioi_counts))

        notes.sort(key=lambda n: (n.onset_s, n.pitch))
        return notes


class MixedNoteSampler:
    def __init__(self, components: Sequence[tuple[str, float, BaseNoteSampler]]):
        filtered = [(name, float(weight), sampler) for name, weight, sampler in components if float(weight) > 0]
        if not filtered:
            raise ValueError("MixedNoteSampler requires at least one component with positive weight")
        self.names = [name for name, _, _ in filtered]
        self.weights = [weight for _, weight, _ in filtered]
        self.samplers = [sampler for _, _, sampler in filtered]

    def sample(self, rng: torch.Generator) -> list[NoteEvent]:
        sampler = self.samplers[int(_weighted_choice(rng, range(len(self.samplers)), self.weights))]
        return sampler.sample(rng)


def _make_mixed_sampler(cfg: Mapping) -> BaseNoteSampler:
    components_cfg = cfg.get("components", {})
    if not isinstance(components_cfg, Mapping):
        raise ValueError("mixed sampler expects a mapping under 'components'")

    components: list[tuple[str, float, BaseNoteSampler]] = []
    for name, component_cfg in components_cfg.items():
        if not isinstance(component_cfg, Mapping):
            continue
        child_cfg = dict(component_cfg)
        weight = float(child_cfg.pop("weight", child_cfg.pop("prob", 0.0)) or 0.0)
        if weight <= 0:
            continue
        components.append((str(name), weight, make_sampler(child_cfg)))

    return MixedNoteSampler(components)


def make_sampler(cfg: dict) -> BaseNoteSampler:
    sampler_type = str(cfg.get("type", "coverage") or "coverage").strip().lower()
    cfg_no_type = dict(cfg)
    cfg_no_type.pop("type", None)

    if sampler_type == "coverage":
        return CoverageNoteSampler(CoverageSamplerConfig(**cfg_no_type))
    if sampler_type == "realism":
        return RealismNoteSampler(RealismSamplerConfig(**cfg_no_type))
    if sampler_type == "mixed":
        return _make_mixed_sampler(cfg_no_type)
    raise ValueError(f"Unknown sampler type: {sampler_type}")
