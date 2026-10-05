import torch

from data.note_samplers import (
    CoverageNoteSampler,
    CoverageSamplerConfig,
    MixedNoteSampler,
    RealismNoteSampler,
    make_sampler,
)


def _max_polyphony(notes):
    events = []
    for n in notes:
        events.append((n.onset_s, +1))
        events.append((n.onset_s + n.dur_s, -1))
    # Endpoints that only touch at the same timestamp should not count as
    # overlapping polyphony, so process note offsets before note onsets.
    events.sort(key=lambda x: (x[0], x[1]))
    active = 0
    max_active = 0
    for _, delta in events:
        active += delta
        max_active = max(max_active, active)
    return max_active


def test_coverage_sampler_constraints():
    cfg = CoverageSamplerConfig(
        seg_len_s=2.0,
        pitch_range=(60, 62),
        max_notes=128,
        polyphony_limit=4,
        chord_prob=0.5,
        max_chord_size=4,
        duration_range=(0.05, 0.4),
        ioi_range=(0.01, 0.1),
    )
    sampler = CoverageNoteSampler(cfg)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(123)

    notes = sampler.sample(rng)
    assert len(notes) > 0

    for n in notes:
        assert 60 <= n.pitch <= 62
        assert 0.0 <= n.onset_s <= cfg.seg_len_s
        assert n.dur_s > 0.0
        assert 0.0 <= n.velocity_01 <= 1.0
        assert n.onset_s + n.dur_s <= cfg.seg_len_s + 1e-6

    assert _max_polyphony(notes) <= cfg.polyphony_limit


def test_coverage_sampler_independent_chord_velocity_has_diversity():
    cfg = CoverageSamplerConfig(
        seg_len_s=2.0,
        pitch_range=(60, 72),
        max_notes=128,
        polyphony_limit=6,
        chord_prob=1.0,
        max_chord_size=4,
        duration_range=(0.1, 0.2),
        ioi_range=(0.15, 0.2),
        chord_velocity_mode="independent",
    )
    sampler = CoverageNoteSampler(cfg)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(7)

    notes = sampler.sample(rng)
    by_onset = {}
    for note in notes:
        by_onset.setdefault(round(note.onset_s, 4), []).append(note.velocity_01)

    multi_note_onsets = [vels for vels in by_onset.values() if len(vels) >= 2]
    assert multi_note_onsets, "expected at least one chord onset"
    assert any(len({round(v, 4) for v in vels}) >= 2 for vels in multi_note_onsets)


def test_make_sampler_mixed_builds_and_samples():
    sampler = make_sampler(
        {
            "type": "mixed",
            "components": {
                "boundary": {
                    "weight": 0.5,
                    "type": "coverage",
                    "seg_len_s": 2.0,
                    "pitch_range": [60, 72],
                    "max_notes": 32,
                    "polyphony_limit": 2,
                    "chord_prob": 0.1,
                    "max_chord_size": 2,
                    "duration_range": [0.05, 0.2],
                    "ioi_range": [0.05, 0.2],
                    "chord_velocity_mode": "independent",
                    "p_uniform": 0.15,
                    "p_near_boundary": 0.60,
                    "p_extreme": 0.25,
                },
                "coverage": {
                    "weight": 0.5,
                    "type": "coverage",
                    "seg_len_s": 2.0,
                    "pitch_range": [60, 72],
                    "max_notes": 32,
                    "polyphony_limit": 4,
                    "chord_prob": 0.5,
                    "max_chord_size": 4,
                    "duration_range": [0.05, 0.4],
                    "ioi_range": [0.01, 0.1],
                    "chord_velocity_mode": "mixed",
                },
            },
        }
    )
    assert isinstance(sampler, MixedNoteSampler)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(11)
    notes = sampler.sample(rng)
    assert notes
    assert all(0.0 <= n.velocity_01 <= 1.0 for n in notes)
