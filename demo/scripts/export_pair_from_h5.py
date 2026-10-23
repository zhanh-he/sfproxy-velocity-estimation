#!/usr/bin/env python3
"""Export one aligned WAV/MIDI pair from the external Score-HPT HDF5 cache.

This is useful when a research machine retains packed MAESTRO data but no
longer has the original WAV/MIDI files. The exported pair can be passed to
scripts/infer_compare.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="Packed Score-HPT HDF5 file")
    parser.add_argument("out", type=Path, help="Output basename without extension")
    args = parser.parse_args()

    import h5py
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(ROOT / "score_hpt/pytorch"))
    from direct_invension.common import compose_cfg
    from utilities import original_score_events, write_events_to_midi

    source = args.h5.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as cache:
        waveform = np.asarray(cache["waveform"], dtype=np.int16)
        times = np.asarray(cache["midi_event_time"], dtype=float)
        events = [e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in cache["midi_event"][:]]
    cfg = compose_cfg([], job_name="export_h5_pair")
    sample_rate = int(cfg.feature.sample_rate)
    duration = len(waveform) / sample_rate
    if len(waveform) == 0 or len(times) == 0:
        raise ValueError("Cache contains empty audio or MIDI events")
    note_events, pedal_events = original_score_events(cfg, times, events, duration)
    sf.write(str(out.with_suffix(".wav")), waveform, sample_rate, subtype="PCM_16")
    write_events_to_midi(0.0, note_events, pedal_events, str(out.with_suffix(".mid")))
    print(f"Exported {len(note_events)} notes and {duration:.2f} s audio to {out}.[wav|mid]")


if __name__ == "__main__":
    main()
