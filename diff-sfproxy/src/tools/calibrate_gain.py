from __future__ import annotations

"""Gain calibration utility.

This script is optional. It can be used to pick a per-SF2 gain_db that places
rendered audio RMS in a reasonable range.

Phase-1 keeps this tool minimal.
"""

import argparse
from pathlib import Path

import numpy as np

from data.note_samplers import CoverageNoteSampler, CoverageSamplerConfig
from renderers.fluidsynth_sf2 import FluidSynthConfig, FluidSynthSF2Renderer


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf2", type=str, required=True)
    ap.add_argument("--bank", type=int, default=0)
    ap.add_argument("--program", type=int, default=0)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--seg_len", type=float, default=2.0)
    ap.add_argument("--gain_db", type=float, default=-6.0)
    ap.add_argument("--tries", type=int, default=16)
    args = ap.parse_args()

    sampler = CoverageNoteSampler(CoverageSamplerConfig(seg_len_s=args.seg_len))

    cfg = FluidSynthConfig(
        sf2_path=args.sf2,
        bank=args.bank,
        program=args.program,
        gain_db=args.gain_db,
    )
    r = FluidSynthSF2Renderer(cfg)

    rng = np.random.RandomState(0)
    rms_list = []
    for i in range(args.tries):
        # Use torch generator via sampler; here we just reuse a fixed seed.
        import torch

        g = torch.Generator(device="cpu")
        g.manual_seed(int(rng.randint(0, 2**31 - 1)))
        notes = sampler.sample(g)
        audio = r.render_segment(notes, sr=args.sr, seg_len_s=args.seg_len)
        rms_list.append(rms(audio))

    print(f"gain_db={args.gain_db} mean_rms={float(np.mean(rms_list)):.4f} std={float(np.std(rms_list)):.4f}")


if __name__ == "__main__":
    main()
