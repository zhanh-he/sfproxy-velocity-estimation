import math

import torch

from features.dynamics import DynamicsFeatureConfig, extract_note_features_padded


def test_dynamics_feature_energy_increases_with_amplitude():
    sr = 32000
    seg_len_s = 2.0
    t = torch.arange(int(sr * seg_len_s), dtype=torch.float32) / sr

    # Two A4 notes (440 Hz) with different amplitudes
    f0 = 440.0
    # note 1: 0.1-0.6s, amp=0.2
    m1 = (t >= 0.1) & (t < 0.6)
    # note 2: 1.0-1.5s, amp=0.4
    m2 = (t >= 1.0) & (t < 1.5)

    audio = torch.zeros_like(t)
    audio[m1] = 0.2 * torch.sin(2 * math.pi * f0 * t[m1])
    audio[m2] = 0.4 * torch.sin(2 * math.pi * f0 * t[m2])

    nmax = 2
    pitch = torch.tensor([69, 69], dtype=torch.long)
    cont = torch.tensor(
        [
            [0.1, 0.5, 0.3],
            [1.0, 0.5, 0.8],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([True, True])

    cfg = DynamicsFeatureConfig(n_fft=2048, hop=256, harmonic_count=3)
    feats, _ = extract_note_features_padded(
        audio=audio,
        pitch=pitch,
        cont=cont,
        mask=mask,
        sr=sr,
        seg_len_s=seg_len_s,
        cfg=cfg,
    )

    # harmonic energy is feature[0]
    assert feats[1, 0] > feats[0, 0]
