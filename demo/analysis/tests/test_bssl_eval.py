import math
import unittest
from unittest import mock

import numpy as np
import torch

from data_analysis.evaluation import bssl_eval


class _FakeExtractor:
    def __init__(self, *args, **kwargs):
        self.sample_rate = 22050
        self.hop_size = 22050

    def to(self, device):
        return self

    def __call__(self, waveform):
        marker = float(waveform.reshape(-1)[0].item())
        if marker == 1.0:
            values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
        else:
            values = torch.tensor([[[4.0, 5.0, 7.0, 9.0]]], dtype=torch.float32)
        return values


def _fake_load_audio_mono(wav_path, *, target_sample_rate, device):
    marker = 1.0 if "pred" in str(wav_path) else 2.0
    waveform = torch.tensor([[marker]], dtype=torch.float32, device=device)
    return waveform, int(target_sample_rate)


class EvaluateBsslPairTests(unittest.TestCase):
    @mock.patch.object(bssl_eval, "_load_audio_mono", side_effect=_fake_load_audio_mono)
    @mock.patch.object(bssl_eval, "PsychoFeatureExtractor", _FakeExtractor)
    def test_evaluate_bssl_pair_reports_raw_ntot_cosine_in_summary(self, _load_audio_mono_mock):
        result = bssl_eval.evaluate_bssl_pair(
            pred_wav="pred.wav",
            gt_wav="gt.wav",
            sample_rate=22050,
            num_samples=4,
            normalization="zscore",
        )

        pred = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        gt = np.array([4.0, 5.0, 7.0, 9.0], dtype=np.float64)
        raw_cosine = float(np.dot(pred, gt) / (np.linalg.norm(pred) * np.linalg.norm(gt)))
        pearson = float(np.corrcoef(pred, gt)[0, 1])

        self.assertFalse(math.isclose(raw_cosine, pearson))
        self.assertAlmostEqual(result["summary"]["ntot_cosine_sim"], raw_cosine)
        self.assertAlmostEqual(result["summary"]["ntot_cosine_sim_raw"], raw_cosine)
        self.assertAlmostEqual(result["summary"]["ntot_cosine_sim_normalized"], pearson)


if __name__ == "__main__":
    unittest.main()
