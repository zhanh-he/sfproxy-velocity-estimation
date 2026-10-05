import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

from losses import get_audio_loss_name
from proxy.common import (
    resolve_backend_segment_seconds,
    resolve_supervision_fft_size,
    resolve_supervision_frame_rate,
    resolve_supervision_hop_size,
    resolve_supervision_sample_rate,
)
from proxy.naming import normalize_backend_type


class BackendConfigUsageTest(unittest.TestCase):
    def test_backend_segment_resolution_reads_backend_namespace(self):
        cfg = OmegaConf.create(
            {
                "feature": {"segment_seconds": 10.0},
                "backend": {"backend_segment_seconds": 5.0},
            }
        )

        self.assertEqual(resolve_backend_segment_seconds(cfg), 5.0)

    def test_supervision_contract_reads_backend_namespace(self):
        cfg = OmegaConf.create(
            {
                "feature": {"sample_rate": 22050, "frames_per_second": 100.0, "fft_size": 2048},
                "backend": {
                    "supervision": {
                        "sample_rate": 16000,
                        "frame_rate": 80.0,
                        "fft_size": 1024,
                        "hop_size": 200,
                    }
                },
            }
        )

        self.assertEqual(resolve_supervision_sample_rate(cfg), 16000)
        self.assertEqual(resolve_supervision_frame_rate(cfg), 80.0)
        self.assertEqual(resolve_supervision_fft_size(cfg), 1024)
        self.assertEqual(resolve_supervision_hop_size(cfg), 200)

    def test_audio_loss_name_reads_backend_namespace(self):
        cfg = OmegaConf.create(
            {
                "backend": {
                    "audio_loss": {
                        "type": "piano_ssm_spectral_plus_log_rms",
                    }
                }
            }
        )

        self.assertEqual(get_audio_loss_name(cfg), "piano_ssm_spectral_plus_log_rms")

    def test_backend_type_only_accepts_canonical_names(self):
        self.assertEqual(normalize_backend_type("diffproxy"), "diffproxy")
        with self.assertRaises(ValueError):
            normalize_backend_type("sfproxy")


if __name__ == "__main__":
    unittest.main()
