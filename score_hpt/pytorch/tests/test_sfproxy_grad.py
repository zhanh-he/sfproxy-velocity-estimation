import sys
import unittest
from pathlib import Path

import torch


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

from proxy.sfproxy import SFProxyObjective


class SFProxyGradientTest(unittest.TestCase):
    def _make_objective_stub(self):
        obj = object.__new__(SFProxyObjective)
        obj.device = torch.device("cpu")
        obj.src_frames_per_second = 100.0
        obj.min_duration_frames = 1
        obj.begin_note = 21
        obj.max_notes = 512
        obj.onset_threshold = 0.5
        obj.frame_threshold = 0.5
        obj.loss_type = "smooth_l1"
        obj.loss_beta = 1.0
        obj.feature_weights = None
        obj.use_onset_window_pooling = False
        return obj

    def test_note_events_path_preserves_velocity_gradient(self):
        obj = self._make_objective_stub()
        vel_pred = torch.zeros((200, 88), requires_grad=True)
        note_events = [
            {"midi_note": 60, "onset_time": 0.12, "offset_time": 0.40, "velocity": 64},
            {"midi_note": 64, "onset_time": 0.50, "offset_time": 0.80, "velocity": 80},
        ]

        _, _, cont_norm, _, length = SFProxyObjective._extract_note_list_from_events(
            obj,
            note_events=note_events,
            vel_pred=vel_pred,
            segment_seconds=2.0,
        )

        self.assertEqual(length, 2)
        self.assertTrue(cont_norm.requires_grad)

        loss = cont_norm[:, 2].sum()
        loss.backward()

        self.assertIsNotNone(vel_pred.grad)
        self.assertGreater(float(vel_pred.grad.abs().sum().item()), 0.0)

    def test_masked_loss_ignores_nonfinite_entries(self):
        obj = self._make_objective_stub()
        pred = torch.tensor([[[1.0, float("nan")], [2.0, 3.0]]], requires_grad=True)
        target = torch.tensor([[[0.0, 5.0], [1.0, float("nan")]]])
        mask = torch.tensor([[True, True]])

        loss, mae = SFProxyObjective._masked_loss(obj, pred, target, mask)

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(mae))
        self.assertAlmostEqual(float(loss.item()), 0.5, places=6)
        self.assertAlmostEqual(float(mae.item()), 1.0, places=6)

        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_roll_fallback_preserves_velocity_gradient(self):
        obj = self._make_objective_stub()
        vel_pred = torch.zeros((100, 88), requires_grad=True)
        onset_roll = torch.zeros((100, 88))
        frame_roll = torch.zeros((100, 88))
        onset_roll[10, 39] = 1.0
        frame_roll[10:20, 39] = 1.0

        _, _, cont_norm, _, length = SFProxyObjective._extract_note_list(
            obj,
            onset_roll=onset_roll,
            frame_roll=frame_roll,
            vel_pred=vel_pred,
            segment_seconds=1.0,
        )

        self.assertEqual(length, 1)
        self.assertTrue(cont_norm.requires_grad)

        loss = cont_norm[:, 2].sum()
        loss.backward()

        self.assertIsNotNone(vel_pred.grad)
        self.assertGreater(float(vel_pred.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
