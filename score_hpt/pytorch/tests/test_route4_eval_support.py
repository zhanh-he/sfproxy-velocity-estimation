import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

from direct_invension.common import resolve_dataset_split
from direct_invension.eval_framework import VelocityAlignmentResult, evaluate_prediction_item


def _fake_eval_payload():
    return {
        "bssl": {
            "flattened_raw_metrics": {
                "pearson_correlation": 0.11,
                "mean_absolute_error": 0.22,
                "cosine_sim": 0.33,
                "spearman_correlation": 0.44,
            }
        },
        "ntot": {
            "curve_metrics_raw": {
                "pearson_correlation": 0.55,
                "mean_absolute_error": 0.66,
                "cosine_sim": 0.77,
                "spearman_correlation": 0.88,
            }
        },
    }


class Route4EvalSupportTest(unittest.TestCase):
    def test_resolve_dataset_split_maps_full_to_all(self):
        self.assertEqual(resolve_dataset_split("full"), "all")
        self.assertEqual(resolve_dataset_split("test"), "test")
        self.assertEqual(resolve_dataset_split("all"), "all")
        self.assertEqual(resolve_dataset_split("valid"), "validation")

    def test_evaluate_prediction_item_skips_velocity_alignment_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gt_midi = root / "gt.mid"
            pred_midi = root / "pred.mid"
            real_audio = root / "real.wav"
            instrument = root / "instrument.sf2"
            for path in (gt_midi, pred_midi, real_audio, instrument):
                path.touch()

            item = {
                "key": "demo",
                "label": "route4",
                "gt_midi": str(gt_midi),
                "pred_midi": str(pred_midi),
                "real_audio": str(real_audio),
            }

            with patch(
                "direct_invension.eval_framework._import_bssl_eval",
                return_value=lambda **kwargs: _fake_eval_payload(),
            ), patch(
                "direct_invension.eval_framework.render_midi_to_audio",
                return_value={"status": "rendered"},
            ), patch(
                "direct_invension.eval_framework.align_note_velocities",
                side_effect=AssertionError("align_note_velocities should not run"),
            ):
                result = evaluate_prediction_item(
                    item,
                    instrument_path=instrument,
                    out_dir=root / "out",
                    compute_velocity_mae=False,
                )

        self.assertIsNone(result["velocity"])
        self.assertNotIn("velocity_mae", result["summary"])
        self.assertAlmostEqual(result["summary"]["synth_ref_bssl_pearson_correlation"], 0.11)
        self.assertAlmostEqual(result["summary"]["real_ref_bstl_cosine_similarity"], 0.77)

    def test_evaluate_prediction_item_computes_velocity_alignment_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gt_midi = root / "gt.mid"
            pred_midi = root / "pred.mid"
            instrument = root / "instrument.sf2"
            for path in (gt_midi, pred_midi, instrument):
                path.touch()

            item = {
                "key": "demo",
                "label": "route4",
                "gt_midi": str(gt_midi),
                "pred_midi": str(pred_midi),
            }
            alignment = VelocityAlignmentResult(
                num_gt_notes=10,
                num_pred_notes=10,
                num_matched_notes=10,
                mae=12.5,
                matched_in_exact_order=True,
                unmatched_gt=0,
                unmatched_pred=0,
            )

            with patch(
                "direct_invension.eval_framework._import_bssl_eval",
                return_value=lambda **kwargs: _fake_eval_payload(),
            ), patch(
                "direct_invension.eval_framework.render_midi_to_audio",
                return_value={"status": "rendered"},
            ), patch(
                "direct_invension.eval_framework.align_note_velocities",
                return_value=alignment,
            ):
                result = evaluate_prediction_item(
                    item,
                    instrument_path=instrument,
                    out_dir=root / "out",
                    compute_velocity_mae=True,
                )

        self.assertEqual(result["velocity"]["num_gt_notes"], 10)
        self.assertAlmostEqual(result["summary"]["velocity_mae"], 12.5)
        self.assertEqual(result["summary"]["num_matched_notes"], 10)

    def test_evaluate_prediction_item_updates_score_progress(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gt_midi = root / "gt.mid"
            pred_midi = root / "pred.mid"
            real_audio = root / "real.wav"
            instrument = root / "instrument.sf2"
            for path in (gt_midi, pred_midi, real_audio, instrument):
                path.touch()

            item = {
                "key": "demo",
                "label": "route4",
                "gt_midi": str(gt_midi),
                "pred_midi": str(pred_midi),
                "real_audio": str(real_audio),
            }
            progress = unittest.mock.Mock()

            with patch(
                "direct_invension.eval_framework._import_bssl_eval",
                return_value=lambda **kwargs: _fake_eval_payload(),
            ), patch(
                "direct_invension.eval_framework.render_midi_to_audio",
                return_value={"status": "rendered"},
            ):
                evaluate_prediction_item(
                    item,
                    instrument_path=instrument,
                    out_dir=root / "out",
                    compute_velocity_mae=False,
                    score_progress=progress,
                )

        self.assertEqual(progress.update.call_count, 2)


if __name__ == "__main__":
    unittest.main()
