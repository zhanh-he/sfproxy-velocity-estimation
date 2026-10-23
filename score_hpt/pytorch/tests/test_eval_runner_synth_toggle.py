import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

import direct_invension.eval_framework as eval_framework
import direct_invension.eval_runner as eval_runner
from direct_invension.eval_runner import format_evaluation_summary


class EvalRunnerSynthToggleTest(unittest.TestCase):
    def test_format_evaluation_summary_omits_synth_blocks_when_disabled(self):
        summary_text = format_evaluation_summary(
            dataset_type="maestro",
            instrument_path=Path("/tmp/piano.sfz"),
            prediction_dir=Path("/tmp/pred"),
            out_dir=Path("/tmp/out"),
            manifest_path=Path("/tmp/out/manifest.json"),
            summary={
                "label": "demo",
                "num_items": 1,
                "num_ok": 1,
                "num_fail": 0,
                "velocity_mae_enabled": True,
                "velocity_mae": 1.0,
                "real_ref_bssl_pearson_correlation": 0.1,
                "real_ref_bssl_cosine_similarity": 0.2,
                "real_ref_bssl_mean_absolute_error": 0.3,
                "real_ref_bssl_spearman_correlation": 0.4,
                "real_ref_bstl_pearson_correlation": 0.5,
                "real_ref_bstl_cosine_similarity": 0.6,
                "real_ref_bstl_mean_absolute_error": 0.7,
                "real_ref_bstl_spearman_correlation": 0.8,
            },
        )

        self.assertNotIn("synth_gt_vs_synth_pred_bssl", summary_text)
        self.assertNotIn("synth_gt_vs_synth_pred_bstl", summary_text)
        self.assertIn("real_vs_synth_pred_bssl", summary_text)
        self.assertIn("real_vs_synth_pred_bstl", summary_text)

    def test_evaluate_prediction_item_skips_gt_render_and_synth_metrics_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gt_midi = tmp_path / "gt.mid"
            pred_midi = tmp_path / "pred.mid"
            real_audio = tmp_path / "real.wav"
            gt_midi.touch()
            pred_midi.touch()
            real_audio.touch()

            render_calls = []
            eval_calls = []

            def fake_render_midi_to_audio(**kwargs):
                render_calls.append(kwargs)
                return {"wav_path": str(kwargs["wav_path"])}

            def fake_bssl_pair(**kwargs):
                eval_calls.append(kwargs)
                return {
                    "bssl": {
                        "flattened_raw_metrics": {
                            "pearson_correlation": 0.1,
                            "mean_absolute_error": 0.2,
                            "cosine_sim": 0.3,
                            "spearman_correlation": 0.4,
                        }
                    },
                    "ntot": {
                        "curve_metrics_raw": {
                            "pearson_correlation": 0.5,
                            "mean_absolute_error": 0.6,
                            "cosine_sim": 0.7,
                            "spearman_correlation": 0.8,
                        }
                    },
                }

            with mock.patch.object(eval_framework, "_import_bssl_eval", return_value=fake_bssl_pair), mock.patch.object(
                eval_framework, "render_midi_to_audio", side_effect=fake_render_midi_to_audio
            ), mock.patch.object(
                eval_framework,
                "align_note_velocities",
                return_value=eval_framework.VelocityAlignmentResult(1, 1, 1, 2.0, True, 0, 0),
            ):
                result = eval_framework.evaluate_prediction_item(
                    {
                        "key": "demo",
                        "gt_midi": str(gt_midi),
                        "pred_midi": str(pred_midi),
                        "real_audio": str(real_audio),
                    },
                    instrument_path=tmp_path / "piano.sfz",
                    out_dir=tmp_path / "out",
                    compute_synth_gt_metrics=False,
                )

        self.assertEqual(len(render_calls), 1)
        self.assertEqual(Path(render_calls[0]["midi_path"]), pred_midi)
        self.assertEqual(len(eval_calls), 1)
        self.assertEqual(Path(eval_calls[0]["gt_wav"]), real_audio)
        self.assertIsNone(result["render"]["gt"])
        self.assertIsNone(result["synth_ref"])
        self.assertIsNotNone(result["real_ref"])
        self.assertNotIn("synth_ref_bssl_pearson_correlation", result["summary"])
        self.assertIn("real_ref_bssl_pearson_correlation", result["summary"])

    def test_run_evaluation_disables_synth_metrics_for_route3(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pred_dir = tmp_path / "pred"
            gt_dir = tmp_path / "gt"
            instrument = tmp_path / "piano.sfz"
            pred_dir.mkdir()
            gt_dir.mkdir()
            instrument.touch()

            cfg = SimpleNamespace(backend=SimpleNamespace(supervision=SimpleNamespace(hop_size=512)))
            eval_cfg = SimpleNamespace(
                out_dir=str(tmp_path / "out"),
                pred_midi_dir=str(pred_dir),
                instrument_path=str(instrument),
                manifest_mode="folder",
                audio_reference_mode="none",
                label="demo",
                split="test",
                gt_midi_dir=str(gt_dir),
                render_sr=44100,
                eval_sr=22050,
                frames_per_second=50.0,
                fft_size=1024,
                bssl_mode="sone",
                num_samples=2048,
                normalization="zscore",
                backend="auto",
                overwrite_render=False,
                compute_velocity_mae=True,
            )

            with mock.patch.object(eval_runner, "validate_hop_contract"), mock.patch.object(
                eval_runner, "build_folder_prediction_manifest", return_value={"items": [], "label": "demo"}
            ), mock.patch.object(
                eval_runner, "evaluate_prediction_manifest", return_value={"summary": {}, "per_file_results_dir": str(tmp_path / "per_file")}
            ) as mock_eval_manifest:
                eval_runner.run_evaluation(
                    cfg=cfg,
                    dataset_type="maestro",
                    eval_cfg=eval_cfg,
                    config_prefix="route3.eval",
                    route_name="Route III eval",
                    run_json_name="route3_eval_run.json",
                )

        self.assertFalse(mock_eval_manifest.call_args.kwargs["compute_synth_gt_metrics"])


if __name__ == "__main__":
    unittest.main()
