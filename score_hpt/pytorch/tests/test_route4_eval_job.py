import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
from omegaconf import OmegaConf


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

import direct_invension.route4_eval_job as route4_eval_job
from direct_invension.route4_eval_job import Route4EvalJobRequest, _request_from_args, build_parser, run_route4_eval_job


class Route4EvalJobTest(unittest.TestCase):
    def _prepare_project_layout(self, shared_root: Path) -> Path:
        project_root = shared_root / "202604_midiproxy"
        (project_root / "score_hpt").mkdir(parents=True)
        (shared_root / "Dataset" / "maestro-v3.0.0").mkdir(parents=True)
        (shared_root / "Dataset" / "GAPS").mkdir(parents=True)
        (shared_root / "202604_midiproxy_data" / "score_hpt" / "workspaces").mkdir(parents=True)
        piano_sf = shared_root / "202604_midiproxy_data" / "soundfont" / "SalamanderGrandPiano" / "SalamanderGrandPianoV3.sfz"
        guitar_sf = (
            shared_root / "202604_midiproxy_data" / "soundfont" / "SpanishClassicalGuitar" / "SpanishClassicalGuitar-20190618.sfz"
        )
        piano_sf.parent.mkdir(parents=True)
        guitar_sf.parent.mkdir(parents=True)
        piano_sf.touch()
        guitar_sf.touch()
        return project_root

    def test_parser_accepts_requested_flags(self):
        args = build_parser().parse_args(
            [
                "--ckpt_path",
                "/tmp/model.pth",
                "--dataset",
                "maestro",
                "--eval_scope",
                "one",
                "--compute_velo_mae",
                "--instrument",
                "piano",
            ]
        )
        self.assertEqual(args.ckpt_path, "/tmp/model.pth")
        self.assertEqual(args.dataset, "maestro")
        self.assertEqual(args.eval_scope, "one")
        self.assertTrue(args.compute_velo_mae)
        self.assertEqual(args.instrument, "piano")

    def test_request_from_args_converts_namespace(self):
        request = _request_from_args(
            SimpleNamespace(
                ckpt_path="/tmp/model.pth",
                dataset="maestro",
                eval_scope="test",
                compute_velo_mae=True,
                instrument="piano",
            )
        )
        self.assertEqual(
            request,
            Route4EvalJobRequest(
                ckpt_path="/tmp/model.pth",
                dataset="maestro",
                eval_scope="test",
                compute_velo_mae=True,
                instrument="piano",
            ),
        )

    def test_run_route4_eval_job_resolves_paths_writes_summary_txt_and_infers_note_editor_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shared_root = Path(tmp_dir)
            project_root = self._prepare_project_layout(shared_root)
            ckpt = (
                project_root
                / "ckpts"
                / "hpt+onset+score_note_editor+backend_diffproxy"
                / "120000_iterations.pth"
            )
            ckpt.parent.mkdir(parents=True)
            ckpt.touch()
            request = Route4EvalJobRequest(
                ckpt_path=ckpt,
                dataset="maestro",
                eval_scope="full",
                compute_velo_mae=True,
                instrument="piano",
            )
            cfg = OmegaConf.create(
                {
                    "route4": {
                        "infer": {"velocity_method": "onset_only", "overwrite": False},
                        "eval": {},
                    }
                }
            )
            eval_payload = {
                "payload": {"summary": {"label": "route4_test", "num_items": 5, "velocity_mae": 7.5}},
                "summary_text": "Summary\n  dataset: maestro",
            }

            with mock.patch.object(route4_eval_job, "repo_root", return_value=project_root), mock.patch.object(
                route4_eval_job, "compose_cfg", return_value=cfg
            ) as mock_compose, mock.patch.object(
                route4_eval_job, "predict_route4_dataset"
            ) as mock_predict, mock.patch.object(
                route4_eval_job, "run_evaluation", return_value=eval_payload
            ) as mock_eval, mock.patch.object(
                route4_eval_job, "evaluation_results_to_dataframe", return_value=pd.DataFrame([{"label": "route4_test"}])
            ):
                result = run_route4_eval_job(request)

            overrides = mock_compose.call_args.kwargs["overrides"] if "overrides" in mock_compose.call_args.kwargs else mock_compose.call_args.args[0]
            self.assertIn("score_informed.method=note_editor", overrides)
            self.assertIn("model.input2=onset", overrides)
            self.assertEqual(result.dataset_type, "maestro")
            self.assertEqual(result.requested_split, "full")
            self.assertEqual(result.resolved_split, "all")
            self.assertEqual(result.instrument_key, "piano")
            self.assertEqual(result.dataset_dir, shared_root / "Dataset" / "maestro-v3.0.0")
            self.assertTrue(result.txt_path.exists())
            self.assertEqual(result.txt_path.name, "result_summary.txt")
            self.assertFalse(any(result.eval_out_dir.glob("*.csv")))
            self.assertIn("maestro_full_piano", str(result.eval_out_dir))
            txt = result.txt_path.read_text(encoding="utf-8")
            self.assertIn("CHECKPOINT_PATH =", txt)
            self.assertIn("COMPUTE_VELOCITY_MAE = True", txt)
            mock_predict.assert_called_once()
            self.assertEqual(mock_predict.call_args.kwargs["split"], "full")
            self.assertEqual(mock_predict.call_args.kwargs["dataset_dir"], shared_root / "Dataset" / "maestro-v3.0.0")
            mock_eval.assert_called_once()

    def test_run_route4_eval_job_scope_one_runs_single_debug_item(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shared_root = Path(tmp_dir)
            project_root = self._prepare_project_layout(shared_root)
            ckpt = (
                project_root
                / "ckpts"
                / "hpt+onset+score_note_editor+backend_diffproxy"
                / "120000_iterations.pth"
            )
            ckpt.parent.mkdir(parents=True)
            ckpt.touch()
            request = Route4EvalJobRequest(
                ckpt_path=ckpt,
                dataset="maestro",
                eval_scope="one",
                compute_velo_mae=False,
                instrument="piano",
            )
            cfg = OmegaConf.create(
                {
                    "route4": {
                        "infer": {"velocity_method": "onset_only", "overwrite": False},
                        "eval": {},
                    }
                }
            )
            eval_payload = {
                "payload": {"summary": {"label": "route4_one", "num_items": 1, "velocity_mae_enabled": False}},
                "summary_text": "Summary\n  dataset: maestro",
            }

            with mock.patch.object(route4_eval_job, "repo_root", return_value=project_root), mock.patch.object(
                route4_eval_job, "compose_cfg", return_value=cfg
            ) as mock_compose, mock.patch.object(
                route4_eval_job, "predict_route4_dataset"
            ) as mock_predict, mock.patch.object(
                route4_eval_job, "run_evaluation", return_value=eval_payload
            ) as mock_eval, mock.patch.object(
                route4_eval_job, "evaluation_results_to_dataframe", return_value=pd.DataFrame([{"label": "route4_one"}])
            ):
                result = run_route4_eval_job(request)

            overrides = mock_compose.call_args.kwargs["overrides"] if "overrides" in mock_compose.call_args.kwargs else mock_compose.call_args.args[0]
            self.assertIn("route4.infer.split=test", overrides)
            self.assertIn("route4.eval.split=test", overrides)
            self.assertEqual(result.requested_split, "one")
            self.assertEqual(result.resolved_split, "test")
            self.assertEqual(mock_predict.call_args.kwargs["split"], "test")
            self.assertEqual(mock_predict.call_args.kwargs["max_items"], 1)
            self.assertEqual(mock_eval.call_args.kwargs["max_items"], 1)


if __name__ == "__main__":
    unittest.main()
