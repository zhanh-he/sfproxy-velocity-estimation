import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))


class Route34EvalSupportTest(unittest.TestCase):
    def test_route3_infer_delegates_to_route2_dataset_predictor(self):
        from direct_invension.route3_infer import predict_route3_dataset

        cfg = object()
        checkpoint_path = Path("/tmp/route3.ckpt")
        out_dir = Path("/tmp/route3_out")
        dataset_dir = Path("/tmp/dataset")

        with patch(
            "direct_invension.route3_infer.predict_route2_dataset",
            return_value={"label": "route3"},
        ) as mock_predict:
            result = predict_route3_dataset(
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                dataset_type="maestro",
                dataset_dir=dataset_dir,
                split="full",
                maps_pianos="AkPnBcht",
                velocity_method="max_frame",
                skip_existing=False,
            )

        self.assertEqual(result["label"], "route3")
        mock_predict.assert_called_once_with(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            dataset_type="maestro",
            dataset_dir=dataset_dir,
            split="all",
            maps_pianos="AkPnBcht",
            velocity_method="max_frame",
            skip_existing=False,
            max_items=None,
            label="route3",
            manifest_name="route3_manifest.json",
            file_summary_name="route3_predictions.json",
        )

    def test_infer_route_dataset_dispatches_to_route3_and_route4_predictors(self):
        from direct_invension.route34_eval_support import infer_route_dataset

        cfg = object()
        checkpoint_path = Path("/tmp/model.ckpt")
        out_dir = Path("/tmp/out")
        dataset_dir = Path("/tmp/dataset")

        with patch(
            "direct_invension.route34_eval_support.predict_route3_dataset",
            return_value={"label": "route3", "pred_midi_dir": "/tmp/out/pred_midis"},
        ) as mock_route3, patch(
            "direct_invension.route34_eval_support.predict_route4_dataset",
            return_value={"label": "route4", "pred_midi_dir": "/tmp/out/pred_midis"},
        ) as mock_route4:
            route3_result = infer_route_dataset(
                route_name="route3",
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                dataset_type="gaps",
                dataset_dir=dataset_dir,
                split="test",
                maps_pianos="both",
                overwrite=False,
            )
            route4_result = infer_route_dataset(
                route_name="route4",
                cfg=cfg,
                checkpoint_path=checkpoint_path,
                out_dir=out_dir,
                dataset_type="maestro",
                dataset_dir=dataset_dir,
                split="full",
                maps_pianos="AkPnBcht",
                overwrite=True,
            )

        self.assertEqual(route3_result["label"], "route3")
        self.assertEqual(route4_result["label"], "route4")
        mock_route3.assert_called_once_with(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            dataset_type="gaps",
            dataset_dir=dataset_dir,
            split="test",
            maps_pianos="both",
            velocity_method="onset_only",
            skip_existing=True,
        )
        mock_route4.assert_called_once_with(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            out_dir=out_dir,
            dataset_type="maestro",
            dataset_dir=dataset_dir,
            split="full",
            maps_pianos="AkPnBcht",
            velocity_method="onset_only",
            skip_existing=False,
        )

    def test_infer_route_dataset_rejects_unknown_route(self):
        from direct_invension.route34_eval_support import infer_route_dataset

        with self.assertRaisesRegex(ValueError, "route_name"):
            infer_route_dataset(
                route_name="route9",
                cfg=object(),
                checkpoint_path="/tmp/model.ckpt",
                out_dir="/tmp/out",
                dataset_type="maestro",
                dataset_dir="/tmp/dataset",
            )

    def test_checkpoint_model_overrides_follow_checkpoint_folder_name(self):
        from direct_invension.route34_eval_support import checkpoint_model_overrides

        note_editor_ckpt = Path(
            "/tmp/371311_0/checkpoints/"
            "hpt+onset+score_note_editor+backend_diffsynth_piano+piano_ssm_spectral/"
            "60000_iterations.pth"
        )
        direct_hpt_ckpt = Path(
            "/tmp/route3/checkpoints/hpt+backend_diffsynth_piano/60000_iterations.pth"
        )

        self.assertEqual(
            checkpoint_model_overrides(note_editor_ckpt),
            [
                "model.type=hpt",
                "score_informed.method=note_editor",
                "model.input2=onset",
                "model.input3=null",
            ],
        )
        self.assertEqual(
            checkpoint_model_overrides(direct_hpt_ckpt),
            [
                "model.type=hpt",
                "score_informed.method=direct",
                "model.input2=null",
                "model.input3=null",
            ],
        )

    def test_route3_evaluate_main_delegates_to_shared_runner(self):
        from direct_invension import route3_evaluate

        with patch.object(
            route3_evaluate,
            "run_route_evaluate_main",
        ) as mock_run:
            route3_evaluate.main()

        mock_run.assert_called_once_with(
            argv=sys.argv[1:],
            route_name="route3",
            route_title="Route III eval",
        )

    def test_route4_evaluate_main_delegates_to_shared_runner(self):
        from direct_invension import route4_evaluate

        with patch.object(
            route4_evaluate,
            "run_route_evaluate_main",
        ) as mock_run:
            route4_evaluate.main()

        mock_run.assert_called_once_with(
            argv=sys.argv[1:],
            route_name="route4",
            route_title="Route IV eval",
        )

    def test_route4_infer_main_delegates_to_shared_runner(self):
        from direct_invension import route4_infer

        with patch.object(
            route4_infer,
            "run_route_infer_main",
        ) as mock_run:
            route4_infer.main()

        mock_run.assert_called_once_with(
            argv=sys.argv[1:],
            route_name="route4",
            predict_fn=route4_infer.predict_route4_dataset,
        )


if __name__ == "__main__":
    unittest.main()
