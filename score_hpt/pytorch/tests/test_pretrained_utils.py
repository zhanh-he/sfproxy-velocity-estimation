import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

from velo_model.pretrained_utils import (
    get_frontend_pretrained_value,
    load_score_wrapper_state,
    resolve_frontend_pretrained_checkpoint,
    resolve_pretrained_checkpoint,
    select_prefixed_substate,
    unwrap_checkpoint_state_dict,
)


class PretrainedUtilsTest(unittest.TestCase):
    @contextmanager
    def _chdir(self, path: Path):
        old_cwd = Path.cwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(old_cwd)

    @staticmethod
    def _model_cfg(**kwargs):
        defaults = {
            "type": "hpt",
            "input2": None,
            "frontend_pretrained_mode": "scratch",
            "frontend_pretrained": "",
        }
        defaults.update(kwargs)
        return type("ModelCfg", (), defaults)()

    def test_get_frontend_pretrained_value_reads_frontend_pretrained(self):
        value = get_frontend_pretrained_value(
            type(
                "ModelCfg",
                (),
                {
                    "frontend_pretrained": "/tmp/new_name.pth",
                },
            )()
        )
        self.assertEqual(value, "/tmp/new_name.pth")

    def test_get_frontend_pretrained_value_missing_field_returns_empty_string(self):
        value = get_frontend_pretrained_value(
            type("ModelCfg", (), {})()
        )
        self.assertEqual(value, "")

    def test_resolve_frontend_pretrained_checkpoint_prefers_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            explicit_path = root / "manual.pth"
            explicit_path.touch()

            resolved = resolve_frontend_pretrained_checkpoint(
                self._model_cfg(
                    frontend_pretrained_mode="route2_piano_specific",
                    frontend_pretrained=str(explicit_path),
                ),
                model_label="hpt",
                required=False,
            )

        self.assertEqual(resolved, explicit_path.resolve())

    def test_resolve_frontend_pretrained_checkpoint_resolves_relative_path_from_cwd(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "pretrained_checkpoints" / "hpt" / "120000_iterations.pth"
            target.parent.mkdir(parents=True)
            target.touch()

            with self._chdir(root):
                resolved = resolve_frontend_pretrained_checkpoint(
                    self._model_cfg(
                        frontend_pretrained_mode="route2_piano_specific",
                        frontend_pretrained="pretrained_checkpoints/hpt/120000_iterations.pth",
                    ),
                    model_label="hpt",
                    required=False,
                )

        self.assertEqual(resolved, target.resolve())

    def test_resolve_frontend_pretrained_checkpoint_auto_selects_latest_hpt_note_editor(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "pretrained_checkpoints" / "hpt+onset+score_note_editor"
            root.mkdir(parents=True)
            (root / "100000_iterations.pth").touch()
            latest = root / "120000_iterations.pth"
            latest.touch()

            with self._chdir(root.parent.parent):
                resolved = resolve_frontend_pretrained_checkpoint(
                    self._model_cfg(
                        input2="onset",
                        frontend_pretrained_mode="route2_piano_auto",
                    ),
                    model_label="hpt",
                    required=False,
                )

        self.assertEqual(resolved, latest.resolve())

    def test_resolve_frontend_pretrained_checkpoint_specific_mode_resolves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "pretrained_checkpoints" / "hpt+onset+score_note_editor" / "120000_iterations.pth"
            target.parent.mkdir(parents=True)
            target.touch()

            with self._chdir(root):
                resolved = resolve_frontend_pretrained_checkpoint(
                    self._model_cfg(
                        input2="onset",
                        frontend_pretrained_mode="route2_piano_specific",
                        frontend_pretrained="pretrained_checkpoints/hpt+onset+score_note_editor/120000_iterations.pth",
                    ),
                    model_label="hpt",
                    required=False,
                )

        self.assertEqual(resolved, target.resolve())

    def test_resolve_frontend_pretrained_checkpoint_scratch_returns_none(self):
        resolved = resolve_frontend_pretrained_checkpoint(
            self._model_cfg(
                input2="onset",
                frontend_pretrained_mode="scratch",
                frontend_pretrained="",
            ),
            model_label="hpt",
            required=False,
        )
        self.assertIsNone(resolved)

    def test_resolve_pretrained_checkpoint_existing_path_returns_path(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as fp:
            resolved = resolve_pretrained_checkpoint(fp.name, model_label="hpt_pretrained", required=True)
        self.assertEqual(resolved, Path(fp.name).resolve())

    def test_unwrap_checkpoint_state_dict_strips_module_prefix(self):
        state = unwrap_checkpoint_state_dict({
            "state_dict": {
                "module.layer.weight": torch.tensor([1.0]),
            }
        })
        self.assertIn("layer.weight", state)
        self.assertNotIn("module.layer.weight", state)

    def test_select_prefixed_substate_prefers_requested_prefix(self):
        state = {
            "base_adapter.model.weight": torch.tensor([1.0]),
            "base_adapter.model.bias": torch.tensor([2.0]),
            "other.weight": torch.tensor([3.0]),
        }
        substate = select_prefixed_substate(
            state,
            model_keys=["weight", "bias"],
            prefixes=("base_adapter.model.", "model.", ""),
        )
        self.assertEqual(set(substate.keys()), {"weight", "bias"})
        self.assertEqual(float(substate["weight"].item()), 1.0)
        self.assertEqual(float(substate["bias"].item()), 2.0)

    def test_load_score_wrapper_state_loads_full_wrapper(self):
        class DummyWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_adapter = torch.nn.Linear(1, 1)
                self.post = torch.nn.Linear(1, 1)

        with tempfile.NamedTemporaryFile(suffix=".pth") as fp:
            checkpoint = {
                "model": {
                    "base_adapter.weight": torch.tensor([[3.0]]),
                    "base_adapter.bias": torch.tensor([4.0]),
                    "post.weight": torch.tensor([[5.0]]),
                    "post.bias": torch.tensor([6.0]),
                }
            }
            torch.save(checkpoint, fp.name)

            model = DummyWrapper()
            load_score_wrapper_state(model, Path(fp.name))

        self.assertEqual(float(model.base_adapter.weight.item()), 3.0)
        self.assertEqual(float(model.base_adapter.bias.item()), 4.0)
        self.assertEqual(float(model.post.weight.item()), 5.0)
        self.assertEqual(float(model.post.bias.item()), 6.0)


if __name__ == "__main__":
    unittest.main()
