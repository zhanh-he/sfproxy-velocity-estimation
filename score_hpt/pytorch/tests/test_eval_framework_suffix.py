import sys
import unittest
from pathlib import Path


PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(PYTORCH_DIR))

from direct_invension.eval_framework import _normalize_stem


class EvalFrameworkSuffixTest(unittest.TestCase):
    def test_normalize_stem_strips_dot_route_suffix(self):
        self.assertEqual(_normalize_stem("demo.route2"), "demo")
        self.assertEqual(_normalize_stem("demo.route12"), "demo")


if __name__ == "__main__":
    unittest.main()
