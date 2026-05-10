from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrendV3ExtendedTargetsTest(unittest.TestCase):
    def test_makefile_extended_targets_and_dirs_exist(self) -> None:
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("research-trend-v3-extended:", makefile)
        self.assertIn("compare-trend-v3-extended:", makefile)
        self.assertIn("--split-scheme extended", makefile)
        self.assertIn("reports/research/trend_following_v3_extended/$(EXT_SPLIT)", makefile)
        self.assertIn("reports/research/trend_following_v3_extended/train_ext", makefile)
        self.assertIn("reports/research/trend_following_v3_extended/validation_ext", makefile)
        self.assertIn("reports/research/trend_following_v3_extended/oos_ext", makefile)
        self.assertIn("reports/research/trend_following_v3_extended_compare", makefile)


if __name__ == "__main__":
    unittest.main()
