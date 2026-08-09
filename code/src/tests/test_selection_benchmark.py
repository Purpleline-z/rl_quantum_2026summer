import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_selection_benchmark as benchmark


class SelectionBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.original = benchmark.STUDIES
        self.temp = tempfile.TemporaryDirectory()
        benchmark.STUDIES = Path(self.temp.name)

    def tearDown(self):
        benchmark.STUDIES = self.original
        self.temp.cleanup()

    def test_stage1_manifest_has_exactly_sixty_missing_cells(self):
        path = benchmark.generate_stage1()
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(value["jobs"]), 60)
        self.assertEqual({x["strategy"] for x in value["jobs"]}, {"uncertainty_diversity", "cluster_quota_uncertainty", "core_set"})
        self.assertEqual({x["budget"] for x in value["jobs"]}, {10, 25, 50, 75})
        self.assertEqual({x["symmetry_mode"] for x in value["jobs"]}, {"none"})

    def test_stage2_manifest_has_exactly_two_hundred_fifty_cells(self):
        path = benchmark.generate_stage2()
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(value["jobs"]), 250)
        self.assertEqual({x["strategy"] for x in value["jobs"]}, set(benchmark.SELECTORS))
        self.assertEqual({x["symmetry_mode"] for x in value["jobs"]}, {"left_half_mirror", "symmetric_average"})

    def test_grid_validator_rejects_missing_cell(self):
        import pandas as pd
        row = pd.DataFrame([{ "seed": 42, "budget": 10, "strategy": "random", "symmetry_mode": "none" }])
        with self.assertRaises(ValueError):
            benchmark._validate_grid(row, ("random",), (10,), ("none",))


if __name__ == "__main__":
    unittest.main()
