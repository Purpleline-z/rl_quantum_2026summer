import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import measure_accuracy_at_each_new_label_budget as study


class BudgetCurveManifestFilterTests(unittest.TestCase):
    def test_explicit_low_budget_manifest_has_only_requested_jobs(self):
        original = study.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                study.OUT = Path(directory)
                path = study.generate_manifest(
                    budgets=(10,), seeds=(404, 505), strategies=("random", "uncertainty"),
                    manifest_name="low_budget_extension_manifest",
                )
                content = __import__("json").loads(path.read_text(encoding="utf-8"))
        finally:
            study.OUT = original
        self.assertEqual(len(content["jobs"]), 4)
        self.assertEqual({job["budget"] for job in content["jobs"]}, {10})
        self.assertEqual({job["seed"] for job in content["jobs"]}, {404, 505})
        self.assertEqual({job["strategy"] for job in content["jobs"]}, {"random", "uncertainty"})

    def test_parse_rejects_duplicate_values(self):
        with self.assertRaises(ValueError):
            study.parse_csv_values("10,10", int, "budgets")


if __name__ == "__main__":
    unittest.main()
