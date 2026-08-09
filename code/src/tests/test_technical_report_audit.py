from __future__ import annotations

import re
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
REPORT = CODE_ROOT / "TECHNICAL_REPORT.md"


class TechnicalReportAuditTests(unittest.TestCase):
    def setUp(self):
        self.text = REPORT.read_text(encoding="utf-8")

    def test_report_uses_completed_experiments_not_obsolete_root(self):
        self.assertNotIn("results/published", self.text)
        self.assertIn("results/completed_experiments", self.text)

    def test_report_documents_result_oriented_benchmark(self):
        self.assertIn("Result-Oriented RHEED Active Pair Selection", self.text)
        self.assertIn("five selectors", self.text.lower())
        self.assertIn("Benchmark status matrix", self.text)
        self.assertIn("Uncertainty + diversity", self.text)
        self.assertIn("Cluster quota uncertainty", self.text)
        self.assertIn("Core-set", self.text)
        self.assertIn("post_test_accuracy_by_acquisition_budget.png", self.text)
        self.assertIn("SimCLR is image-only self-supervised pretraining, not pairwise preference learning", self.text)
        self.assertIn("There is no separate pairwise validation partition", self.text)

    def test_all_markdown_artifact_links_resolve(self):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", self.text):
            if "://" not in target:
                self.assertTrue((CODE_ROOT / target).exists(), target)


if __name__ == "__main__":
    unittest.main()
