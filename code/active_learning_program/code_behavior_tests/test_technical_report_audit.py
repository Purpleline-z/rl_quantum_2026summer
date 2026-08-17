from __future__ import annotations

import re
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
REPORT = CODE_ROOT / "TECHNICAL_REPORT.md"


class TechnicalReportAuditTests(unittest.TestCase):
    def setUp(self):
        self.text = REPORT.read_text(encoding="utf-8")

    def test_report_uses_study_owned_results(self):
        self.assertIn("active_learning_studies", self.text)
        self.assertNotIn("results/completed_experiments", self.text)

    def test_report_documents_result_oriented_benchmark(self):
        self.assertIn("Technical Report: RHEED Active Pair Selection", self.text)
        self.assertIn("five-selector", self.text.lower())
        self.assertIn("Completed symmetry factorial", self.text)
        self.assertIn("Uncertainty + diversity", self.text)
        self.assertIn("Cluster-quota uncertainty", self.text)
        self.assertIn("Core-set", self.text)
        self.assertIn("post_test_accuracy_by_budget.png", self.text)
        self.assertIn("SimCLR is image-only self-supervised pretraining, not pairwise preference learning", self.text)
        self.assertIn("There is no separate pairwise validation partition", self.text)
        self.assertIn("Stage 2 completed", self.text)
        self.assertNotIn("Stage 2 symmetry factorial is not yet complete", self.text)
        self.assertNotIn("90%+", self.text)

    def test_all_markdown_artifact_links_resolve(self):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", self.text):
            if "://" not in target:
                self.assertTrue((CODE_ROOT / target).exists(), target)

    def test_report_has_no_stale_or_mojibake_claims(self):
        self.assertNotIn("results/published", self.text)
        self.assertNotIn("not yet complete/aggregated", self.text.lower())
        self.assertNotRegex(self.text, r"[\uFFFD\uE000-\uF8FF]")


if __name__ == "__main__":
    unittest.main()
