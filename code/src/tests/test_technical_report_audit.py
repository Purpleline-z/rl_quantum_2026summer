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

    def test_report_answers_core_review_questions(self):
        self.assertIn("Direct Responses to Active-Learning Review Questions", self.text)
        self.assertIn("post_test_accuracy_by_acquisition_budget.png", self.text)
        self.assertIn("self-supervised contrastive learning on RHEED images, not pairwise preference learning", self.text)
        self.assertIn("exact image population used to produce the shipped checkpoint is not established", self.text)
        self.assertIn("There is **no separate pairwise validation partition**", self.text)

    def test_all_markdown_artifact_links_resolve(self):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", self.text):
            if "://" not in target:
                self.assertTrue((CODE_ROOT / target).exists(), target)


if __name__ == "__main__":
    unittest.main()
