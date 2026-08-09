import json
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_selection_benchmark as benchmark
import run_budget_curve_study as budget_runner


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

    def test_mc_manifest_has_configuration_and_thirty_jobs(self):
        path = benchmark.generate_mc_screen()
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(value["jobs"]), 30)
        self.assertEqual({x["strategy"] for x in value["jobs"]}, set(benchmark.MC_SELECTORS))
        self.assertEqual({x["budget"] for x in value["jobs"]}, {10, 25})
        self.assertTrue(all(x["dropout_p"] == .2 and x["mc_samples"] == 20 for x in value["jobs"]))
        self.assertTrue(all(not x["run_nmf_diagnostic"] for x in value["jobs"]))

    def test_cluster_margin_manifest_has_twenty_five_jobs(self):
        path = benchmark.generate_cluster_margin()
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(value["jobs"]), 25)
        self.assertEqual({x["strategy"] for x in value["jobs"]}, {"cluster_margin_pairwise"})
        self.assertEqual({x["budget"] for x in value["jobs"]}, {10, 25, 50, 75, 100})

    def test_run_manifest_skips_completed_jobs(self):
        path = benchmark.create_manifest("resume_test", ("cluster_margin_pairwise",), (10,), ("none",))
        value = json.loads(path.read_text(encoding="utf-8"))
        completed = value["jobs"][0]
        result = path.parent / "jobs" / completed["job_id"] / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}", encoding="utf-8")
        called = []
        original = benchmark.run_job
        try:
            benchmark.run_job = lambda manifest_path, requested, data_root=None: called.append(requested)
            benchmark.run_manifest(path)
        finally:
            benchmark.run_job = original
        self.assertNotIn(completed["job_id"], called)
        self.assertEqual(len(called), len(value["jobs"]) - 1)

    def test_controlled_job_path_does_not_use_checkpoint_writing_runner(self):
        source = inspect.getsource(budget_runner.run_job)
        self.assertNotIn("run_strategy", source)
        self.assertNotIn("torch.save", source)
        self.assertNotIn(".pth", source)

    def test_mc_manifest_has_configuration_and_thirty_jobs(self):
        path = benchmark.generate_mc_screen()
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(value["jobs"]), 30)
        self.assertEqual({x["strategy"] for x in value["jobs"]}, set(benchmark.MC_SELECTORS))
        self.assertEqual({x["budget"] for x in value["jobs"]}, {10, 25})
        self.assertTrue(all(x["dropout_p"] == .2 and x["mc_samples"] == 20 for x in value["jobs"]))
        self.assertTrue(all(not x["run_nmf_diagnostic"] for x in value["jobs"]))

    def test_cluster_margin_manifest_has_twenty_five_jobs(self):
        path = benchmark.generate_cluster_margin()
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(value["jobs"]), 25)
        self.assertEqual({x["strategy"] for x in value["jobs"]}, {"cluster_margin_pairwise"})
        self.assertEqual({x["budget"] for x in value["jobs"]}, {10, 25, 50, 75, 100})

    def test_run_manifest_skips_completed_jobs(self):
        path = benchmark.create_manifest("resume_test", ("cluster_margin_pairwise",), (10,), ("none",))
        value = json.loads(path.read_text(encoding="utf-8"))
        completed = value["jobs"][0]
        result = path.parent / "jobs" / completed["job_id"] / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text("{}", encoding="utf-8")
        called = []
        original = benchmark.run_job
        try:
            benchmark.run_job = lambda manifest_path, requested, data_root=None: called.append(requested)
            benchmark.run_manifest(path)
        finally:
            benchmark.run_job = original
        self.assertNotIn(completed["job_id"], called)
        self.assertEqual(len(called), len(value["jobs"]) - 1)


if __name__ == "__main__":
    unittest.main()
