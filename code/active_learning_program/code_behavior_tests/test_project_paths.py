import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from project_file_locations import resolve_data_root


class ProjectPathTests(unittest.TestCase):
    def test_explicit_temporary_data_root_overrides_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stable_data"
            (root / "original data").mkdir(parents=True)
            self.assertEqual(resolve_data_root(root), root.resolve())

    def test_environment_override_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("RHEED_DATA_ROOT")
            os.environ["RHEED_DATA_ROOT"] = directory
            try:
                self.assertEqual(resolve_data_root(), Path(directory).resolve())
            finally:
                if previous is None:
                    os.environ.pop("RHEED_DATA_ROOT", None)
                else:
                    os.environ["RHEED_DATA_ROOT"] = previous

    def test_pipeline_initializes_against_temporary_override(self):
        from pairwise_active_learning_pipeline import Config, Experiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original data").mkdir()
            experiment = Experiment(Config(data_root=str(root)))
            self.assertEqual(experiment.data_root, root.resolve() / "original data")


if __name__ == "__main__":
    unittest.main()
