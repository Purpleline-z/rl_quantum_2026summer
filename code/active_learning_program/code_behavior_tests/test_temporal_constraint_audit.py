import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_rheed_trajectory_ordering import build_trajectory_manifest, parse_frame


class TemporalConstraintAuditTests(unittest.TestCase):
    def test_filename_parser_uses_observed_index_not_physical_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "2022-02-04" / "001_RR220204A_933C_0006.bmp"
            frame.parent.mkdir()
            frame.touch()
            row = parse_frame(frame, root)
        self.assertEqual(row["trajectory_id"], "2022-02-04/RR220204A")
        self.assertEqual(row["frame_index"], 1)
        self.assertEqual(row["ordering_source"], "leading_filename_index")
        self.assertNotIn("reconstruction_type", row)

    def test_manifest_reports_numeric_gaps_without_inferring_a_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "2022-02-04"
            folder.mkdir()
            for name in ("001_RR220204A_933C_0006.bmp", "003_RR220204A_933C_0008.bmp"):
                (folder / name).touch()
            frames, summary = build_trajectory_manifest(root)
        self.assertEqual(len(frames), 2)
        self.assertEqual(int(summary.iloc[0].missing_index_count), 1)
        self.assertFalse(bool(summary.iloc[0].order_ambiguous))


if __name__ == "__main__":
    unittest.main()
