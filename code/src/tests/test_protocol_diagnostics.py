import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import run_protocol_diagnostics as diagnostics


class ProtocolDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original = diagnostics.STUDIES
        diagnostics.STUDIES = Path(self.temp.name)

    def tearDown(self):
        diagnostics.STUDIES = self.original
        self.temp.cleanup()

    def test_lr_calibration_manifest_is_inner_validation_only(self):
        path = diagnostics.generate_lr_calibration()
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["jobs"]), 24)
        self.assertTrue(all(job["phase"] == "calibration" for job in payload["jobs"]))
        self.assertFalse(payload["protocol"]["outer_test_access_during_calibration"])
        self.assertEqual(set(job["lr"] for job in payload["jobs"]), set(diagnostics.LRS))

    def test_encoder_manifest_is_inner_validation_only(self):
        path = diagnostics.generate_encoder_screen()
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["jobs"]), 12)
        self.assertEqual({job["encoder_initialization"] for job in payload["jobs"]}, {"simclr", "imagenet"})
        self.assertFalse(payload["protocol"]["outer_test_access_during_screening"])

    def test_manifest_is_immutable(self):
        path = diagnostics.generate_encoder_screen()
        payload = json.loads(path.read_text(encoding="utf-8"))
        diagnostics.atomic_json({**payload, "kind": "changed"}, path)
        with self.assertRaises(FileExistsError):
            diagnostics.generate_encoder_screen()

    def test_diagnostic_runner_does_not_write_model_checkpoints(self):
        source = Path(diagnostics.__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.save", source)
        self.assertNotIn(".pth\"", source)


if __name__ == "__main__":
    unittest.main()
