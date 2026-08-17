import csv
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

STUDY = Path(__file__).parents[2] / "active_learning_studies" / "simulator_augmented_classifier2_preliminary"
sys.path.insert(0, str(STUDY))
from simulator_study_common import SYNTHETIC_TO_REAL, synthetic_rows, validate_synthetic_archive


class SimulatorPreliminaryTests(unittest.TestCase):
    def make_archive(self, directory: Path, bad_hash: bool = False) -> Path:
        source = directory / "source"; root = source / "p656_aligned_peak"; (root / "images").mkdir(parents=True)
        records = []
        for label in SYNTHETIC_TO_REAL:
            group = f"group_{label}"
            for view in range(3):
                relative = f"images/{label}_{view}.png"; path = root / relative
                Image.new("L", (2, 2), color=view).save(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                records.append({"image_path": relative, "image_sha256": "0" * 64 if bad_hash and not records else digest, "reconstruction_label": label, "synthetic_training_only": "true", "same_surface_group": group})
        with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
        manifest = {"generator": "classifier_synthetic_sto_v6", "class_counts": {label: 3 for label in SYNTHETIC_TO_REAL}, "image_count": len(records), "generator_config": {"schema_version": 6}, "detector_contract": {"output": {"bit_depth": 8, "dtype": "uint8", "height": 492, "width": 656}}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        archive = directory / "synthetic.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in root.rglob("*"):
                if path.is_file(): bundle.write(path, path.relative_to(source).as_posix())
        return archive

    def test_validator_accepts_manifest_backed_training_only_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            result = validate_synthetic_archive(self.make_archive(Path(directory)))
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["synthetic_image_count"], 9)
        self.assertEqual(result["same_surface_group_count"], 3)

    def test_validator_rejects_corrupted_image_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): validate_synthetic_archive(self.make_archive(Path(directory), bad_hash=True))

    def test_synthetic_partition_keeps_every_surface_group_together(self):
        rows = []
        for label in SYNTHETIC_TO_REAL:
            for group in range(8):
                for view in range(3): rows.append({"reconstruction_label": label, "same_surface_group": f"{label}_{group}", "image_path": f"unused_{label}_{group}_{view}.png"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            train, validation = synthetic_rows(root, 42)
        self.assertFalse(set(train.same_surface_group) & set(validation.same_surface_group))
        self.assertEqual(len(train) + len(validation), len(rows))


if __name__ == "__main__": unittest.main()
