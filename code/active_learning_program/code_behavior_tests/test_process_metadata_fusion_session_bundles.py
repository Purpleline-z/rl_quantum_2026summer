from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

STUDY = Path(__file__).resolve().parents[2] / "active_learning_studies" / "process_metadata_fusion_for_rheed_prediction"
sys.path.insert(0, str(STUDY))
from process_metadata_fusion_core import (MetadataOnlyBradleyTerryRewardModel, MetadataSafetyError,
    PRIMARY_FEATURE_COLUMNS, align_heartbeat_frames_causally, build_run_disjoint_split,
    fit_training_only_scaler, read_session_bundle, validate_requested_feature_columns)


def write_bundle(path: Path, sample_id: str = "AJ002_STO") -> None:
    sensor = """timestamp,elapsed_s,pyrometer_temp_C,mistral_v_actual_V,mistral_i_actual_A,chamber_pressure_mbar,recon_1x1
2026-08-05T16:50:00,0,100,1,0.1,1e-9,99
2026-08-05T16:50:10,10,110,2,0.3,2e-9,99
2026-08-05T16:50:20,20,130,4,0.7,4e-9,99
"""
    heartbeat = """timestamp,elapsed_s,frame_path
2026-08-05T16:50:05,5,E:\\ChMBE\\frames\\a.bmp
2026-08-05T16:50:15,15,E:\\ChMBE\\frames\\b.bmp
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("session_metadata.json", json.dumps({"sample_id": sample_id, "session_start": "2026-08-05T16:50:00", "session_end": "2026-08-05T16:51:00"}))
        archive.writestr("sensor_log.csv", sensor)
        archive.writestr("heartbeat_log.csv", heartbeat)


class ProcessMetadataSessionBundleTests(unittest.TestCase):
    def test_causal_alignment_and_backward_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "session.zip"; write_bundle(archive)
            aligned = align_heartbeat_frames_causally(read_session_bundle(archive))
            self.assertEqual(len(aligned), 2)
            self.assertTrue((aligned.source_sensor_timestamp <= aligned.frame_timestamp).all())
            self.assertEqual(aligned.sensor_lag_seconds.tolist(), [5.0, 5.0])
            self.assertTrue(np.isnan(aligned.loc[0, "pyrometer_temp_rate_C_per_s"]))
            self.assertAlmostEqual(aligned.loc[1, "pyrometer_temp_rate_C_per_s"], 1.0)

    def test_label_like_features_are_rejected(self):
        with self.assertRaises(MetadataSafetyError):
            validate_requested_feature_columns((*PRIMARY_FEATURE_COLUMNS[:-1], "classifier_status"))

    def test_one_session_and_unresolved_paths_block_split(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "session.zip"; write_bundle(archive)
            aligned = align_heartbeat_frames_causally(read_session_bundle(archive))
            with self.assertRaisesRegex(MetadataSafetyError, "unresolved"):
                build_run_disjoint_split(aligned)

    def test_training_only_scaler_has_missingness_indicators(self):
        rows = pd.DataFrame([{name: float(index) for index, name in enumerate(PRIMARY_FEATURE_COLUMNS)},
                             {name: np.nan for name in PRIMARY_FEATURE_COLUMNS}])
        scaler = fit_training_only_scaler(rows)
        transformed = scaler.transform(rows)
        self.assertEqual(transformed.shape, (2, 16))
        self.assertTrue((transformed[1, 8:] == 1).all())

    def test_metadata_only_arm_retains_five_reward_outputs(self):
        model = MetadataOnlyBradleyTerryRewardModel()
        self.assertEqual(tuple(model(torch.zeros(3, 16)).shape), (3, 5))

