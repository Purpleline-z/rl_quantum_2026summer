import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from decode_rheed_trajectory_with_higher_order_constraints import (
    STATES,
    decode_one_trajectory,
    decode_score_table,
)


def configuration():
    return {
        "penalty_rules": {
            "initial_state_not_1x1": {"base_log_penalty": 1.0},
            "leave_htr": {"base_log_penalty": 1.0},
            "skip_bad_before_other_reconstruction": {"base_log_penalty": 1.0},
        },
        "sensitivity_levels": {
            "weak": {"multiplier": 0.223143551},
            "moderate": {"multiplier": 0.693147181},
            "strong": {"multiplier": 1.609437912},
        },
    }


class HigherOrderConstraintDecoderTests(unittest.TestCase):
    def test_moderate_penalty_prefers_bad_before_another_reconstruction(self):
        rows = np.array([
            [.90, .02, .02, .02, .02, .02],
            [.01, .42, .53, .01, .01, .02],
            [.01, .04, .04, .04, .04, .83],
        ])
        decoded = decode_one_trajectory(rows, configuration(), "moderate")
        self.assertEqual(decoded.states, ["(1 x 1)", "Bad", "HTR"])
        self.assertEqual(decoded.changed_frame_indices, [1])

    def test_htr_can_be_overridden_but_leaving_it_is_penalized(self):
        rows = np.array([
            [.85, .03, .03, .03, .03, .03],
            [.02, .90, .02, .02, .02, .02],
            [.02, .02, .02, .02, .02, .90],
            [.01, .01, .60, .01, .01, .36],
        ])
        decoded = decode_one_trajectory(rows, configuration(), "strong")
        self.assertEqual(decoded.states[-2:], ["HTR", "HTR"])

    def test_output_keeps_raw_predictions_and_all_sensitivity_columns(self):
        rows = []
        for frame_index, values in enumerate((
            [.90, .02, .02, .02, .02, .02],
            [.01, .42, .53, .01, .01, .02],
        ), start=1):
            row = {"trajectory_id": "trajectory_a", "frame_index": frame_index, "relative_image_path": f"a_{frame_index}.bmp"}
            row.update({f"probability_{name}": value for name, value in zip(("1x1", "bad", "twinned", "c6x2", "sqrt13", "htr"), values)})
            rows.append(row)
        frame_output, summary = decode_score_table(pd.DataFrame(rows), configuration())
        self.assertEqual(frame_output.raw_framewise_argmax.tolist(), ["(1 x 1)", "Twinned(2 x 1)"])
        self.assertTrue({"decoded_weak_state", "decoded_moderate_state", "decoded_strong_state"}.issubset(frame_output.columns))
        self.assertEqual(len(summary), 1)


if __name__ == "__main__":
    unittest.main()
