from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))
import extend_budget_100_comparison_to_15_seeds as followup


class EndpointExtensionTests(unittest.TestCase):
    def test_exact_fixed_seed_set(self):
        self.assertEqual(followup.SEEDS, [42, 79, 123, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313])
        self.assertEqual(len(set(followup.SEEDS)), 15)

    def test_all_strategies_are_compared(self):
        self.assertEqual(set(followup.STRATEGIES), {"random", "uncertainty", "cluster_quota_uncertainty", "uncertainty_diversity", "core_set"})


if __name__ == "__main__":
    unittest.main()
