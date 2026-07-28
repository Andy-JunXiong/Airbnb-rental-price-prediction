from __future__ import annotations

import unittest

import numpy as np

from inside_airbnb_upper_tail_challenger import (
    candidate_decision,
    tail_sample_weights,
)


class UpperTailChallengerTest(unittest.TestCase):
    def test_tail_weights_are_positive_and_capped(self) -> None:
        weights = tail_sample_weights(
            np.asarray([1.0, 100.0, 1_000_000.0], dtype=float)
        )
        self.assertTrue(np.all(weights >= 0.75))
        self.assertTrue(np.all(weights <= 4.0))
        self.assertGreater(weights[-1], weights[0])

    def test_candidate_must_pass_every_promotion_condition(self) -> None:
        incumbent = {
            "summary": {
                "overall_mae": {"mean": 100.0},
                "upper_tail_mae": {"mean": 500.0},
                "upper_tail_median_bias": {"mean": -300.0},
            },
            "folds": [{"upper_tail_mae": 500.0} for _ in range(5)],
        }
        candidate = {
            "summary": {
                "overall_mae": {"mean": 101.0},
                "upper_tail_mae": {"mean": 400.0},
                "upper_tail_median_bias": {"mean": -200.0},
            },
            "folds": [{"upper_tail_mae": 400.0} for _ in range(5)],
        }
        self.assertTrue(candidate_decision(incumbent, candidate)["qualifies"])
        candidate["summary"]["overall_mae"]["mean"] = 103.0
        self.assertFalse(candidate_decision(incumbent, candidate)["qualifies"])


if __name__ == "__main__":
    unittest.main()
