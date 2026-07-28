from __future__ import annotations

import unittest

import numpy as np

from inside_airbnb_error_analysis import (
    diagnostic_flags,
    finite_metrics,
    price_band,
    price_band_definitions,
)


class ErrorAnalysisTest(unittest.TestCase):
    def test_price_bands_are_defined_only_from_training_values(self) -> None:
        definitions = price_band_definitions(np.arange(1, 101, dtype=float))
        self.assertEqual(price_band(20.0, definitions), "up_to_p50")
        self.assertEqual(price_band(1000.0, definitions), "above_p99")

    def test_metrics_use_predicted_minus_actual_bias_direction(self) -> None:
        actual = np.asarray([100.0, 200.0])
        predicted = np.asarray([90.0, 180.0])
        baseline = np.asarray([80.0, 160.0])
        lower = np.asarray([70.0, 150.0])
        upper = np.asarray([120.0, 220.0])
        result = finite_metrics(actual, predicted, baseline, lower, upper)
        self.assertEqual(result["mean_error_predicted_minus_actual"], -15.0)
        self.assertEqual(result["underprediction_rate"], 1.0)
        self.assertEqual(result["interval_coverage"], 1.0)

    def test_diagnostics_ignore_small_segments(self) -> None:
        groups = {
            "room_type": [
                {
                    "room_type": "small",
                    "rows": 50,
                    "interval_coverage": 0.50,
                    "relative_mae_improvement_vs_market": -0.50,
                    "median_error_predicted_minus_actual": -100.0,
                }
            ]
        }
        self.assertEqual(diagnostic_flags(groups), [])


if __name__ == "__main__":
    unittest.main()
