from __future__ import annotations

import unittest

import numpy as np

from inside_airbnb_interval_challenger import (
    asymmetric_quantiles,
    banded_interval_bounds,
    fit_banded_asymmetric_calibration,
    predicted_price_band,
)


class IntervalChallengerTest(unittest.TestCase):
    def test_prediction_band_uses_prediction_not_label(self) -> None:
        thresholds = [100.0, 200.0, 300.0, 400.0]
        self.assertEqual(predicted_price_band(150.0, thresholds), "p50_p75")
        self.assertEqual(predicted_price_band(500.0, thresholds), "above_p95")

    def test_asymmetric_quantiles_allow_larger_upper_adjustment(self) -> None:
        predicted = np.log1p(np.asarray([100.0, 100.0, 100.0, 100.0]))
        actual = np.log1p(np.asarray([90.0, 120.0, 150.0, 200.0]))
        lower, upper = asymmetric_quantiles(actual, predicted)
        self.assertGreater(upper, lower)

    def test_banded_bounds_are_non_negative_and_ordered(self) -> None:
        calibration_target = np.linspace(50.0, 500.0, 200)
        calibration_prediction = np.log1p(calibration_target * 0.9)
        calibration = fit_banded_asymmetric_calibration(
            calibration_target,
            calibration_prediction,
            [100.0, 200.0, 300.0, 400.0],
        )
        lower, upper, _ = banded_interval_bounds(
            np.log1p(np.asarray([80.0, 450.0])), calibration
        )
        self.assertTrue(np.all(lower >= 0))
        self.assertTrue(np.all(upper >= lower))


if __name__ == "__main__":
    unittest.main()
