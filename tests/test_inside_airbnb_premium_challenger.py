from __future__ import annotations

import unittest

import numpy as np

from inside_airbnb_premium_challenger import hard_mixture, soft_mixture


class PremiumChallengerTest(unittest.TestCase):
    def test_soft_mixture_uses_probability_as_blend_weight(self) -> None:
        general = np.asarray([100.0, 100.0, 100.0])
        expert = np.asarray([300.0, 300.0, 300.0])
        probability = np.asarray([0.0, 0.5, 1.0])
        np.testing.assert_allclose(
            soft_mixture(general, expert, probability),
            np.asarray([100.0, 200.0, 300.0]),
        )

    def test_hard_mixture_routes_only_above_threshold(self) -> None:
        general = np.asarray([100.0, 100.0])
        expert = np.asarray([300.0, 300.0])
        probability = np.asarray([0.24, 0.25])
        np.testing.assert_allclose(
            hard_mixture(general, expert, probability),
            np.asarray([100.0, 300.0]),
        )


if __name__ == "__main__":
    unittest.main()
