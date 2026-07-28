from __future__ import annotations

import math
import unittest

from sydney_geography import (
    GEOGRAPHIC_FEATURES,
    SYDNEY_CBD,
    geographic_features,
    haversine_km,
)


class SydneyGeographyTest(unittest.TestCase):
    def test_haversine_is_zero_for_same_point(self) -> None:
        self.assertAlmostEqual(
            haversine_km(
                SYDNEY_CBD[1],
                SYDNEY_CBD[2],
                SYDNEY_CBD[1],
                SYDNEY_CBD[2],
            ),
            0.0,
        )

    def test_cbd_coordinate_creates_finite_distances(self) -> None:
        result = geographic_features(SYDNEY_CBD[1], SYDNEY_CBD[2])
        self.assertEqual(result["distance_to_sydney_cbd_km"], 0.0)
        self.assertEqual(set(result), set(GEOGRAPHIC_FEATURES))
        self.assertTrue(
            all(math.isfinite(float(value)) for value in result.values())
        )

    def test_invalid_coordinates_create_missing_features(self) -> None:
        result = geographic_features("", "not-a-number")
        self.assertTrue(all(value == "" for value in result.values()))


if __name__ == "__main__":
    unittest.main()
