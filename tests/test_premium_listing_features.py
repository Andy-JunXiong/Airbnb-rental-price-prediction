from __future__ import annotations

import unittest

from premium_listing_features import (
    PREMIUM_FIELDS,
    bathroom_privacy,
    premium_features,
    property_group,
)


class PremiumListingFeaturesTest(unittest.TestCase):
    def test_semantic_flags_exclude_pool_table_false_positive(self) -> None:
        row = {
            "amenities": (
                '["Pool table", "Private outdoor pool - heated", '
                '"Ocean view", "Free parking on premises"]'
            ),
            "bathrooms": "2",
            "bedrooms": "3",
            "beds": "4",
            "accommodates": "6",
            "bathrooms_text": "2 private baths",
            "property_type": "Entire villa",
        }
        result = premium_features(row)
        self.assertEqual(result["has_pool"], 1)
        self.assertEqual(result["has_water_view"], 1)
        self.assertEqual(result["has_on_premises_parking"], 1)
        self.assertEqual(result["bathroom_privacy"], "private")
        self.assertEqual(result["property_group"], "house")
        self.assertAlmostEqual(result["bathrooms_per_guest"], 1 / 3, places=5)
        self.assertEqual(set(result), set(PREMIUM_FIELDS))

    def test_only_pool_table_does_not_set_pool(self) -> None:
        result = premium_features({"amenities": '["Pool table"]'})
        self.assertEqual(result["has_pool"], 0)

    def test_bathroom_and_property_groups_are_stable(self) -> None:
        self.assertEqual(bathroom_privacy("1 shared bath"), "shared")
        self.assertEqual(
            bathroom_privacy("1 bath"), "exclusive_or_unspecified"
        )
        self.assertEqual(property_group("Room in boutique hotel"), "hospitality")
        self.assertEqual(property_group("Entire rental unit"), "apartment")


if __name__ == "__main__":
    unittest.main()
