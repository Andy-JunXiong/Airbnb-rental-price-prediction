from __future__ import annotations

import unittest

from prepare_inside_airbnb_quotes import SILVER_FIELDS, transform_listing


def source_row() -> dict[str, str]:
    return {
        "id": "10",
        "host_id": "20",
        "scrape_id": "20260616",
        "last_scraped": "2026-06-17",
        "source": "city scrape",
        "price_quote_checkin_date": "2026-07-01",
        "price_quote_checkout_date": "2026-07-04",
        "price_quote_price_per_night": "250.00",
        "price_quote_raw": '{"quote": {"currency": "AUD"}}',
        "neighbourhood_cleansed": "Sydney",
        "property_type": "Entire rental unit",
        "room_type": "Entire home/apt",
        "latitude": "-33.87",
        "longitude": "151.21",
        "accommodates": "2",
        "bathrooms": "1",
        "bedrooms": "1",
        "beds": "1",
        "amenities": '["Wifi", "Kitchen"]',
        "minimum_nights": "1",
        "maximum_nights": "365",
        "host_total_listings_count": "",
        "calculated_host_listings_count": "1",
        "host_is_superhost": "t",
        "instant_bookable": "",
        "availability_30": "20",
        "availability_60": "40",
        "availability_90": "60",
        "availability_365": "200",
        "number_of_reviews_ltm": "12",
        "number_of_reviews_l30d": "1",
    }


class SilverQuoteTransformTest(unittest.TestCase):
    def test_complete_aud_quote_is_training_eligible(self) -> None:
        transformed = transform_listing(source_row(), "2026-06-16")
        self.assertEqual(transformed["training_eligible"], "1")
        self.assertEqual(transformed["quote_lead_days"], 14)
        self.assertEqual(transformed["stay_nights"], 3)
        self.assertEqual(transformed["amenities_count"], 2)
        self.assertEqual(transformed["target_quoted_price_per_night"], 250.0)
        self.assertAlmostEqual(
            float(transformed["distance_to_sydney_cbd_km"]), 0.16, delta=0.2
        )

    def test_missing_quote_is_retained_but_ineligible(self) -> None:
        row = source_row()
        row["price_quote_price_per_night"] = ""
        transformed = transform_listing(row, "2026-06-16")
        self.assertEqual(transformed["training_eligible"], "0")
        self.assertIn(
            "missing_or_invalid_quote_price",
            transformed["eligibility_reason"],
        )

    def test_silver_contract_excludes_direct_text_and_names(self) -> None:
        forbidden = {"host_name", "name", "description", "reviewer_name", "comments"}
        self.assertFalse(forbidden & set(SILVER_FIELDS))


if __name__ == "__main__":
    unittest.main()
