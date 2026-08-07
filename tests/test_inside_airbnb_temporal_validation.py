from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from inside_airbnb_quote_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _LGBM_AVAILABLE,
    _SKLEARN_HGB_AVAILABLE,
)
from inside_airbnb_temporal_validation import validate_temporally

_PIPELINE_AVAILABLE = _LGBM_AVAILABLE or _SKLEARN_HGB_AVAILABLE


class TemporalValidationIntegrationTest(unittest.TestCase):
    def write_silver(
        self, path: Path, snapshot: str, as_of: str, newer: bool
    ) -> None:
        fixed = [
            "snapshot_label",
            "training_eligible",
            "target_quoted_price_per_night",
            "host_id",
            "listing_id",
            "currency",
            "as_of_date",
        ]
        fields = fixed + NUMERIC_FEATURES + CATEGORICAL_FEATURES
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(300):
                host_number = index // 5
                host_id = (
                    f"new-{host_number}"
                    if newer and index % 2
                    else f"host-{host_number}"
                )
                accommodates = 1 + index % 6
                row = {
                    "snapshot_label": snapshot,
                    "training_eligible": "1",
                    "target_quoted_price_per_night": str(
                        80 + accommodates * 35 + (index % 7)
                    ),
                    "host_id": host_id,
                    "listing_id": (
                        f"listing-{index}" if not newer else f"new-listing-{index}"
                    ),
                    "currency": "AUD",
                    "as_of_date": as_of,
                    "quote_lead_days": str(10 + index % 30),
                    "stay_nights": str(2 + index % 4),
                    "checkin_month": "7",
                    "checkin_day_of_week": str(index % 7),
                    "checkin_is_weekend": str(int(index % 7 >= 5)),
                    "latitude": str(-33.9 + (index % 20) / 100),
                    "longitude": str(151.1 + (index % 20) / 100),
                    "distance_to_sydney_cbd_km": str(index % 20),
                    "distance_to_sydney_airport_km": str(5 + index % 20),
                    "distance_to_nearest_reference_beach_km": str(index % 10),
                    "distance_to_nearest_major_hub_km": str(index % 8),
                    "accommodates": str(accommodates),
                    "bathrooms": str(1 + index % 2),
                    "bedrooms": str(1 + index % 3),
                    "beds": str(1 + index % 4),
                    "amenities_count": str(10 + index % 20),
                    "minimum_nights": "1",
                    "maximum_nights": "365",
                    "calculated_host_listings_count": "5",
                    "neighbourhood": f"area-{index % 3}",
                    "property_type": "Entire rental unit",
                    "room_type": (
                        "Entire home/apt" if index % 4 else "Private room"
                    ),
                    "host_is_superhost": "t" if index % 2 else "f",
                }
                writer.writerow(row)

    @unittest.skipUnless(_PIPELINE_AVAILABLE, "Requires LightGBM or sklearn>=1.5")
    def test_end_to_end_temporal_report_stays_research_only_when_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "older.csv"
            newer = root / "newer.csv"
            report = root / "report.json"
            self.write_silver(older, "2026-01-01", "2026-01-02", False)
            self.write_silver(newer, "2026-02-01", "2026-02-02", True)
            result = validate_temporally(older, newer, report)
            self.assertTrue(result["protocol"]["strict_forward_time"])
            self.assertEqual(
                result["authority"]["current_quote_model"], "research_only"
            )
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
