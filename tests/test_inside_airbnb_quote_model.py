from __future__ import annotations

import unittest

import numpy as np

from inside_airbnb_quote_model import (
    category_inventory,
    feature_row_from_request,
    finite_sample_quantile,
    fit_market_baseline,
    market_prediction,
    refusal_reasons,
)


class QuoteModelGovernanceTest(unittest.TestCase):
    def test_finite_sample_conformal_quantile_is_conservative(self) -> None:
        residuals = np.arange(1, 11, dtype=float)
        self.assertEqual(finite_sample_quantile(residuals, alpha=0.10), 10.0)

    def test_market_baseline_uses_exact_comparable_market(self) -> None:
        records = [
            {"neighbourhood": "Sydney", "room_type": "Entire home/apt"}
            for _ in range(12)
        ]
        y = np.arange(100, 112, dtype=float)
        indices = np.arange(12)
        baseline = fit_market_baseline(records, y, indices)
        prediction, count, level = market_prediction(baseline, records[0])
        self.assertEqual(prediction, 105.5)
        self.assertEqual(count, 12)
        self.assertEqual(level, "neighbourhood_room")

    def test_request_dates_create_leakage_safe_derived_features(self) -> None:
        payload = {
            "as_of_date": "2026-07-01",
            "quote_checkin_date": "2026-07-11",
            "quote_checkout_date": "2026-07-14",
        }
        row, _ = feature_row_from_request(payload)
        self.assertEqual(row["quote_lead_days"], 10)
        self.assertEqual(row["stay_nights"], 3)
        self.assertIn("distance_to_sydney_cbd_km", row)

    def test_gate_refuses_unseen_and_low_evidence_request(self) -> None:
        row = {
            "neighbourhood": "Unknown",
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "accommodates": 2.0,
            "quote_lead_days": 10.0,
            "stay_nights": 2.0,
        }
        inventory = {
            "neighbourhood": ["Sydney"],
            "property_type": ["Entire rental unit"],
            "room_type": ["Entire home/apt"],
            "host_is_superhost": ["t"],
        }
        reasons = refusal_reasons(
            row,
            predicted=300.0,
            lower=50.0,
            upper=900.0,
            comparable_count=0,
            category_values=inventory,
            price_range=[20.0, 5000.0],
            snapshot_age_days=10,
        )
        self.assertIn("unseen_category:neighbourhood", reasons)
        self.assertIn("insufficient_comparables", reasons)
        self.assertIn("prediction_interval_too_wide", reasons)


if __name__ == "__main__":
    unittest.main()
