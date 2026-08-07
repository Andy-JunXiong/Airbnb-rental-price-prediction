from __future__ import annotations

import unittest

from inside_airbnb_explain import (
    _comparable_blurb,
    _fmt_aud,
    _nearest_anchor,
    _price_band_label,
    explain_prediction,
)


class ExplainPredictionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = {
            "status": "ok",
            "estimated_price": 245.50,
            "prediction_interval": [180.00, 330.00],
            "comparable_count": 87,
            "comparable_level": "neighbourhood_room",
            "evidence_level": "moderate",
            "snapshot_staleness_warning": False,
            "snapshot_age_days": 45,
            "authority_warning": None,
        }
        self.row = {
            "neighbourhood": "Manly",
            "room_type": "Entire home/apt",
            "property_type": "Entire condominium",
            "accommodates": "4",
            "bedrooms": "2",
            "bathrooms": "1.5",
            "stay_nights": "5",
            "quote_lead_days": "14",
            "latitude": "-33.798",
            "longitude": "151.287",
        }
        self.artifact = {
            "market_baseline": {
                "exact": {
                    ("Manly", "Entire home/apt"): {"median": 210.0, "count": 87}
                },
                "global": 150.0,
            },
            "supported_price_range": [60.0, 700.0],
        }

    def test_template_explanation_has_key_sections(self) -> None:
        explanation = explain_prediction(
            self.result, self.row, self.artifact, prefer_llm=False
        )
        self.assertEqual(explanation["mode"], "template")
        text = explanation["text"]
        self.assertIn("AUD 246", text)
        self.assertIn("AUD 180", text)
        self.assertIn("AUD 330", text)
        self.assertIn("Market Comparison", text)
        self.assertIn("Manly", text)
        self.assertIn("Location Context", text)
        self.assertIn("Listing Details", text)
        self.assertIn("Limitations", text)

    def test_refused_prediction_shows_reasons(self) -> None:
        refused = {
            **self.result,
            "status": "refused",
            "estimated_price": None,
            "prediction_interval": None,
            "refusal_reasons": ["missing_critical:accommodates", "snapshot_too_old"],
        }
        explanation = explain_prediction(
            refused, self.row, self.artifact, prefer_llm=False
        )
        self.assertIn("Prediction Refused", explanation["text"])
        self.assertIn("missing_critical:accommodates", explanation["text"])

    def test_price_band_budget(self) -> None:
        label = _price_band_label(80.0, [60.0, 700.0])
        self.assertIn("budget", label)

    def test_price_band_luxury(self) -> None:
        label = _price_band_label(800.0, [60.0, 700.0])
        self.assertIn("luxury", label)

    def test_comparable_blurb_high_count(self) -> None:
        blurb = _comparable_blurb(120, "neighbourhood_room")
        self.assertIn("120", blurb)

    def test_comparable_blurb_low_count(self) -> None:
        blurb = _comparable_blurb(15, "global")
        self.assertIn("broader", blurb)

    def test_fmt_aud(self) -> None:
        self.assertEqual(_fmt_aud(245.50), "AUD 246")

    def test_nearest_anchor_returns_distances(self) -> None:
        anchors = _nearest_anchor(-33.8688, 151.2093)
        self.assertLess(anchors["cbd_km"], 1.0)
        self.assertGreater(anchors["airport_km"], 5.0)
        self.assertIn("nearest_beach", anchors)
        self.assertIn("nearest_hub", anchors)


if __name__ == "__main__":
    unittest.main()
