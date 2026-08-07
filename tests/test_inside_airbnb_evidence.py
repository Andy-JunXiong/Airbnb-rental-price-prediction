from __future__ import annotations

import unittest

from inside_airbnb_evidence import assess_evidence


class EvidencePolicyTest(unittest.TestCase):
    def test_high_evidence_with_strong_signals(self) -> None:
        result = assess_evidence(
            estimated_price=200,
            prediction_interval=(150, 260),
            comparable_count=150,
            snapshot_age_days=30,
            deployment_authority="temporally_validated",
            refusal_reasons=[],
        )
        self.assertEqual(result["tier"], "HIGH")
        self.assertEqual(result["tier_label"], "Well-supported estimate")

    def test_medium_with_moderate_comparables(self) -> None:
        result = assess_evidence(
            estimated_price=200,
            prediction_interval=(150, 260),
            comparable_count=50,
            snapshot_age_days=30,
            deployment_authority="temporally_validated",
            refusal_reasons=[],
        )
        self.assertEqual(result["tier"], "MEDIUM")

    def test_low_with_few_comparables(self) -> None:
        result = assess_evidence(
            estimated_price=200,
            prediction_interval=(150, 260),
            comparable_count=10,
            snapshot_age_days=30,
            deployment_authority="temporally_validated",
            refusal_reasons=[],
        )
        self.assertEqual(result["tier"], "LOW")

    def test_refuse_when_refusal_reasons_present(self) -> None:
        result = assess_evidence(
            estimated_price=0,
            prediction_interval=None,
            comparable_count=0,
            snapshot_age_days=30,
            deployment_authority="research_only",
            refusal_reasons=["missing_critical:accommodates"],
        )
        self.assertEqual(result["tier"], "REFUSE")

    def test_wide_interval_downgrades(self) -> None:
        result = assess_evidence(
            estimated_price=200,
            prediction_interval=(0, 800),
            comparable_count=150,
            snapshot_age_days=30,
            deployment_authority="temporally_validated",
            refusal_reasons=[],
        )
        self.assertEqual(result["tier"], "LOW")

    def test_old_snapshot_downgrades(self) -> None:
        result = assess_evidence(
            estimated_price=200,
            prediction_interval=(150, 260),
            comparable_count=150,
            snapshot_age_days=120,
            deployment_authority="temporally_validated",
            refusal_reasons=[],
        )
        self.assertEqual(result["tier"], "MEDIUM")

    def test_research_only_capped_at_medium(self) -> None:
        result = assess_evidence(
            estimated_price=200,
            prediction_interval=(150, 260),
            comparable_count=150,
            snapshot_age_days=30,
            deployment_authority="research_only",
            refusal_reasons=[],
        )
        self.assertEqual(result["tier"], "MEDIUM")
        self.assertIn("research_only", result["reasons"][0])

    def test_upper_tail_downgrades(self) -> None:
        result = assess_evidence(
            estimated_price=900,
            prediction_interval=(700, 1100),
            comparable_count=150,
            snapshot_age_days=30,
            deployment_authority="temporally_validated",
            refusal_reasons=[],
            training_price_quantiles={"p90": 600},
        )
        self.assertEqual(result["tier"], "MEDIUM")
        self.assertIn("Premium listing", result["reasons"][0])

    def test_no_interval_lowers_confidence(self) -> None:
        result = assess_evidence(
            estimated_price=0,
            prediction_interval=None,
            comparable_count=0,
            snapshot_age_days=30,
            deployment_authority="research_only",
            refusal_reasons=["snapshot_too_old"],
        )
        self.assertEqual(result["tier"], "REFUSE")


if __name__ == "__main__":
    unittest.main()
