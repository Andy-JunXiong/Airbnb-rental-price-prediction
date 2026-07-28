from __future__ import annotations

import unittest

from compare_inside_airbnb_snapshots import compatibility_decision
from inside_airbnb_snapshot_discovery import (
    discover_sydney_dates,
    discovery_decision,
    requires_action,
)
from inside_airbnb_temporal_validation import temporal_gate


class SnapshotCompatibilityTest(unittest.TestCase):
    def test_price_validation_is_blocked_without_common_target(self) -> None:
        older = {
            "quote_price_with_context": False,
            "listing_price": False,
            "calendar_daily_price": False,
            "availability_proxy": True,
        }
        newer = {
            "quote_price_with_context": True,
            "listing_price": True,
            "calendar_daily_price": False,
            "availability_proxy": True,
        }
        status, blockers, common = compatibility_decision(older, newer)
        self.assertEqual(status, "TEMPORAL_PRICE_VALIDATION_BLOCKED")
        self.assertIn("no_common_non_null_price_target", blockers)
        self.assertEqual(common, [])

    def test_shared_quote_target_allows_temporal_validation(self) -> None:
        older = {
            "quote_price_with_context": True,
            "listing_price": True,
            "calendar_daily_price": False,
            "availability_proxy": True,
        }
        newer = dict(older)
        status, blockers, common = compatibility_decision(older, newer)
        self.assertEqual(status, "TEMPORAL_PRICE_VALIDATION_READY")
        self.assertEqual(blockers, [])
        self.assertIn("quote_price_with_context", common)

    def test_discovery_extracts_and_orders_official_sydney_dates(self) -> None:
        page = """
        <a href="https://data.insideairbnb.com/australia/nsw/sydney/2026-06-16/data/listings.csv.gz">new</a>
        <a href="/australia/nsw/sydney/2025-09-12/visualisations/listings.csv">old</a>
        <a href="/australia/vic/melbourne/2026-06-16/data/listings.csv.gz">other</a>
        """
        self.assertEqual(
            discover_sydney_dates(page), ["2025-09-12", "2026-06-16"]
        )

    def test_discovery_does_not_claim_a_new_snapshot_when_registry_is_current(self) -> None:
        decision = discovery_decision(
            ["2025-09-12", "2026-06-16"],
            ["2025-09-12", "2026-06-16"],
        )
        self.assertEqual(decision["status"], "NO_NEWER_SNAPSHOT")
        self.assertEqual(decision["newer_candidates"], [])
        self.assertFalse(requires_action(decision))

    def test_discovery_requires_action_for_new_snapshot_or_parse_failure(self) -> None:
        newer = discovery_decision(
            ["2026-06-16", "2026-09-15"],
            ["2026-06-16"],
        )
        missing = discovery_decision([], ["2026-06-16"])
        self.assertTrue(requires_action(newer))
        self.assertTrue(requires_action(missing))

    def test_temporal_gate_requires_cold_start_and_overall_evidence(self) -> None:
        passed = temporal_gate(True, 3000, 0.20, 0.89, 500, 0.05)
        self.assertTrue(passed["passed"])
        failed = temporal_gate(True, 3000, 0.20, 0.89, 500, -0.01)
        self.assertFalse(failed["passed"])
        self.assertEqual(
            failed["recommended_current_quote_model_authority"],
            "research_only",
        )


if __name__ == "__main__":
    unittest.main()
