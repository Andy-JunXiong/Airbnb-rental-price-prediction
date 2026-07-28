from __future__ import annotations

import unittest

from inside_airbnb_eda import (
    categorical_summary,
    numeric_summary,
    paired_correlation,
)


class ModernEdaTest(unittest.TestCase):
    def test_numeric_summary_reports_missingness_and_median(self) -> None:
        rows = [{"x": "1"}, {"x": ""}, {"x": "3"}]
        summary = numeric_summary(rows, "x")
        self.assertEqual(summary["non_missing"], 2)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["p50"], 2.0)

    def test_categorical_summary_counts_missing_explicitly(self) -> None:
        rows = [{"x": "a"}, {"x": ""}, {"x": "a"}]
        summary = categorical_summary(rows, "x")
        self.assertEqual(summary["unique_values"], 2)
        self.assertEqual(summary["missing"], 1)

    def test_correlation_uses_log_target_and_complete_pairs(self) -> None:
        rows = [
            {"x": "1", "target_quoted_price_per_night": "10"},
            {"x": "2", "target_quoted_price_per_night": "20"},
            {"x": "3", "target_quoted_price_per_night": "40"},
            {"x": "", "target_quoted_price_per_night": "80"},
        ]
        result = paired_correlation(rows, "x")
        self.assertEqual(result["rows"], 3)
        self.assertGreater(result["pearson_with_log1p_target"], 0.99)


if __name__ == "__main__":
    unittest.main()
