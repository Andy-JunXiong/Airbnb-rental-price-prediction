from __future__ import annotations

import unittest

from inside_airbnb_text_features import merge_text_fields, reference_manifest


TEXT_FIELDS = ("description", "neighborhood_overview", "name")


class TextFeaturesTest(unittest.TestCase):
    def test_merge_empty_fields_returns_empty_string(self) -> None:
        row = {"description": "", "neighborhood_overview": "", "name": ""}
        result = merge_text_fields(row, TEXT_FIELDS)
        self.assertEqual(result, "")

    def test_merge_concatenates_non_empty_fields(self) -> None:
        row = {
            "description": "Beautiful apartment near the beach",
            "neighborhood_overview": "",
            "name": "Cozy Beach Studio",
        }
        result = merge_text_fields(row, TEXT_FIELDS)
        self.assertIn("Beautiful apartment", result)
        self.assertIn("Cozy Beach Studio", result)

    def test_merge_handles_missing_keys(self) -> None:
        row = {"name": "Only name"}
        result = merge_text_fields(row, TEXT_FIELDS)
        self.assertEqual(result, "Only name")

    def test_merge_strips_whitespace(self) -> None:
        row = {"description": "  padded text  ", "neighborhood_overview": "", "name": ""}
        result = merge_text_fields(row, TEXT_FIELDS)
        self.assertEqual(result, "padded text")

    def test_reference_manifest_is_valid(self) -> None:
        manifest = reference_manifest()
        self.assertEqual(manifest["version"], 1)
        self.assertIsInstance(manifest["available"], bool)
        self.assertIn("text_fields", manifest)
        self.assertIn("preferred", manifest)


if __name__ == "__main__":
    unittest.main()
