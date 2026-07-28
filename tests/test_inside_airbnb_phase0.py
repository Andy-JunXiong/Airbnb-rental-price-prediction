from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from inside_airbnb_phase0 import audit_snapshot, build_manifest, write_json_atomic


PROVIDER = {"name": "Inside Airbnb"}


def write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    if path.suffix == ".gz":
        handle_context = gzip.open(path, "wt", encoding="utf-8", newline="")
    else:
        handle_context = path.open("w", encoding="utf-8", newline="")
    with handle_context as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class PhaseZeroAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.snapshot = {
            "city": "sydney",
            "display_name": "Sydney",
            "snapshot_date": "2026-06-16",
            "calendar_horizon_days": 2,
            "files": [
                {
                    "name": "listings.csv.gz",
                    "kind": "listings",
                    "url": "https://example.test/listings.csv.gz",
                    "required_columns": [
                        "id",
                        "scrape_id",
                        "last_scraped",
                        "source",
                        "host_id",
                        "neighbourhood_cleansed",
                        "latitude",
                        "longitude",
                        "room_type",
                        "accommodates",
                        "price",
                        "price_quote_checkin_date",
                        "price_quote_checkout_date",
                        "price_quote_price_per_night",
                        "minimum_nights",
                        "maximum_nights",
                        "calendar_last_scraped",
                        "number_of_reviews",
                    ],
                },
                {
                    "name": "calendar.csv.gz",
                    "kind": "calendar",
                    "url": "https://example.test/calendar.csv.gz",
                    "required_columns": [
                        "listing_id",
                        "date",
                        "available",
                        "price",
                        "minimum_nights",
                        "maximum_nights",
                    ],
                },
                {
                    "name": "reviews.csv",
                    "kind": "reviews",
                    "url": "https://example.test/reviews.csv",
                    "required_columns": ["listing_id", "date"],
                },
                {
                    "name": "neighbourhoods.csv",
                    "kind": "neighbourhoods",
                    "url": "https://example.test/neighbourhoods.csv",
                    "required_columns": ["neighbourhood", "neighbourhood_group"],
                },
                {
                    "name": "neighbourhoods.geojson",
                    "kind": "neighbourhood_geojson",
                    "url": "https://example.test/neighbourhoods.geojson",
                    "required_columns": ["neighbourhood"],
                },
            ],
        }
        self.raw = (
            self.data_root / "sydney" / "snapshot_date=2026-06-16"
        )
        self.raw.mkdir(parents=True)
        listing_columns = self.snapshot["files"][0]["required_columns"]
        write_csv(
            self.raw / "listings.csv.gz",
            listing_columns,
            [
                [
                    "1", "s1", "2026-06-16", "city scrape", "h1", "Alpha",
                    "-33.8", "151.2", "Entire home/apt", "2", "$100.00",
                    "2026-06-17", "2026-06-18", "100.00", "1", "30",
                    "2026-06-16", "4",
                ],
                [
                    "2", "s1", "2026-06-16", "city scrape", "h2", "Beta",
                    "-33.9", "151.1", "Private room", "1", "$80.00",
                    "2026-06-17", "2026-06-18", "80.00", "2", "20",
                    "2026-06-16", "2",
                ],
            ],
        )
        calendar_columns = self.snapshot["files"][1]["required_columns"]
        write_csv(
            self.raw / "calendar.csv.gz",
            calendar_columns,
            [
                ["1", "2026-06-16", "t", "$100", "1", "30"],
                ["1", "2026-06-17", "f", "$110", "1", "30"],
                ["2", "2026-06-16", "t", "$80", "2", "20"],
                ["2", "2026-06-17", "t", "$85", "2", "20"],
            ],
        )
        write_csv(
            self.raw / "reviews.csv",
            ["listing_id", "date"],
            [["1", "2026-06-01"], ["2", "2026-05-01"]],
        )
        write_csv(
            self.raw / "neighbourhoods.csv",
            ["neighbourhood", "neighbourhood_group"],
            [["Alpha", ""], ["Beta", ""]],
        )
        (self.raw / "neighbourhoods.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"neighbourhood": "Alpha"},
                            "geometry": {"type": "Polygon", "coordinates": []},
                        },
                        {
                            "type": "Feature",
                            "properties": {"neighbourhood": "Beta"},
                            "geometry": {"type": "Polygon", "coordinates": []},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifest = build_manifest(PROVIDER, self.snapshot, self.raw)
        write_json_atomic(self.raw / "manifest.json", manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def audit(self) -> dict:
        return audit_snapshot(
            PROVIDER,
            self.snapshot,
            self.data_root,
            self.root / "report.json",
        )

    def test_valid_snapshot_is_approved_for_mvp(self) -> None:
        report = self.audit()
        self.assertEqual(report["decision"]["status"], "GO_FULL_PRICE_MVP")
        self.assertEqual(report["datasets"]["listings"]["rows"], 2)
        self.assertEqual(report["datasets"]["calendar"]["rows"], 4)

    def test_duplicate_calendar_key_blocks_snapshot(self) -> None:
        with gzip.open(
            self.raw / "calendar.csv.gz", "at", encoding="utf-8", newline=""
        ) as handle:
            csv.writer(handle).writerow(
                ["1", "2026-06-16", "t", "$100", "1", "30"]
            )
        manifest = build_manifest(PROVIDER, self.snapshot, self.raw)
        write_json_atomic(self.raw / "manifest.json", manifest)
        report = self.audit()
        self.assertEqual(report["decision"]["status"], "NO_GO")
        self.assertIn("calendar.primary_key", report["decision"]["blockers"])

    def test_calendar_without_price_limits_mvp_to_quote_level(self) -> None:
        self.snapshot["files"][1]["required_columns"].remove("price")
        write_csv(
            self.raw / "calendar.csv.gz",
            self.snapshot["files"][1]["required_columns"],
            [
                ["1", "2026-06-16", "t", "1", "30"],
                ["1", "2026-06-17", "f", "1", "30"],
                ["2", "2026-06-16", "t", "2", "20"],
                ["2", "2026-06-17", "t", "2", "20"],
            ],
        )
        manifest = build_manifest(PROVIDER, self.snapshot, self.raw)
        write_json_atomic(self.raw / "manifest.json", manifest)
        report = self.audit()
        self.assertEqual(report["decision"]["status"], "GO_QUOTE_LEVEL_MVP")
        self.assertIn(
            "calendar.daily_listed_price",
            report["decision"]["capability_blockers"],
        )

    def test_historical_snapshot_without_quote_fields_uses_daily_price_scope(
        self,
    ) -> None:
        quote_fields = {
            "price_quote_checkin_date",
            "price_quote_checkout_date",
            "price_quote_price_per_night",
        }
        listing_source = self.snapshot["files"][0]
        listing_source["required_columns"] = [
            name
            for name in listing_source["required_columns"]
            if name not in quote_fields
        ]
        with gzip.open(
            self.raw / "listings.csv.gz",
            "rt",
            encoding="utf-8",
            newline="",
        ) as handle:
            original_rows = list(csv.DictReader(handle))
        write_csv(
            self.raw / "listings.csv.gz",
            listing_source["required_columns"],
            [
                [row[name] for name in listing_source["required_columns"]]
                for row in original_rows
            ],
        )
        manifest = build_manifest(PROVIDER, self.snapshot, self.raw)
        write_json_atomic(self.raw / "manifest.json", manifest)
        report = self.audit()
        self.assertEqual(
            report["decision"]["status"], "GO_DAILY_PRICE_HISTORICAL"
        )
        self.assertIn(
            "listings.quote_fields",
            report["decision"]["capability_blockers"],
        )


if __name__ == "__main__":
    unittest.main()
