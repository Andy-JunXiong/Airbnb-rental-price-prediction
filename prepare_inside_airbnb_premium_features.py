"""Join privacy-minimised premium listing features onto the governed Silver table."""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path
from typing import Any

from inside_airbnb_phase0 import ROOT, sha256_file, utc_now, write_json_atomic
from inside_airbnb_quote_model import DEFAULT_SILVER
from premium_listing_features import PREMIUM_FIELDS, premium_features
from prepare_inside_airbnb_quotes import DEFAULT_SOURCE


DEFAULT_OUTPUT = DEFAULT_SILVER.with_name("listing_quotes_premium.csv")
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_2026-06-16_premium_features.json"
)


def build_premium_silver(
    source_listings: Path,
    base_silver: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    feature_by_id: dict[str, dict[str, Any]] = {}
    with gzip.open(
        source_listings, "rt", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            listing_id = (row.get("id") or "").strip()
            if not listing_id:
                continue
            if listing_id in feature_by_id:
                raise ValueError(f"Duplicate raw listing ID: {listing_id}")
            feature_by_id[listing_id] = premium_features(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    missing_ids = []
    flag_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = {
        "bathroom_privacy": Counter(),
        "property_group": Counter(),
    }
    rows = 0
    with base_silver.open(
        encoding="utf-8", newline=""
    ) as source_handle, output.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        original_fields = list(reader.fieldnames or [])
        collisions = sorted(set(original_fields) & set(PREMIUM_FIELDS))
        if collisions:
            raise ValueError(f"Premium fields already exist: {collisions}")
        writer = csv.DictWriter(
            output_handle, fieldnames=[*original_fields, *PREMIUM_FIELDS]
        )
        writer.writeheader()
        for row in reader:
            rows += 1
            listing_id = row["listing_id"]
            features = feature_by_id.get(listing_id)
            if features is None:
                missing_ids.append(listing_id)
                features = {field: "" for field in PREMIUM_FIELDS}
            else:
                flag_counts.update(
                    {
                        field: int(features[field])
                        for field in PREMIUM_FIELDS
                        if field.startswith("has_")
                    }
                )
                for field in category_counts:
                    category_counts[field][str(features[field])] += 1
            writer.writerow({**row, **features})
    if missing_ids:
        raise ValueError(
            f"Raw listings do not cover {len(missing_ids)} Silver IDs"
        )
    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "source_listings": str(source_listings.relative_to(ROOT)),
        "source_listings_sha256": sha256_file(source_listings),
        "base_silver": str(base_silver.relative_to(ROOT)),
        "base_silver_sha256": sha256_file(base_silver),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256_file(output),
        "rows": rows,
        "raw_listing_ids": len(feature_by_id),
        "missing_join_ids": len(missing_ids),
        "premium_fields": PREMIUM_FIELDS,
        "flag_counts": dict(sorted(flag_counts.items())),
        "category_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in category_counts.items()
        },
        "privacy": {
            "raw_amenity_strings_retained": False,
            "free_text_retained": False,
            "direct_identifiers_added": False,
        },
    }
    write_json_atomic(report_path, report)
    print(f"premium  {output}")
    print(f"rows     {rows:,}")
    print(f"report   {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-listings", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base-silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_premium_silver(
        args.source_listings, args.base_silver, args.output, args.report
    )


if __name__ == "__main__":
    main()
