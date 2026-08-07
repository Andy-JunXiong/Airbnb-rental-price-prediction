"""Join text embeddings onto the governed Silver quote table.

Creates listing_quotes_text.csv: the base Silver table augmented with text
embedding columns, one row per training-eligible quote.

Requires the raw listings file (for text extraction) and the base Silver table.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from inside_airbnb_phase0 import (
    ROOT,
    active_silver_dir,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from inside_airbnb_quote_model import DEFAULT_SILVER as BASE_SILVER
from inside_airbnb_text_features import (
    TEXT_FEATURE_NAMES,
    embedding_for_listing,
    fit_embeddings,
    reference_manifest,
)
from prepare_inside_airbnb_quotes import DEFAULT_SOURCE as RAW_LISTINGS


DEFAULT_OUTPUT = BASE_SILVER.with_name("listing_quotes_text.csv")
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / f"sydney_{active_silver_dir().name.split('=')[-1]}_text_features.json"
)


def build_text_silver(
    raw_listings: Path,
    base_silver: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    print(f"fitting text embeddings from {raw_listings.name} ...")
    lookup, id_list, embedding_manifest = fit_embeddings(raw_listings)
    print(f"  {len(id_list):,} listings, {lookup.shape[1]} dimensions")

    output.parent.mkdir(parents=True, exist_ok=True)
    missing_text = 0
    rows = 0
    with base_silver.open(encoding="utf-8", newline="") as source, output.open(
        "w", encoding="utf-8", newline=""
    ) as dest:
        reader = csv.DictReader(source)
        base_fields = list(reader.fieldnames or [])
        all_fields = base_fields + list(TEXT_FEATURE_NAMES)
        writer = csv.DictWriter(dest, fieldnames=all_fields)
        writer.writeheader()
        for row in reader:
            rows += 1
            listing_id = row["listing_id"]
            embeddings = embedding_for_listing(listing_id, lookup, id_list)
            if any(v == "" for v in embeddings.values()):
                missing_text += 1
            row.update(embeddings)
            writer.writerow(row)

    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "base_silver": str(base_silver.relative_to(ROOT)),
        "base_silver_sha256": sha256_file(base_silver),
        "raw_listings": str(raw_listings.relative_to(ROOT)),
        "raw_listings_sha256": sha256_file(raw_listings),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256_file(output),
        "text_features": {
            "names": list(TEXT_FEATURE_NAMES),
            "count": len(TEXT_FEATURE_NAMES),
            "missing_count": missing_text,
            "missing_rate": missing_text / rows if rows else 0,
        },
        "embedding_manifest": embedding_manifest,
        "rows": rows,
    }
    write_json_atomic(report_path, report)
    print(f"text silver {output}")
    print(f"  {rows} rows, {missing_text} missing text ({missing_text / rows:.1%})")
    print(f"report      {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-listings", type=Path, default=RAW_LISTINGS)
    parser.add_argument("--base-silver", type=Path, default=BASE_SILVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_listings.exists():
        raise FileNotFoundError(
            f"Raw listings not found at {args.raw_listings}. "
            "Run inside_airbnb_phase0.py download first."
        )
    if not args.base_silver.exists():
        raise FileNotFoundError(
            f"Base Silver not found at {args.base_silver}. "
            "Run prepare_inside_airbnb_quotes.py first."
        )
    build_text_silver(args.raw_listings, args.base_silver, args.output, args.report)


if __name__ == "__main__":
    main()
