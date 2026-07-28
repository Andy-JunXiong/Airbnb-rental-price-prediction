"""Build a privacy-minimised Silver table for public Airbnb quote modelling."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from inside_airbnb_phase0 import ROOT, parse_money, utc_now, write_json_atomic
from sydney_geography import (
    GEOGRAPHIC_FEATURES,
    geographic_features,
    reference_manifest,
)


DEFAULT_SNAPSHOT_DATE = "2026-06-16"
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "raw"
    / "inside_airbnb"
    / "sydney"
    / f"snapshot_date={DEFAULT_SNAPSHOT_DATE}"
    / "listings.csv.gz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "silver"
    / "inside_airbnb"
    / "sydney"
    / f"snapshot_date={DEFAULT_SNAPSHOT_DATE}"
    / "listing_quotes.csv"
)
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_2026-06-16_silver_quotes.json"
)

SILVER_FIELDS = [
    "snapshot_label",
    "scrape_id",
    "listing_id",
    "host_id",
    "as_of_date",
    "source",
    "training_eligible",
    "eligibility_reason",
    "target_quoted_price_per_night",
    "currency",
    "quote_checkin_date",
    "quote_checkout_date",
    "quote_lead_days",
    "stay_nights",
    "checkin_month",
    "checkin_day_of_week",
    "checkin_is_weekend",
    "neighbourhood",
    "property_type",
    "room_type",
    "latitude",
    "longitude",
    *GEOGRAPHIC_FEATURES,
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "amenities_count",
    "minimum_nights",
    "maximum_nights",
    "host_total_listings_count",
    "calculated_host_listings_count",
    "host_is_superhost",
    "instant_bookable",
    "proxy_available_days_30",
    "proxy_available_days_60",
    "proxy_available_days_90",
    "proxy_available_days_365",
    "proxy_review_count_ltm",
    "proxy_review_count_l30d",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_number(value: str | None) -> str:
    cleaned = (value or "").strip().replace(",", "")
    if not cleaned:
        return ""
    try:
        parsed = float(cleaned)
    except ValueError:
        return ""
    return str(parsed) if math.isfinite(parsed) else ""


def amenity_count(value: str | None) -> int:
    cleaned = (value or "").strip()
    if not cleaned:
        return 0
    try:
        parsed = json.loads(cleaned)
        return len(parsed) if isinstance(parsed, list) else 0
    except json.JSONDecodeError:
        return 0


def quote_currency(value: str | None) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return "AUD"
    try:
        document = json.loads(cleaned)
    except json.JSONDecodeError:
        return "AUD"
    currency = ((document.get("quote") or {}).get("currency") or "").strip()
    return currency or "AUD"


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def transform_listing(row: dict[str, str], snapshot_label: str) -> dict[str, Any]:
    reasons: list[str] = []
    as_of: date | None = None
    checkin: date | None = None
    checkout: date | None = None
    target: float | None = None
    try:
        as_of = date.fromisoformat((row.get("last_scraped") or "").strip())
    except ValueError:
        reasons.append("invalid_as_of_date")
    try:
        checkin = date.fromisoformat(
            (row.get("price_quote_checkin_date") or "").strip()
        )
        checkout = date.fromisoformat(
            (row.get("price_quote_checkout_date") or "").strip()
        )
    except ValueError:
        reasons.append("missing_or_invalid_quote_dates")
    try:
        target = parse_money(row.get("price_quote_price_per_night"))
        if target is None or target <= 0:
            reasons.append("missing_or_invalid_quote_price")
    except ValueError:
        reasons.append("missing_or_invalid_quote_price")
        target = None

    lead_days: int | None = None
    stay_nights: int | None = None
    if as_of is not None and checkin is not None:
        lead_days = (checkin - as_of).days
        if lead_days < 0:
            reasons.append("quote_precedes_as_of_date")
    if checkin is not None and checkout is not None:
        stay_nights = (checkout - checkin).days
        if stay_nights <= 0:
            reasons.append("non_positive_stay")

    currency = quote_currency(row.get("price_quote_raw"))
    if currency != "AUD":
        reasons.append("non_aud_currency")

    latitude = clean_number(row.get("latitude"))
    longitude = clean_number(row.get("longitude"))
    geography = geographic_features(latitude, longitude)

    return {
        "snapshot_label": snapshot_label,
        "scrape_id": (row.get("scrape_id") or "").strip(),
        "listing_id": (row.get("id") or "").strip(),
        "host_id": (row.get("host_id") or "").strip(),
        "as_of_date": as_of.isoformat() if as_of else "",
        "source": (row.get("source") or "").strip(),
        "training_eligible": "1" if not reasons else "0",
        "eligibility_reason": "|".join(sorted(set(reasons))),
        "target_quoted_price_per_night": target if target is not None else "",
        "currency": currency,
        "quote_checkin_date": checkin.isoformat() if checkin else "",
        "quote_checkout_date": checkout.isoformat() if checkout else "",
        "quote_lead_days": lead_days if lead_days is not None else "",
        "stay_nights": stay_nights if stay_nights is not None else "",
        "checkin_month": checkin.month if checkin else "",
        "checkin_day_of_week": checkin.weekday() if checkin else "",
        "checkin_is_weekend": int(checkin.weekday() >= 5) if checkin else "",
        "neighbourhood": (row.get("neighbourhood_cleansed") or "").strip(),
        "property_type": (row.get("property_type") or "").strip(),
        "room_type": (row.get("room_type") or "").strip(),
        "latitude": latitude,
        "longitude": longitude,
        **geography,
        "accommodates": clean_number(row.get("accommodates")),
        "bathrooms": clean_number(row.get("bathrooms")),
        "bedrooms": clean_number(row.get("bedrooms")),
        "beds": clean_number(row.get("beds")),
        "amenities_count": amenity_count(row.get("amenities")),
        "minimum_nights": clean_number(row.get("minimum_nights")),
        "maximum_nights": clean_number(row.get("maximum_nights")),
        "host_total_listings_count": clean_number(
            row.get("host_total_listings_count")
        ),
        "calculated_host_listings_count": clean_number(
            row.get("calculated_host_listings_count")
        ),
        "host_is_superhost": (row.get("host_is_superhost") or "").strip().lower(),
        "instant_bookable": (row.get("instant_bookable") or "").strip().lower(),
        "proxy_available_days_30": clean_number(row.get("availability_30")),
        "proxy_available_days_60": clean_number(row.get("availability_60")),
        "proxy_available_days_90": clean_number(row.get("availability_90")),
        "proxy_available_days_365": clean_number(row.get("availability_365")),
        "proxy_review_count_ltm": clean_number(row.get("number_of_reviews_ltm")),
        "proxy_review_count_l30d": clean_number(row.get("number_of_reviews_l30d")),
    }


def build_silver_quotes(
    source: Path, output: Path, report_path: Path, snapshot_label: str
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    eligibility_reasons: Counter[str] = Counter()
    missing_by_feature: Counter[str] = Counter()
    ids: set[str] = set()
    duplicate_ids = 0
    target_values: list[float] = []
    rows = 0
    eligible = 0

    with gzip.open(
        source, "rt", encoding="utf-8-sig", newline=""
    ) as source_handle, output.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=SILVER_FIELDS)
        writer.writeheader()
        for raw in reader:
            rows += 1
            transformed = transform_listing(raw, snapshot_label)
            listing_id = str(transformed["listing_id"])
            if listing_id in ids:
                duplicate_ids += 1
            ids.add(listing_id)
            if transformed["training_eligible"] == "1":
                eligible += 1
                target_values.append(
                    float(transformed["target_quoted_price_per_night"])
                )
            else:
                reasons = str(transformed["eligibility_reason"]).split("|")
                eligibility_reasons.update(reason for reason in reasons if reason)
            for name, value in transformed.items():
                if value == "":
                    missing_by_feature[name] += 1
            writer.writerow(transformed)

    report = {
        "report_version": 2,
        "generated_at_utc": utc_now(),
        "snapshot_label": snapshot_label,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": file_sha256(source),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": file_sha256(output),
        "rows": rows,
        "unique_listing_ids": len(ids),
        "duplicate_listing_ids": duplicate_ids,
        "training_eligible": eligible,
        "training_ineligible": rows - eligible,
        "eligibility_reason_counts": dict(sorted(eligibility_reasons.items())),
        "target_quoted_price_per_night": {
            "min": min(target_values) if target_values else None,
            "p01": quantile(target_values, 0.01),
            "median": statistics.median(target_values) if target_values else None,
            "p99": quantile(target_values, 0.99),
            "max": max(target_values) if target_values else None,
        },
        "missing_value_counts": dict(sorted(missing_by_feature.items())),
        "privacy": {
            "direct_names_or_text_included": False,
            "host_id_purpose": "group-disjoint evaluation only",
            "proxy_features_excluded_from_primary_model": [
                name for name in SILVER_FIELDS if name.startswith("proxy_")
            ],
        },
        "geographic_features": reference_manifest(),
    }
    if duplicate_ids:
        raise ValueError(f"Silver table has {duplicate_ids} duplicate listing IDs")
    if eligible == 0:
        raise ValueError("Silver table has no training-eligible quotes")
    write_json_atomic(report_path, report)
    print(f"silver   {output}")
    print(f"eligible {eligible:,}/{rows:,}")
    print(f"report   {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--snapshot-label", default=DEFAULT_SNAPSHOT_DATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_silver_quotes(args.source, args.output, args.report, args.snapshot_label)


if __name__ == "__main__":
    main()
