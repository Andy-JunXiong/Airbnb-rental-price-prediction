"""Download and audit a pinned Inside Airbnb snapshot.

The Phase 0 workflow deliberately downloads each source file once, keeps raw
files out of Git, records immutable hashes, and performs streaming validation
before any modelling work begins.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = ROOT / "config" / "inside_airbnb_snapshots.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "raw" / "inside_airbnb"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "inside_airbnb"
BUFFER_SIZE = 1024 * 1024
MONEY_PATTERN = re.compile(r"[^0-9.+-]")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_registry(
    path: Path, city: str, snapshot_date: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        snapshot
        for snapshot in registry["snapshots"]
        if snapshot["city"] == city and snapshot["snapshot_date"] == snapshot_date
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one registry entry for {city}/{snapshot_date}; found {len(matches)}"
        )
    return registry["provider"], matches[0]


def snapshot_directory(data_root: Path, snapshot: dict[str, Any]) -> Path:
    return data_root / snapshot["city"] / f"snapshot_date={snapshot['snapshot_date']}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_fingerprint(columns: Iterable[str]) -> str:
    schema = json.dumps(list(columns), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def open_csv_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def source_schema(path: Path, kind: str) -> list[str]:
    if kind == "neighbourhood_geojson":
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        properties: set[str] = set()
        for feature in document.get("features", []):
            properties.update((feature.get("properties") or {}).keys())
        return sorted(properties)
    with open_csv_text(path) as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def download_file(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Airbnb-price-research/phase0 (single snapshot download)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            expected_size = response.headers.get("Content-Length")
            with temporary.open("wb") as output:
                while chunk := response.read(BUFFER_SIZE):
                    output.write(chunk)
        actual_size = temporary.stat().st_size
        if expected_size is not None and actual_size != int(expected_size):
            raise IOError(
                f"Incomplete download for {url}: expected {expected_size}, got {actual_size}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "downloaded", "downloaded_at_utc": utc_now()}


def build_manifest(
    provider: dict[str, Any],
    snapshot: dict[str, Any],
    raw_directory: Path,
    existing_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous_files = {
        row["name"]: row for row in (existing_manifest or {}).get("files", [])
    }
    files = []
    for source in snapshot["files"]:
        path = raw_directory / source["name"]
        columns = source_schema(path, source["kind"])
        previous = previous_files.get(source["name"], {})
        files.append(
            {
                "name": source["name"],
                "kind": source["kind"],
                "source_url": source["url"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "columns": columns,
                "schema_fingerprint": schema_fingerprint(columns),
                "downloaded_at_utc": previous.get("downloaded_at_utc", utc_now()),
            }
        )
    return {
        "manifest_version": 1,
        "provider": provider,
        "city": snapshot["city"],
        "display_name": snapshot["display_name"],
        "snapshot_date": snapshot["snapshot_date"],
        "calendar_horizon_days": snapshot["calendar_horizon_days"],
        "generated_at_utc": utc_now(),
        "files": files,
    }


def download_snapshot(
    provider: dict[str, Any], snapshot: dict[str, Any], data_root: Path
) -> Path:
    raw_directory = snapshot_directory(data_root, snapshot)
    raw_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_directory / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    expected_hashes = {
        row["name"]: row["sha256"]
        for row in (existing_manifest or {}).get("files", [])
    }

    for source in snapshot["files"]:
        destination = raw_directory / source["name"]
        if destination.exists():
            expected = expected_hashes.get(source["name"])
            if expected and sha256_file(destination) != expected:
                raise ValueError(
                    f"Existing raw file differs from its manifest: {destination}"
                )
            print(f"reuse    {source['name']} ({destination.stat().st_size:,} bytes)")
            continue
        print(f"download {source['name']}")
        download_file(source["url"], destination)
        print(f"saved    {source['name']} ({destination.stat().st_size:,} bytes)")

    manifest = build_manifest(provider, snapshot, raw_directory, existing_manifest)
    write_json_atomic(manifest_path, manifest)
    print(f"manifest {manifest_path}")
    return manifest_path


def check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    details: Any,
) -> None:
    checks.append({"name": name, "status": status, "details": details})


def parse_identifier(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def parse_money(value: str | None) -> float | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    normalized = MONEY_PATTERN.sub("", cleaned.replace(",", ""))
    if not normalized:
        raise ValueError(f"Invalid money value: {value!r}")
    result = float(normalized)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite money value: {value!r}")
    return result


def parse_iso_date(value: str | None) -> date | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    return date.fromisoformat(cleaned)


def missing_required(columns: list[str], required: list[str]) -> list[str]:
    return sorted(set(required) - set(columns))


def audit_listings(
    path: Path, required: list[str], checks: list[dict[str, Any]]
) -> tuple[
    dict[str, Any], set[str], set[str], dict[str, date], dict[str, date]
]:
    ids: set[str] = set()
    duplicate_ids = 0
    missing_ids = 0
    host_ids: set[str] = set()
    missing_host_ids = 0
    neighbourhoods: set[str] = set()
    room_types: Counter[str] = Counter()
    price_values: list[float] = []
    price_missing = 0
    price_errors = 0
    invalid_coordinates = 0
    invalid_as_of_dates = 0
    listing_as_of: dict[str, date] = {}
    calendar_as_of: dict[str, date] = {}
    scrape_ids: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    quote_complete = 0
    quote_partial = 0
    quote_invalid = 0
    quote_price_mismatches = 0
    quote_lead_days: list[int] = []
    quote_stay_nights: list[int] = []
    rows = 0

    with open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = missing_required(columns, required)
        check(
            checks,
            "listings.required_columns",
            "fail" if missing else "pass",
            {"missing": missing},
        )
        if missing:
            return (
                {"rows": 0, "columns": columns},
                ids,
                neighbourhoods,
                listing_as_of,
                calendar_as_of,
            )
        quote_fields = {
            "price_quote_checkin_date",
            "price_quote_checkout_date",
            "price_quote_price_per_night",
        }
        has_quote_fields = quote_fields.issubset(columns)
        check(
            checks,
            "listings.quote_fields",
            "pass" if has_quote_fields else "capability_blocker",
            {
                "available": has_quote_fields,
                "missing": sorted(quote_fields - set(columns)),
                "impact": (
                    None
                    if has_quote_fields
                    else "Quote-level target construction is unavailable."
                ),
            },
        )
        for row in reader:
            rows += 1
            listing_id = parse_identifier(row["id"])
            if listing_id is None:
                missing_ids += 1
            elif listing_id in ids:
                duplicate_ids += 1
            else:
                ids.add(listing_id)
            host_id = parse_identifier(row["host_id"])
            if host_id is None:
                missing_host_ids += 1
            else:
                host_ids.add(host_id)
            scrape_ids[(row["scrape_id"] or "").strip() or "<missing>"] += 1
            source_counts[(row["source"] or "").strip() or "<missing>"] += 1
            try:
                last_scraped = parse_iso_date(row["last_scraped"])
                calendar_last_scraped = parse_iso_date(row["calendar_last_scraped"])
                if (
                    listing_id is None
                    or last_scraped is None
                    or calendar_last_scraped is None
                ):
                    raise ValueError("Missing as-of date")
                listing_as_of[listing_id] = last_scraped
                calendar_as_of[listing_id] = calendar_last_scraped
            except ValueError:
                invalid_as_of_dates += 1
            neighbourhood = (row["neighbourhood_cleansed"] or "").strip()
            if neighbourhood:
                neighbourhoods.add(neighbourhood)
            room_types[(row["room_type"] or "").strip() or "<missing>"] += 1
            try:
                parsed_price = parse_money(row["price"])
                if parsed_price is None:
                    price_missing += 1
                else:
                    price_values.append(parsed_price)
            except ValueError:
                price_errors += 1
                parsed_price = None

            if has_quote_fields:
                quote_values = [
                    (row["price_quote_checkin_date"] or "").strip(),
                    (row["price_quote_checkout_date"] or "").strip(),
                    (row["price_quote_price_per_night"] or "").strip(),
                ]
                if all(quote_values):
                    try:
                        checkin = date.fromisoformat(quote_values[0])
                        checkout = date.fromisoformat(quote_values[1])
                        quoted_price = parse_money(quote_values[2])
                        if (
                            quoted_price is None
                            or checkout <= checkin
                            or listing_id is None
                            or listing_id not in listing_as_of
                        ):
                            raise ValueError("Invalid quote")
                        quote_complete += 1
                        quote_stay_nights.append((checkout - checkin).days)
                        quote_lead_days.append(
                            (checkin - listing_as_of[listing_id]).days
                        )
                        if (
                            parsed_price is not None
                            and not math.isclose(
                                parsed_price,
                                quoted_price,
                                rel_tol=0,
                                abs_tol=0.011,
                            )
                        ):
                            quote_price_mismatches += 1
                    except ValueError:
                        quote_invalid += 1
                elif any(quote_values):
                    quote_partial += 1
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    invalid_coordinates += 1
            except (TypeError, ValueError):
                invalid_coordinates += 1

    check(
        checks,
        "listings.primary_key",
        "fail" if duplicate_ids or missing_ids else "pass",
        {"duplicate_ids": duplicate_ids, "missing_ids": missing_ids},
    )
    check(
        checks,
        "listings.price_parse",
        "fail" if price_errors else ("warn" if price_missing else "pass"),
        {"parse_errors": price_errors, "missing": price_missing},
    )
    check(
        checks,
        "listings.coordinates",
        "fail" if invalid_coordinates else "pass",
        {"invalid_or_missing": invalid_coordinates},
    )
    check(
        checks,
        "listings.as_of_dates",
        "fail" if invalid_as_of_dates else "pass",
        {
            "invalid_or_missing": invalid_as_of_dates,
            "min": min(listing_as_of.values()).isoformat() if listing_as_of else None,
            "max": max(listing_as_of.values()).isoformat() if listing_as_of else None,
        },
    )
    if has_quote_fields:
        check(
            checks,
            "listings.quote_contract",
            (
                "fail"
                if quote_invalid or quote_price_mismatches
                else ("warn" if quote_partial else "pass")
            ),
            {
                "complete": quote_complete,
                "partial": quote_partial,
                "invalid": quote_invalid,
                "price_mismatches": quote_price_mismatches,
            },
        )
    return (
        {
            "rows": rows,
            "unique_listing_ids": len(ids),
            "unique_host_ids": len(host_ids),
            "missing_host_ids": missing_host_ids,
            "scrape_id_counts": dict(sorted(scrape_ids.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "as_of_date_range": {
                "min": min(listing_as_of.values()).isoformat()
                if listing_as_of
                else None,
                "max": max(listing_as_of.values()).isoformat()
                if listing_as_of
                else None,
            },
            "neighbourhood_count": len(neighbourhoods),
            "room_type_counts": dict(sorted(room_types.items())),
            "price": {
                "non_null": len(price_values),
                "missing": price_missing,
                "min": min(price_values) if price_values else None,
                "median": statistics.median(price_values) if price_values else None,
                "max": max(price_values) if price_values else None,
            },
            "quote": {
                "fields_available": has_quote_fields,
                "complete": quote_complete,
                "partial": quote_partial,
                "invalid": quote_invalid,
                "checkin_lead_days": {
                    "min": min(quote_lead_days) if quote_lead_days else None,
                    "median": statistics.median(quote_lead_days)
                    if quote_lead_days
                    else None,
                    "max": max(quote_lead_days) if quote_lead_days else None,
                },
                "stay_nights": {
                    "min": min(quote_stay_nights) if quote_stay_nights else None,
                    "median": statistics.median(quote_stay_nights)
                    if quote_stay_nights
                    else None,
                    "max": max(quote_stay_nights) if quote_stay_nights else None,
                },
            },
        },
        ids,
        neighbourhoods,
        listing_as_of,
        calendar_as_of,
    )


def audit_calendar(
    path: Path,
    required: list[str],
    listing_ids: set[str],
    calendar_as_of: dict[str, date],
    horizon_days: int,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    listing_counts: defaultdict[str, int] = defaultdict(int)
    date_masks: defaultdict[str, int] = defaultdict(int)
    duplicate_keys = 0
    invalid_dates = 0
    outside_horizon = 0
    orphan_rows = 0
    orphan_ids: set[str] = set()
    calendar_listing_ids: set[str] = set()
    availability: Counter[str] = Counter()
    price_non_null = 0
    price_missing = 0
    price_errors = 0
    price_min: float | None = None
    price_max: float | None = None
    minimum_date: date | None = None
    maximum_date: date | None = None
    rows = 0

    with open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = missing_required(columns, required)
        check(
            checks,
            "calendar.required_columns",
            "fail" if missing else "pass",
            {"missing": missing},
        )
        if missing:
            return {"rows": 0, "columns": columns}
        has_daily_price = "price" in columns
        for row in reader:
            rows += 1
            listing_id = parse_identifier(row["listing_id"])
            if listing_id is None:
                orphan_rows += 1
                continue
            calendar_listing_ids.add(listing_id)
            listing_counts[listing_id] += 1
            if listing_id not in listing_ids:
                orphan_rows += 1
                if len(orphan_ids) < 20:
                    orphan_ids.add(listing_id)
            try:
                calendar_date = parse_iso_date(row["date"])
            except ValueError:
                calendar_date = None
            if calendar_date is None:
                invalid_dates += 1
            else:
                minimum_date = (
                    calendar_date
                    if minimum_date is None
                    else min(minimum_date, calendar_date)
                )
                maximum_date = (
                    calendar_date
                    if maximum_date is None
                    else max(maximum_date, calendar_date)
                )
                as_of = calendar_as_of.get(listing_id)
                offset = (calendar_date - as_of).days if as_of else -1
                if as_of is not None and 0 <= offset <= horizon_days:
                    bit = 1 << offset
                    if date_masks[listing_id] & bit:
                        duplicate_keys += 1
                    date_masks[listing_id] |= bit
                else:
                    outside_horizon += 1
            availability[(row["available"] or "").strip().lower()] += 1
            if has_daily_price:
                try:
                    parsed_price = parse_money(row["price"])
                    if parsed_price is None:
                        price_missing += 1
                    else:
                        price_non_null += 1
                        price_min = (
                            parsed_price
                            if price_min is None
                            else min(price_min, parsed_price)
                        )
                        price_max = (
                            parsed_price
                            if price_max is None
                            else max(price_max, parsed_price)
                        )
                except ValueError:
                    price_errors += 1

    coverage = list(listing_counts.values())
    missing_calendar_listings = len(listing_ids - calendar_listing_ids)
    invalid_availability = sorted(set(availability) - {"t", "f"})
    daily_price_available = has_daily_price and price_non_null > 0
    check(
        checks,
        "calendar.daily_listed_price",
        "pass" if daily_price_available else "capability_blocker",
        {
            "column_present": has_daily_price,
            "non_null_rows": price_non_null,
            "available": daily_price_available,
            "impact": (
                None
                if daily_price_available
                else "Date-level listed-price modelling is unavailable."
            ),
        },
    )
    check(
        checks,
        "calendar.primary_key",
        "fail" if duplicate_keys or invalid_dates else "pass",
        {"duplicate_listing_dates": duplicate_keys, "invalid_dates": invalid_dates},
    )
    check(
        checks,
        "calendar.listing_foreign_key",
        "fail" if orphan_rows else "pass",
        {
            "orphan_rows": orphan_rows,
            "orphan_id_examples": sorted(orphan_ids),
            "listings_without_calendar": missing_calendar_listings,
        },
    )
    check(
        checks,
        "calendar.date_horizon",
        "fail" if outside_horizon else "pass",
        {
            "outside_horizon": outside_horizon,
            "min": minimum_date.isoformat() if minimum_date else None,
            "max": maximum_date.isoformat() if maximum_date else None,
        },
    )
    check(
        checks,
        "calendar.availability_values",
        "fail" if invalid_availability else "pass",
        {"invalid": invalid_availability},
    )
    check(
        checks,
        "calendar.price_parse",
        "fail" if price_errors else ("warn" if price_missing else "pass"),
        {"parse_errors": price_errors, "missing": price_missing},
    )
    return {
        "rows": rows,
        "unique_listing_ids": len(calendar_listing_ids),
        "listings_without_calendar": missing_calendar_listings,
        "rows_per_listing": {
            "min": min(coverage) if coverage else None,
            "median": statistics.median(coverage) if coverage else None,
            "max": max(coverage) if coverage else None,
        },
        "date_range": {
            "min": minimum_date.isoformat() if minimum_date else None,
            "max": maximum_date.isoformat() if maximum_date else None,
        },
        "availability_counts": dict(sorted(availability.items())),
        "price": {
            "non_null": price_non_null,
            "missing": price_missing,
            "min": price_min,
            "max": price_max,
        },
    }


def audit_reviews(
    path: Path,
    required: list[str],
    listing_ids: set[str],
    listing_as_of: dict[str, date],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    review_listing_ids: set[str] = set()
    orphan_rows = 0
    future_dates = 0
    invalid_dates = 0
    minimum_date: date | None = None
    maximum_date: date | None = None
    rows = 0
    with open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = missing_required(columns, required)
        check(
            checks,
            "reviews.required_columns",
            "fail" if missing else "pass",
            {"missing": missing},
        )
        if missing:
            return {"rows": 0, "columns": columns}
        for row in reader:
            rows += 1
            listing_id = parse_identifier(row["listing_id"])
            if listing_id is None or listing_id not in listing_ids:
                orphan_rows += 1
            else:
                review_listing_ids.add(listing_id)
            try:
                review_date = parse_iso_date(row["date"])
            except ValueError:
                review_date = None
            if review_date is None:
                invalid_dates += 1
                continue
            minimum_date = (
                review_date if minimum_date is None else min(minimum_date, review_date)
            )
            maximum_date = (
                review_date if maximum_date is None else max(maximum_date, review_date)
            )
            as_of = listing_as_of.get(listing_id or "")
            if as_of is not None and review_date > as_of:
                future_dates += 1
    check(
        checks,
        "reviews.listing_foreign_key",
        "fail" if orphan_rows else "pass",
        {"orphan_rows": orphan_rows},
    )
    check(
        checks,
        "reviews.as_of_date",
        "fail" if future_dates or invalid_dates else "pass",
        {"future_dates": future_dates, "invalid_dates": invalid_dates},
    )
    return {
        "rows": rows,
        "unique_listing_ids": len(review_listing_ids),
        "date_range": {
            "min": minimum_date.isoformat() if minimum_date else None,
            "max": maximum_date.isoformat() if maximum_date else None,
        },
    }


def audit_neighbourhoods(
    csv_path: Path,
    geojson_path: Path,
    csv_required: list[str],
    geojson_required: list[str],
    listing_neighbourhoods: set[str],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    csv_names: set[str] = set()
    duplicate_rows = 0
    rows = 0
    with open_csv_text(csv_path) as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = missing_required(columns, csv_required)
        check(
            checks,
            "neighbourhoods.required_columns",
            "fail" if missing else "pass",
            {"missing": missing},
        )
        if not missing:
            for row in reader:
                rows += 1
                name = (row["neighbourhood"] or "").strip()
                if name in csv_names:
                    duplicate_rows += 1
                if name:
                    csv_names.add(name)

    document = json.loads(geojson_path.read_text(encoding="utf-8-sig"))
    geojson_names: set[str] = set()
    geometry_types: Counter[str] = Counter()
    property_names: set[str] = set()
    for feature in document.get("features", []):
        properties = feature.get("properties") or {}
        property_names.update(properties)
        name = str(properties.get("neighbourhood") or "").strip()
        if name:
            geojson_names.add(name)
        geometry_types[str((feature.get("geometry") or {}).get("type") or "<missing>")] += 1

    missing_geo_columns = sorted(set(geojson_required) - property_names)
    check(
        checks,
        "neighbourhood_geojson.required_properties",
        "fail" if missing_geo_columns else "pass",
        {"missing": missing_geo_columns},
    )
    check(
        checks,
        "neighbourhoods.unique",
        "fail" if duplicate_rows else "pass",
        {"duplicate_rows": duplicate_rows},
    )
    csv_geo_difference = sorted(csv_names ^ geojson_names)
    check(
        checks,
        "neighbourhoods.csv_geojson_alignment",
        "fail" if csv_geo_difference else "pass",
        {"symmetric_difference": csv_geo_difference},
    )
    uncovered_listings = sorted(listing_neighbourhoods - csv_names)
    check(
        checks,
        "neighbourhoods.listing_coverage",
        "fail" if uncovered_listings else "pass",
        {"uncovered": uncovered_listings},
    )
    return {
        "csv_rows": rows,
        "csv_unique_neighbourhoods": len(csv_names),
        "geojson_features": len(document.get("features", [])),
        "geojson_unique_neighbourhoods": len(geojson_names),
        "geometry_type_counts": dict(sorted(geometry_types.items())),
    }


def verify_manifest(
    manifest: dict[str, Any], raw_directory: Path, checks: list[dict[str, Any]]
) -> None:
    mismatches = []
    for row in manifest.get("files", []):
        path = raw_directory / row["name"]
        if not path.exists():
            mismatches.append({"name": row["name"], "reason": "missing"})
            continue
        actual_hash = sha256_file(path)
        if actual_hash != row["sha256"]:
            mismatches.append(
                {
                    "name": row["name"],
                    "reason": "sha256",
                    "expected": row["sha256"],
                    "actual": actual_hash,
                }
            )
    check(
        checks,
        "manifest.file_integrity",
        "fail" if mismatches else "pass",
        {"mismatches": mismatches},
    )


def audit_snapshot(
    provider: dict[str, Any],
    snapshot: dict[str, Any],
    data_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    raw_directory = snapshot_directory(data_root, snapshot)
    manifest_path = raw_directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest; run download first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {source["kind"]: source for source in snapshot["files"]}
    checks: list[dict[str, Any]] = []
    verify_manifest(manifest, raw_directory, checks)

    (
        listings,
        listing_ids,
        listing_neighbourhoods,
        listing_as_of,
        calendar_as_of,
    ) = audit_listings(
        raw_directory / sources["listings"]["name"],
        sources["listings"]["required_columns"],
        checks,
    )
    calendar = audit_calendar(
        raw_directory / sources["calendar"]["name"],
        sources["calendar"]["required_columns"],
        listing_ids,
        calendar_as_of,
        snapshot["calendar_horizon_days"],
        checks,
    )
    reviews = audit_reviews(
        raw_directory / sources["reviews"]["name"],
        sources["reviews"]["required_columns"],
        listing_ids,
        listing_as_of,
        checks,
    )
    neighbourhoods = audit_neighbourhoods(
        raw_directory / sources["neighbourhoods"]["name"],
        raw_directory / sources["neighbourhood_geojson"]["name"],
        sources["neighbourhoods"]["required_columns"],
        sources["neighbourhood_geojson"]["required_columns"],
        listing_neighbourhoods,
        checks,
    )

    blockers = [row["name"] for row in checks if row["status"] == "fail"]
    capability_blockers = [
        row["name"] for row in checks if row["status"] == "capability_blocker"
    ]
    warnings = [row["name"] for row in checks if row["status"] == "warn"]
    calendar_daily_price_available = not any(
        row["name"] == "calendar.daily_listed_price"
        and row["status"] == "capability_blocker"
        for row in checks
    )
    quote_fields_available = not any(
        row["name"] == "listings.quote_fields"
        and row["status"] == "capability_blocker"
        for row in checks
    )
    if blockers:
        decision_status = "NO_GO"
    elif quote_fields_available and not calendar_daily_price_available:
        decision_status = "GO_QUOTE_LEVEL_MVP"
    elif calendar_daily_price_available and not quote_fields_available:
        decision_status = "GO_DAILY_PRICE_HISTORICAL"
    elif calendar_daily_price_available and quote_fields_available:
        decision_status = "GO_FULL_PRICE_MVP"
    else:
        decision_status = "GO_AVAILABILITY_ONLY"
    if decision_status == "GO_QUOTE_LEVEL_MVP":
        scope = (
            "Public quoted-price estimation only; the current calendar has "
            "availability but no daily price. Availability and reviews remain "
            "explicitly labelled proxies."
        )
    elif decision_status == "GO_DAILY_PRICE_HISTORICAL":
        scope = (
            "Historical listing-date advertised-price analysis; quote-level "
            "target construction is unavailable."
        )
    elif decision_status == "GO_FULL_PRICE_MVP":
        scope = "Both quote-level and listing-date advertised-price analysis."
    else:
        scope = "Availability and review proxy analysis only."
    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "provider": provider["name"],
        "city": snapshot["city"],
        "snapshot_date": snapshot["snapshot_date"],
        "source_manifest": str(manifest_path.relative_to(ROOT))
        if manifest_path.is_relative_to(ROOT)
        else str(manifest_path),
        "source_files": manifest["files"],
        "decision": {
            "status": decision_status,
            "blockers": blockers,
            "capability_blockers": capability_blockers,
            "warnings": warnings,
            "scope": scope,
        },
        "checks": checks,
        "datasets": {
            "listings": listings,
            "calendar": calendar,
            "reviews": reviews,
            "neighbourhoods": neighbourhoods,
        },
    }
    write_json_atomic(report_path, report)
    print(f"decision {report['decision']['status']}")
    print(f"report   {report_path}")
    if blockers:
        print(f"blockers {', '.join(blockers)}")
    if capability_blockers:
        print(f"limited  {', '.join(capability_blockers)}")
    if warnings:
        print(f"warnings {', '.join(warnings)}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and audit a pinned Inside Airbnb snapshot."
    )
    parser.add_argument("command", choices=("download", "audit", "all"))
    parser.add_argument("--city", default="sydney")
    parser.add_argument("--snapshot-date", default="2026-06-16")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider, snapshot = load_registry(args.registry, args.city, args.snapshot_date)
    report = args.report or (
        DEFAULT_REPORT_ROOT
        / f"{snapshot['city']}_{snapshot['snapshot_date']}_phase0_audit.json"
    )
    if args.command in {"download", "all"}:
        download_snapshot(provider, snapshot, args.data_root)
    if args.command in {"audit", "all"}:
        result = audit_snapshot(provider, snapshot, args.data_root, report)
        if result["decision"]["status"] == "NO_GO":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
