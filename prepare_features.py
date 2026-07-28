"""Repair processed geographic features by joining postcode data on listing ID."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AREA_COLUMNS = ("area_Manly", "area_Pittwater", "area_Warringah")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def postcode_area(value: str) -> str:
    postcode = int(float(value))
    if postcode in {2092, 2094, 2095}:
        return "Manly"
    if postcode in {2093, 2085, 2086, 2087, 2096, 2097, 2099, 2100, 2101, 2084}:
        return "Warringah"
    if postcode in {2102, 2103, 2104, 2105, 2106, 2107, 2108}:
        return "Pittwater"
    raise ValueError(f"Postcode {postcode} is outside the expected council areas")


def location_index() -> dict[int, str]:
    _, extracted = read_rows(ROOT / "Feature engineering" / "extracted_location.csv")
    by_id = {int(row["id"]): row["postcode"] for row in extracted}
    if len(by_id) != len(extracted):
        raise ValueError("Duplicate IDs in extracted_location.csv")

    # These 383 rows are independently checked-in address records and are the
    # authoritative source for labeled listings.
    _, labeled = read_rows(ROOT / "Datasets" / "train.csv")
    for row in labeled:
        by_id[int(row["Id"])] = row["Post Code"]
    if set(by_id) != set(range(953)):
        raise ValueError("Location data must contain every ID from 0 through 952")
    return by_id


def repair(source: Path, destination: Path, postcodes: dict[int, str]) -> dict[str, int]:
    fields, rows = read_rows(source)
    if "Id" not in fields or not set(AREA_COLUMNS).issubset(fields):
        raise ValueError(f"Unexpected processed feature schema in {source}")

    changed = 0
    for row in rows:
        listing_id = int(float(row["Id"]))
        area = postcode_area(postcodes[listing_id])
        expected = {column: "1" if column == f"area_{area}" else "0" for column in AREA_COLUMNS}
        if any(float(row[column]) != float(expected[column]) for column in AREA_COLUMNS):
            changed += 1
        row.update(expected)

    ids = [int(float(row["Id"])) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate listing IDs in {source}")
    if any(sum(float(row[column]) for column in AREA_COLUMNS) != 1 for row in rows):
        raise ValueError("Every row must belong to exactly one area")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "area_rows_changed": changed}


def main() -> None:
    postcodes = location_index()
    jobs = {
        "train": (
            ROOT / "Exploratory Data Analysis" / "EDA_X_train.csv",
            ROOT / "data" / "processed" / "train_features.csv",
        ),
        "test": (
            ROOT / "Exploratory Data Analysis" / "EDA_X_test.csv",
            ROOT / "data" / "processed" / "test_features.csv",
        ),
    }
    for name, (source, destination) in jobs.items():
        stats = repair(source, destination, postcodes)
        print(f"{name}: {stats['rows']} rows; corrected area on {stats['area_rows_changed']} rows")
        print(f"  -> {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
