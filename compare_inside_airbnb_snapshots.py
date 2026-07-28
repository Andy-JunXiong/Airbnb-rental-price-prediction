"""Assess whether two Inside Airbnb snapshots support temporal price validation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import date
from pathlib import Path
from typing import Any

from inside_airbnb_phase0 import ROOT, utc_now, write_json_atomic


DEFAULT_OLDER_DATE = "2025-09-12"
DEFAULT_NEWER_DATE = "2026-06-16"
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_snapshot_target_compatibility.json"
)


def phase0_report(snapshot_date: str) -> Path:
    return (
        ROOT
        / "reports"
        / "inside_airbnb"
        / f"sydney_{snapshot_date}_phase0_audit.json"
    )


def raw_listings(snapshot_date: str) -> Path:
    return (
        ROOT
        / "data"
        / "raw"
        / "inside_airbnb"
        / "sydney"
        / f"snapshot_date={snapshot_date}"
        / "listings.csv.gz"
    )


def target_capabilities(report: dict[str, Any]) -> dict[str, bool]:
    listings = report["datasets"]["listings"]
    calendar = report["datasets"]["calendar"]
    return {
        "quote_price_with_context": bool(
            listings["quote"].get("fields_available")
            and listings["quote"].get("complete", 0) > 0
        ),
        "listing_price": listings["price"].get("non_null", 0) > 0,
        "calendar_daily_price": calendar["price"].get("non_null", 0) > 0,
        "availability_proxy": calendar.get("rows", 0) > 0,
    }


def source_schemas(report: dict[str, Any]) -> dict[str, set[str]]:
    return {
        row["kind"]: set(row["columns"])
        for row in report.get("source_files", [])
    }


def identities(path: Path) -> tuple[set[str], set[str]]:
    listings: set[str] = set()
    hosts: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            listing_id = (row.get("id") or "").strip()
            host_id = (row.get("host_id") or "").strip()
            if listing_id:
                listings.add(listing_id)
            if host_id:
                hosts.add(host_id)
    return listings, hosts


def compatibility_decision(
    older_capabilities: dict[str, bool],
    newer_capabilities: dict[str, bool],
) -> tuple[str, list[str], list[str]]:
    price_targets = [
        "quote_price_with_context",
        "listing_price",
        "calendar_daily_price",
    ]
    common = [
        target
        for target in price_targets
        if older_capabilities[target] and newer_capabilities[target]
    ]
    blockers = []
    if not common:
        blockers.append("no_common_non_null_price_target")
    if not any(older_capabilities[target] for target in price_targets):
        blockers.append("older_snapshot_has_no_non_null_price_labels")
    if not any(newer_capabilities[target] for target in price_targets):
        blockers.append("newer_snapshot_has_no_non_null_price_labels")
    status = (
        "TEMPORAL_PRICE_VALIDATION_READY"
        if not blockers
        else "TEMPORAL_PRICE_VALIDATION_BLOCKED"
    )
    return status, blockers, common


def compare_snapshots(
    older_date: str,
    newer_date: str,
    older_report_path: Path,
    newer_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    older_report = json.loads(older_report_path.read_text(encoding="utf-8"))
    newer_report = json.loads(newer_report_path.read_text(encoding="utf-8"))
    older_capabilities = target_capabilities(older_report)
    newer_capabilities = target_capabilities(newer_report)
    status, blockers, common_targets = compatibility_decision(
        older_capabilities, newer_capabilities
    )
    older_schemas = source_schemas(older_report)
    newer_schemas = source_schemas(newer_report)
    kinds = sorted(set(older_schemas) | set(newer_schemas))
    schema_drift = {
        kind: {
            "added_in_newer": sorted(newer_schemas.get(kind, set()) - older_schemas.get(kind, set())),
            "removed_in_newer": sorted(older_schemas.get(kind, set()) - newer_schemas.get(kind, set())),
        }
        for kind in kinds
    }
    older_listings, older_hosts = identities(raw_listings(older_date))
    newer_listings, newer_hosts = identities(raw_listings(newer_date))
    listing_overlap = older_listings & newer_listings
    host_overlap = older_hosts & newer_hosts
    older_as_of = older_report["datasets"]["listings"]["as_of_date_range"]
    newer_as_of = newer_report["datasets"]["listings"]["as_of_date_range"]
    chronology_valid = date.fromisoformat(older_as_of["max"]) < date.fromisoformat(
        newer_as_of["min"]
    )

    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "city": "sydney",
        "snapshots": {
            "older": {
                "snapshot_label": older_date,
                "as_of_date_range": older_as_of,
                "decision": older_report["decision"]["status"],
                "target_capabilities": older_capabilities,
            },
            "newer": {
                "snapshot_label": newer_date,
                "as_of_date_range": newer_as_of,
                "decision": newer_report["decision"]["status"],
                "target_capabilities": newer_capabilities,
            },
        },
        "chronology": {
            "valid": chronology_valid,
            "gap_days": (
                date.fromisoformat(newer_as_of["min"])
                - date.fromisoformat(older_as_of["max"])
            ).days,
        },
        "population_overlap": {
            "older_listings": len(older_listings),
            "newer_listings": len(newer_listings),
            "listing_overlap": len(listing_overlap),
            "older_listing_retention_rate": len(listing_overlap)
            / max(len(older_listings), 1),
            "newer_listing_seen_rate": len(listing_overlap)
            / max(len(newer_listings), 1),
            "older_hosts": len(older_hosts),
            "newer_hosts": len(newer_hosts),
            "host_overlap": len(host_overlap),
            "newer_host_seen_rate": len(host_overlap) / max(len(newer_hosts), 1),
        },
        "schema_drift": schema_drift,
        "target_compatibility": {
            "status": status,
            "common_non_null_price_targets": common_targets,
            "blockers": blockers,
            "interpretation": (
                "The older free snapshot has no non-null price labels, while "
                "the newer snapshot introduces quote-context labels and removes "
                "calendar price. A price backtest across these snapshots would "
                "change the supervised target and is therefore invalid."
                if blockers
                else "At least one price target is comparable across time."
            ),
        },
        "authority": {
            "current_quote_model": "research_only",
            "next_evidence_required": (
                "A later Sydney snapshot with the same non-null quote fields as "
                "2026-06-16, or an approved archived dataset with compatible "
                "price labels."
                if blockers
                else (
                    "Run the governed out-of-time evaluation. Target "
                    "compatibility alone does not validate model performance."
                )
            ),
        },
    }
    if not chronology_valid:
        report["target_compatibility"]["status"] = "TEMPORAL_PRICE_VALIDATION_BLOCKED"
        report["target_compatibility"]["blockers"].append(
            "snapshot_as_of_ranges_not_strictly_ordered"
        )
        report["authority"]["current_quote_model"] = "research_only"
    write_json_atomic(output, report)
    print(f"decision {report['target_compatibility']['status']}")
    print(f"report   {output}")
    if report["target_compatibility"]["blockers"]:
        print(
            "blockers "
            + ", ".join(report["target_compatibility"]["blockers"])
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-date", default=DEFAULT_OLDER_DATE)
    parser.add_argument("--newer-date", default=DEFAULT_NEWER_DATE)
    parser.add_argument("--older-report", type=Path)
    parser.add_argument("--newer-report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_snapshots(
        args.older_date,
        args.newer_date,
        args.older_report or phase0_report(args.older_date),
        args.newer_report or phase0_report(args.newer_date),
        args.output,
    )


if __name__ == "__main__":
    main()
