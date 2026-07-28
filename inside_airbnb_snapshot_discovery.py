"""Discover official Sydney snapshots without silently mutating the registry."""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from inside_airbnb_phase0 import ROOT, utc_now, write_json_atomic


DEFAULT_INDEX_URL = "https://insideairbnb.com/get-the-data/"
DEFAULT_REGISTRY = ROOT / "config" / "inside_airbnb_snapshots.json"
DEFAULT_REPORT = (
    ROOT / "reports" / "inside_airbnb" / "sydney_snapshot_discovery.json"
)
DATE_PATH_PATTERN = re.compile(
    r"(?:https?://[^\"'<>\s]+)?"
    r"/australia/nsw/sydney/(\d{4}-\d{2}-\d{2})/"
    r"(?:data|visualisations)/[^\"'<>\s]+",
    re.IGNORECASE,
)


def fetch_index(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Airbnb-price-research/snapshot-discovery (read-only)"
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def discover_sydney_dates(index_html: str) -> list[str]:
    decoded = html.unescape(index_html)
    return sorted(set(DATE_PATH_PATTERN.findall(decoded)))


def registered_sydney_dates(registry_path: Path) -> list[str]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return sorted(
        snapshot["snapshot_date"]
        for snapshot in registry["snapshots"]
        if snapshot["city"] == "sydney"
    )


def discovery_decision(
    official_dates: list[str], registered_dates: list[str]
) -> dict[str, Any]:
    latest_official = max(official_dates) if official_dates else None
    latest_registered = max(registered_dates) if registered_dates else None
    newer = [
        snapshot_date
        for snapshot_date in official_dates
        if latest_registered is None or snapshot_date > latest_registered
    ]
    if not official_dates:
        status = "OFFICIAL_INDEX_HAS_NO_SYDNEY_SNAPSHOTS"
        action = "Investigate official index parsing before changing the registry."
    elif newer:
        status = "NEWER_SNAPSHOT_DISCOVERED"
        action = (
            "Review and pin the candidate source URLs, then run Phase 0 audit "
            "before downloading modelling outputs."
        )
    else:
        status = "NO_NEWER_SNAPSHOT"
        action = "Keep research_only authority and check again after the next release."
    return {
        "status": status,
        "latest_official_snapshot": latest_official,
        "latest_registered_snapshot": latest_registered,
        "newer_candidates": newer,
        "next_action": action,
    }


def discover(
    index_url: str,
    registry_path: Path,
    report_path: Path,
    index_html_path: Path | None = None,
) -> dict[str, Any]:
    index_html = (
        index_html_path.read_text(encoding="utf-8")
        if index_html_path
        else fetch_index(index_url)
    )
    official_dates = discover_sydney_dates(index_html)
    registered_dates = registered_sydney_dates(registry_path)
    decision = discovery_decision(official_dates, registered_dates)
    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "provider": "Inside Airbnb",
        "official_index_url": index_url,
        "city": "sydney",
        "official_snapshot_dates": official_dates,
        "registered_snapshot_dates": registered_dates,
        "decision": decision,
        "safety": {
            "registry_mutated": False,
            "files_downloaded": False,
            "automatic_authority_change": False,
        },
    }
    write_json_atomic(report_path, report)
    print(f"decision {decision['status']}")
    print(f"latest   {decision['latest_official_snapshot']}")
    print(f"report   {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--index-html",
        type=Path,
        help="Optional saved HTML for deterministic/offline testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discover(args.index_url, args.registry, args.report, args.index_html)


if __name__ == "__main__":
    main()
