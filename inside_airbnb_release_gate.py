"""Enforce research or production release authority for the quote model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib

from inside_airbnb_manifest import REPORT_PATHS, display_path
from inside_airbnb_phase0 import ROOT, sha256_file, utc_now, write_json_atomic
from inside_airbnb_quote_model import DEFAULT_ARTIFACT, DEFAULT_REPORT, DEFAULT_SILVER


DEFAULT_RESEARCH_OUTPUT = (
    ROOT / "reports" / "inside_airbnb" / "research_release_gate.json"
)
DEFAULT_PRODUCTION_OUTPUT = (
    ROOT / "reports" / "inside_airbnb" / "production_release_gate.json"
)
DEFAULT_PIPELINE_RUN = (
    ROOT / "reports" / "inside_airbnb" / "pipeline_run.json"
)


def check(name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def finalize_gate(target: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = all(row["passed"] for row in checks)
    return {
        "target": target,
        "passed": passed,
        "status": (
            f"{target.upper()}_RELEASE_ALLOWED"
            if passed
            else f"{target.upper()}_RELEASE_BLOCKED"
        ),
        "failed_checks": [
            row["name"] for row in checks if not row["passed"]
        ],
        "checks": checks,
    }


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def silver_snapshot_label(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    return row.get("snapshot_label") if row else None


def evaluate_release(
    target: str,
    silver_path: Path = DEFAULT_SILVER,
    artifact_path: Path = DEFAULT_ARTIFACT,
    model_report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    checks = []
    model_report = read_json(model_report_path)
    artifact = joblib.load(artifact_path) if artifact_path.exists() else None
    silver_hash = sha256_file(silver_path) if silver_path.exists() else None
    snapshot_label = silver_snapshot_label(silver_path)
    pipeline_run = read_json(DEFAULT_PIPELINE_RUN)
    command_status = {
        row.get("name"): row.get("status")
        for row in (pipeline_run or {}).get("commands", [])
    }

    checks.extend(
        [
            check("silver_exists", silver_path.exists(), silver_path.exists(), True),
            check(
                "primary_artifact_exists",
                artifact is not None,
                artifact is not None,
                True,
            ),
            check(
                "model_report_exists",
                model_report is not None,
                model_report is not None,
                True,
            ),
            check(
                "pipeline_compile_passed",
                command_status.get("compile") == "passed",
                command_status.get("compile"),
                "passed",
            ),
            check(
                "pipeline_unit_tests_passed",
                command_status.get("unit_tests") == "passed",
                command_status.get("unit_tests"),
                "passed",
            ),
        ]
    )
    if artifact and model_report and silver_hash:
        model_mae = model_report["model"]["test_metrics_all"]["mae"]
        baseline_mae = model_report["market_median_baseline"]["test_metrics_all"][
            "mae"
        ]
        checks.extend(
            [
                check(
                    "artifact_version_supported",
                    artifact.get("artifact_version", 0) >= 2,
                    artifact.get("artifact_version"),
                    ">=2",
                ),
                check(
                    "artifact_silver_hash_matches",
                    artifact.get("training_silver_sha256") == silver_hash,
                    artifact.get("training_silver_sha256"),
                    silver_hash,
                ),
                check(
                    "artifact_snapshot_matches_silver",
                    artifact.get("snapshot_label") == snapshot_label,
                    artifact.get("snapshot_label"),
                    snapshot_label,
                ),
                check(
                    "model_beats_market_baseline",
                    model_mae < baseline_mae,
                    model_mae,
                    f"<{baseline_mae}",
                ),
                check(
                    "artifact_authority_matches_report",
                    artifact.get("deployment_authority")
                    == model_report["temporal_validation"][
                        "deployment_authority"
                    ],
                    artifact.get("deployment_authority"),
                    model_report["temporal_validation"][
                        "deployment_authority"
                    ],
                ),
            ]
        )

    for name in (
        "upper_tail_challenger",
        "premium_challenger",
        "interval_challenger",
    ):
        document = read_json(REPORT_PATHS[name])
        checks.append(
            check(
                f"{name}_did_not_write_primary_artifact",
                bool(
                    document
                    and not document.get("governance", {}).get(
                        "primary_artifact_written", True
                    )
                ),
                (
                    document.get("governance", {}).get(
                        "primary_artifact_written"
                    )
                    if document
                    else None
                ),
                False,
            )
        )

    if target == "production":
        temporal = read_json(REPORT_PATHS["temporal_validation"])
        compatibility = read_json(REPORT_PATHS["target_compatibility"])
        temporal_passed = bool(
            temporal and temporal.get("evidence_gate", {}).get("passed")
        )
        checks.extend(
            [
                check(
                    "artifact_is_temporally_validated",
                    bool(
                        artifact
                        and artifact.get("deployment_authority")
                        == "temporally_validated"
                    ),
                    artifact.get("deployment_authority") if artifact else None,
                    "temporally_validated",
                ),
                check(
                    "model_report_temporal_gate_passed",
                    bool(
                        model_report
                        and model_report.get("temporal_validation", {}).get(
                            "out_of_time_evidence_gate_passed"
                        )
                    ),
                    (
                        model_report.get("temporal_validation", {}).get(
                            "out_of_time_evidence_gate_passed"
                        )
                        if model_report
                        else None
                    ),
                    True,
                ),
                check(
                    "temporal_report_exists_and_passes",
                    temporal_passed,
                    temporal_passed,
                    True,
                ),
                check(
                    "temporal_report_matches_training_silver",
                    bool(
                        temporal
                        and temporal.get("snapshots", {})
                        .get("older", {})
                        .get("source_sha256")
                        == silver_hash
                    ),
                    (
                        temporal.get("snapshots", {})
                        .get("older", {})
                        .get("source_sha256")
                        if temporal
                        else None
                    ),
                    silver_hash,
                ),
                check(
                    "target_compatibility_ready",
                    bool(
                        compatibility
                        and compatibility.get("target_compatibility", {}).get(
                            "status"
                        )
                        == "TEMPORAL_PRICE_VALIDATION_READY"
                    ),
                    (
                        compatibility.get("target_compatibility", {}).get(
                            "status"
                        )
                        if compatibility
                        else None
                    ),
                    "TEMPORAL_PRICE_VALIDATION_READY",
                ),
            ]
        )
    elif target != "research":
        raise ValueError(f"Unknown release target: {target}")

    decision = finalize_gate(target, checks)
    return {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "artifact": display_path(artifact_path),
        "silver": display_path(silver_path),
        "model_report": display_path(model_report_path),
        **decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("research", "production"), required=True)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit non-zero when the requested release target is blocked.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (
        DEFAULT_RESEARCH_OUTPUT
        if args.target == "research"
        else DEFAULT_PRODUCTION_OUTPUT
    )
    report = evaluate_release(
        args.target, args.silver, args.artifact, args.model_report
    )
    write_json_atomic(output, report)
    print(f"decision {report['status']}")
    print(f"report   {output}")
    if args.enforce and not report["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
