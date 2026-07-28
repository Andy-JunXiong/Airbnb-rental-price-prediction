"""Build a machine-readable reproducibility manifest for the Sydney pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from inside_airbnb_phase0 import ROOT, sha256_file, utc_now, write_json_atomic


DEFAULT_OUTPUT = (
    ROOT / "reports" / "inside_airbnb" / "reproducibility_manifest.json"
)

INPUT_PATHS = {
    "snapshot_registry": ROOT / "config" / "inside_airbnb_snapshots.json",
    "older_raw_manifest": (
        ROOT
        / "data"
        / "raw"
        / "inside_airbnb"
        / "sydney"
        / "snapshot_date=2025-09-12"
        / "manifest.json"
    ),
    "current_raw_manifest": (
        ROOT
        / "data"
        / "raw"
        / "inside_airbnb"
        / "sydney"
        / "snapshot_date=2026-06-16"
        / "manifest.json"
    ),
    "silver_quotes": (
        ROOT
        / "data"
        / "silver"
        / "inside_airbnb"
        / "sydney"
        / "snapshot_date=2026-06-16"
        / "listing_quotes.csv"
    ),
    "premium_silver_quotes": (
        ROOT
        / "data"
        / "silver"
        / "inside_airbnb"
        / "sydney"
        / "snapshot_date=2026-06-16"
        / "listing_quotes_premium.csv"
    ),
}

ARTIFACT_PATHS = {
    "quote_model": ROOT / "artifacts" / "inside_airbnb_quote_mvp.joblib",
    "example_prediction": (
        ROOT / "predictions" / "inside_airbnb_quote_example.json"
    ),
}

REPORT_PATHS = {
    "phase0_current": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_phase0_audit.json"
    ),
    "snapshot_discovery": (
        ROOT / "reports" / "inside_airbnb" / "sydney_snapshot_discovery.json"
    ),
    "target_compatibility": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_snapshot_target_compatibility.json"
    ),
    "quote_model_evaluation": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_quote_mvp_evaluation.json"
    ),
    "modern_eda": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_modern_eda.json"
    ),
    "feature_ablation": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_feature_ablation.json"
    ),
    "error_analysis": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_error_analysis.json"
    ),
    "upper_tail_challenger": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_upper_tail_challenger.json"
    ),
    "premium_challenger": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_premium_challenger.json"
    ),
    "interval_challenger": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_2026-06-16_interval_challenger.json"
    ),
    "temporal_validation": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "sydney_temporal_quote_validation.json"
    ),
    "research_release_gate": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "research_release_gate.json"
    ),
    "production_release_gate": (
        ROOT
        / "reports"
        / "inside_airbnb"
        / "production_release_gate.json"
    ),
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": display_path(path), "exists": False}
    return {
        "path": display_path(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def json_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"parse_error": True}
    summary: dict[str, Any] = {"report_version": document.get("report_version")}
    decision = document.get("decision")
    if isinstance(decision, dict):
        summary["decision"] = (
            decision.get("status")
            or decision.get("recommendation")
            or decision.get("passed")
        )
    if "target_compatibility" in document:
        summary["target_compatibility"] = document["target_compatibility"].get(
            "status"
        )
    if "temporal_validation" in document:
        summary["deployment_authority"] = document["temporal_validation"].get(
            "deployment_authority"
        )
    if "authority" in document:
        summary["authority"] = document["authority"].get("current_quote_model")
    if "evidence_gate" in document:
        summary["evidence_gate_passed"] = document["evidence_gate"].get("passed")
    if "status" in document:
        summary["status"] = document["status"]
    return summary


def git_state() -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    lines = status.stdout.splitlines() if status.returncode == 0 else []
    return {
        "available": commit.returncode == 0,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(lines),
        "changed_path_count": len(lines),
    }


def dependency_versions() -> dict[str, str | None]:
    result = {}
    for package in ("numpy", "scikit-learn", "joblib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def build_manifest(
    output: Path = DEFAULT_OUTPUT,
    run_report: Path | None = None,
) -> dict[str, Any]:
    run_document = (
        json.loads(run_report.read_text(encoding="utf-8"))
        if run_report and run_report.exists()
        else None
    )
    manifest = {
        "manifest_version": 1,
        "generated_at_utc": utc_now(),
        "project": "Airbnb Sydney public quote-price research pipeline",
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "git": git_state(),
        "random_seeds": {
            "governed_test": 42,
            "conformal_calibration": 43,
            "development_folds": [101, 102, 103, 104, 105],
            "model": 42,
        },
        "inputs": {
            name: file_record(path) for name, path in INPUT_PATHS.items()
        },
        "artifacts": {
            name: file_record(path) for name, path in ARTIFACT_PATHS.items()
        },
        "reports": {
            name: {
                **file_record(path),
                "summary": json_summary(path),
            }
            for name, path in REPORT_PATHS.items()
        },
        "pipeline_run": (
            {
                "report": file_record(run_report),
                "mode": run_document.get("mode"),
                "status": run_document.get("status"),
                "commands": [
                    {
                        "name": command.get("name"),
                        "status": command.get("status"),
                        "returncode": command.get("returncode"),
                        "duration_seconds": command.get("duration_seconds"),
                    }
                    for command in run_document.get("commands", [])
                ],
            }
            if run_document
            else None
        ),
        "governance": {
            "raw_data_in_git_expected": False,
            "ci_downloads_raw_data": False,
            "production_requires_temporal_gate": True,
            "challengers_may_overwrite_primary_artifact": False,
        },
    }
    write_json_atomic(output, manifest)
    print(f"manifest {output}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(args.output, args.run_report)


if __name__ == "__main__":
    main()
