"""Run the Sydney research pipeline, CI checks, or a pinned-source refresh."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inside_airbnb_manifest import DEFAULT_OUTPUT as DEFAULT_MANIFEST
from inside_airbnb_manifest import build_manifest
from inside_airbnb_phase0 import (
    ROOT,
    active_snapshot_date,
    sha256_file,
    utc_now,
    write_json_atomic,
)


DEFAULT_RUN_REPORT = (
    ROOT / "reports" / "inside_airbnb" / "pipeline_run.json"
)
PRIMARY_ARTIFACT = ROOT / "artifacts" / "inside_airbnb_quote_mvp.joblib"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    arguments: tuple[str, ...]
    preserve_primary_artifact: bool = False


def python_command(name: str, *arguments: str, preserve: bool = False) -> CommandSpec:
    return CommandSpec(
        name=name,
        arguments=(sys.executable, *arguments),
        preserve_primary_artifact=preserve,
    )


def compile_arguments() -> tuple[str, ...]:
    sources = sorted(
        [
            str(path.relative_to(ROOT))
            for path in [*ROOT.glob("*.py"), *(ROOT / "tests").glob("test_*.py")]
        ]
    )
    return ("-m", "compileall", "-q", *sources)


def pipeline_commands(mode: str) -> list[CommandSpec]:
    ci = [
        python_command("compile", *compile_arguments()),
        python_command("unit_tests", "-m", "unittest", "discover", "-s", "tests", "-v"),
    ]
    if mode == "ci":
        return ci
    research = [
        python_command("snapshot_compatibility", "compare_inside_airbnb_snapshots.py"),
        python_command("prepare_silver", "prepare_inside_airbnb_quotes.py"),
        python_command("modern_eda", "inside_airbnb_eda.py"),
        python_command("feature_ablation", "inside_airbnb_feature_ablation.py"),
        python_command("train_primary_model", "inside_airbnb_quote_model.py", "train"),
        python_command(
            "upper_tail_challenger",
            "inside_airbnb_upper_tail_challenger.py",
            preserve=True,
        ),
        python_command(
            "prepare_premium_silver",
            "prepare_inside_airbnb_premium_features.py",
            preserve=True,
        ),
        python_command(
            "premium_challenger",
            "inside_airbnb_premium_challenger.py",
            preserve=True,
        ),
        python_command(
            "interval_challenger",
            "inside_airbnb_interval_challenger.py",
            preserve=True,
        ),
        python_command(
            "error_analysis",
            "inside_airbnb_error_analysis.py",
            preserve=True,
        ),
        python_command(
            "multi_snapshot_training",
            "inside_airbnb_multi_snapshot.py",
            preserve=True,
        ),
        python_command(
            "example_prediction",
            "inside_airbnb_quote_model.py",
            "predict",
            "--input",
            "examples/inside_airbnb_quote_request.json",
            "--output",
            "predictions/inside_airbnb_quote_example.json",
            preserve=True,
        ),
        *ci,
        python_command(
            "research_release_gate",
            "inside_airbnb_release_gate.py",
            "--target",
            "research",
            "--enforce",
            preserve=True,
        ),
        python_command(
            "production_release_gate_report",
            "inside_airbnb_release_gate.py",
            "--target",
            "production",
            preserve=True,
        ),
    ]
    if mode == "research":
        return research
    if mode == "refresh":
        return [
            python_command(
                "snapshot_discovery", "inside_airbnb_snapshot_discovery.py"
            ),
            python_command(
                "phase0_older",
                "inside_airbnb_phase0.py",
                "all",
                "--snapshot-date",
                "2025-09-12",
                "--report",
                "reports/inside_airbnb/sydney_2025-09-12_phase0_audit.json",
            ),
            python_command(
                "phase0_current",
                "inside_airbnb_phase0.py",
                "all",
                "--snapshot-date",
                active_snapshot_date(),
            ),
            *research,
        ]
    raise ValueError(f"Unknown pipeline mode: {mode}")


def hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def run_pipeline(
    mode: str,
    run_report_path: Path = DEFAULT_RUN_REPORT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "mode": mode,
        "status": "running",
        "network_or_raw_download_allowed": mode == "refresh",
        "commands": [],
    }
    write_json_atomic(run_report_path, report)
    failed = False
    try:
        for spec in pipeline_commands(mode):
            artifact_before = hash_or_none(PRIMARY_ARTIFACT)
            started = time.monotonic()
            print(f"\n[{spec.name}] {' '.join(spec.arguments)}")
            completed = subprocess.run(
                list(spec.arguments),
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            duration = time.monotonic() - started
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
            artifact_after = hash_or_none(PRIMARY_ARTIFACT)
            preserved = (
                not spec.preserve_primary_artifact
                or artifact_before == artifact_after
            )
            status = (
                "passed"
                if completed.returncode == 0 and preserved
                else "failed"
            )
            command_result = {
                "name": spec.name,
                "arguments": list(spec.arguments),
                "status": status,
                "returncode": completed.returncode,
                "duration_seconds": round(duration, 3),
                "preserve_primary_artifact_required": (
                    spec.preserve_primary_artifact
                ),
                "primary_artifact_preserved": preserved,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
            report["commands"].append(command_result)
            write_json_atomic(run_report_path, report)
            if completed.returncode != 0:
                failed = True
                report["failure"] = (
                    f"Command {spec.name} returned {completed.returncode}"
                )
                break
            if not preserved:
                failed = True
                report["failure"] = (
                    f"Command {spec.name} changed the primary artifact"
                )
                break
    finally:
        report["status"] = "failed" if failed else "passed"
        report["completed_at_utc"] = utc_now()
        write_json_atomic(run_report_path, report)
        build_manifest(manifest_path, run_report_path)
    print(f"pipeline {report['status']}")
    print(f"run      {run_report_path}")
    print(f"manifest {manifest_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ci", "research", "refresh"))
    parser.add_argument("--run-report", type=Path, default=DEFAULT_RUN_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.mode, args.run_report, args.manifest)
    if result["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
