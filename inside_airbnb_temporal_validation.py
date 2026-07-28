"""Evaluate a quote model strictly forward in time on a compatible Silver snapshot."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from inside_airbnb_phase0 import ROOT, sha256_file, utc_now, write_json_atomic
from inside_airbnb_quote_model import (
    ALPHA,
    CATEGORICAL_FEATURES,
    DEFAULT_SILVER,
    NUMERIC_FEATURES,
    build_pipeline,
    calibration_quantiles,
    feature_matrix,
    fit_market_baseline,
    group_split,
    interval_bounds,
    load_silver,
    market_prediction,
    price_metrics,
    quantiles_for_records,
)


DEFAULT_REPORT = (
    ROOT / "reports" / "inside_airbnb" / "sydney_temporal_quote_validation.json"
)
MIN_NEWER_ROWS = 2000
MIN_COLD_START_HOST_ROWS = 200
MIN_RELATIVE_MAE_IMPROVEMENT = 0.10
MIN_INTERVAL_COVERAGE = 0.85


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def temporal_gate(
    chronology_valid: bool,
    newer_rows: int,
    relative_mae_improvement: float,
    interval_coverage: float,
    cold_start_host_rows: int,
    cold_start_relative_mae_improvement: float | None,
) -> dict[str, Any]:
    checks = [
        {
            "name": "strict_chronology",
            "passed": chronology_valid,
            "observed": chronology_valid,
            "threshold": True,
        },
        {
            "name": "minimum_newer_rows",
            "passed": newer_rows >= MIN_NEWER_ROWS,
            "observed": newer_rows,
            "threshold": MIN_NEWER_ROWS,
        },
        {
            "name": "mae_improvement_vs_market_baseline",
            "passed": relative_mae_improvement >= MIN_RELATIVE_MAE_IMPROVEMENT,
            "observed": relative_mae_improvement,
            "threshold": MIN_RELATIVE_MAE_IMPROVEMENT,
        },
        {
            "name": "conformal_interval_coverage",
            "passed": interval_coverage >= MIN_INTERVAL_COVERAGE,
            "observed": interval_coverage,
            "threshold": MIN_INTERVAL_COVERAGE,
        },
        {
            "name": "minimum_cold_start_host_rows",
            "passed": cold_start_host_rows >= MIN_COLD_START_HOST_ROWS,
            "observed": cold_start_host_rows,
            "threshold": MIN_COLD_START_HOST_ROWS,
        },
        {
            "name": "cold_start_host_mae_improvement",
            "passed": (
                cold_start_relative_mae_improvement is not None
                and cold_start_relative_mae_improvement > 0
            ),
            "observed": cold_start_relative_mae_improvement,
            "threshold": 0.0,
        },
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "checks": checks,
        "recommended_current_quote_model_authority": (
            "temporally_validated" if passed else "research_only"
        ),
    }


def cohort_result(
    mask: np.ndarray,
    y: np.ndarray,
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    covered: np.ndarray,
) -> dict[str, Any]:
    rows = int(np.sum(mask))
    if not rows:
        return {
            "rows": 0,
            "model_metrics": None,
            "baseline_metrics": None,
            "relative_mae_improvement": None,
            "interval_coverage": None,
        }
    model_metrics = price_metrics(y[mask], model_prediction[mask])
    baseline_metrics = price_metrics(y[mask], baseline_prediction[mask])
    return {
        "rows": rows,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "relative_mae_improvement": float(
            1 - model_metrics["mae"] / baseline_metrics["mae"]
        ),
        "interval_coverage": float(np.mean(covered[mask])),
    }


def category_drift(
    older_records: list[dict[str, str]],
    older_indices: np.ndarray,
    newer_records: list[dict[str, str]],
) -> dict[str, Any]:
    result = {}
    for field in CATEGORICAL_FEATURES:
        known = {older_records[int(index)][field] for index in older_indices}
        unseen = Counter(
            row[field] for row in newer_records if row[field] not in known
        )
        result[field] = {
            "training_unique": len(known),
            "newer_unseen_rows": sum(unseen.values()),
            "newer_unseen_rate": sum(unseen.values()) / len(newer_records),
            "unseen_values": dict(unseen.most_common(20)),
        }
    return result


def validate_temporally(
    older_silver: Path,
    newer_silver: Path,
    report_path: Path,
) -> dict[str, Any]:
    older_records, older_X, older_y, older_groups = load_silver(older_silver)
    newer_records, newer_X, newer_y, _ = load_silver(newer_silver)
    older_dates = [date.fromisoformat(row["as_of_date"]) for row in older_records]
    newer_dates = [date.fromisoformat(row["as_of_date"]) for row in newer_records]
    chronology_valid = max(older_dates) < min(newer_dates)
    if not chronology_valid:
        raise ValueError("Newer Silver rows must all occur after older Silver rows")

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing_older = sorted(required - set(older_records[0]))
    missing_newer = sorted(required - set(newer_records[0]))
    if missing_older or missing_newer:
        raise ValueError(
            f"Incompatible Silver feature contract: older={missing_older}, "
            f"newer={missing_newer}"
        )
    currencies = {
        row["currency"] for row in [*older_records, *newer_records]
    }
    if currencies != {"AUD"}:
        raise ValueError(f"Temporal validation requires only AUD; got {currencies}")

    train_indices, calibration_indices = group_split(
        older_X, older_groups, 0.20, 43
    )
    pipeline = build_pipeline()
    pipeline.fit(older_X[train_indices], np.log1p(older_y[train_indices]))
    calibration_log = pipeline.predict(older_X[calibration_indices])
    calibration_residuals = np.abs(
        np.log1p(older_y[calibration_indices]) - calibration_log
    )
    global_quantile, segment_quantiles, room_quantiles = calibration_quantiles(
        older_records, calibration_indices, calibration_residuals
    )

    newer_log = pipeline.predict(newer_X)
    model_prediction = np.maximum(0.0, np.expm1(newer_log))
    newer_quantiles = quantiles_for_records(
        newer_records,
        np.arange(len(newer_records)),
        global_quantile,
        segment_quantiles,
        room_quantiles,
    )
    lower, upper = interval_bounds(newer_log, newer_quantiles)
    covered = (newer_y >= lower) & (newer_y <= upper)

    baseline = fit_market_baseline(older_records, older_y, train_indices)
    baseline_prediction = np.asarray(
        [market_prediction(baseline, row)[0] for row in newer_records],
        dtype=float,
    )
    overall_model = price_metrics(newer_y, model_prediction)
    overall_baseline = price_metrics(newer_y, baseline_prediction)
    relative_improvement = float(
        1 - overall_model["mae"] / overall_baseline["mae"]
    )

    older_listing_ids = {row["listing_id"] for row in older_records}
    older_host_ids = {row["host_id"] for row in older_records}
    seen_listing = np.asarray(
        [row["listing_id"] in older_listing_ids for row in newer_records]
    )
    seen_host = np.asarray(
        [row["host_id"] in older_host_ids for row in newer_records]
    )
    cohorts = {
        "seen_listing": cohort_result(
            seen_listing,
            newer_y,
            model_prediction,
            baseline_prediction,
            covered,
        ),
        "new_listing": cohort_result(
            ~seen_listing,
            newer_y,
            model_prediction,
            baseline_prediction,
            covered,
        ),
        "seen_host": cohort_result(
            seen_host,
            newer_y,
            model_prediction,
            baseline_prediction,
            covered,
        ),
        "new_host": cohort_result(
            ~seen_host,
            newer_y,
            model_prediction,
            baseline_prediction,
            covered,
        ),
    }
    cold_start = cohorts["new_host"]
    gate = temporal_gate(
        chronology_valid,
        len(newer_records),
        relative_improvement,
        float(np.mean(covered)),
        cold_start["rows"],
        cold_start["relative_mae_improvement"],
    )

    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "target": {
            "name": "target_quoted_price_per_night",
            "definition": "Public quoted price per night in AUD",
            "contract_compatible": True,
        },
        "snapshots": {
            "older": {
                "label": older_records[0]["snapshot_label"],
                "source": display_path(older_silver),
                "source_sha256": sha256_file(older_silver),
                "rows": len(older_records),
                "as_of_min": min(older_dates).isoformat(),
                "as_of_max": max(older_dates).isoformat(),
            },
            "newer": {
                "label": newer_records[0]["snapshot_label"],
                "source": display_path(newer_silver),
                "source_sha256": sha256_file(newer_silver),
                "rows": len(newer_records),
                "as_of_min": min(newer_dates).isoformat(),
                "as_of_max": max(newer_dates).isoformat(),
            },
        },
        "protocol": {
            "strict_forward_time": chronology_valid,
            "older_training_rows": len(train_indices),
            "older_calibration_rows": len(calibration_indices),
            "older_train_calibration_host_overlap": len(
                set(older_groups[train_indices])
                & set(older_groups[calibration_indices])
            ),
            "newer_rows_used_for_selection": False,
            "model_configuration_fixed": True,
        },
        "overall": {
            "model_metrics": overall_model,
            "market_baseline_metrics": overall_baseline,
            "relative_mae_improvement": relative_improvement,
            "interval_target_coverage": 1 - ALPHA,
            "interval_observed_coverage": float(np.mean(covered)),
            "interval_average_width": float(np.mean(upper - lower)),
        },
        "cohorts": cohorts,
        "drift": {
            "older_target_median": float(np.median(older_y)),
            "newer_target_median": float(np.median(newer_y)),
            "target_median_ratio": float(
                np.median(newer_y) / np.median(older_y)
            ),
            "categorical": category_drift(
                older_records, train_indices, newer_records
            ),
        },
        "evidence_gate": gate,
        "authority": {
            "current_quote_model": gate[
                "recommended_current_quote_model_authority"
            ],
            "interpretation": (
                "All predeclared temporal evidence gates passed."
                if gate["passed"]
                else "At least one temporal evidence gate failed; keep research_only."
            ),
        },
    }
    write_json_atomic(report_path, report)
    print(
        f"temporal MAE {overall_model['mae']:.2f}; "
        f"baseline {overall_baseline['mae']:.2f}"
    )
    print(f"coverage {np.mean(covered):.3f}; gate {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"report   {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--older-silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--newer-silver", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_temporally(args.older_silver, args.newer_silver, args.report)


if __name__ == "__main__":
    main()
