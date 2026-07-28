"""Benchmark predicted-price-band asymmetric conformal intervals on development only."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from inside_airbnb_feature_ablation import FOLD_SEEDS
from inside_airbnb_phase0 import ROOT, utc_now, write_json_atomic
from inside_airbnb_quote_model import (
    ALPHA,
    DEFAULT_SILVER,
    build_pipeline,
    calibration_quantiles,
    finite_sample_quantile,
    group_split,
    interval_bounds,
    load_silver,
    quantiles_for_records,
)
from inside_airbnb_upper_tail_challenger import UPPER_TAIL_QUANTILE


DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_2026-06-16_interval_challenger.json"
)
DEFAULT_MARKDOWN = ROOT / "docs" / "inside_airbnb_interval_challenger.md"
PRICE_BAND_PROBABILITIES = (0.50, 0.75, 0.90, 0.95)
PRICE_BAND_LABELS = ("up_to_p50", "p50_p75", "p75_p90", "p90_p95", "above_p95")
MIN_BAND_CALIBRATION = 80
MIN_OVERALL_COVERAGE = 0.88
MIN_UPPER_TAIL_COVERAGE_IMPROVEMENT = 0.10
MAX_AVERAGE_WIDTH_RATIO = 1.25
MAX_UPPER_TAIL_WIDTH_RATIO = 1.50
MIN_BETTER_FOLDS = 4


def price_thresholds(training_target: np.ndarray) -> list[float]:
    return [
        float(np.quantile(training_target, probability))
        for probability in PRICE_BAND_PROBABILITIES
    ]


def predicted_price_band(value: float, thresholds: list[float]) -> str:
    for label, threshold in zip(PRICE_BAND_LABELS, thresholds):
        if value <= threshold:
            return label
    return PRICE_BAND_LABELS[-1]


def asymmetric_quantiles(
    true_log: np.ndarray,
    predicted_log: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    lower_scores = predicted_log - true_log
    upper_scores = true_log - predicted_log
    lower = max(0.0, finite_sample_quantile(lower_scores, alpha / 2))
    upper = max(0.0, finite_sample_quantile(upper_scores, alpha / 2))
    return lower, upper


def fit_banded_asymmetric_calibration(
    calibration_target: np.ndarray,
    calibration_log_prediction: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    true_log = np.log1p(calibration_target)
    global_lower, global_upper = asymmetric_quantiles(
        true_log, calibration_log_prediction
    )
    positions: defaultdict[str, list[int]] = defaultdict(list)
    calibration_price_prediction = np.maximum(
        0.0, np.expm1(calibration_log_prediction)
    )
    for position, prediction in enumerate(calibration_price_prediction):
        positions[predicted_price_band(float(prediction), thresholds)].append(position)
    bands = {}
    for band, selected_positions in positions.items():
        if len(selected_positions) < MIN_BAND_CALIBRATION:
            continue
        selected = np.asarray(selected_positions, dtype=int)
        lower, upper = asymmetric_quantiles(
            true_log[selected], calibration_log_prediction[selected]
        )
        bands[band] = {
            "rows": len(selected),
            "lower_log_adjustment": lower,
            "upper_log_adjustment": upper,
        }
    return {
        "global": {
            "rows": len(calibration_target),
            "lower_log_adjustment": global_lower,
            "upper_log_adjustment": global_upper,
        },
        "bands": bands,
        "thresholds_aud": thresholds,
    }


def banded_interval_bounds(
    predicted_log: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lower_values = []
    upper_values = []
    bands = []
    for value in predicted_log:
        predicted_price = max(0.0, math.expm1(float(value)))
        band = predicted_price_band(
            predicted_price, calibration["thresholds_aud"]
        )
        adjustment = calibration["bands"].get(band, calibration["global"])
        lower_values.append(
            max(0.0, math.expm1(float(value) - adjustment["lower_log_adjustment"]))
        )
        upper_values.append(
            max(0.0, math.expm1(float(value) + adjustment["upper_log_adjustment"]))
        )
        bands.append(band)
    return (
        np.asarray(lower_values, dtype=float),
        np.asarray(upper_values, dtype=float),
        bands,
    )


def interval_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    upper_tail: np.ndarray,
) -> dict[str, float | int]:
    covered = (actual >= lower) & (actual <= upper)
    below = actual < lower
    above = actual > upper
    return {
        "rows": len(actual),
        "coverage": float(np.mean(covered)),
        "average_width": float(np.mean(upper - lower)),
        "lower_miss_rate": float(np.mean(below)),
        "upper_miss_rate": float(np.mean(above)),
        "upper_tail_rows": int(np.sum(upper_tail)),
        "upper_tail_coverage": float(np.mean(covered[upper_tail])),
        "upper_tail_average_width": float(
            np.mean((upper - lower)[upper_tail])
        ),
        "upper_tail_upper_miss_rate": float(np.mean(above[upper_tail])),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "standard_deviation": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def challenger_decision(
    incumbent_folds: list[dict[str, Any]],
    challenger_folds: list[dict[str, Any]],
) -> dict[str, Any]:
    incumbent_coverage = statistics.mean(row["coverage"] for row in incumbent_folds)
    incumbent_width = statistics.mean(row["average_width"] for row in incumbent_folds)
    incumbent_tail_coverage = statistics.mean(
        row["upper_tail_coverage"] for row in incumbent_folds
    )
    incumbent_tail_width = statistics.mean(
        row["upper_tail_average_width"] for row in incumbent_folds
    )
    challenger_coverage = statistics.mean(row["coverage"] for row in challenger_folds)
    challenger_width = statistics.mean(row["average_width"] for row in challenger_folds)
    challenger_tail_coverage = statistics.mean(
        row["upper_tail_coverage"] for row in challenger_folds
    )
    challenger_tail_width = statistics.mean(
        row["upper_tail_average_width"] for row in challenger_folds
    )
    better_folds = sum(
        challenger["upper_tail_coverage"] > incumbent["upper_tail_coverage"]
        for incumbent, challenger in zip(incumbent_folds, challenger_folds)
    )
    improvement = challenger_tail_coverage - incumbent_tail_coverage
    width_ratio = challenger_width / incumbent_width
    tail_width_ratio = challenger_tail_width / incumbent_tail_width
    checks = {
        "overall_coverage_passes": challenger_coverage >= MIN_OVERALL_COVERAGE,
        "upper_tail_coverage_improvement_passes": (
            improvement >= MIN_UPPER_TAIL_COVERAGE_IMPROVEMENT
        ),
        "average_width_ratio_passes": width_ratio <= MAX_AVERAGE_WIDTH_RATIO,
        "upper_tail_width_ratio_passes": (
            tail_width_ratio <= MAX_UPPER_TAIL_WIDTH_RATIO
        ),
        "better_fold_count_passes": better_folds >= MIN_BETTER_FOLDS,
    }
    return {
        "qualifies": all(checks.values()),
        "overall_coverage": challenger_coverage,
        "upper_tail_coverage_absolute_improvement": improvement,
        "average_width_ratio": width_ratio,
        "upper_tail_width_ratio": tail_width_ratio,
        "upper_tail_better_fold_count": better_folds,
        "checks": checks,
    }


def summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "coverage",
        "average_width",
        "lower_miss_rate",
        "upper_miss_rate",
        "upper_tail_coverage",
        "upper_tail_average_width",
        "upper_tail_upper_miss_rate",
    )
    return {
        metric: mean_std([float(fold[metric]) for fold in folds])
        for metric in metrics
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    incumbent = report["methods"]["incumbent_hierarchical_symmetric"]["summary"]
    challenger = report["methods"]["predicted_band_asymmetric"]["summary"]
    decision = report["decision"]
    lines = [
        "# Inside Airbnb interval challenger",
        "",
        "This experiment uses nested host-disjoint train/calibration/validation splits inside development data. The governed test set and primary artifact are untouched.",
        "",
        "## Results",
        "",
        "| Method | Overall coverage | Average width | Upper-tail coverage | Upper-tail width | Upper-tail upper-miss rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Incumbent hierarchical symmetric | {incumbent['coverage']['mean']:.1%} | "
        f"{incumbent['average_width']['mean']:.0f} | "
        f"{incumbent['upper_tail_coverage']['mean']:.1%} | "
        f"{incumbent['upper_tail_average_width']['mean']:.0f} | "
        f"{incumbent['upper_tail_upper_miss_rate']['mean']:.1%} |",
        f"| Predicted-band asymmetric | {challenger['coverage']['mean']:.1%} | "
        f"{challenger['average_width']['mean']:.0f} | "
        f"{challenger['upper_tail_coverage']['mean']:.1%} | "
        f"{challenger['upper_tail_average_width']['mean']:.0f} | "
        f"{challenger['upper_tail_upper_miss_rate']['mean']:.1%} |",
        "",
        "## Predeclared qualification rule",
        "",
        "- Overall coverage must be at least 88%.",
        "- Upper-tail coverage must improve by at least 10 percentage points.",
        "- Average width may increase by no more than 25%.",
        "- Upper-tail width may increase by no more than 50%.",
        "- Upper-tail coverage must improve in at least four of five folds.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`.",
        f"- Qualifies: **{'yes' if decision['qualifies'] else 'no'}**.",
        f"- Primary artifact changed: **{'yes' if decision['primary_artifact_changed'] else 'no'}**.",
        f"- Next evidence: {decision['next_evidence']}",
        "",
        "## Boundaries",
        "",
        "- Price bands are assigned from model predictions, never from unknown inference-time labels.",
        "- Band thresholds use only the corresponding fit split's target distribution.",
        "- Calibration residuals are asymmetric, allowing a wider upper correction than lower correction.",
        "- This development result cannot replace future temporal validation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    silver_path: Path, report_path: Path, markdown_path: Path
) -> dict[str, Any]:
    records, X, y, groups = load_silver(silver_path)
    development, governed_test = group_split(X, groups, 0.20, 42)
    incumbent_folds = []
    challenger_folds = []
    calibration_inventory = []
    for seed in FOLD_SEEDS:
        outer_train_relative, validation_relative = group_split(
            X[development], groups[development], 0.20, seed
        )
        outer_train = development[outer_train_relative]
        validation = development[validation_relative]
        fit_relative, calibration_relative = group_split(
            X[outer_train], groups[outer_train], 0.20, seed + 1000
        )
        fit = outer_train[fit_relative]
        calibration = outer_train[calibration_relative]
        if (
            set(groups[fit]) & set(groups[calibration])
            or set(groups[fit]) & set(groups[validation])
            or set(groups[calibration]) & set(groups[validation])
        ):
            raise AssertionError("Nested host groups overlap")

        pipeline = build_pipeline()
        pipeline.fit(X[fit], np.log1p(y[fit]))
        calibration_log = pipeline.predict(X[calibration])
        validation_log = pipeline.predict(X[validation])
        validation_prediction = np.maximum(0.0, np.expm1(validation_log))
        upper_threshold = float(np.quantile(y[fit], UPPER_TAIL_QUANTILE))
        upper_tail = y[validation] > upper_threshold

        absolute_residuals = np.abs(
            np.log1p(y[calibration]) - calibration_log
        )
        global_q, segment_q, room_q = calibration_quantiles(
            records, calibration, absolute_residuals
        )
        incumbent_q = quantiles_for_records(
            records, validation, global_q, segment_q, room_q
        )
        incumbent_lower, incumbent_upper = interval_bounds(
            validation_log, incumbent_q
        )
        incumbent_folds.append(
            {
                "seed": seed,
                **interval_metrics(
                    y[validation],
                    incumbent_lower,
                    incumbent_upper,
                    upper_tail,
                ),
            }
        )

        thresholds = price_thresholds(y[fit])
        banded = fit_banded_asymmetric_calibration(
            y[calibration], calibration_log, thresholds
        )
        challenger_lower, challenger_upper, _ = banded_interval_bounds(
            validation_log, banded
        )
        challenger_folds.append(
            {
                "seed": seed,
                **interval_metrics(
                    y[validation],
                    challenger_lower,
                    challenger_upper,
                    upper_tail,
                ),
            }
        )
        calibration_inventory.append(
            {
                "seed": seed,
                "fit_rows": len(fit),
                "calibration_rows": len(calibration),
                "validation_rows": len(validation),
                "calibrated_bands": {
                    name: values["rows"]
                    for name, values in sorted(banded["bands"].items())
                },
                "global_lower_log_adjustment": banded["global"][
                    "lower_log_adjustment"
                ],
                "global_upper_log_adjustment": banded["global"][
                    "upper_log_adjustment"
                ],
            }
        )
    decision_metrics = challenger_decision(incumbent_folds, challenger_folds)
    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "source": str(silver_path.relative_to(ROOT)),
        "protocol": {
            "development_rows": len(development),
            "governed_test_rows_reserved_and_unused": len(governed_test),
            "governed_test_unique_hosts": len(set(groups[governed_test])),
            "fold_seeds": list(FOLD_SEEDS),
            "nested_group": "host_id",
            "alpha": ALPHA,
            "minimum_band_calibration_rows": MIN_BAND_CALIBRATION,
            "price_band_probabilities": list(PRICE_BAND_PROBABILITIES),
            "upper_tail_definition": "validation target above fit-split p90",
        },
        "methods": {
            "incumbent_hierarchical_symmetric": {
                "folds": incumbent_folds,
                "summary": summarize(incumbent_folds),
            },
            "predicted_band_asymmetric": {
                "folds": challenger_folds,
                "summary": summarize(challenger_folds),
                "calibration_inventory": calibration_inventory,
            },
        },
        "qualification_thresholds": {
            "minimum_overall_coverage": MIN_OVERALL_COVERAGE,
            "minimum_upper_tail_coverage_absolute_improvement": (
                MIN_UPPER_TAIL_COVERAGE_IMPROVEMENT
            ),
            "maximum_average_width_ratio": MAX_AVERAGE_WIDTH_RATIO,
            "maximum_upper_tail_width_ratio": MAX_UPPER_TAIL_WIDTH_RATIO,
            "minimum_better_folds": MIN_BETTER_FOLDS,
        },
        "decision": {
            **decision_metrics,
            "status": (
                "INTERVAL_CHALLENGER_IDENTIFIED"
                if decision_metrics["qualifies"]
                else "INTERVAL_CHALLENGER_REJECTED"
            ),
            "primary_artifact_changed": False,
            "next_evidence": (
                "Evaluate the interval challenger on a compatible future "
                "snapshot before promotion."
                if decision_metrics["qualifies"]
                else (
                    "Retain incumbent intervals and investigate richer "
                    "scale/quantile models on development data."
                )
            ),
        },
        "governance": {
            "governed_test_used": False,
            "primary_artifact_written": False,
            "deployment_authority_changed": False,
        },
    }
    write_json_atomic(report_path, report)
    write_markdown(markdown_path, report)
    print(f"decision {report['decision']['status']}")
    print(
        "tail coverage "
        f"{summarize(incumbent_folds)['upper_tail_coverage']['mean']:.3f} -> "
        f"{summarize(challenger_folds)['upper_tail_coverage']['mean']:.3f}"
    )
    print(f"report   {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(args.silver, args.report, args.markdown)


if __name__ == "__main__":
    main()
