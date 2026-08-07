"""Benchmark upper-tail challengers on development folds without touching test."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from inside_airbnb_feature_ablation import FOLD_SEEDS
from inside_airbnb_phase0 import (
    ROOT,
    active_snapshot_date,
    utc_now,
    write_json_atomic,
)
from inside_airbnb_quote_model import (
    DEFAULT_SILVER,
    build_pipeline,
    group_split,
    load_silver,
    price_metrics,
)


_SNAPSHOT = active_snapshot_date()
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / f"sydney_{_SNAPSHOT}_upper_tail_challenger.json"
)
DEFAULT_MARKDOWN = ROOT / "docs" / "inside_airbnb_upper_tail_challenger.md"
UPPER_TAIL_QUANTILE = 0.90
MAX_OVERALL_MAE_DEGRADATION = 0.02
MIN_UPPER_TAIL_MAE_IMPROVEMENT = 0.10
MIN_ABSOLUTE_BIAS_IMPROVEMENT = 0.10
MIN_BETTER_FOLDS = 4

CANDIDATES = [
    {
        "name": "incumbent_log_absolute",
        "loss": "absolute_error",
        "target_scale": "log1p",
        "sample_weight": "uniform",
    },
    {
        "name": "log_squared_error",
        "loss": "squared_error",
        "target_scale": "log1p",
        "sample_weight": "uniform",
    },
    {
        "name": "log_tail_weighted_absolute",
        "loss": "absolute_error",
        "target_scale": "log1p",
        "sample_weight": "sqrt_price_capped",
    },
    {
        "name": "raw_poisson",
        "loss": "poisson",
        "target_scale": "raw",
        "sample_weight": "uniform",
    },
    {
        "name": "raw_squared_error",
        "loss": "squared_error",
        "target_scale": "raw",
        "sample_weight": "uniform",
    },
]


def tail_sample_weights(target: np.ndarray) -> np.ndarray:
    median = max(float(np.median(target)), 1.0)
    return np.clip(np.sqrt(target / median), 0.75, 4.0)


def candidate_decision(
    incumbent: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    incumbent_overall = incumbent["summary"]["overall_mae"]["mean"]
    incumbent_tail = incumbent["summary"]["upper_tail_mae"]["mean"]
    incumbent_bias = incumbent["summary"]["upper_tail_median_bias"]["mean"]
    candidate_overall = candidate["summary"]["overall_mae"]["mean"]
    candidate_tail = candidate["summary"]["upper_tail_mae"]["mean"]
    candidate_bias = candidate["summary"]["upper_tail_median_bias"]["mean"]
    overall_change = candidate_overall / incumbent_overall - 1
    tail_improvement = 1 - candidate_tail / incumbent_tail
    bias_improvement = 1 - abs(candidate_bias) / max(abs(incumbent_bias), 1e-9)
    better_folds = sum(
        candidate_fold["upper_tail_mae"] < incumbent_fold["upper_tail_mae"]
        for candidate_fold, incumbent_fold in zip(
            candidate["folds"], incumbent["folds"]
        )
    )
    checks = {
        "overall_mae_degradation_within_limit": (
            overall_change <= MAX_OVERALL_MAE_DEGRADATION
        ),
        "upper_tail_mae_improvement_passes": (
            tail_improvement >= MIN_UPPER_TAIL_MAE_IMPROVEMENT
        ),
        "absolute_bias_improvement_passes": (
            bias_improvement >= MIN_ABSOLUTE_BIAS_IMPROVEMENT
        ),
        "better_fold_count_passes": better_folds >= MIN_BETTER_FOLDS,
    }
    return {
        "qualifies": all(checks.values()),
        "overall_mae_relative_change": overall_change,
        "upper_tail_mae_relative_improvement": tail_improvement,
        "upper_tail_absolute_bias_relative_improvement": bias_improvement,
        "upper_tail_better_fold_count": better_folds,
        "checks": checks,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "standard_deviation": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def evaluate_candidate(
    X: np.ndarray,
    y: np.ndarray,
    development: np.ndarray,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    candidate: dict[str, str],
) -> dict[str, Any]:
    fold_results = []
    for seed, train_relative, validation_relative in folds:
        train = development[train_relative]
        validation = development[validation_relative]
        pipeline = build_pipeline()
        pipeline.set_params(model__loss=candidate["loss"])
        training_target = (
            np.log1p(y[train])
            if candidate["target_scale"] == "log1p"
            else y[train]
        )
        fit_kwargs: dict[str, Any] = {}
        if candidate["sample_weight"] == "sqrt_price_capped":
            fit_kwargs["model__sample_weight"] = tail_sample_weights(y[train])
        pipeline.fit(X[train], training_target, **fit_kwargs)
        raw_prediction = pipeline.predict(X[validation])
        predicted = (
            np.expm1(raw_prediction)
            if candidate["target_scale"] == "log1p"
            else raw_prediction
        )
        predicted = np.maximum(0.0, predicted)
        threshold = float(np.quantile(y[train], UPPER_TAIL_QUANTILE))
        upper = y[validation] > threshold
        upper_error = predicted[upper] - y[validation][upper]
        fold_results.append(
            {
                "seed": seed,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "upper_tail_threshold_aud": threshold,
                "upper_tail_rows": int(np.sum(upper)),
                "overall_mae": price_metrics(y[validation], predicted)["mae"],
                "upper_tail_mae": float(np.mean(np.abs(upper_error))),
                "upper_tail_median_bias": float(np.median(upper_error)),
                "upper_tail_severe_underprediction_rate": float(
                    np.mean(predicted[upper] < y[validation][upper] * 0.70)
                ),
            }
        )
    return {
        **candidate,
        "folds": fold_results,
        "summary": {
            metric: mean_std([fold[metric] for fold in fold_results])
            for metric in (
                "overall_mae",
                "upper_tail_mae",
                "upper_tail_median_bias",
                "upper_tail_severe_underprediction_rate",
            )
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Inside Airbnb upper-tail challenger benchmark",
        "",
        "This experiment uses development data only. The governed test set is reserved and no primary artifact is replaced.",
        "",
        "## Predeclared promotion rule",
        "",
        "- Overall MAE may degrade by no more than 2%.",
        "- Upper-tail MAE must improve by at least 10%.",
        "- Absolute upper-tail median bias must improve by at least 10%.",
        "- Upper-tail MAE must improve in at least four of five host-disjoint folds.",
        "",
        "## Results",
        "",
        "| Candidate | Overall MAE | Upper-tail MAE | Tail median bias | Severe underprediction | Qualifies |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in report["candidates"]:
        decision = candidate.get("decision")
        lines.append(
            f"| `{candidate['name']}` | "
            f"{candidate['summary']['overall_mae']['mean']:.2f} | "
            f"{candidate['summary']['upper_tail_mae']['mean']:.2f} | "
            f"{candidate['summary']['upper_tail_median_bias']['mean']:.2f} | "
            f"{candidate['summary']['upper_tail_severe_underprediction_rate']['mean']:.1%} | "
            f"{'incumbent' if decision is None else ('yes' if decision['qualifies'] else 'no')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: `{report['decision']['status']}`.",
            f"- Selected challenger: `{report['decision']['selected_challenger']}`.",
            f"- Primary model replaced: **{'yes' if report['decision']['primary_model_replaced'] else 'no'}**.",
            f"- Next evidence: {report['decision']['next_evidence']}",
            "",
            "## Boundaries",
            "",
            "- The upper tail is defined independently inside every fold using that fold's training-target p90.",
            "- Validation hosts never appear in the corresponding training fold.",
            "- No availability or review proxy is introduced.",
            "- This benchmark cannot restore temporal authority or make the spent governed test set fresh again.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    silver_path: Path, report_path: Path, markdown_path: Path
) -> dict[str, Any]:
    _, X, y, groups = load_silver(silver_path)
    development, governed_test = group_split(X, groups, 0.20, 42)
    folds = []
    for seed in FOLD_SEEDS:
        train_relative, validation_relative = group_split(
            X[development], groups[development], 0.20, seed
        )
        folds.append((seed, train_relative, validation_relative))
    evaluated = [
        evaluate_candidate(X, y, development, folds, candidate)
        for candidate in CANDIDATES
    ]
    incumbent = evaluated[0]
    for candidate in evaluated[1:]:
        candidate["decision"] = candidate_decision(incumbent, candidate)
    qualifying = [
        candidate
        for candidate in evaluated[1:]
        if candidate["decision"]["qualifies"]
    ]
    selected = (
        min(
            qualifying,
            key=lambda candidate: candidate["summary"]["upper_tail_mae"]["mean"],
        )["name"]
        if qualifying
        else None
    )
    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "source": str(silver_path.relative_to(ROOT)),
        "protocol": {
            "governed_test_rows_reserved_and_unused": len(governed_test),
            "governed_test_unique_hosts": len(set(groups[governed_test])),
            "development_rows": len(development),
            "fold_seeds": list(FOLD_SEEDS),
            "group": "host_id",
            "upper_tail_definition": "validation target above fold-training p90",
        },
        "promotion_thresholds": {
            "maximum_overall_mae_degradation": MAX_OVERALL_MAE_DEGRADATION,
            "minimum_upper_tail_mae_improvement": MIN_UPPER_TAIL_MAE_IMPROVEMENT,
            "minimum_absolute_bias_improvement": MIN_ABSOLUTE_BIAS_IMPROVEMENT,
            "minimum_better_folds": MIN_BETTER_FOLDS,
        },
        "candidates": evaluated,
        "decision": {
            "status": (
                "CHALLENGER_IDENTIFIED"
                if selected
                else "NO_CHALLENGER_PASSED_ALL_GATES"
            ),
            "selected_challenger": selected,
            "primary_model_replaced": False,
            "next_evidence": (
                "Evaluate the selected challenger on a compatible future "
                "Sydney snapshot before considering promotion."
                if selected
                else (
                    "Retain the incumbent and investigate new feature/data "
                    "families on development folds."
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
    print(f"selected {selected}")
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
