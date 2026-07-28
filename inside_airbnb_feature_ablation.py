"""Run leakage-safe, host-disjoint feature ablations on development data only."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from inside_airbnb_phase0 import ROOT, utc_now, write_json_atomic
from inside_airbnb_quote_model import (
    DEFAULT_SILVER,
    build_pipeline,
    feature_matrix,
    group_split,
    load_silver,
    price_metrics,
)
from sydney_geography import GEOGRAPHIC_FEATURES, reference_manifest


DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_2026-06-16_feature_ablation.json"
)
DEFAULT_MARKDOWN = ROOT / "docs" / "inside_airbnb_feature_ablation.md"
FOLD_SEEDS = (101, 102, 103, 104, 105)

CONTEXT_AND_STRUCTURE = [
    "quote_lead_days",
    "stay_nights",
    "checkin_month",
    "checkin_day_of_week",
    "checkin_is_weekend",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "amenities_count",
    "minimum_nights",
    "maximum_nights",
    "calculated_host_listings_count",
]
LISTING_CATEGORICAL = ["property_type", "room_type", "host_is_superhost"]

VARIANTS = [
    {
        "name": "listing_and_quote_context",
        "numeric": CONTEXT_AND_STRUCTURE,
        "categorical": LISTING_CATEGORICAL,
        "question": "How much can quote context and listing structure explain?",
    },
    {
        "name": "plus_neighbourhood",
        "numeric": CONTEXT_AND_STRUCTURE,
        "categorical": [*LISTING_CATEGORICAL, "neighbourhood"],
        "question": "Does the governed neighbourhood category add signal?",
    },
    {
        "name": "plus_raw_coordinates",
        "numeric": [*CONTEXT_AND_STRUCTURE, "latitude", "longitude"],
        "categorical": [*LISTING_CATEGORICAL, "neighbourhood"],
        "question": "Do approximate raw coordinates add signal beyond neighbourhood?",
    },
    {
        "name": "plus_engineered_geography",
        "numeric": [
            *CONTEXT_AND_STRUCTURE,
            "latitude",
            "longitude",
            *GEOGRAPHIC_FEATURES,
        ],
        "categorical": [*LISTING_CATEGORICAL, "neighbourhood"],
        "question": "Do transparent reference-distance features add incremental signal?",
    },
]


def mean_and_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "standard_deviation": float(statistics.stdev(values))
        if len(values) > 1
        else 0.0,
    }


def evaluate_variant(
    records: list[dict[str, str]],
    y: np.ndarray,
    groups: np.ndarray,
    development_indices: np.ndarray,
    variant: dict[str, Any],
    folds: list[tuple[int, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    numeric = list(variant["numeric"])
    categorical = list(variant["categorical"])
    X = feature_matrix(records, numeric, categorical)
    fold_results = []
    for seed, train_relative, validation_relative in folds:
        train = development_indices[train_relative]
        validation = development_indices[validation_relative]
        pipeline = build_pipeline(numeric, categorical)
        pipeline.fit(X[train], np.log1p(y[train]))
        predicted = np.maximum(0.0, np.expm1(pipeline.predict(X[validation])))
        fold_results.append(
            {
                "seed": seed,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "host_overlap": len(set(groups[train]) & set(groups[validation])),
                **price_metrics(y[validation], predicted),
            }
        )
    return {
        **variant,
        "folds": fold_results,
        "summary": {
            metric: mean_and_std([fold[metric] for fold in fold_results])
            for metric in ("mae", "median_absolute_error", "rmse", "r2")
        },
    }


def held_out_permutation_importance(
    records: list[dict[str, str]],
    y: np.ndarray,
    development_indices: np.ndarray,
    fold: tuple[int, np.ndarray, np.ndarray],
    variant: dict[str, Any],
) -> list[dict[str, Any]]:
    seed, train_relative, validation_relative = fold
    train = development_indices[train_relative]
    validation = development_indices[validation_relative]
    numeric = list(variant["numeric"])
    categorical = list(variant["categorical"])
    names = numeric + categorical
    X = feature_matrix(records, numeric, categorical)
    pipeline = build_pipeline(numeric, categorical)
    pipeline.fit(X[train], np.log1p(y[train]))

    def negative_price_mae(estimator: Any, features: np.ndarray, target: np.ndarray) -> float:
        predicted = np.maximum(0.0, np.expm1(estimator.predict(features)))
        return -float(np.mean(np.abs(target - predicted)))

    result = permutation_importance(
        pipeline,
        X[validation],
        y[validation],
        scoring=negative_price_mae,
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    rows = [
        {
            "feature": name,
            "mae_increase_mean": float(result.importances_mean[index]),
            "mae_increase_standard_deviation": float(result.importances_std[index]),
        }
        for index, name in enumerate(names)
    ]
    return sorted(rows, key=lambda row: -row["mae_increase_mean"])


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Inside Airbnb feature ablation",
        "",
        "This diagnostic uses only development data. The governed seed-42 test set remains untouched.",
        "Every comparison uses the same five host-disjoint folds and the same model configuration.",
        "",
        "## Results",
        "",
        "| Variant | Features | MAE | RMSE | R² | Δ MAE vs previous |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    previous_mae = None
    for variant in report["variants"]:
        mae = variant["summary"]["mae"]["mean"]
        delta = "—" if previous_mae is None else f"{mae - previous_mae:+.2f}"
        lines.append(
            f"| `{variant['name']}` | "
            f"{len(variant['numeric']) + len(variant['categorical'])} | "
            f"{mae:.2f} ± {variant['summary']['mae']['standard_deviation']:.2f} | "
            f"{variant['summary']['rmse']['mean']:.2f} | "
            f"{variant['summary']['r2']['mean']:.3f} | {delta} |"
        )
        previous_mae = mae
    decision = report["engineered_geography_decision"]
    lines.extend(
        [
            "",
            "## Engineered-geography decision",
            "",
            f"- Mean MAE delta versus raw coordinates: {decision['mean_mae_delta_vs_raw_coordinates']:+.2f}.",
            f"- Better folds: {decision['better_fold_count']} of {len(FOLD_SEEDS)}.",
            f"- Primary-model adoption: **{'yes' if decision['adopt'] else 'no'}**.",
            f"- Rule: {decision['rule']}",
            "",
            "## Held-out permutation importance",
            "",
            "Positive values mean validation MAE increased when the feature was shuffled. This is predictive association, not causality.",
            "",
            "| Feature | Mean MAE increase | Standard deviation |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{row['feature']}` | {row['mae_increase_mean']:.2f} | "
        f"{row['mae_increase_standard_deviation']:.2f} |"
        for row in report["held_out_permutation_importance"][:15]
    )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Hosts never overlap between a fold's training and validation rows.",
            "- The experiment does not introduce availability or review proxy features.",
            "- Approximate coordinates and fixed reference distances do not represent route distance or travel time.",
            "- The final deployment authority remains governed by temporal validation, not this ablation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(
    silver_path: Path, report_path: Path, markdown_path: Path
) -> dict[str, Any]:
    records, full_X, y, groups = load_silver(silver_path)
    development, governed_test = group_split(full_X, groups, 0.20, 42)
    folds = []
    for seed in FOLD_SEEDS:
        train_relative, validation_relative = group_split(
            full_X[development], groups[development], 0.20, seed
        )
        folds.append((seed, train_relative, validation_relative))
    variants = [
        evaluate_variant(records, y, groups, development, variant, folds)
        for variant in VARIANTS
    ]
    raw = variants[-2]
    engineered = variants[-1]
    raw_fold_mae = [fold["mae"] for fold in raw["folds"]]
    engineered_fold_mae = [fold["mae"] for fold in engineered["folds"]]
    delta = (
        engineered["summary"]["mae"]["mean"] - raw["summary"]["mae"]["mean"]
    )
    better_folds = sum(
        engineered_mae < raw_mae
        for engineered_mae, raw_mae in zip(engineered_fold_mae, raw_fold_mae)
    )
    adopt = delta < 0 and better_folds >= 3
    importance_variant = engineered if adopt else raw
    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "source": str(silver_path.relative_to(ROOT)),
        "protocol": {
            "governed_test_rows_reserved_and_unused": len(governed_test),
            "governed_test_unique_hosts": len(set(groups[governed_test])),
            "development_rows": len(development),
            "fold_seeds": list(FOLD_SEEDS),
            "validation_fraction_per_fold": 0.20,
            "group": "host_id",
            "target_transform": "log1p",
            "model_configuration_fixed": True,
        },
        "variants": variants,
        "engineered_geography_decision": {
            "adopt": adopt,
            "mean_mae_delta_vs_raw_coordinates": delta,
            "better_fold_count": better_folds,
            "rule": (
                "Adopt only if mean development MAE improves and at least "
                "three of five host-disjoint folds improve."
            ),
            "reference_manifest": reference_manifest(),
        },
        "permutation_importance_variant": importance_variant["name"],
        "permutation_importance_fold_seed": folds[0][0],
        "held_out_permutation_importance": held_out_permutation_importance(
            records, y, development, folds[0], importance_variant
        ),
        "governance": {
            "governed_test_used_for_selection": False,
            "proxy_features_included": False,
            "causal_claim": False,
        },
    }
    write_json_atomic(report_path, report)
    write_markdown(markdown_path, report)
    print(f"ablation {report_path}")
    print(f"docs     {markdown_path}")
    print(
        "geography "
        f"{'adopt' if adopt else 'do not adopt'} "
        f"(MAE delta {delta:+.2f}; {better_folds}/{len(FOLD_SEEDS)} folds)"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ablation(args.silver, args.report, args.markdown)


if __name__ == "__main__":
    main()
