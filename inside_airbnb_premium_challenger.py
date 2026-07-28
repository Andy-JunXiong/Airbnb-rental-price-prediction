"""Benchmark premium semantic features and two-stage experts on development only."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from inside_airbnb_feature_ablation import FOLD_SEEDS
from inside_airbnb_phase0 import ROOT, utc_now, write_json_atomic
from inside_airbnb_quote_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    feature_matrix,
    group_split,
    load_silver,
    price_metrics,
)
from inside_airbnb_upper_tail_challenger import (
    MAX_OVERALL_MAE_DEGRADATION,
    MIN_ABSOLUTE_BIAS_IMPROVEMENT,
    MIN_BETTER_FOLDS,
    MIN_UPPER_TAIL_MAE_IMPROVEMENT,
    UPPER_TAIL_QUANTILE,
    candidate_decision,
    tail_sample_weights,
)
from premium_listing_features import (
    PREMIUM_CATEGORICAL_FEATURES,
    PREMIUM_NUMERIC_FEATURES,
)
from prepare_inside_airbnb_premium_features import DEFAULT_OUTPUT as DEFAULT_SILVER


DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_2026-06-16_premium_challenger.json"
)
DEFAULT_MARKDOWN = ROOT / "docs" / "inside_airbnb_premium_challenger.md"
TAIL_EXPERT_TRAINING_QUANTILE = 0.75
HARD_GATE_PROBABILITY = 0.25

EXTENDED_NUMERIC = [*NUMERIC_FEATURES, *PREMIUM_NUMERIC_FEATURES]
EXTENDED_CATEGORICAL = [*CATEGORICAL_FEATURES, *PREMIUM_CATEGORICAL_FEATURES]

CANDIDATES = [
    "incumbent_base_features",
    "premium_log_absolute",
    "premium_tail_weighted",
    "premium_two_stage_soft",
    "premium_two_stage_hard",
]


def soft_mixture(
    general_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    tail_probability: np.ndarray,
) -> np.ndarray:
    probability = np.clip(tail_probability, 0.0, 1.0)
    return general_prediction + probability * (
        expert_prediction - general_prediction
    )


def hard_mixture(
    general_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    tail_probability: np.ndarray,
    threshold: float = HARD_GATE_PROBABILITY,
) -> np.ndarray:
    return np.where(
        tail_probability >= threshold, expert_prediction, general_prediction
    )


def classifier_pipeline() -> Any:
    pipeline = build_pipeline(EXTENDED_NUMERIC, EXTENDED_CATEGORICAL)
    pipeline.set_params(
        model=HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=5.0,
            random_state=42,
        )
    )
    return pipeline


def fit_predict_single(
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    weighted: bool,
) -> np.ndarray:
    pipeline = build_pipeline(numeric, categorical)
    fit_kwargs = (
        {"model__sample_weight": tail_sample_weights(y[train])}
        if weighted
        else {}
    )
    pipeline.fit(X[train], np.log1p(y[train]), **fit_kwargs)
    return np.maximum(0.0, np.expm1(pipeline.predict(X[validation])))


def fit_predict_two_stage(
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    hard_gate: bool,
) -> np.ndarray:
    generalist = build_pipeline(EXTENDED_NUMERIC, EXTENDED_CATEGORICAL)
    generalist.fit(X[train], np.log1p(y[train]))
    general_prediction = np.maximum(
        0.0, np.expm1(generalist.predict(X[validation]))
    )

    classification_threshold = float(
        np.quantile(y[train], UPPER_TAIL_QUANTILE)
    )
    tail_labels = (y[train] > classification_threshold).astype(int)
    router = classifier_pipeline()
    router.fit(X[train], tail_labels)
    tail_probability = router.predict_proba(X[validation])[:, 1]

    expert_threshold = float(
        np.quantile(y[train], TAIL_EXPERT_TRAINING_QUANTILE)
    )
    expert_train = train[y[train] > expert_threshold]
    expert = build_pipeline(EXTENDED_NUMERIC, EXTENDED_CATEGORICAL)
    expert.set_params(model__loss="squared_error")
    expert.fit(X[expert_train], np.log1p(y[expert_train]))
    expert_prediction = np.maximum(
        0.0, np.expm1(expert.predict(X[validation]))
    )
    if hard_gate:
        return hard_mixture(
            general_prediction, expert_prediction, tail_probability
        )
    return soft_mixture(
        general_prediction, expert_prediction, tail_probability
    )


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "standard_deviation": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def evaluate(
    records: list[dict[str, str]],
    y: np.ndarray,
    development: np.ndarray,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    candidate: str,
) -> dict[str, Any]:
    if candidate == "incumbent_base_features":
        numeric, categorical = NUMERIC_FEATURES, CATEGORICAL_FEATURES
    else:
        numeric, categorical = EXTENDED_NUMERIC, EXTENDED_CATEGORICAL
    X = feature_matrix(records, numeric, categorical)
    fold_results = []
    for seed, train_relative, validation_relative in folds:
        train = development[train_relative]
        validation = development[validation_relative]
        if candidate == "incumbent_base_features":
            predicted = fit_predict_single(
                X, y, train, validation, numeric, categorical, False
            )
        elif candidate == "premium_log_absolute":
            predicted = fit_predict_single(
                X, y, train, validation, numeric, categorical, False
            )
        elif candidate == "premium_tail_weighted":
            predicted = fit_predict_single(
                X, y, train, validation, numeric, categorical, True
            )
        elif candidate == "premium_two_stage_soft":
            predicted = fit_predict_two_stage(
                X, y, train, validation, hard_gate=False
            )
        elif candidate == "premium_two_stage_hard":
            predicted = fit_predict_two_stage(
                X, y, train, validation, hard_gate=True
            )
        else:
            raise ValueError(f"Unknown candidate: {candidate}")
        threshold = float(np.quantile(y[train], UPPER_TAIL_QUANTILE))
        upper = y[validation] > threshold
        upper_error = predicted[upper] - y[validation][upper]
        fold_results.append(
            {
                "seed": seed,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "upper_tail_rows": int(np.sum(upper)),
                "upper_tail_threshold_aud": threshold,
                "overall_mae": price_metrics(y[validation], predicted)["mae"],
                "upper_tail_mae": float(np.mean(np.abs(upper_error))),
                "upper_tail_median_bias": float(np.median(upper_error)),
                "upper_tail_severe_underprediction_rate": float(
                    np.mean(predicted[upper] < y[validation][upper] * 0.70)
                ),
            }
        )
    return {
        "name": candidate,
        "numeric_features": numeric,
        "categorical_features": categorical,
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
        "# Inside Airbnb premium-feature and two-stage challenger",
        "",
        "All selection is performed on five host-disjoint development folds. The governed test set and primary artifact remain untouched.",
        "",
        "## Candidate design",
        "",
        "- Premium semantics: pool, hot tub, waterfront, beach access, water view, on-premises parking, gym, sauna, indoor fireplace, and private outdoor space.",
        "- Structural interactions: per-guest bathroom/bedroom/bed ratios, accommodates per bedroom, and bedroom-bathroom interaction.",
        "- Hierarchies: bathroom privacy and stable property group.",
        "- Two-stage router: classify fold-training p90, then blend or route to a log-squared expert trained above fold-training p75.",
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
            "## Predeclared promotion rule",
            "",
            "- Overall MAE may degrade by no more than 2%.",
            "- Upper-tail MAE must improve by at least 10%.",
            "- Absolute upper-tail median bias must improve by at least 10%.",
            "- Upper-tail MAE must improve in at least four of five folds.",
            "",
            "## Decision",
            "",
            f"- Status: `{report['decision']['status']}`.",
            f"- Selected challenger: `{report['decision']['selected_challenger']}`.",
            "- Primary model replaced: **no**.",
            f"- Next evidence: {report['decision']['next_evidence']}",
            "",
            "## Governance",
            "",
            "- Every upper-tail threshold is fitted inside the corresponding training fold.",
            "- Router labels and expert training membership never use validation targets.",
            "- Raw amenity strings and listing text are not retained in the enriched Silver table.",
            "- A challenger cannot be promoted without a compatible future temporal snapshot.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    silver_path: Path, report_path: Path, markdown_path: Path
) -> dict[str, Any]:
    records, base_X, y, groups = load_silver(silver_path)
    missing = sorted(
        set(PREMIUM_NUMERIC_FEATURES + PREMIUM_CATEGORICAL_FEATURES)
        - set(records[0])
    )
    if missing:
        raise ValueError(f"Premium Silver is missing fields: {missing}")
    development, governed_test = group_split(base_X, groups, 0.20, 42)
    folds = []
    for seed in FOLD_SEEDS:
        train_relative, validation_relative = group_split(
            base_X[development], groups[development], 0.20, seed
        )
        folds.append((seed, train_relative, validation_relative))
    candidates = [
        evaluate(records, y, development, folds, candidate)
        for candidate in CANDIDATES
    ]
    incumbent = candidates[0]
    for candidate in candidates[1:]:
        candidate["decision"] = candidate_decision(incumbent, candidate)
    qualifying = [
        candidate
        for candidate in candidates[1:]
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
            "development_rows": len(development),
            "governed_test_rows_reserved_and_unused": len(governed_test),
            "governed_test_unique_hosts": len(set(groups[governed_test])),
            "fold_seeds": list(FOLD_SEEDS),
            "group": "host_id",
            "upper_tail_definition": "validation target above fold-training p90",
            "tail_expert_training_definition": "fold-training target above p75",
            "hard_gate_probability": HARD_GATE_PROBABILITY,
        },
        "premium_feature_contract": {
            "numeric": PREMIUM_NUMERIC_FEATURES,
            "categorical": PREMIUM_CATEGORICAL_FEATURES,
            "raw_amenity_strings_retained": False,
        },
        "promotion_thresholds": {
            "maximum_overall_mae_degradation": MAX_OVERALL_MAE_DEGRADATION,
            "minimum_upper_tail_mae_improvement": (
                MIN_UPPER_TAIL_MAE_IMPROVEMENT
            ),
            "minimum_absolute_bias_improvement": MIN_ABSOLUTE_BIAS_IMPROVEMENT,
            "minimum_better_folds": MIN_BETTER_FOLDS,
        },
        "candidates": candidates,
        "decision": {
            "status": (
                "CHALLENGER_IDENTIFIED_FOR_FUTURE_TEMPORAL_EVALUATION"
                if selected
                else "NO_CHALLENGER_PASSED_ALL_GATES"
            ),
            "selected_challenger": selected,
            "primary_model_replaced": False,
            "next_evidence": (
                "Rebuild the enriched Silver table on a compatible future "
                "snapshot and run strict out-of-time evaluation."
                if selected
                else (
                    "Retain the incumbent; richer licensed market/context data "
                    "is required for a material upper-tail improvement."
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
