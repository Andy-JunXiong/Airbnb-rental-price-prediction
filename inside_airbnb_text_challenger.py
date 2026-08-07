"""Benchmark text-embedding features on development folds without touching test."""

from __future__ import annotations

import argparse
import csv
import json
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
    CATEGORICAL_FEATURES,
    DEFAULT_SILVER as BASE_SILVER,
    NUMERIC_FEATURES,
    build_pipeline,
    feature_matrix,
    group_split,
    load_silver,
    price_metrics,
)
from inside_airbnb_text_features import (
    TEXT_FEATURE_NAMES,
    reference_manifest as text_reference_manifest,
)
from prepare_inside_airbnb_text_features import DEFAULT_OUTPUT as TEXT_SILVER


_SNAPSHOT = active_snapshot_date()
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / f"sydney_{_SNAPSHOT}_text_challenger.json"
)
DEFAULT_MARKDOWN = ROOT / "docs" / "inside_airbnb_text_challenger.md"
MIN_RELATIVE_MAE_IMPROVEMENT = 0.02
MIN_BETTER_FOLDS = 3
MAX_MAE_DEGRADATION = 0.02


def load_text_silver(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["training_eligible"] == "1"
        ]


def evaluate_text_candidate(
    base_records: list[dict[str, str]],
    text_records: list[dict[str, str]],
    y: np.ndarray,
    groups: np.ndarray,
    development_indices: np.ndarray,
    fold: tuple[int, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    seed, train_rel, val_rel = fold
    train = development_indices[train_rel]
    val = development_indices[val_rel]

    # --- incumbent: base features only ---
    X_base = feature_matrix(base_records, list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES))
    pipe_base = build_pipeline(list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES))
    pipe_base.fit(X_base[train], np.log1p(y[train]))
    base_pred = np.maximum(0.0, np.expm1(pipe_base.predict(X_base[val])))
    base_metrics = price_metrics(y[val], base_pred)

    # --- challenger: base + text ---
    extended_numeric = list(NUMERIC_FEATURES) + list(TEXT_FEATURE_NAMES)
    X_text = feature_matrix(text_records, extended_numeric, list(CATEGORICAL_FEATURES))
    pipe_text = build_pipeline(extended_numeric, list(CATEGORICAL_FEATURES))
    pipe_text.fit(X_text[train], np.log1p(y[train]))
    text_pred = np.maximum(0.0, np.expm1(pipe_text.predict(X_text[val])))
    text_metrics = price_metrics(y[val], text_pred)

    return {
        "seed": seed,
        "train_rows": len(train),
        "val_rows": len(val),
        "host_overlap": len(set(groups[train]) & set(groups[val])),
        "incumbent": base_metrics,
        "challenger": text_metrics,
        "mae_delta": float(text_metrics["mae"] - base_metrics["mae"]),
    }


def run_text_challenger(
    base_silver: Path = BASE_SILVER,
    text_silver: Path = TEXT_SILVER,
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    if not text_silver.exists():
        raise FileNotFoundError(
            f"Text Silver not found at {text_silver}. "
            "Run prepare_inside_airbnb_text_features.py first."
        )
    base_records, X, y, groups = load_silver(base_silver)
    text_records = load_text_silver(text_silver)
    if len(base_records) != len(text_records):
        raise ValueError("Base and text Silver tables have different row counts")

    development, governed_test = group_split(X, groups, 0.20, 42)

    folds = []
    for seed in FOLD_SEEDS:
        train_rel, val_rel = group_split(
            X[development], groups[development], 0.20, seed
        )
        folds.append((seed, train_rel, val_rel))

    results = [
        evaluate_text_candidate(
            base_records, text_records, y, groups, development, fold
        )
        for fold in folds
    ]

    mae_deltas = [r["mae_delta"] for r in results]
    mean_delta = statistics.mean(mae_deltas)
    better_folds = sum(1 for d in mae_deltas if d < 0)
    inc_mae = statistics.mean(r["incumbent"]["mae"] for r in results)
    chal_mae = statistics.mean(r["challenger"]["mae"] for r in results)

    relative_improvement = (inc_mae - chal_mae) / inc_mae
    promote = (
        mean_delta < 0
        and relative_improvement >= MIN_RELATIVE_MAE_IMPROVEMENT
        and better_folds >= MIN_BETTER_FOLDS
        and chal_mae / inc_mae - 1 <= MAX_MAE_DEGRADATION
    )

    report: dict[str, Any] = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "candidate": {
            "name": "text_embeddings",
            "description": (
                "Sentence-transformer or TF-IDF embeddings of listing "
                "description, neighborhood overview, name, and host about text."
            ),
            "feature_count": len(TEXT_FEATURE_NAMES),
            "text_reference": text_reference_manifest(),
        },
        "protocol": {
            "governed_test_rows_reserved": len(governed_test),
            "folds": len(FOLD_SEEDS),
            "host_disjoint": True,
        },
        "results": {
            "incumbent_mean_mae": inc_mae,
            "challenger_mean_mae": chal_mae,
            "mean_mae_delta": mean_delta,
            "relative_mae_improvement": float(
                (inc_mae - chal_mae) / inc_mae
            ),
            "better_folds": better_folds,
            "per_fold": results,
        },
        "decision": {
            "status": (
                "TEXT_FEATURES_PROMOTED"
                if promote
                else "TEXT_FEATURES_REJECTED"
            ),
            "promote": promote,
            "thresholds": {
                "min_relative_mae_improvement": MIN_RELATIVE_MAE_IMPROVEMENT,
                "min_better_folds": MIN_BETTER_FOLDS,
                "max_overall_mae_degradation": MAX_OVERALL_MAE_DEGRADATION,
            },
        },
        "governance": {
            "governed_test_used": False,
            "primary_artifact_written": False,
            "promotion_rule": (
                "Mean development MAE must improve by at least "
                f"{MIN_RELATIVE_MAE_IMPROVEMENT:.0%} and at least "
                f"{MIN_BETTER_FOLDS} of {len(FOLD_SEEDS)} folds must improve."
            ),
        },
    }
    write_json_atomic(report_path, report)
    write_challenger_markdown(markdown_path, report)
    print(
        f"text challenger: MAE {inc_mae:.2f} → {chal_mae:.2f} "
        f"({mean_delta:+.2f}, {better_folds}/{len(FOLD_SEEDS)} folds) → "
        f"{report['decision']['status']}"
    )
    return report


def write_challenger_markdown(path: Path, report: dict[str, Any]) -> None:
    d = report["decision"]
    r = report["results"]
    lines = [
        "# Text-feature challenger",
        "",
        f"Decision: **{d['status']}**",
        "",
        "## Summary",
        "",
        f"| Metric | Incumbent | +Text | Δ |",
        "| --- | ---: | ---: | ---: |",
        f"| MAE | {r['incumbent_mean_mae']:.2f} | {r['challenger_mean_mae']:.2f} | "
        f"{r['mean_mae_delta']:+.2f} |",
        f"| Better folds | — | {r['better_folds']}/{report['protocol']['folds']} | — |",
        f"| Relative improvement | — | {r['relative_mae_improvement']:.1%} | — |",
        "",
        "## Per-fold results",
        "",
        "| Fold | Incumbent MAE | +Text MAE | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for fold in r["per_fold"]:
        lines.append(
            f"| {fold['seed']} | {fold['incumbent']['mae']:.2f} | "
            f"{fold['challenger']['mae']:.2f} | {fold['mae_delta']:+.2f} |"
        )
    lines.extend([
        "",
        "## Governance",
        "",
        "- Governed test set was not used.",
        "- Primary artifact was not written.",
        "- Text features are privacy-minimised: raw text is not stored.",
        "- Missing text (empty descriptions) are encoded as NaN and imputed by median.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-silver", type=Path, default=BASE_SILVER)
    parser.add_argument("--text-silver", type=Path, default=TEXT_SILVER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_text_challenger(args.base_silver, args.text_silver, args.report, args.markdown)


if __name__ == "__main__":
    main()
