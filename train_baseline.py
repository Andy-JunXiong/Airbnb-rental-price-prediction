"""Reproducible baselines for the processed Airbnb price data.

The repository does not contain the original modelling data, so this script uses
the checked-in processed feature matrices. It verifies label alignment before
running repeated cross-validation and writes machine-readable results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "compact": ROOT / "Models" / "Trees_features.csv",
    "full": ROOT / "Exploratory Data Analysis" / "EDA_X_train.csv",
    "corrected": ROOT / "data" / "processed" / "train_features.csv",
}
LABELS = ROOT / "Models" / "Trees_labels.csv"


def read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        rows = [[float(row[name]) for name in names] for row in reader]
    return names, np.asarray(rows, dtype=float)


def load_dataset(kind: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if kind == "corrected" and not DATASETS[kind].exists():
        raise FileNotFoundError("Run `python prepare_features.py` before using the corrected dataset")
    feature_names, feature_rows = read_numeric_csv(DATASETS[kind])
    label_names, label_rows = read_numeric_csv(LABELS)
    if "Id" not in feature_names or label_names != ["x", "y"]:
        raise ValueError("Unexpected feature or label schema")

    # Trees_features contains train followed by the unlabeled competition test
    # set. EDA_X_train contains only the labeled rows.
    train_rows = feature_rows[: len(label_rows)]
    ids = train_rows[:, feature_names.index("Id")]
    if not np.array_equal(ids, label_rows[:, 0]):
        raise ValueError("Feature Id values do not align exactly with label x values")

    keep = [i for i, name in enumerate(feature_names) if name != "Id"]
    X = train_rows[:, keep]
    y = label_rows[:, 1]
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("The processed modelling data contains missing/non-finite values")
    return X, y, [feature_names[i] for i in keep]


def models() -> dict[str, object]:
    return {
        "median": DummyRegressor(strategy="median"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
            l2_regularization=5.0, random_state=42,
        ),
        "extra_trees": ExtraTreesRegressor(
            # Modal choice across the five independent inner searches in
            # robust_evaluation.py.
            n_estimators=500, max_depth=12, min_samples_leaf=5,
            max_features=0.6, n_jobs=-1, random_state=42,
        ),
    }


def evaluate(X: np.ndarray, y: np.ndarray) -> list[dict[str, object]]:
    splitter = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    results = []
    for name, estimator in models().items():
        folds = []
        for train_index, test_index in splitter.split(X):
            fitted = clone(estimator).fit(X[train_index], y[train_index])
            predicted = fitted.predict(X[test_index])
            folds.append({
                "rmse": float(mean_squared_error(y[test_index], predicted) ** 0.5),
                "mae": float(mean_absolute_error(y[test_index], predicted)),
                "r2": float(r2_score(y[test_index], predicted)),
            })
        summary = {"model": name, "folds": len(folds)}
        for metric in ("rmse", "mae", "r2"):
            values = np.asarray([fold[metric] for fold in folds])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        results.append(summary)
    return sorted(results, key=lambda row: row["rmse_mean"])


def feature_importance(
    X: np.ndarray, y: np.ndarray, names: list[str], model_name: str
) -> list[dict[str, float | str]]:
    estimator = models()[model_name]
    estimator.fit(X, y)
    scores = permutation_importance(
        estimator, X, y, scoring="neg_root_mean_squared_error",
        n_repeats=20, random_state=42, n_jobs=-1,
    )
    order = np.argsort(scores.importances_mean)[::-1][:10]
    return [
        {"feature": names[i], "importance": float(scores.importances_mean[i])}
        for i in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="full")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "baseline_results.json")
    args = parser.parse_args()

    X, y, names = load_dataset(args.dataset)
    results = evaluate(X, y)
    best_name = next(row["model"] for row in results if row["model"] != "median")
    report = {
        "dataset": args.dataset,
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "validation": "RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)",
        "models": results,
        "top_permutation_features": feature_importance(X, y, names, best_name),
        "limitations": [
            "Only processed features are available; preprocessing cannot be cross-validated.",
            "The dataset has no host identifier, so host-grouped validation is impossible.",
            "Permutation importance is descriptive because it is computed on all training rows.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Verified {X.shape[0]} aligned rows with {X.shape[1]} features ({args.dataset}).")
    print(f"{'model':24} {'RMSE':>15} {'MAE':>15} {'R2':>15}")
    for row in results:
        print(
            f"{row['model']:24} "
            f"{row['rmse_mean']:7.2f} +/- {row['rmse_std']:.2f} "
            f"{row['mae_mean']:7.2f} +/- {row['mae_std']:.2f} "
            f"{row['r2_mean']:7.3f} +/- {row['r2_std']:.3f}"
        )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
