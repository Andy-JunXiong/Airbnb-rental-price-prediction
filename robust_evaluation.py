"""Nested model selection and leave-one-area-out stress testing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from train_baseline import ROOT, load_dataset


PARAM_GRID = {
    "n_estimators": [300],
    "max_features": [0.6, 0.8, 1.0],
    "min_samples_leaf": [2, 3, 5],
    "max_depth": [None, 12],
}


def metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, predicted) ** 0.5),
        "mae": float(mean_absolute_error(y_true, predicted)),
        "r2": float(r2_score(y_true, predicted)),
    }


def nested_cv(X: np.ndarray, y: np.ndarray) -> dict[str, object]:
    outer = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    out_of_fold = np.empty_like(y, dtype=float)
    for fold, (train_index, test_index) in enumerate(outer.split(X), start=1):
        search = GridSearchCV(
            ExtraTreesRegressor(n_jobs=-1, random_state=42),
            PARAM_GRID,
            scoring="neg_root_mean_squared_error",
            cv=KFold(n_splits=5, shuffle=True, random_state=100 + fold),
            n_jobs=-1,
        )
        search.fit(X[train_index], y[train_index])
        predicted = search.predict(X[test_index])
        out_of_fold[test_index] = predicted
        folds.append({
            "fold": fold,
            **metrics(y[test_index], predicted),
            "best_inner_rmse": float(-search.best_score_),
            "best_parameters": search.best_params_,
        })
    absolute_errors = np.abs(y - out_of_fold)
    return {
        "protocol": "5-fold outer CV with independent 5-fold inner GridSearchCV",
        "parameter_grid": PARAM_GRID,
        "folds": folds,
        "aggregate_out_of_fold": metrics(y, out_of_fold),
        "absolute_error_quantiles": {
            "p50": float(np.quantile(absolute_errors, 0.50)),
            "p80": float(np.quantile(absolute_errors, 0.80)),
            "p90": float(np.quantile(absolute_errors, 0.90)),
            "p95": float(np.quantile(absolute_errors, 0.95)),
        },
    }


def area_holdout(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict[str, object]:
    area_columns = [name for name in feature_names if name.startswith("area_")]
    results = []
    for area_column in area_columns:
        area_index = feature_names.index(area_column)
        test_mask = X[:, area_index] == 1
        train_mask = ~test_mask
        model = ExtraTreesRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=5, max_features=0.6,
            n_jobs=-1, random_state=42,
        ).fit(X[train_mask], y[train_mask])
        predicted = model.predict(X[test_mask])
        results.append({
            "held_out_area": area_column.removeprefix("area_"),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            **metrics(y[test_mask], predicted),
            "actual_mean": float(y[test_mask].mean()),
            "predicted_mean": float(predicted.mean()),
        })
    return {
        "protocol": "Train on two areas and test on the entirely unseen third area",
        "warning": "This is a geographic transfer stress test, not the primary CV score.",
        "areas": results,
    }


def main() -> None:
    X, y, names = load_dataset("corrected")
    report = {
        "samples": int(len(y)),
        "features": int(X.shape[1]),
        "nested_model_selection": nested_cv(X, y),
        "leave_one_area_out": area_holdout(X, y, names),
    }
    output = ROOT / "reports" / "robust_evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    nested = report["nested_model_selection"]["aggregate_out_of_fold"]
    print(
        f"Nested OOF: RMSE {nested['rmse']:.2f}, MAE {nested['mae']:.2f}, "
        f"R2 {nested['r2']:.3f}"
    )
    for row in report["leave_one_area_out"]["areas"]:
        print(
            f"Hold out {row['held_out_area']}: n={row['test_rows']}, "
            f"RMSE {row['rmse']:.2f}, MAE {row['mae']:.2f}, R2 {row['r2']:.3f}"
        )
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
