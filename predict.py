"""Fit the selected baseline on all labeled rows and predict the test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from train_baseline import ROOT, load_dataset, models, read_numeric_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-features",
        type=Path,
        default=ROOT / "data" / "processed" / "test_features.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "predictions" / "extra_trees_predictions.csv"
    )
    args = parser.parse_args()

    X_train, y_train, train_names = load_dataset("corrected")
    test_names, test_rows = read_numeric_csv(args.test_features)
    if "Id" not in test_names:
        raise ValueError("Test features must contain an Id column")
    ids = test_rows[:, test_names.index("Id")].astype(int)
    keep = [i for i, name in enumerate(test_names) if name != "Id"]
    feature_names = [test_names[i] for i in keep]
    if feature_names != train_names:
        raise ValueError("Train and test feature schemas differ")
    if len(ids) != len(set(ids)) or set(ids) != set(range(570)):
        raise ValueError("Expected each test ID from 0 through 569 exactly once")

    model = models()["extra_trees"].fit(X_train, y_train)
    predictions = np.maximum(model.predict(test_rows[:, keep]), 0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "price"])
        writer.writerows(sorted(zip(ids, predictions), key=lambda row: row[0]))
    print(
        f"Wrote {len(predictions)} predictions to {args.output}; "
        f"range ${predictions.min():.2f}-${predictions.max():.2f}, "
        f"median ${np.median(predictions):.2f}"
    )


if __name__ == "__main__":
    main()
