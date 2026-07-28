"""Create a compact, dependency-light audit of the corrected modelling data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from train_baseline import ROOT, load_dataset, read_numeric_csv


def main() -> None:
    X, y, names = load_dataset("corrected")
    raw_names, raw_rows = read_numeric_csv(ROOT / "data" / "processed" / "train_features.csv")
    ids = raw_rows[:, raw_names.index("Id")].astype(int)
    area_names = [name for name in names if name.startswith("area_")]
    area_indices = [names.index(name) for name in area_names]
    area_counts = Counter(
        area_names[int(np.argmax(row[area_indices]))].removeprefix("area_") for row in X
    )
    correlations = []
    for index, name in enumerate(names):
        correlation = float(np.corrcoef(X[:, index], y)[0, 1])
        if np.isfinite(correlation):
            correlations.append({"feature": name, "correlation": correlation})

    report = {
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "id": {"unique": len(set(ids)), "min": int(ids.min()), "max": int(ids.max())},
        "target_price": {
            "min": float(y.min()), "q1": float(np.quantile(y, 0.25)),
            "median": float(np.median(y)), "mean": float(y.mean()),
            "q3": float(np.quantile(y, 0.75)), "max": float(y.max()),
            "std": float(y.std(ddof=1)),
        },
        "area_counts": dict(sorted(area_counts.items())),
        "top_absolute_price_correlations": sorted(
            correlations, key=lambda row: abs(row["correlation"]), reverse=True
        )[:10],
        "quality_checks": {
            "finite_features": bool(np.isfinite(X).all()),
            "finite_target": bool(np.isfinite(y).all()),
            "unique_ids": len(set(ids)) == len(ids),
            "one_area_per_row": bool(np.all(X[:, area_indices].sum(axis=1) == 1)),
        },
    }
    output = ROOT / "reports" / "data_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
