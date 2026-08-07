"""Train the quote model on multiple compatible Inside Airbnb snapshots.

The multi-snapshot protocol:
- Collects all quote-compatible Silver tables (June 2026).
- Splits by snapshot date (earliest → train, middle → calibration, latest → test).
- Enforces host-disjoint splits across time.
- Trains with the same LightGBM pipeline as the single-snapshot MVP.
- Compares performance against single-snapshot baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from inside_airbnb_phase0 import (
    ROOT,
    active_snapshot_date,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from inside_airbnb_quote_model import (
    CATEGORICAL_FEATURES,
    DEFAULT_ARTIFACT,
    DEFAULT_REPORT,
    DEFAULT_SILVER,
    MIN_COMPARABLES,
    NUMERIC_FEATURES,
    TARGET,
    _LGBM_AVAILABLE,
    build_pipeline,
    calibration_quantiles,
    feature_matrix,
    finite_sample_quantile,
    fit_market_baseline,
    interval_bounds,
    market_prediction,
    price_metrics,
    quantiles_for_records,
    refusal_reasons,
    supported_price_range,
)


DEFAULT_MULTI_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_multi_snapshot_evaluation.json"
)
SNAPSHOT_DATES = [
    "2026-06-15",
    "2026-06-16",
    "2026-06-19",
    "2026-06-22",
    "2026-06-25",
    "2026-06-28",
]
TRAIN_FRAC = 0.65
CALIBRATION_FRAC = 0.15


def silver_path_for_date(snapshot_date: str) -> Path:
    return (
        ROOT
        / "data"
        / "silver"
        / "inside_airbnb"
        / "sydney"
        / f"snapshot_date={snapshot_date}"
        / "listing_quotes.csv"
    )


def load_eligible(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["training_eligible"] == "1"
        ]


def load_multi_silver(
    snapshot_dates: list[str],
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and merge all compatible Silver tables. Returns (records, X, y, groups, snapshot_index)."""
    all_records: list[dict[str, str]] = []
    all_X_parts: list[np.ndarray] = []
    all_y_parts: list[np.ndarray] = []
    all_groups_parts: list[np.ndarray] = []
    all_snapshot_parts: list[np.ndarray] = []
    for idx, snapshot_date in enumerate(snapshot_dates):
        path = silver_path_for_date(snapshot_date)
        if not path.exists():
            print(f"skip missing {path}")
            continue
        records = load_eligible(path)
        if not records:
            print(f"skip empty {snapshot_date}")
            continue
        X = feature_matrix(records, list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES))
        y = np.asarray([float(row[TARGET]) for row in records], dtype=float)
        groups = np.asarray([row["host_id"] for row in records], dtype=object)
        snapshot_idx = np.full(len(records), idx, dtype=int)
        all_records.extend(records)
        all_X_parts.append(X)
        all_y_parts.append(y)
        all_groups_parts.append(groups)
        all_snapshot_parts.append(snapshot_idx)
        print(f"loaded {snapshot_date}: {len(records):,} eligible rows")

    if len(all_records) == 0:
        raise ValueError("No compatible Silver tables found")
    if len(all_X_parts) <= 1:
        print(
            "MULTI_SNAPSHOT_SKIP: only one compatible Silver table available. "
            "Download additional snapshots first."
        )
        return (
            [],
            np.empty((0, 0)),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,)),
        )

    return (
        all_records,
        np.concatenate(all_X_parts),
        np.concatenate(all_y_parts),
        np.concatenate(all_groups_parts),
        np.concatenate(all_snapshot_parts),
    )


def temporal_host_disjoint_split(
    groups: np.ndarray,
    snapshot_indices: np.ndarray,
    unique_snapshots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by snapshot index, enforcing host-disjoint across splits."""
    n_snapshots = len(unique_snapshots)
    train_cut = max(1, int(n_snapshots * TRAIN_FRAC))
    cal_cut = max(train_cut + 1, int(n_snapshots * (TRAIN_FRAC + CALIBRATION_FRAC)))

    train_mask = np.isin(snapshot_indices, unique_snapshots[:train_cut])
    cal_mask = np.isin(snapshot_indices, unique_snapshots[train_cut:cal_cut])
    test_mask = np.isin(snapshot_indices, unique_snapshots[cal_cut:])

    train_hosts = set(groups[train_mask])
    cal_hosts = set(groups[cal_mask]) - train_hosts
    test_hosts = set(groups[test_mask]) - train_hosts - cal_hosts

    cal_mask = cal_mask & np.isin(groups, list(cal_hosts))
    test_mask = test_mask & np.isin(groups, list(test_hosts))

    train_idx = np.where(train_mask)[0]
    cal_idx = np.where(cal_mask)[0]
    test_idx = np.where(test_mask)[0]

    overlap_tc = len(set(groups[train_idx]) & set(groups[cal_idx]))
    overlap_tt = len(set(groups[train_idx]) & set(groups[test_idx]))
    overlap_ct = len(set(groups[cal_idx]) & set(groups[test_idx]))
    if overlap_tc or overlap_tt or overlap_ct:
        raise AssertionError(
            f"Host overlap: train-cal={overlap_tc}, train-test={overlap_tt}, cal-test={overlap_ct}"
        )
    return train_idx, cal_idx, test_idx


def train_multi_snapshot(
    snapshot_dates: list[str] | None = None,
    report_path: Path = DEFAULT_MULTI_REPORT,
) -> dict[str, Any]:
    snapshot_dates = snapshot_dates or SNAPSHOT_DATES
    records, X, y, groups, snapshot_indices = load_multi_silver(snapshot_dates)
    if len(records) == 0:
        report = {
            "report_version": 1,
            "generated_at_utc": utc_now(),
            "status": "skipped",
            "reason": "Insufficient compatible Silver tables (need >=2).",
        }
        write_json_atomic(report_path, report)
        print(f"multi-snapshot skipped (need >=2 Silver tables)")
        return report

    unique_snapshots = np.unique(snapshot_indices)

    train_idx, cal_idx, test_idx = temporal_host_disjoint_split(
        groups, snapshot_indices, unique_snapshots
    )

    pipeline = build_pipeline()
    pipeline.fit(X[train_idx], np.log1p(y[train_idx]))

    cal_log = pipeline.predict(X[cal_idx])
    cal_residuals = np.abs(np.log1p(y[cal_idx]) - cal_log)
    global_quantile, segment_quantiles, room_quantiles = calibration_quantiles(
        records, cal_idx, cal_residuals
    )

    test_log = pipeline.predict(X[test_idx])
    test_prediction = np.maximum(0.0, np.expm1(test_log))
    test_q = quantiles_for_records(
        records, test_idx, global_quantile, segment_quantiles, room_quantiles
    )
    lower, upper = interval_bounds(test_log, test_q)

    baseline = fit_market_baseline(records, y, train_idx)
    baseline_prediction = np.asarray(
        [market_prediction(baseline, records[int(i)])[0] for i in test_idx],
        dtype=float,
    )

    covered = (y[test_idx] >= lower) & (y[test_idx] <= upper)

    train_snapshots = sorted(set(
        records[int(i)]["snapshot_label"] for i in train_idx
    ))
    cal_snapshots = sorted(set(
        records[int(i)]["snapshot_label"] for i in cal_idx
    ))
    test_snapshots = sorted(set(
        records[int(i)]["snapshot_label"] for i in test_idx
    ))

    train_p01 = float(np.quantile(y[train_idx], 0.01))
    train_p99 = float(np.quantile(y[train_idx], 0.99))
    core_market = (y[test_idx] >= train_p01) & (y[test_idx] <= train_p99)

    report: dict[str, Any] = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "protocol": {
            "type": "multi-snapshot temporal host-disjoint",
            "train_snapshots": train_snapshots,
            "calibration_snapshots": cal_snapshots,
            "test_snapshots": test_snapshots,
            "host_disjoint": True,
            "rows": {
                "total": len(y),
                "train": len(train_idx),
                "calibration": len(cal_idx),
                "test": len(test_idx),
            },
        },
        "model": {
            "algorithm": (
                "LightGBM LGBMRegressor on log1p target (multi-snapshot)"
                if _LGBM_AVAILABLE
                else "HistGradientBoostingRegressor on log1p target (multi-snapshot)"
            ),
            "test_metrics_all": price_metrics(y[test_idx], test_prediction),
            "test_metrics_core_market": price_metrics(
                y[test_idx][core_market], test_prediction[core_market]
            ),
        },
        "market_baseline": {
            "test_metrics_all": price_metrics(y[test_idx], baseline_prediction),
        },
        "comparison": {
            "relative_mae_improvement_vs_market": float(
                1
                - price_metrics(y[test_idx], test_prediction)["mae"]
                / price_metrics(y[test_idx], baseline_prediction)["mae"]
            ),
        },
        "conformal_interval": {
            "target_coverage": 0.90,
            "test_coverage_all": float(np.mean(covered)),
            "test_average_width_all": float(np.mean(upper - lower)),
        },
        "snapshot_composition": {
            str(records[int(i)]["snapshot_label"]): int(
                np.sum(snapshot_indices == idx)
            )
            for idx, _ in enumerate(unique_snapshots)
        },
    }

    write_json_atomic(report_path, report)
    print(
        f"multi-snapshot MAE {report['model']['test_metrics_all']['mae']:.2f} "
        f"(baseline {report['market_baseline']['test_metrics_all']['mae']:.2f})"
    )
    print(f"coverage {report['conformal_interval']['test_coverage_all']:.3f}")
    print(f"train: {train_snapshots}  cal: {cal_snapshots}  test: {test_snapshots}")
    print(f"report {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dates", nargs="*", default=SNAPSHOT_DATES)
    parser.add_argument("--report", type=Path, default=DEFAULT_MULTI_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_multi_snapshot(args.snapshot_dates, args.report)


if __name__ == "__main__":
    main()
