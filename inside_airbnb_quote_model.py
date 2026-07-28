"""Train and serve the evidence-aware Inside Airbnb quote-level MVP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from inside_airbnb_phase0 import ROOT, sha256_file, utc_now, write_json_atomic
from sydney_geography import (
    GEOGRAPHIC_FEATURES,
    geographic_features,
    reference_manifest,
)


DEFAULT_SILVER = (
    ROOT
    / "data"
    / "silver"
    / "inside_airbnb"
    / "sydney"
    / "snapshot_date=2026-06-16"
    / "listing_quotes.csv"
)
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_2026-06-16_quote_mvp_evaluation.json"
)
DEFAULT_ARTIFACT = ROOT / "artifacts" / "inside_airbnb_quote_mvp.joblib"
DEFAULT_TEMPORAL_COMPATIBILITY = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / "sydney_snapshot_target_compatibility.json"
)
DEFAULT_TEMPORAL_EVALUATION = (
    ROOT / "reports" / "inside_airbnb" / "sydney_temporal_quote_validation.json"
)

NUMERIC_FEATURES = [
    "quote_lead_days",
    "stay_nights",
    "checkin_month",
    "checkin_day_of_week",
    "checkin_is_weekend",
    "latitude",
    "longitude",
    *GEOGRAPHIC_FEATURES,
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "amenities_count",
    "minimum_nights",
    "maximum_nights",
    "calculated_host_listings_count",
]
CATEGORICAL_FEATURES = [
    "neighbourhood",
    "property_type",
    "room_type",
    "host_is_superhost",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CRITICAL_FEATURES = [
    "neighbourhood",
    "property_type",
    "room_type",
    "accommodates",
    "quote_lead_days",
    "stay_nights",
]
TARGET = "target_quoted_price_per_night"
GROUP = "host_id"
ALPHA = 0.10
MIN_SEGMENT_CALIBRATION = 50
MIN_COMPARABLES = 20
MAX_RELATIVE_INTERVAL_WIDTH = 2.0
MAX_SNAPSHOT_AGE_DAYS = 120


def finite_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return np.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return np.nan
    return parsed if math.isfinite(parsed) else np.nan


def load_silver(
    path: Path,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        eligible = [
            row
            for row in csv.DictReader(handle)
            if row["training_eligible"] == "1"
        ]
    X = feature_matrix(eligible, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    y = np.asarray([float(row[TARGET]) for row in eligible], dtype=float)
    groups = np.asarray([row[GROUP] for row in eligible], dtype=object)
    return eligible, X, y, groups


def feature_matrix(
    records: list[dict[str, Any]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    matrix = [
        [finite_float(row.get(name)) for name in numeric_features]
        + [str(row.get(name, "") or "").strip() for name in categorical_features]
        for row in records
    ]
    return np.asarray(matrix, dtype=object)


def build_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    feature_count = len(numeric_features) + len(categorical_features)
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, list(range(len(numeric_features)))),
            (
                "categorical",
                categorical,
                list(range(len(numeric_features), feature_count)),
            ),
        ],
        remainder="drop",
    )
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=5.0,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def group_split(
    X: np.ndarray, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train, test = next(splitter.split(X, groups=groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("Host groups overlap across split")
    return train, test


def price_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predicted)),
        "median_absolute_error": float(median_absolute_error(y_true, predicted)),
        "rmse": float(mean_squared_error(y_true, predicted) ** 0.5),
        "r2": float(r2_score(y_true, predicted)),
    }


def finite_sample_quantile(residuals: np.ndarray, alpha: float = ALPHA) -> float:
    if len(residuals) == 0:
        raise ValueError("Conformal calibration requires residuals")
    probability = min(1.0, math.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
    return float(np.quantile(residuals, probability, method="higher"))


def fit_market_baseline(
    records: list[dict[str, str]], y: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    exact: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    neighbourhood: defaultdict[str, list[float]] = defaultdict(list)
    room: defaultdict[str, list[float]] = defaultdict(list)
    for index in indices:
        row = records[int(index)]
        value = float(y[int(index)])
        key = (row["neighbourhood"], row["room_type"])
        exact[key].append(value)
        neighbourhood[row["neighbourhood"]].append(value)
        room[row["room_type"]].append(value)
    return {
        "global": float(np.median(y[indices])),
        "exact": {
            key: {"median": statistics.median(values), "count": len(values)}
            for key, values in exact.items()
        },
        "neighbourhood": {
            key: {"median": statistics.median(values), "count": len(values)}
            for key, values in neighbourhood.items()
        },
        "room": {
            key: {"median": statistics.median(values), "count": len(values)}
            for key, values in room.items()
        },
    }


def market_prediction(
    baseline: dict[str, Any], row: dict[str, str]
) -> tuple[float, int, str]:
    exact = baseline["exact"].get((row["neighbourhood"], row["room_type"]))
    if exact and exact["count"] >= 10:
        return float(exact["median"]), int(exact["count"]), "neighbourhood_room"
    neighbourhood = baseline["neighbourhood"].get(row["neighbourhood"])
    if neighbourhood and neighbourhood["count"] >= 10:
        return (
            float(neighbourhood["median"]),
            int(neighbourhood["count"]),
            "neighbourhood",
        )
    room = baseline["room"].get(row["room_type"])
    if room and room["count"] >= 10:
        return float(room["median"]), int(room["count"]), "room_type"
    return float(baseline["global"]), 0, "global"


def interval_bounds(
    predicted_log: np.ndarray, quantiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(0.0, np.expm1(predicted_log - quantiles))
    upper = np.expm1(predicted_log + quantiles)
    return lower, upper


def category_inventory(
    records: list[dict[str, str]], indices: np.ndarray
) -> dict[str, list[str]]:
    return {
        name: sorted({records[int(index)][name] for index in indices})
        for name in CATEGORICAL_FEATURES
    }


def supported_price_range(y: np.ndarray, indices: np.ndarray) -> list[float]:
    return [
        float(np.quantile(y[indices], 0.005)),
        float(np.quantile(y[indices], 0.995)),
    ]


def calibration_quantiles(
    records: list[dict[str, str]],
    indices: np.ndarray,
    residuals: np.ndarray,
) -> tuple[
    float,
    dict[str, dict[str, float | int]],
    dict[str, dict[str, float | int]],
]:
    global_quantile = finite_sample_quantile(residuals)
    by_segment: defaultdict[str, list[float]] = defaultdict(list)
    by_room_type: defaultdict[str, list[float]] = defaultdict(list)
    for position, index in enumerate(indices):
        row = records[int(index)]
        segment = f"{row['neighbourhood']}|||{row['room_type']}"
        by_segment[segment].append(float(residuals[position]))
        by_room_type[row["room_type"]].append(float(residuals[position]))
    segment_quantiles = {
        segment: {
            "count": len(values),
            "quantile": finite_sample_quantile(np.asarray(values)),
        }
        for segment, values in by_segment.items()
        if len(values) >= MIN_SEGMENT_CALIBRATION
    }
    room_type_quantiles = {
        room_type: {
            "count": len(values),
            "quantile": finite_sample_quantile(np.asarray(values)),
        }
        for room_type, values in by_room_type.items()
        if len(values) >= MIN_SEGMENT_CALIBRATION
    }
    return global_quantile, segment_quantiles, room_type_quantiles


def quantiles_for_records(
    records: list[dict[str, str]],
    indices: np.ndarray,
    global_quantile: float,
    segment_quantiles: dict[str, dict[str, float | int]],
    room_type_quantiles: dict[str, dict[str, float | int]],
) -> np.ndarray:
    values = []
    for index in indices:
        row = records[int(index)]
        segment = f"{row['neighbourhood']}|||{row['room_type']}"
        if segment in segment_quantiles:
            quantile = segment_quantiles[segment]["quantile"]
        elif row["room_type"] in room_type_quantiles:
            quantile = room_type_quantiles[row["room_type"]]["quantile"]
        else:
            quantile = global_quantile
        values.append(float(quantile))
    return np.asarray(values, dtype=float)


def refusal_reasons(
    row: dict[str, Any],
    predicted: float,
    lower: float,
    upper: float,
    comparable_count: int,
    category_values: dict[str, list[str]],
    price_range: list[float],
    snapshot_age_days: int,
) -> list[str]:
    reasons = []
    for feature in CRITICAL_FEATURES:
        value = row.get(feature)
        if value is None or value == "" or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            reasons.append(f"missing_critical:{feature}")
    for feature in ("neighbourhood", "property_type", "room_type"):
        if str(row.get(feature, "")) not in category_values[feature]:
            reasons.append(f"unseen_category:{feature}")
    if comparable_count < MIN_COMPARABLES:
        reasons.append("insufficient_comparables")
    relative_width = (upper - lower) / max(predicted, 1.0)
    if relative_width > MAX_RELATIVE_INTERVAL_WIDTH:
        reasons.append("prediction_interval_too_wide")
    if not (price_range[0] <= predicted <= price_range[1]):
        reasons.append("prediction_outside_supported_price_range")
    if snapshot_age_days > MAX_SNAPSHOT_AGE_DAYS:
        reasons.append("snapshot_too_old")
    lead_days = finite_float(row.get("quote_lead_days"))
    if math.isfinite(lead_days) and not (0 <= lead_days <= 365):
        reasons.append("outside_supported_quote_horizon")
    return sorted(set(reasons))


def segment_coverage(
    records: list[dict[str, str]],
    indices: np.ndarray,
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    field: str,
) -> list[dict[str, Any]]:
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(indices):
        positions[records[int(index)][field]].append(position)
    results = []
    for value, segment_positions in positions.items():
        if len(segment_positions) < 30:
            continue
        selected = np.asarray(segment_positions, dtype=int)
        covered = (y[indices][selected] >= lower[selected]) & (
            y[indices][selected] <= upper[selected]
        )
        results.append(
            {
                field: value,
                "rows": len(selected),
                "coverage": float(np.mean(covered)),
                "average_interval_width": float(
                    np.mean(upper[selected] - lower[selected])
                ),
            }
        )
    return sorted(results, key=lambda row: (-row["rows"], str(row[field])))


def train_mvp(
    silver_path: Path, report_path: Path, artifact_path: Path
) -> dict[str, Any]:
    records, X, y, groups = load_silver(silver_path)
    dev_indices, test_indices = group_split(X, groups, 0.20, 42)
    train_relative, calibration_relative = group_split(
        X[dev_indices], groups[dev_indices], 0.20, 43
    )
    train_indices = dev_indices[train_relative]
    calibration_indices = dev_indices[calibration_relative]

    pipeline = build_pipeline()
    pipeline.fit(X[train_indices], np.log1p(y[train_indices]))
    calibration_log = pipeline.predict(X[calibration_indices])
    calibration_residuals = np.abs(
        np.log1p(y[calibration_indices]) - calibration_log
    )
    global_quantile, segment_quantiles, room_type_quantiles = calibration_quantiles(
        records, calibration_indices, calibration_residuals
    )

    test_log = pipeline.predict(X[test_indices])
    test_prediction = np.maximum(0.0, np.expm1(test_log))
    test_quantiles = quantiles_for_records(
        records,
        test_indices,
        global_quantile,
        segment_quantiles,
        room_type_quantiles,
    )
    lower, upper = interval_bounds(test_log, test_quantiles)

    baseline = fit_market_baseline(records, y, train_indices)
    baseline_values = []
    comparable_counts = []
    baseline_levels: Counter[str] = Counter()
    for index in test_indices:
        prediction, _, level = market_prediction(baseline, records[int(index)])
        baseline_values.append(prediction)
        exact = baseline["exact"].get(
            (
                records[int(index)]["neighbourhood"],
                records[int(index)]["room_type"],
            )
        )
        comparable_counts.append(int(exact["count"]) if exact else 0)
        baseline_levels[level] += 1
    baseline_prediction = np.asarray(baseline_values, dtype=float)

    category_values = category_inventory(records, train_indices)
    price_range = supported_price_range(y, train_indices)
    latest_as_of = max(
        date.fromisoformat(records[int(index)]["as_of_date"]) for index in train_indices
    )
    snapshot_age_days = (date.today() - latest_as_of).days
    test_refusals = []
    for position, index in enumerate(test_indices):
        test_refusals.append(
            refusal_reasons(
                records[int(index)],
                float(test_prediction[position]),
                float(lower[position]),
                float(upper[position]),
                int(comparable_counts[position]),
                category_values,
                price_range,
                snapshot_age_days,
            )
        )
    accepted = np.asarray([not reasons for reasons in test_refusals])
    covered = (y[test_indices] >= lower) & (y[test_indices] <= upper)
    train_p01 = float(np.quantile(y[train_indices], 0.01))
    train_p99 = float(np.quantile(y[train_indices], 0.99))
    core_market = (y[test_indices] >= train_p01) & (y[test_indices] <= train_p99)
    temporal_compatibility = (
        json.loads(DEFAULT_TEMPORAL_COMPATIBILITY.read_text(encoding="utf-8"))
        if DEFAULT_TEMPORAL_COMPATIBILITY.exists()
        else None
    )
    temporal_evaluation = (
        json.loads(DEFAULT_TEMPORAL_EVALUATION.read_text(encoding="utf-8"))
        if DEFAULT_TEMPORAL_EVALUATION.exists()
        else None
    )
    temporal_status = (
        temporal_compatibility["target_compatibility"]["status"]
        if temporal_compatibility
        else "NOT_ASSESSED"
    )
    temporal_evaluation_matches_training = bool(
        temporal_evaluation
        and temporal_evaluation.get("snapshots", {})
        .get("older", {})
        .get("label")
        == records[0]["snapshot_label"]
        and temporal_evaluation.get("snapshots", {})
        .get("older", {})
        .get("source_sha256")
        == sha256_file(silver_path)
    )
    temporal_evaluation_passed = bool(
        temporal_evaluation_matches_training
        and temporal_evaluation.get("evidence_gate", {}).get("passed")
    )
    deployment_authority = (
        temporal_evaluation["authority"]["current_quote_model"]
        if temporal_evaluation_passed
        else "research_only"
    )

    report = {
        "report_version": 2,
        "generated_at_utc": utc_now(),
        "target": {
            "name": TARGET,
            "definition": "Public quoted price per night in AUD",
            "not_claimed": [
                "realised booking price",
                "optimal price",
                "verified occupancy",
                "realised revenue",
            ],
        },
        "temporal_validation": {
            "status": temporal_status,
            "deployment_authority": deployment_authority,
            "compatibility_report": (
                str(DEFAULT_TEMPORAL_COMPATIBILITY.relative_to(ROOT))
                if temporal_compatibility
                else None
            ),
            "blockers": (
                temporal_compatibility["target_compatibility"]["blockers"]
                if temporal_compatibility
                else ["snapshot_target_compatibility_not_assessed"]
            ),
            "out_of_time_evaluation_report": (
                str(DEFAULT_TEMPORAL_EVALUATION.relative_to(ROOT))
                if temporal_evaluation
                else None
            ),
            "out_of_time_evidence_gate_passed": temporal_evaluation_passed,
            "out_of_time_evidence_matches_training_snapshot": (
                temporal_evaluation_matches_training
            ),
        },
        "evaluation_protocol": {
            "type": "single-snapshot host-disjoint train/calibration/test",
            "random_seeds": {"test": 42, "calibration": 43},
            "temporal_backtest": False,
            "limitation": (
                "A later independent snapshot is required for temporal validation."
            ),
            "rows": {
                "total_eligible": len(y),
                "train": len(train_indices),
                "calibration": len(calibration_indices),
                "test": len(test_indices),
            },
            "unique_hosts": {
                "train": len(set(groups[train_indices])),
                "calibration": len(set(groups[calibration_indices])),
                "test": len(set(groups[test_indices])),
            },
            "host_overlap": {
                "train_calibration": len(
                    set(groups[train_indices]) & set(groups[calibration_indices])
                ),
                "train_test": len(
                    set(groups[train_indices]) & set(groups[test_indices])
                ),
                "calibration_test": len(
                    set(groups[calibration_indices]) & set(groups[test_indices])
                ),
            },
        },
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "geographic_reference_manifest": reference_manifest(),
            "excluded_proxies": [
                "availability_30/60/90/365",
                "review counts and review velocity",
            ],
            "excluded_identifiers": ["listing_id", "host_id"],
        },
        "model": {
            "algorithm": "HistGradientBoostingRegressor on log1p target",
            "selection": "predeclared; no test-set tuning",
            "test_metrics_all": price_metrics(y[test_indices], test_prediction),
            "test_metrics_core_market_p01_p99": price_metrics(
                y[test_indices][core_market], test_prediction[core_market]
            ),
        },
        "market_median_baseline": {
            "hierarchy": [
                "neighbourhood + room_type",
                "neighbourhood",
                "room_type",
                "global",
            ],
            "test_metrics_all": price_metrics(y[test_indices], baseline_prediction),
            "test_metrics_core_market_p01_p99": price_metrics(
                y[test_indices][core_market], baseline_prediction[core_market]
            ),
            "fallback_counts": dict(sorted(baseline_levels.items())),
        },
        "comparison": {
            "relative_mae_improvement_vs_market_median": float(
                1
                - price_metrics(y[test_indices], test_prediction)["mae"]
                / price_metrics(y[test_indices], baseline_prediction)["mae"]
            )
        },
        "conformal_interval": {
            "target_coverage": 1 - ALPHA,
            "method": "split conformal absolute residuals in log1p price space",
            "global_log_residual_quantile": global_quantile,
            "calibrated_segments": len(segment_quantiles),
            "calibrated_room_types": len(room_type_quantiles),
            "test_coverage_all": float(np.mean(covered)),
            "test_average_width_all": float(np.mean(upper - lower)),
            "test_coverage_accepted": float(np.mean(covered[accepted]))
            if np.any(accepted)
            else None,
            "test_average_width_accepted": float(
                np.mean((upper - lower)[accepted])
            )
            if np.any(accepted)
            else None,
            "room_type_coverage": segment_coverage(
                records, test_indices, y, lower, upper, "room_type"
            ),
        },
        "evidence_gate": {
            "thresholds": {
                "minimum_comparables": MIN_COMPARABLES,
                "maximum_relative_interval_width": MAX_RELATIVE_INTERVAL_WIDTH,
                "maximum_snapshot_age_days": MAX_SNAPSHOT_AGE_DAYS,
                "supported_prediction_price_range": price_range,
            },
            "snapshot_age_days_at_evaluation": snapshot_age_days,
            "accepted_rows": int(np.sum(accepted)),
            "refused_rows": int(np.sum(~accepted)),
            "refusal_rate": float(np.mean(~accepted)),
            "reason_counts": dict(
                sorted(Counter(reason for reasons in test_refusals for reason in reasons).items())
            ),
            "accepted_test_metrics": price_metrics(
                y[test_indices][accepted], test_prediction[accepted]
            )
            if np.any(accepted)
            else None,
        },
    }

    artifact = {
        "artifact_version": 2,
        "created_at_utc": utc_now(),
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "geographic_reference_manifest": reference_manifest(),
        "global_conformal_quantile": global_quantile,
        "segment_conformal_quantiles": segment_quantiles,
        "room_type_conformal_quantiles": room_type_quantiles,
        "market_baseline": baseline,
        "category_inventory": category_values,
        "supported_price_range": price_range,
        "latest_training_as_of_date": latest_as_of.isoformat(),
        "snapshot_label": records[0]["snapshot_label"],
        "training_silver_sha256": sha256_file(silver_path),
        "target_definition": report["target"]["definition"],
        "temporal_validation_status": temporal_status,
        "deployment_authority": deployment_authority,
        "gate_thresholds": report["evidence_gate"]["thresholds"],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    report["artifact"] = {
        "path": str(artifact_path.relative_to(ROOT)),
        "bytes": artifact_path.stat().st_size,
    }
    write_json_atomic(report_path, report)
    print(
        f"model    MAE {report['model']['test_metrics_all']['mae']:.2f}; "
        f"baseline {report['market_median_baseline']['test_metrics_all']['mae']:.2f}"
    )
    print(
        f"interval coverage {report['conformal_interval']['test_coverage_all']:.3f}; "
        f"refusal {report['evidence_gate']['refusal_rate']:.3f}"
    )
    print(f"artifact {artifact_path}")
    print(f"report   {report_path}")
    return report


def feature_row_from_request(payload: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    as_of = date.fromisoformat(str(payload["as_of_date"]))
    checkin = date.fromisoformat(str(payload["quote_checkin_date"]))
    checkout = date.fromisoformat(str(payload["quote_checkout_date"]))
    if checkin < as_of:
        raise ValueError("quote_checkin_date must not precede as_of_date")
    if checkout <= checkin:
        raise ValueError("quote_checkout_date must be after quote_checkin_date")
    derived = dict(payload)
    derived.update(
        {
            "quote_lead_days": (checkin - as_of).days,
            "stay_nights": (checkout - checkin).days,
            "checkin_month": checkin.month,
            "checkin_day_of_week": checkin.weekday(),
            "checkin_is_weekend": int(checkin.weekday() >= 5),
        }
    )
    derived.update(
        geographic_features(derived.get("latitude"), derived.get("longitude"))
    )
    values = [finite_float(derived.get(name)) for name in NUMERIC_FEATURES] + [
        str(derived.get(name, "") or "").strip() for name in CATEGORICAL_FEATURES
    ]
    return derived, np.asarray([values], dtype=object)


def predict_request(artifact: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    row, X = feature_row_from_request(payload)
    predicted_log = float(artifact["pipeline"].predict(X)[0])
    predicted = max(0.0, math.expm1(predicted_log))
    segment = f"{row.get('neighbourhood', '')}|||{row.get('room_type', '')}"
    if segment in artifact["segment_conformal_quantiles"]:
        quantile = float(
            artifact["segment_conformal_quantiles"][segment]["quantile"]
        )
    elif str(row.get("room_type", "")) in artifact["room_type_conformal_quantiles"]:
        quantile = float(
            artifact["room_type_conformal_quantiles"][
                str(row.get("room_type", ""))
            ]["quantile"]
        )
    else:
        quantile = float(artifact["global_conformal_quantile"])
    lower_array, upper_array = interval_bounds(
        np.asarray([predicted_log]), np.asarray([quantile])
    )
    lower = float(lower_array[0])
    upper = float(upper_array[0])
    _, _, comparable_level = market_prediction(
        artifact["market_baseline"],
        {
            "neighbourhood": str(row.get("neighbourhood", "")),
            "room_type": str(row.get("room_type", "")),
        },
    )
    exact_market = artifact["market_baseline"]["exact"].get(
        (
            str(row.get("neighbourhood", "")),
            str(row.get("room_type", "")),
        )
    )
    comparable_count = int(exact_market["count"]) if exact_market else 0
    snapshot_age_days = (
        date.today() - date.fromisoformat(artifact["latest_training_as_of_date"])
    ).days
    reasons = refusal_reasons(
        row,
        predicted,
        lower,
        upper,
        comparable_count,
        artifact["category_inventory"],
        artifact["supported_price_range"],
        snapshot_age_days,
    )
    relative_width = (upper - lower) / max(predicted, 1.0)
    evidence_level = (
        "high"
        if comparable_count >= 100 and relative_width <= 1.0
        else ("moderate" if comparable_count >= MIN_COMPARABLES else "low")
    )
    result: dict[str, Any] = {
        "status": "refused" if reasons else "ok",
        "currency": "AUD",
        "target_definition": artifact["target_definition"],
        "deployment_authority": artifact.get(
            "deployment_authority", "research_only"
        ),
        "temporal_validation_status": artifact.get(
            "temporal_validation_status", "NOT_ASSESSED"
        ),
        "snapshot_label": artifact["snapshot_label"],
        "latest_training_as_of_date": artifact["latest_training_as_of_date"],
        "comparable_count": comparable_count,
        "comparable_level": comparable_level,
        "evidence_level": evidence_level,
        "refusal_reasons": reasons,
        "authority_warning": (
            "Research use only: temporal price validation is blocked by "
            "incompatible historical source labels."
            if artifact.get("deployment_authority", "research_only")
            == "research_only"
            else None
        ),
        "disclaimer": (
            "This is a public quoted-price estimate, not a realised booking "
            "price, optimal price, occupancy estimate, or revenue guarantee."
        ),
    }
    if reasons:
        result.update(
            {
                "estimated_price": None,
                "prediction_interval": None,
            }
        )
    else:
        result.update(
            {
                "estimated_price": round(predicted, 2),
                "prediction_interval": [round(lower, 2), round(upper, 2)],
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    train.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    train.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_mvp(args.silver, args.report, args.artifact)
        return
    artifact = joblib.load(args.artifact)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = predict_request(artifact, payload)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
