"""Generate a minimal CI test fixture artifact."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

rng = np.random.RandomState(42)
n = 100
X_num = rng.randn(n, 18).astype(float)
X_cat = rng.randint(0, 3, (n, 4)).astype(str)
y = 100 + 50 * X_num[:, 0] + rng.randn(n) * 20

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), list(range(18))),
    ("cat", OrdinalEncoder(), list(range(18, 22))),
])
model = ExtraTreesRegressor(n_estimators=10, random_state=42)
pipeline = Pipeline([("pre", preprocessor), ("mdl", model)])
X = np.hstack([X_num, X_cat.astype(object)])
pipeline.fit(X, np.log1p(np.maximum(y, 1)))

artifact = {
    "artifact_version": 2,
    "pipeline": pipeline,
    "numeric_features": [
        "quote_lead_days", "stay_nights", "checkin_month", "checkin_day_of_week",
        "checkin_is_weekend", "latitude", "longitude", "distance_to_sydney_cbd_km",
        "distance_to_sydney_airport_km", "distance_to_nearest_reference_beach_km",
        "distance_to_nearest_major_hub_km", "accommodates", "bathrooms", "bedrooms",
        "beds", "amenities_count", "minimum_nights", "maximum_nights",
        "calculated_host_listings_count",
    ],
    "categorical_features": ["neighbourhood", "property_type", "room_type", "host_is_superhost"],
    "global_conformal_quantile": 0.35,
    "segment_conformal_quantiles": {"Sydney|||Entire home/apt": {"count": 50, "quantile": 0.30}},
    "room_type_conformal_quantiles": {"Entire home/apt": {"count": 60, "quantile": 0.32}},
    "market_baseline": {
        "global": 150.0,
        "exact": {("Sydney", "Entire home/apt"): {"median": 200.0, "count": 50}},
        "neighbourhood": {},
        "room": {},
    },
    "category_inventory": {
        "neighbourhood": ["Sydney", "Manly", "Bondi"],
        "property_type": ["Entire rental unit", "Entire condominium"],
        "room_type": ["Entire home/apt", "Private room"],
        "host_is_superhost": ["t", "f"],
    },
    "supported_price_range": [50.0, 800.0],
    "training_price_quantiles": {"p50": 150.0, "p90": 400.0, "p95": 550.0, "p99": 700.0},
    "latest_training_as_of_date": "2026-06-29",
    "snapshot_label": "ci-fixture",
    "training_silver_sha256": "ci-fixture-sha256",
    "target_definition": "Public quoted price per night in AUD (CI fixture)",
    "temporal_validation_status": "NOT_ASSESSED",
    "deployment_authority": "research_only",
    "gate_thresholds": {"minimum_comparables": 20},
}

dest = Path(__file__).resolve().parent / "fixtures"
dest.mkdir(parents=True, exist_ok=True)
out = dest / "minimal_artifact.joblib"
joblib.dump(artifact, out, compress=3)
print(f"Fixture created: {out}")
print(f"Size: {out.stat().st_size:,} bytes")
