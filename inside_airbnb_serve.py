"""FastAPI service for the Airbnb Sydney quote-price prediction model.

Start the server:
    python inside_airbnb_serve.py
    python inside_airbnb_serve.py --port 8080 --artifact artifacts/inside_airbnb_quote_mvp.joblib

API endpoints:
    POST /predict          — Predict price for a single listing
    GET  /health           — Liveness check
    GET  /model-info       — Model metadata and limitations
    GET  /docs             — Interactive OpenAPI documentation
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except ImportError:
    print(
        "fastapi and uvicorn are required. Install with:\n"
        "  pip install fastapi uvicorn[standard]"
    )
    sys.exit(1)

from inside_airbnb_explain import explain_prediction
from inside_airbnb_phase0 import ROOT
from inside_airbnb_quote_model import (
    DEFAULT_ARTIFACT,
    feature_row_from_request,
    predict_request,
)

# ---------------------------------------------------------------------------
# Pydantic models (input / output schemas)
# ---------------------------------------------------------------------------


class QuoteRequest(BaseModel):
    as_of_date: str = Field(..., description="Date the quote was requested (ISO 8601)")
    quote_checkin_date: str = Field(..., description="Check-in date (ISO 8601)")
    quote_checkout_date: str = Field(..., description="Check-out date (ISO 8601)")
    neighbourhood: str = Field(..., description="Neighbourhood name (e.g. Manly)")
    property_type: str = Field(..., description="Property type (e.g. Entire rental unit)")
    room_type: str = Field(..., description="Room type (e.g. Entire home/apt)")
    latitude: float = Field(..., ge=-90, le=90, description="Listing latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Listing longitude")
    accommodates: int = Field(..., ge=1, description="Maximum guest count")
    bathrooms: float = Field(0.0, ge=0, description="Number of bathrooms")
    bedrooms: int = Field(0, ge=0, description="Number of bedrooms")
    beds: int = Field(0, ge=0, description="Number of beds")
    amenities_count: int = Field(0, ge=0, description="Total amenity count")
    minimum_nights: int = Field(1, ge=1, description="Minimum stay (nights)")
    maximum_nights: int = Field(365, ge=1, description="Maximum stay (nights)")
    calculated_host_listings_count: int = Field(1, ge=1, description="Host's listing count")
    host_is_superhost: str = Field("f", description="t or f")


class PredictionInterval(BaseModel):
    lower: float
    upper: float


class Explanation(BaseModel):
    mode: str
    text: str


class PredictionResponse(BaseModel):
    status: str
    estimated_price: float | None = None
    prediction_interval: PredictionInterval | None = None
    currency: str = "AUD"
    comparable_count: int
    evidence_level: str
    refusal_reasons: list[str] = []
    authority_warning: str | None = None
    snapshot_staleness_warning: bool = False
    explanation: Explanation | None = None


class HealthResponse(BaseModel):
    status: str
    snapshot_label: str
    deployment_authority: str


class ModelInfoResponse(BaseModel):
    snapshot_label: str
    deployment_authority: str
    temporal_validation_status: str
    target_definition: str
    latest_training_as_of_date: str = ""
    features: dict[str, list[str]]
    limitations: list[str]
    artifact_path: str
    python: str = ""
    sklearn: str = ""
    lightgbm: str = ""


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_ARTIFACT: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ARTIFACT
    artifact_path = Path(app.state.artifact_path)
    if not artifact_path.exists():
        print(f"Artifact not found: {artifact_path}")
        print("Train the model first: python inside_airbnb_quote_model.py train")
        _ARTIFACT = {}
    else:
        _ARTIFACT = joblib.load(artifact_path)
        print(f"Loaded artifact: {artifact_path}")
        print(f"  snapshot: {_ARTIFACT.get('snapshot_label', 'unknown')}")
        print(f"  authority: {_ARTIFACT.get('deployment_authority', 'unknown')}")
    yield


app = FastAPI(
    title="Airbnb Sydney Quote-Price Predictor",
    description=(
        "Estimate a public quoted nightly price in AUD for a Sydney Airbnb "
        "listing. Research prototype — not a pricing recommendation."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def _require_artifact() -> dict[str, Any]:
    if not _ARTIFACT:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not loaded. Train first: python inside_airbnb_quote_model.py train",
        )
    return _ARTIFACT


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check. Returns 200 when the model is loaded."""
    if not _ARTIFACT:
        raise HTTPException(status_code=503, detail="Artifact not loaded")
    return HealthResponse(
        status="healthy",
        snapshot_label=_ARTIFACT["snapshot_label"],
        deployment_authority=_ARTIFACT.get("deployment_authority", "unknown"),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return metadata about the loaded model."""
    artifact = _require_artifact()
    import platform

    sklearn_ver = "unknown"
    try:
        import sklearn
        sklearn_ver = sklearn.__version__
    except Exception:
        pass
    lgb_ver = "not installed"
    try:
        import lightgbm
        lgb_ver = lightgbm.__version__
    except Exception:
        pass

    return ModelInfoResponse(
        snapshot_label=artifact["snapshot_label"],
        deployment_authority=artifact.get("deployment_authority", "unknown"),
        temporal_validation_status=artifact.get(
            "temporal_validation_status", "NOT_ASSESSED"
        ),
        target_definition=artifact.get(
            "target_definition", "Public quoted price per night in AUD"
        ),
        latest_training_as_of_date=artifact.get("latest_training_as_of_date", ""),
        features={
            "numeric": artifact.get("numeric_features", []),
            "categorical": artifact.get("categorical_features", []),
        },
        limitations=[
            "Research use only — not a pricing recommendation.",
            "Predicts quoted (listed) price, not realised booking revenue.",
            "Upper-tail luxury listings are systematically underestimated.",
            "Geographic distances are straight-line, not route distance.",
            "Sydney market only. Geographic transfer is untested.",
        ],
        artifact_path=artifact.get("training_silver_sha256", ""),
        python=platform.python_version(),
        sklearn=sklearn_ver,
        lightgbm=lgb_ver,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: QuoteRequest, explain: bool = False) -> PredictionResponse:
    """Predict the quoted nightly price for a listing.

    Set `?explain=true` to include a human-readable explanation.
    """
    artifact = _require_artifact()
    payload = request.model_dump()
    row, _ = feature_row_from_request(payload)
    result = predict_request(artifact, payload)

    response = PredictionResponse(
        status=result["status"],
        estimated_price=result.get("estimated_price"),
        prediction_interval=(
            PredictionInterval(
                lower=result["prediction_interval"][0],
                upper=result["prediction_interval"][1],
            )
            if result.get("prediction_interval")
            else None
        ),
        comparable_count=result.get("comparable_count", 0),
        evidence_level=result.get("evidence_level", "low"),
        refusal_reasons=result.get("refusal_reasons", []),
        authority_warning=result.get("authority_warning"),
        snapshot_staleness_warning=result.get("snapshot_staleness_warning", False),
    )

    if explain:
        explanation = explain_prediction(result, row, artifact, prefer_llm=False)
        response.explanation = Explanation(
            mode=explanation["mode"], text=explanation["text"]
        )

    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the interactive dark-theme dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import uvicorn

    app.state.artifact_path = str(args.artifact)

    print(f"Starting Airbnb Sydney Quote-Price Predictor")
    print(f"  http://{args.host}:{args.port}")
    print(f"  docs: http://{args.host}:{args.port}/docs")
    print(f"  artifact: {args.artifact}")

    uvicorn.run(
        "inside_airbnb_serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
