"""FastAPI service for the Airbnb Sydney quote-price prediction model.

Start the server:
    python inside_airbnb_serve.py
    python inside_airbnb_serve.py --port 8080
    MODEL_ARTIFACT_PATH=artifacts/inside_airbnb_quote_mvp.joblib uvicorn inside_airbnb_serve:app

Artifact contract:
    The model artifact is loaded at startup from the first available source:
    1. MODEL_ARTIFACT_PATH environment variable
    2. --artifact CLI argument (sets the env var before uvicorn starts)
    3. Default: artifacts/inside_airbnb_quote_mvp.joblib

    Missing artifact → health returns 503, model-info returns 503,
    predict returns 503. Container is not "ready" until artifact loads.

API endpoints:
    POST /predict          — Predict price for a single listing
    GET  /health           — Liveness + readiness check
    GET  /model-info       — Model metadata and version info
    GET  /dashboard        — Interactive dark-theme dashboard
    GET  /docs             — OpenAPI documentation
"""

from __future__ import annotations

import argparse
import json
import os
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
from inside_airbnb_phase0 import ROOT, sha256_file
from inside_airbnb_quote_model import (
    DEFAULT_ARTIFACT,
    feature_row_from_request,
    predict_request,
)

ENV_ARTIFACT_PATH = "MODEL_ARTIFACT_PATH"

# ---------------------------------------------------------------------------
# Pydantic schemas — shared contract between API and UI
# ---------------------------------------------------------------------------


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
    comparable_level: str = ""
    evidence_level: str = "low"
    evidence_tier: str = ""
    evidence_tier_label: str = ""
    evidence_reasons: list[str] = []
    evidence_recommendation: str = ""
    snapshot_label: str = ""
    deployment_authority: str = "research_only"
    temporal_validation_status: str = "NOT_ASSESSED"
    snapshot_age_days: int = 0
    snapshot_staleness_warning: bool = False
    authority_warning: str | None = None
    refusal_reasons: list[str] = []
    disclaimer: str = ""
    explanation: Explanation | None = None


class HealthResponse(BaseModel):
    status: str
    artifact_loaded: bool
    snapshot_label: str = ""
    deployment_authority: str = ""


class ModelInfoResponse(BaseModel):
    model_name: str = "airbnb-sydney-quote-predictor"
    model_version: str = "0.2.0"
    model_family: str = ""
    artifact_sha256: str = ""
    artifact_path: str = ""
    training_silver_sha256: str = ""
    snapshot_label: str = ""
    training_as_of_date: str = ""
    deployment_authority: str = "research_only"
    temporal_validation_status: str = "NOT_ASSESSED"
    release_gate_status: dict[str, str] = {}
    features: dict[str, list[str]] = {}
    limitations: list[str] = []
    environment: dict[str, str] = {}


# ---------------------------------------------------------------------------
# App factory — configuration resolved independently of CLI
# ---------------------------------------------------------------------------


def _resolve_artifact_path() -> Path:
    """Resolve artifact path from env var or default. Does NOT check existence."""
    return Path(os.environ.get(ENV_ARTIFACT_PATH, str(DEFAULT_ARTIFACT)))


def _try_load_artifact(path: Path) -> dict[str, Any] | None:
    """Load artifact if it exists, else return None."""
    if not path.exists():
        return None
    return joblib.load(path)


def create_app(artifact_path: Path | None = None) -> FastAPI:
    """Create the FastAPI application with the given artifact path.

    If artifact_path is None, resolves from MODEL_ARTIFACT_PATH env var or default.
    """
    resolved = artifact_path or _resolve_artifact_path()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        artifact = _try_load_artifact(resolved)
        if artifact is None:
            print(f"WARNING: artifact not found at {resolved}")
            print(f"  Set {ENV_ARTIFACT_PATH} env var or train first:")
            print(f"  python inside_airbnb_quote_model.py train")
        else:
            print(f"Loaded artifact: {resolved}")
            print(f"  snapshot: {artifact.get('snapshot_label', 'unknown')}")
            print(f"  authority: {artifact.get('deployment_authority', 'unknown')}")
        app.state.artifact = artifact or {}
        app.state.artifact_path = str(resolved)
        app.state.artifact_sha256 = sha256_file(resolved) if resolved.exists() else ""
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

    # -----------------------------------------------------------------------
    # Dependency
    # -----------------------------------------------------------------------

    def _require_artifact(request: Any = None) -> dict[str, Any]:
        artifact = getattr(request.app.state, "artifact", {}) if request else {}
        if not artifact:
            artifact = getattr(app.state, "artifact", {})
        if not artifact:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model artifact not loaded. "
                    f"Set {ENV_ARTIFACT_PATH} or train: "
                    "python inside_airbnb_quote_model.py train"
                ),
            )
        return artifact

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        artifact = app.state.artifact
        if not artifact:
            return HealthResponse(
                status="not_ready",
                artifact_loaded=False,
                snapshot_label="",
                deployment_authority="",
            )
        return HealthResponse(
            status="healthy",
            artifact_loaded=True,
            snapshot_label=artifact.get("snapshot_label", ""),
            deployment_authority=artifact.get("deployment_authority", "unknown"),
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        artifact = app.state.artifact
        if not artifact:
            raise HTTPException(
                status_code=503,
                detail=f"Artifact not loaded. Set {ENV_ARTIFACT_PATH}.",
            )
        import platform

        env = {"python": platform.python_version()}
        try:
            import sklearn
            env["sklearn"] = sklearn.__version__
        except Exception:
            env["sklearn"] = "unknown"
        try:
            import lightgbm
            env["lightgbm"] = lightgbm.__version__
        except Exception:
            env["lightgbm"] = "not installed"

        return ModelInfoResponse(
            model_family=(
                "lightgbm"
                if "LGBMRegressor" in str(artifact.get("pipeline", ""))
                else "sklearn-histgb"
            ),
            artifact_sha256=app.state.artifact_sha256 or "",
            artifact_path=app.state.artifact_path or "",
            training_silver_sha256=artifact.get("training_silver_sha256", ""),
            snapshot_label=artifact.get("snapshot_label", ""),
            training_as_of_date=artifact.get("latest_training_as_of_date", ""),
            deployment_authority=artifact.get("deployment_authority", "research_only"),
            temporal_validation_status=artifact.get(
                "temporal_validation_status", "NOT_ASSESSED"
            ),
            release_gate_status={
                "research": (
                    "ALLOWED"
                    if artifact.get("deployment_authority") != "research_only"
                    or artifact.get("deployment_authority") is None
                    else "ALLOWED"
                ),
                "production": (
                    "ALLOWED"
                    if artifact.get("deployment_authority") == "temporally_validated"
                    else "BLOCKED"
                ),
            },
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
            environment=env,
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: QuoteRequest, explain: bool = False) -> PredictionResponse:
        artifact = _require_artifact()
        payload = request.model_dump()
        row, _ = feature_row_from_request(payload)
        result = predict_request(artifact, payload)

        interval = None
        if result.get("prediction_interval"):
            interval = PredictionInterval(
                lower=result["prediction_interval"][0],
                upper=result["prediction_interval"][1],
            )

        from inside_airbnb_evidence import assess_evidence

        evidence = assess_evidence(
            estimated_price=result.get("estimated_price") or 0,
            prediction_interval=(
                (result["prediction_interval"][0], result["prediction_interval"][1])
                if result.get("prediction_interval")
                else None
            ),
            comparable_count=result.get("comparable_count", 0),
            snapshot_age_days=result.get("snapshot_age_days", 0),
            deployment_authority=artifact.get("deployment_authority", "research_only"),
            refusal_reasons=result.get("refusal_reasons", []),
            training_price_quantiles={
                "p90": artifact.get("supported_price_range", [50, 800])[1] * 0.85,
            },
        )

        response = PredictionResponse(
            status=result.get("status", "refused"),
            estimated_price=result.get("estimated_price"),
            prediction_interval=interval,
            comparable_count=result.get("comparable_count", 0),
            comparable_level=result.get("comparable_level", ""),
            evidence_level=result.get("evidence_level", "low"),
            evidence_tier=evidence["tier"],
            evidence_tier_label=evidence["tier_label"],
            evidence_reasons=evidence["reasons"],
            evidence_recommendation=evidence["recommendation"],
            snapshot_label=artifact.get("snapshot_label", ""),
            deployment_authority=artifact.get("deployment_authority", "research_only"),
            temporal_validation_status=artifact.get(
                "temporal_validation_status", "NOT_ASSESSED"
            ),
            snapshot_age_days=result.get("snapshot_age_days", 0),
            snapshot_staleness_warning=result.get("snapshot_staleness_warning", False),
            authority_warning=result.get("authority_warning"),
            refusal_reasons=result.get("refusal_reasons", []),
            disclaimer=result.get("disclaimer", ""),
        )

        if explain:
            explanation = explain_prediction(result, row, artifact, prefer_llm=False)
            response.explanation = Explanation(
                mode=explanation["mode"], text=explanation["text"]
            )

        return response

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        dashboard_path = Path(__file__).parent / "dashboard.html"
        if not dashboard_path.exists():
            raise HTTPException(status_code=404, detail="dashboard.html not found")
        return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))

    return app


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------


class QuoteRequest(BaseModel):
    as_of_date: str = Field(..., description="Date the quote was requested (ISO 8601)")
    quote_checkin_date: str = Field(..., description="Check-in date (ISO 8601)")
    quote_checkout_date: str = Field(..., description="Check-out date (ISO 8601)")
    neighbourhood: str = Field(..., description="Neighbourhood name")
    property_type: str = Field(..., description="Property type")
    room_type: str = Field(..., description="Room type")
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


# ---------------------------------------------------------------------------
# App singleton — used by `uvicorn inside_airbnb_serve:app`
# ---------------------------------------------------------------------------

app = create_app()


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
    os.environ.setdefault(ENV_ARTIFACT_PATH, str(args.artifact))

    import uvicorn

    print(f"Starting Airbnb Sydney Quote-Price Predictor")
    print(f"  artifact: {os.environ[ENV_ARTIFACT_PATH]}")
    print(f"  http://{args.host}:{args.port}")
    print(f"  docs:  http://{args.host}:{args.port}/docs")
    print(f"  dashboard: http://{args.host}:{args.port}/dashboard")

    uvicorn.run(
        "inside_airbnb_serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
