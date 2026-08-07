# Airbnb Sydney Pricing Intelligence

> **Estimate a Sydney Airbnb market quote — with uncertainty you can actually see.**

A governed ML system that predicts public quoted nightly prices for Sydney Airbnb
listings. Built on [Inside Airbnb](https://insideairbnb.com/) public data, with
conformal prediction intervals, evidence-tier assessment, and containerised serving.

[![CI](https://github.com/Andy-JunXiong/Airbnb-rental-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Andy-JunXiong/Airbnb-rental-price-prediction/actions/workflows/ci.yml)

---

## Why This Project Matters

Most ML price-prediction projects return a single number. This one returns:

| Output | Example |
|--------|---------|
| **Market estimate** | AUD 284 / night |
| **Prediction interval** | AUD 215–368 (90% confidence) |
| **Evidence tier** | MEDIUM — Use as a reference point |
| **Comparable support** | 47 similar listings in Manly |
| **Market position** | 16% above neighbourhood median |
| **Explanation** | Human-readable markdown (template or LLM) |
| **Deployment authority** | research_only (temporal validation pending) |

The core principle: **a model returning a prediction is not the same as having enough evidence to trust that prediction operationally.**

---

## Quickstart

```powershell
# Install
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# For serving (API + UI + Docker):
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn[standard] streamlit

# Run tests (78 tests)
python -m unittest discover -s tests -v
```

### One-command demo

```powershell
# Terminal 1: API server
python inside_airbnb_serve.py
# → http://localhost:8000/dashboard
# → http://localhost:8000/docs

# Terminal 2: Interactive UI
streamlit run inside_airbnb_ui.py
# → http://localhost:8501
```

### Docker

```bash
docker build -t airbnb-predictor .
docker run -d -p 8000:8000 \
  -v $(pwd)/tests/fixtures:/models \
  -e MODEL_ARTIFACT_PATH=/models/minimal_artifact.joblib \
  airbnb-predictor
curl http://localhost:8000/health
```

---

## System Overview

```
Inside Airbnb snapshots
        ↓
Phase 0 source audit (integrity + schema check)
        ↓
Privacy-minimised Silver table (no raw text stored)
        ↓
Feature pipeline (geography + NLP embeddings)
        ↓
LightGBM + split conformal calibration
        ↓
Host-disjoint evaluation + challenger benchmarks
        ↓
Model card + reproducibility manifest
        ↓
Research / Production release gates
        ↓
Versioned artifact (joblib)
        ↓
Containerised FastAPI
        ↓
Evidence-tier policy (HIGH / MEDIUM / LOW / REFUSE)
        ↓
Runtime monitoring (13 signals, 3 categories)
        ↓
Future compatible snapshot → forward-time validation → human release decision
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict?explain=true` | Predict price with evidence tier + explanation |
| `GET` | `/health` | Liveness + artifact readiness |
| `GET` | `/model-info` | Model metadata, versions, release gates |
| `GET` | `/monitoring` | Runtime monitoring report (13 signals) |
| `GET` | `/dashboard` | Dark-theme interactive dashboard |
| `GET` | `/docs` | OpenAPI Swagger UI |

---

## Key Capabilities

### Uncertainty-Aware Predictions

Every prediction returns an **evidence tier** computed from verifiable signals —
not a black-box confidence score:

| Tier | Condition |
|------|-----------|
| **HIGH** | ≥100 comparables, narrow interval, fresh snapshot, production authority |
| **MEDIUM** | ≥20 comparables, moderate interval, or research-only authority |
| **LOW** | <20 comparables, wide interval, old snapshot, or upper-tail listing |
| **REFUSE** | Missing critical features, unseen categories, snapshot >120 days |

### NLP Text Features

Sentence-transformer embeddings (384-dim) from listing descriptions,
neighbourhood overviews, and host profiles. Falls back to TF-IDF (100-dim).
Raw text never enters the Silver table.

### Multi-Snapshot Training

Six June 2026 Sydney snapshots registered. Temporal training with
host-disjoint forward-time splits when multiple compatible Silver tables exist.

### Scenario Comparison

```powershell
python inside_airbnb_scenarios.py --input examples/request.json
# Base: 314 AUD (MEDIUM)
#   +1 bedroom: 342 AUD (+28)
#   Manly location: 290 AUD (-24)
#   Superhost: 330 AUD (+16)
```

Model-based sensitivity — explicitly NOT causal uplift.

### LLM Explanations

```powershell
python inside_airbnb_quote_model.py predict --input request.json --explain-llm
```

Template mode always available. LLM mode via local Ollama for natural language.

### Post-Deployment Monitoring

`GET /monitoring` returns 13 signals across input, prediction, and quality
categories. Signals requiring live traffic or outcome labels are explicitly
marked `not_available`.

### Temporal Validation Gates

```
temporal_compatibility  → schema-level: do snapshots share a price target?
temporal_evaluation     → forward-time performance (requires newer snapshot)
deployment_authority    → research_only until gates pass
```

Production release remains BLOCKED until compatible forward-time evidence exists.

---

## Project Structure

```
├── inside_airbnb_phase0.py          # Download + audit raw snapshots
├── prepare_inside_airbnb_quotes.py  # Raw → privacy-minimised Silver
├── inside_airbnb_quote_model.py     # Train/predict (LightGBM + conformal)
├── inside_airbnb_evidence.py        # Evidence-tier policy (HIGH/MEDIUM/LOW/REFUSE)
├── inside_airbnb_explain.py         # Template + LLM explanations
├── inside_airbnb_scenarios.py       # What-if scenario comparison
├── inside_airbnb_monitoring.py      # Runtime monitoring contract (13 signals)
├── inside_airbnb_serve.py           # FastAPI (5 endpoints)
├── inside_airbnb_ui.py              # Streamlit interactive demo
├── dashboard.html                   # Dark-theme dashboard (Tailwind CSS)
├── Dockerfile                       # Containerised serving
│
├── inside_airbnb_multi_snapshot.py  # Multi-snapshot temporal training
├── inside_airbnb_text_features.py   # NLP embeddings (sentence-transformers)
│
├── inside_airbnb_eda.py             # Modern EDA (SVG charts)
├── inside_airbnb_feature_ablation.py
├── inside_airbnb_error_analysis.py  # Model card + diagnostic flags
│
├── inside_airbnb_release_gate.py    # Research/production gate enforcement
├── inside_airbnb_temporal_validation.py
├── compare_inside_airbnb_snapshots.py
├── inside_airbnb_snapshot_discovery.py
├── inside_airbnb_manifest.py        # Reproducibility manifest
├── run_inside_airbnb_pipeline.py    # CI/research/refresh orchestrator
│
├── sydney_geography.py              # Haversine distances (offline)
├── premium_listing_features.py      # Amenity semantics
│
├── config/                          # Snapshot registry
├── tests/                           # 78 tests (unittest)
│   └── fixtures/                    # CI test artifact (39KB)
├── legacy/2019/                     # Historical 2019 baseline (preserved)
├── src/airbnb_pricing/              # Package skeleton
├── docs/                            # Markdown documentation
├── reports/                         # JSON evidence artifacts
└── .github/workflows/               # CI + Docker smoke test
```

---

## Test Status

```
Ran 78 tests — OK (64 pass, 14 skipped locally)
CI: 64+14 pass (fastapi/httpx installed → serving tests execute)
```

14 skipped locally: fastapi not installed. In CI, all 78 execute.

---

## Current Performance

| Metric | Value |
|--------|-------|
| Held-out MAE | 138.14 AUD |
| Market baseline MAE | 222.90 AUD |
| Improvement | 38.0% |
| 90% interval coverage | ~89.5% |
| Evidence gate refusal rate | ~2.0% |

---

## Limitations

- **Research authority**: temporal price validation is blocked by incompatible
  historical snapshot labels. Production release requires forward-time evidence.
- **Upper-tail weakness**: luxury listings are systematically underestimated.
  Evidence tier auto-downgrades premium predictions.
- **Sydney only**: geographic transfer is untested.
- **No daily calendar prices**: the 2026 snapshots lack the calendar price column.
- **Observational model**: scenario comparison is model sensitivity, not causal uplift.
- **Not a pricing recommendation**: predicts quoted (listed) price, not realised
  booking revenue, optimal price, or occupancy.
