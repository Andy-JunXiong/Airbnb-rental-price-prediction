# Airbnb Sydney Rental Price Prediction

Predict nightly Airbnb quote prices in Sydney using governed ML pipelines with
conformal prediction intervals. Built on [Inside Airbnb](https://insideairbnb.com/)
public data snapshots.

## Quickstart

```powershell
# Create environment and install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Optional: install LightGBM for faster training (falls back to sklearn otherwise)
.\.venv\Scripts\python.exe -m pip install lightgbm>=4.0

# Optional: install sentence-transformers for text features
.\.venv\Scripts\python.exe -m pip install sentence-transformers

# Run the full test suite (56 tests)
python -m unittest discover -s tests -v
```

## Primary Pipeline: Inside Airbnb Quote Model

The active research pipeline trains on **quoted listing prices** (not realised
bookings) from the 2026-06-16 Sydney snapshot. All evaluation is host-disjoint.

```powershell
# Download and audit raw data
python inside_airbnb_phase0.py all

# Build privacy-minimised Silver table
python prepare_inside_airbnb_quotes.py

# Train the governed model with conformal intervals
python inside_airbnb_quote_model.py train

# Run error analysis and model card
python inside_airbnb_error_analysis.py

# Predict with explanation
python inside_airbnb_quote_model.py predict \
  --input examples/inside_airbnb_quote_request.json \
  --explain
```

**Current held-out performance** (2026-06-16 snapshot):

| Metric | Model | Market Baseline |
|--------|------:|----------------:|
| MAE | 138.14 AUD | 222.90 AUD |
| Improvement | 38.0% | — |
| 90% interval coverage | ~89.5% | — |

## Key Features

### LightGBM (auto-selected)

The pipeline automatically uses LightGBM when installed, falling back to
sklearn's HistGradientBoostingRegressor. LightGBM provides native categorical
encoding, missing-value handling, and typically better performance.

### Multi-Snapshot Training

When multiple compatible Silver tables are available, the model can train
across time with strict forward-time, host-disjoint splits:

```powershell
python inside_airbnb_multi_snapshot.py
```

Six June 2026 Sydney snapshots are registered in `config/inside_airbnb_snapshots.json`.

### Text Features (NLP)

Text embeddings from listing descriptions, neighbourhood overviews, and host
profiles can be joined onto the Silver table:

```powershell
python prepare_inside_airbnb_text_features.py
python inside_airbnb_text_challenger.py
```

Uses `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) when available, with
a TF-IDF fallback (100 components). Raw text is never stored in the Silver table.

### Prediction Explanations

Every prediction can include a human-readable markdown explanation:

```powershell
python inside_airbnb_quote_model.py predict --input request.json --explain
```

Template mode is always available. LLM mode (via Ollama) produces more natural
language:

```powershell
python inside_airbnb_quote_model.py predict --input request.json --explain-llm
```

### Snapshot Staleness Warning

Training and prediction both warn when the snapshot exceeds 90 days old
(configurable via `SNAPSHOT_STALENESS_WARN_DAYS`).

### Release Governance

```powershell
python run_inside_airbnb_pipeline.py research   # full offline pipeline
python inside_airbnb_release_gate.py --target research --enforce
python inside_airbnb_release_gate.py --target production --enforce
```

Research release: **ALLOWED**. Production release: **BLOCKED** — requires a
compatible forward-time snapshot passing all temporal validation gates.

## Snapshot Management

To rotate to a different snapshot, edit one field in
`config/inside_airbnb_snapshots.json`:

```json
"active_snapshot": { "date": "2026-06-28", "role": "current_training_source" }
```

All modules (`inside_airbnb_quote_model.py`, `prepare_inside_airbnb_quotes.py`,
etc.) read this centrally. No more hunting through six files for hardcoded dates.

## Project Structure

```
├── inside_airbnb_phase0.py        # Download + audit raw snapshots
├── prepare_inside_airbnb_quotes.py # Raw → privacy-minimised Silver CSV
├── inside_airbnb_quote_model.py   # Train/predict with conformal intervals
├── inside_airbnb_multi_snapshot.py # Multi-snapshot temporal training
├── inside_airbnb_eda.py           # Modern EDA with SVG charts
├── inside_airbnb_feature_ablation.py  # Leakage-safe feature ablation
├── inside_airbnb_error_analysis.py    # Held-out error audit + model card
├── inside_airbnb_explain.py       # Template + LLM prediction explanations
├── inside_airbnb_text_features.py # Text embeddings from listing descriptions
├── inside_airbnb_text_challenger.py    # Text-feature development benchmark
├── inside_airbnb_upper_tail_challenger.py   # Upper-tail loss benchmark
├── inside_airbnb_premium_challenger.py      # Premium semantic features
├── inside_airbnb_interval_challenger.py     # Asymmetric conformal intervals
├── inside_airbnb_release_gate.py    # Research/production release enforcement
├── inside_airbnb_temporal_validation.py  # Forward-time evaluation
├── inside_airbnb_snapshot_discovery.py   # Read-only new-snapshot check
├── compare_inside_airbnb_snapshots.py    # Target compatibility assessment
├── inside_airbnb_manifest.py       # Reproducibility manifest
├── run_inside_airbnb_pipeline.py   # One-command orchestrator
├── sydney_geography.py            # Haversine distances (offline, auditable)
├── premium_listing_features.py    # Amenity semantics
├── config/                        # Snapshot registry + config
├── docs/                          # Markdown documentation
├── tests/                         # 56 tests (unittest)
├── reports/                       # JSON evidence artifacts
├── examples/                      # Sample prediction requests
└── predictions/                   # Prediction outputs
```

## Historical Baseline (Legacy)

The repository also contains a reproducible baseline on 383 labelled Northern
Beaches listings. This is preserved for historical comparison but is **not**
the active pipeline:

```powershell
python prepare_features.py
python train_baseline.py --dataset corrected --output reports/baseline_corrected.json
python predict.py
```

Extra Trees: RMSE 68.03, MAE 53.10, R² 0.540 (corrected 35-feature matrix).

## Test Status

```
Ran 56 tests in 0.196s — OK (skipped=1)
```

The single skipped test requires LightGBM or scikit-learn ≥1.5 for an
end-to-end temporal validation integration test.

## Limitations

- **Research-only**: temporal price validation is blocked by incompatible
  historical snapshot labels.
- **Upper-tail underprediction**: the model systematically underestimates
  luxury listings.
- **Single city**: Sydney only. Geographic transfer is untested.
- **No daily calendar prices**: the 2026 snapshots lack the calendar price
  column needed for date-level dynamic pricing.
- **Approximate coordinates**: reference distances are straight-line, not
  route distance or travel time.
