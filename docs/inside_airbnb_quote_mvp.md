# Evidence-aware Sydney quote-price MVP

## Outcome

The MVP estimates a **public quoted nightly price in AUD** for a Sydney
listing and explicit check-in/check-out context. It returns a prediction
interval, comparable evidence, an evidence level, and either an accepted result
or structured refusal reasons.

It does not estimate realised booking price, optimal price, verified occupancy,
or realised revenue.

## Leakage-safe dataset

`prepare_inside_airbnb_quotes.py` creates a privacy-minimised Silver table from
the pinned Inside Airbnb listings file.

- All 20,573 listings are retained for lineage.
- 17,784 rows have complete, positive AUD quote labels and valid quote dates.
- Missing and partial quotes remain present but are marked ineligible.
- `last_scraped` is the row-specific `as_of_date`.
- Quote lead time and stay length are calculated only from information
  available in the quote record.
- Four versioned straight-line geographic distances are calculated locally
  from approximate coordinates. No per-listing reverse-geocoding API is used.
- Direct names, descriptions, profile URLs, and review text are excluded.
- `host_id` is used only to prevent host overlap across evaluation splits.
- Availability and review-count columns are explicitly named `proxy_*` and
  excluded from the primary model.

The generated Silver table and binary model artifact are reproducible outputs
and remain outside Git.

## Evaluation protocol

The 17,784 eligible listings are separated by `host_id`:

| Split | Rows | Unique hosts |
| --- | ---: | ---: |
| Train | 11,375 | 5,317 |
| Conformal calibration | 2,708 | 1,330 |
| Test | 3,701 | 1,662 |

There is zero host overlap between train, calibration, and test. The model and
its configuration are predeclared; the test set is not used for model
selection.

This is a single-snapshot host-disjoint evaluation, not a temporal backtest.
The available 2025-09-12 snapshot was audited, but all of its listing and
calendar price values are null and it has no quote-context fields. A temporal
price backtest against 2026 would therefore change target definitions and has
been blocked. See `docs/inside_airbnb_temporal_readiness.md`.

## Model and baseline

The primary model is a histogram gradient-boosting regressor trained on
`log1p(quoted price)`. Numeric missing values are median-imputed; categoricals
are frequency-limited one-hot encodings. Geographic inputs include
neighbourhood, approximate latitude/longitude, and auditable Haversine
distances to Sydney CBD, Sydney Airport, five reference beaches, and five
major hubs. These are reference distances, not route distance or travel time.

Feature-family selection is performed on five host-disjoint development folds
while the governed test set remains reserved. The engineered geographic
features improved mean development MAE by AUD 0.81 relative to neighbourhood
plus raw coordinates and improved four of five folds, satisfying the
predeclared adoption rule. See `docs/inside_airbnb_feature_ablation.md`.

The market baseline uses the median in this fallback order:

1. neighbourhood and room type;
2. neighbourhood;
3. room type;
4. global market.

Identifiers and availability/review proxies are excluded from the model.

### Held-out results

| Metric | Model | Market median |
| --- | ---: | ---: |
| MAE, all test rows | AUD 138.14 | AUD 222.90 |
| Median absolute error | AUD 50.37 | AUD 91.98 |
| RMSE | AUD 514.52 | AUD 674.40 |
| R² | 0.450 | 0.056 |
| MAE, training-defined p01–p99 market | AUD 101.41 | AUD 173.04 |

The model improves overall MAE by approximately 38.0% relative to the market
median baseline. Extreme public quotes are retained, so RMSE remains sensitive
to the long upper tail.

## Prediction intervals

Intervals use split conformal absolute residuals in log-price space:

- target marginal coverage: 90%;
- held-out marginal coverage: approximately 89.5%;
- calibration hierarchy: neighbourhood-room type, then room type, then global;
- average held-out interval width: approximately AUD 454.

Observed room-type coverage is reported separately. Results within three
percentage points of the target are treated as diagnostic tolerance, not proof
of conditional coverage.

## Evidence sufficiency gate

A result is refused when any of these conditions applies:

- a critical input is missing;
- neighbourhood, property type, or room type was unseen in training;
- fewer than 20 exact neighbourhood-room comparables exist;
- relative prediction-interval width exceeds 2.0;
- predicted price is outside the training-defined 0.5%–99.5% support range;
- the source evidence is more than 120 days old;
- quote lead time is outside 0–365 days.

The held-out refusal rate is approximately 2.0%. Refused responses do not expose
a price recommendation.

Independently of row-level evidence, the artifact has
`deployment_authority=research_only` until a compatible later price snapshot
passes temporal validation. Every prediction exposes this authority and its
temporal-validation status.

## Reproduction

After completing Phase 0:

```powershell
python prepare_inside_airbnb_quotes.py
python inside_airbnb_eda.py
python inside_airbnb_feature_ablation.py
python inside_airbnb_quote_model.py train
python inside_airbnb_error_analysis.py
python inside_airbnb_upper_tail_challenger.py
python prepare_inside_airbnb_premium_features.py
python inside_airbnb_premium_challenger.py
python inside_airbnb_interval_challenger.py
python -m unittest discover -s tests -v
```

Run the example governed prediction:

```powershell
python inside_airbnb_quote_model.py predict `
  --input examples/inside_airbnb_quote_request.json `
  --output predictions/inside_airbnb_quote_example.json
```

The machine-readable evaluation is written to
`reports/inside_airbnb/sydney_2026-06-16_quote_mvp_evaluation.json`.

## Held-out error boundary

The separate error audit reproduces the governed test metrics and reports a
median absolute percentage error of 19.4%. It also finds systematic
underprediction in the training-defined upper price tail:

- p90-p95: median prediction bias approximately -AUD 257 and interval coverage
  approximately 76.2%;
- p95-p99: median prediction bias approximately -AUD 535 and interval coverage
  approximately 67.6%;
- above p99 contains too few rows for an automatic segment rule, but errors are
  extremely large.

The current model is therefore not suitable as a luxury-listing valuation
model. This disclosure does not change the model or evidence gate using test
labels. Any mitigation must first win on host-disjoint development folds and
then pass a compatible future temporal snapshot. See
`docs/inside_airbnb_model_card.md`.

The development-only challenger benchmark tested four alternative
loss/weighting strategies. Tail-weighted log absolute loss reduced development
upper-tail MAE by approximately 4.4% and improved absolute median bias by
approximately 21.6%, but it missed the predeclared 10% tail-MAE improvement
requirement. No challenger was promoted and the governed test set was not
used. See `docs/inside_airbnb_upper_tail_challenger.md`.

The next development-only experiment extracts premium amenity semantics,
bathroom privacy, property hierarchy, capacity ratios, and structural
interactions into a separate privacy-minimised Silver table. A cross-fitted
two-stage model routes likely fold-training-p90 listings to an expert trained
above fold-training p75. The best hard-gated candidate improves overall
development MAE by approximately 1.7%, upper-tail MAE by approximately 6.9%,
and severe underprediction from 50.9% to 40.7%. It still fails the 10%
upper-tail MAE requirement and is not promoted. See
`docs/inside_airbnb_premium_challenger.md`.

A nested development-only interval experiment calibrates asymmetric residuals
within model-predicted price bands. It raises upper-tail coverage from
approximately 72.1% to 76.5%, but the 4.3 percentage-point gain misses the
predeclared 10-point requirement while widening upper-tail intervals by about
37.1%. The incumbent interval method therefore remains active. See
`docs/inside_airbnb_interval_challenger.md`.

## Product boundary

The output wording must remain “estimated public quoted price per night”.
Calendar availability and review activity may be displayed separately as
proxies, but they must not be presented as verified bookings, occupancy,
demand, or revenue. An LLM may explain the structured result but must not
replace, modify, or override the model output or refusal gate.
