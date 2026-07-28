This project analysed a dataset containing Airbnb listings in the Northern Beaches council area of Sydney, with 28 featurs including the number of beds, baths, people, cleaning & deposite fees, reviews, GPS coordinates. I then used these features to train different machine learning models in Python to predict nightly rental price of Airbnb listings.

## Reproducible baseline (2026 refresh)

The repository contains 383 labeled listings. The labels in
`Models/Trees_labels.csv` align exactly by ID with the first 383 rows of
`Models/Trees_features.csv` and with all rows of
`Exploratory Data Analysis/EDA_X_train.csv`. The ID is checked but excluded
from model inputs.

Create an environment, install dependencies, and run the baseline:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_baseline.py --dataset compact
```

The experiment uses 5-fold cross-validation repeated five times with a fixed
random seed. On the compact 22-feature dataset, the current results are:

| Model | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Extra Trees | 70.01 +/- 5.50 | 54.35 +/- 4.90 | 0.513 +/- 0.073 |
| Ridge | 70.10 +/- 6.00 | 54.30 +/- 4.34 | 0.510 +/- 0.084 |
| Histogram Gradient Boosting | 73.58 +/- 6.62 | 56.84 +/- 5.31 | 0.460 +/- 0.096 |
| Median baseline | 103.68 +/- 5.48 | 83.88 +/- 3.51 | -0.062 +/- 0.045 |

Use `--dataset full` to evaluate the 35-feature matrix. JSON reports, including
the top permutation features, are written to `reports/`.

### Corrected geographic features

The historical notebook assigned postcodes by DataFrame row index even though
the location file and concatenated feature table used different row orders.
This gives some listings the wrong area. Rebuild the checked, ID-joined feature
matrices and evaluate them with:

```powershell
python prepare_features.py
python train_baseline.py --dataset corrected --output reports/baseline_corrected.json
```

The repair joins on listing `Id`, verifies uniqueness and complete ID coverage,
and requires every listing to have exactly one area. The original files are
preserved unchanged.

The repair changed the area assignment of 241 out of 383 labeled rows. The
570 test rows were already aligned. On the corrected 35-feature training set,
Extra Trees achieves RMSE `68.03 +/- 5.94`, MAE `53.10 +/- 4.74`, and R2
`0.540 +/- 0.071` under the same repeated cross-validation protocol. This is
the strongest reproducible baseline currently available in the repository.

Generate the data audit and final predictions with:

```powershell
python data_audit.py
python predict.py
```

The prediction command fits the selected Extra Trees configuration on all 383
labeled rows and writes one non-negative price prediction for every test ID
from 0 through 569. Cross-validation metrics remain the measure of expected
generalization; the unlabeled test predictions cannot be scored locally.

For a stricter model-selection estimate and a geographic transfer stress test:

```powershell
python robust_evaluation.py
```

This performs hyperparameter search only inside each outer training fold, then
separately trains on two council areas and tests on the entirely unseen third
area. The latter is intentionally a harder diagnostic and should not be mixed
with the primary random-fold score.

The nested out-of-fold result is RMSE `68.93`, MAE `53.23`, and R2 `0.534`.
Leaving out an entire area produces RMSE `71.58` for Manly, `91.22` for
Pittwater, and `72.51` for Warringah. This confirms that the model is useful as
an in-distribution prototype but is not reliable for a new geographic market,
especially Pittwater. The median absolute error is about $43, while 10% of
predictions miss by more than about $111.

Important limitation: only preprocessed modelling features are present. Since
the original raw listing table is missing, preprocessing cannot yet be moved
inside each cross-validation fold. The historical RMSE of 57.5 below should
therefore be treated as an unverified result rather than directly compared with
the refreshed baseline.

## Inside Airbnb Sydney Phase 0

The modern-data feasibility audit is now reproducible:

```powershell
python inside_airbnb_phase0.py all
```

The pinned 2026-06-16 Sydney source passed integrity, key, lineage, review-time,
and neighbourhood checks. It contains 20,573 listings, 7,509,145 calendar
rows, and 801,398 review-date rows.

The audit also found an important scope constraint: the current calendar file
has availability and stay restrictions but no daily price. The approved next
step is therefore a quote-level listed-price MVP using the 17,784 complete
listing quote records, not a listing-date dynamic-pricing model. Availability
and reviews remain explicitly labelled proxies.

See `docs/inside_airbnb_phase0.md` for the data contract and decision, and
`reports/inside_airbnb/sydney_2026-06-16_phase0_audit.json` for the complete
machine-readable evidence. Raw source files are downloaded once and remain
outside Git.

### Quote-level MVP

Build the privacy-minimised Silver quote table and train the governed model:

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

The evaluation is host-disjoint: 11,375 training rows, 2,708 independent
conformal-calibration rows, and 3,701 test rows have zero host overlap. On the
held-out test set, the model achieves MAE `138.14` versus `222.90` for the
neighbourhood-room market median baseline, a relative improvement of about
38.0%. The 90% conformal interval achieves approximately 89.5% marginal
coverage. The evidence gate refuses about 2.0% of test cases.

The modern EDA pack reports explicit missingness, robust price quantiles,
segment distributions, numeric associations with log price, and spatial
patterns. Its four SVG charts are generated without an online service. The
feature ablation reserves the governed test set and compares five identical
host-disjoint development folds. Transparent offline distances to Sydney CBD,
airport, reference beaches, and major hubs improved mean development MAE in
four of five folds and were therefore retained.

Run a governed example prediction:

```powershell
python inside_airbnb_quote_model.py predict `
  --input examples/inside_airbnb_quote_request.json `
  --output predictions/inside_airbnb_quote_example.json
```

See `docs/inside_airbnb_quote_mvp.md` and
`reports/inside_airbnb/sydney_2026-06-16_quote_mvp_evaluation.json` for the
protocol, segment coverage, refusal rules, and limitations.

See `docs/inside_airbnb_modern_eda.md` and
`docs/inside_airbnb_feature_ablation.md` for the methodology upgrade and
development-only feature evidence.

The held-out error audit reports a median absolute percentage error of 19.4%
but identifies systematic underprediction and weak conditional interval
coverage in the upper price tail. The current model must not be presented as a
luxury-listing valuation model. These findings are disclosed in
`docs/inside_airbnb_model_card.md`; they are not converted into test-set-driven
tuning or refusal rules.

A development-only upper-tail benchmark compared log squared loss, tail
weighting, raw Poisson, and raw squared loss against the incumbent. None met
all predeclared promotion gates, so the primary artifact remains unchanged.
See `docs/inside_airbnb_upper_tail_challenger.md`.

The premium-feature experiment adds privacy-minimised amenity semantics,
bathroom/property hierarchies, capacity ratios, and cross-fitted two-stage
experts in a separate enriched Silver table. The best hard-gated challenger
improves development overall MAE by about 1.7%, upper-tail MAE by about 6.9%,
and severe underprediction from 50.9% to 40.7%. It still misses the
predeclared 10% upper-tail requirement, so it is not promoted. See
`docs/inside_airbnb_premium_challenger.md`.

The development-only predicted-price-band asymmetric conformal challenger
raises upper-tail coverage from about 72.1% to 76.5%, while increasing overall
interval width by about 13.1% and upper-tail width by about 37.1%. The 4.3
percentage-point coverage gain misses the predeclared 10-point requirement, so
the incumbent interval method remains active. See
`docs/inside_airbnb_interval_challenger.md`.

### Temporal-readiness check

The public 2025-09-12 Sydney snapshot has been downloaded and audited as the
earlier candidate for temporal validation. Its chronology is valid, but its
listing and calendar price columns contain no non-null labels, while the 2026
snapshot introduces a different quote-context target. The pipeline therefore
blocks a misleading cross-target backtest:

```powershell
python inside_airbnb_snapshot_discovery.py
python compare_inside_airbnb_snapshots.py
```

The current model artifact and predictions are marked `research_only` with
`TEMPORAL_PRICE_VALIDATION_BLOCKED`. A later snapshot with the same quote
fields, or an approved compatible archive, is required to unlock temporal
production evidence. See `docs/inside_airbnb_temporal_readiness.md` and
`reports/inside_airbnb/sydney_snapshot_target_compatibility.json`.

The live discovery check currently finds no Sydney snapshot newer than
2026-06-16. Once one appears, `inside_airbnb_temporal_validation.py` performs
strict forward-time evaluation, reports seen/new listing and host cohorts,
measures categorical and target drift, and applies predeclared performance and
coverage gates. Compatibility by itself never upgrades model authority.

### Delivery pipeline and release gates

Run offline CI checks or rebuild the complete pinned-data research workflow:

```powershell
python run_inside_airbnb_pipeline.py ci
python run_inside_airbnb_pipeline.py research
```

Only the explicit `refresh` mode may perform live discovery or raw downloads.
The research pipeline hashes the primary artifact around every challenger and
diagnostic command, writes a reproducibility Manifest, and evaluates separate
research and production release gates.

The current verified result is `RESEARCH_RELEASE_ALLOWED` and
`PRODUCTION_RELEASE_BLOCKED`. Production enforcement remains blocked until a
compatible forward-time snapshot passes all temporal gates:

```powershell
python inside_airbnb_release_gate.py --target production --enforce
```

See `docs/inside_airbnb_delivery.md` and
`reports/inside_airbnb/reproducibility_manifest.json`.

•	Exploratory Data Analysis

- Performed statistical imputation to handle significant amount of missing values
- Annalysed the distribution of continuous features, and the correlation between features and rental price.
- Performed log-normal transformation and z-score standardization of the features to handle non-linear relationships and highly-skewed distributions

•	Feature Engineering

- Mapped GPS coordinates into postcodes using geo-location API to generate a new feature representing district area, 
- Calculated distance from each Airbnb listing to 10 popular tourist attractions. 
- Trained a XGBoost model on these distances to predict the rental price and Westfield Shopping Center and Manly Beach have the highest feature importance.

•	Model Selection

- Trained XGBoost, Extremely Randomised Trees, Lasso, OLS, Ridge models and a Generalised Additive Model using Ridge with natural cubic splines. 
- Combined the best models using stacking to improve their predictive accuracy: Gradient boosted Ridge model achieved the best RMSE of 57.5.


