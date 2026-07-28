# Inside Airbnb interval challenger

This experiment uses nested host-disjoint train/calibration/validation splits inside development data. The governed test set and primary artifact are untouched.

## Results

| Method | Overall coverage | Average width | Upper-tail coverage | Upper-tail width | Upper-tail upper-miss rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent hierarchical symmetric | 90.0% | 475 | 72.1% | 1394 | 27.6% |
| Predicted-band asymmetric | 88.7% | 538 | 76.5% | 1912 | 22.4% |

## Predeclared qualification rule

- Overall coverage must be at least 88%.
- Upper-tail coverage must improve by at least 10 percentage points.
- Average width may increase by no more than 25%.
- Upper-tail width may increase by no more than 50%.
- Upper-tail coverage must improve in at least four of five folds.

## Decision

- Status: `INTERVAL_CHALLENGER_REJECTED`.
- Qualifies: **no**.
- Primary artifact changed: **no**.
- Next evidence: Retain incumbent intervals and investigate richer scale/quantile models on development data.

## Boundaries

- Price bands are assigned from model predictions, never from unknown inference-time labels.
- Band thresholds use only the corresponding fit split's target distribution.
- Calibration residuals are asymmetric, allowing a wider upper correction than lower correction.
- This development result cannot replace future temporal validation.
