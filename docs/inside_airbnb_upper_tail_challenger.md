# Inside Airbnb upper-tail challenger benchmark

This experiment uses development data only. The governed test set is reserved and no primary artifact is replaced.

## Predeclared promotion rule

- Overall MAE may degrade by no more than 2%.
- Upper-tail MAE must improve by at least 10%.
- Absolute upper-tail median bias must improve by at least 10%.
- Upper-tail MAE must improve in at least four of five host-disjoint folds.

## Results

| Candidate | Overall MAE | Upper-tail MAE | Tail median bias | Severe underprediction | Qualifies |
| --- | ---: | ---: | ---: | ---: | --- |
| `incumbent_log_absolute` | 134.64 | 756.75 | -339.17 | 50.9% | incumbent |
| `log_squared_error` | 134.12 | 728.03 | -284.19 | 43.6% | no |
| `log_tail_weighted_absolute` | 135.11 | 723.38 | -265.94 | 41.7% | no |
| `raw_poisson` | 148.44 | 737.04 | -236.89 | 39.0% | no |
| `raw_squared_error` | 172.58 | 756.75 | -245.00 | 41.4% | no |

## Decision

- Status: `NO_CHALLENGER_PASSED_ALL_GATES`.
- Selected challenger: `None`.
- Primary model replaced: **no**.
- Next evidence: Retain the incumbent and investigate new feature/data families on development folds.

## Boundaries

- The upper tail is defined independently inside every fold using that fold's training-target p90.
- Validation hosts never appear in the corresponding training fold.
- No availability or review proxy is introduced.
- This benchmark cannot restore temporal authority or make the spent governed test set fresh again.
