# Inside Airbnb premium-feature and two-stage challenger

All selection is performed on five host-disjoint development folds. The governed test set and primary artifact remain untouched.

## Candidate design

- Premium semantics: pool, hot tub, waterfront, beach access, water view, on-premises parking, gym, sauna, indoor fireplace, and private outdoor space.
- Structural interactions: per-guest bathroom/bedroom/bed ratios, accommodates per bedroom, and bedroom-bathroom interaction.
- Hierarchies: bathroom privacy and stable property group.
- Two-stage router: classify fold-training p90, then blend or route to a log-squared expert trained above fold-training p75.

## Results

| Candidate | Overall MAE | Upper-tail MAE | Tail median bias | Severe underprediction | Qualifies |
| --- | ---: | ---: | ---: | ---: | --- |
| `incumbent_base_features` | 134.64 | 756.75 | -339.17 | 50.9% | incumbent |
| `premium_log_absolute` | 132.92 | 745.27 | -333.95 | 48.6% | no |
| `premium_tail_weighted` | 133.41 | 710.28 | -248.82 | 39.5% | no |
| `premium_two_stage_soft` | 131.36 | 705.71 | -275.54 | 41.2% | no |
| `premium_two_stage_hard` | 132.32 | 704.71 | -274.61 | 40.7% | no |

## Predeclared promotion rule

- Overall MAE may degrade by no more than 2%.
- Upper-tail MAE must improve by at least 10%.
- Absolute upper-tail median bias must improve by at least 10%.
- Upper-tail MAE must improve in at least four of five folds.

## Decision

- Status: `NO_CHALLENGER_PASSED_ALL_GATES`.
- Selected challenger: `None`.
- Primary model replaced: **no**.
- Next evidence: Retain the incumbent; richer licensed market/context data is required for a material upper-tail improvement.

## Governance

- Every upper-tail threshold is fitted inside the corresponding training fold.
- Router labels and expert training membership never use validation targets.
- Raw amenity strings and listing text are not retained in the enriched Silver table.
- A challenger cannot be promoted without a compatible future temporal snapshot.
