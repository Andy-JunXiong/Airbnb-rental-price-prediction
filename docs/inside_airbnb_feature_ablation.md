# Inside Airbnb feature ablation

This diagnostic uses only development data. The governed seed-42 test set remains untouched.
Every comparison uses the same five host-disjoint folds and the same model configuration.

## Results

| Variant | Features | MAE | RMSE | R² | Δ MAE vs previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| `listing_and_quote_context` | 16 | 151.51 ± 33.57 | 565.20 | 0.317 | — |
| `plus_neighbourhood` | 17 | 140.15 ± 32.22 | 554.05 | 0.349 | -11.36 |
| `plus_raw_coordinates` | 19 | 135.72 ± 31.60 | 545.89 | 0.371 | -4.43 |
| `plus_engineered_geography` | 23 | 134.91 ± 30.98 | 546.42 | 0.369 | -0.81 |

## Engineered-geography decision

- Mean MAE delta versus raw coordinates: -0.81.
- Better folds: 4 of 5.
- Primary-model adoption: **yes**.
- Rule: Adopt only if mean development MAE improves and at least three of five host-disjoint folds improve.

## Held-out permutation importance

Positive values mean validation MAE increased when the feature was shuffled. This is predictive association, not causality.

| Feature | Mean MAE increase | Standard deviation |
| --- | ---: | ---: |
| `accommodates` | 38.37 | 1.37 |
| `bedrooms` | 37.26 | 1.28 |
| `stay_nights` | 31.59 | 1.20 |
| `bathrooms` | 21.48 | 0.86 |
| `longitude` | 20.70 | 0.77 |
| `room_type` | 15.92 | 1.02 |
| `distance_to_nearest_reference_beach_km` | 5.12 | 0.30 |
| `quote_lead_days` | 4.32 | 0.22 |
| `property_type` | 3.37 | 0.20 |
| `latitude` | 2.66 | 0.35 |
| `checkin_month` | 2.52 | 0.12 |
| `calculated_host_listings_count` | 2.13 | 0.21 |
| `distance_to_sydney_airport_km` | 2.09 | 0.24 |
| `distance_to_sydney_cbd_km` | 1.70 | 0.28 |
| `amenities_count` | 1.63 | 0.22 |

## Boundaries

- Hosts never overlap between a fold's training and validation rows.
- The experiment does not introduce availability or review proxy features.
- Approximate coordinates and fixed reference distances do not represent route distance or travel time.
- The final deployment authority remains governed by temporal validation, not this ablation.
