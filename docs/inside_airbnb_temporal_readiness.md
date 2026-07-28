# Sydney temporal price-validation readiness

## Decision

Temporal price validation is currently **blocked by incompatible source
labels**. The quote model remains `research_only`.

The live discovery check on 2026-07-28 confirmed that the official Inside
Airbnb index still lists 2026-06-16 as Sydney's latest free snapshot. The
discovery command is read-only: it neither changes the pinned registry nor
downloads data.

Inside Airbnb's public index currently exposes two Sydney snapshots within the
available period:

- 2025-09-12
- 2026-06-16

Their effective observation ranges are strictly ordered and separated by 278
days, so the chronology itself is valid. The blocker is the supervised target,
not the dates.

## Target compatibility

| Capability | 2025-09-12 | 2026-06-16 |
| --- | --- | --- |
| Quote price with check-in/check-out context | No | Yes |
| Non-null listing price | No | Yes |
| Non-null calendar daily price | No | No |
| Availability proxy | Yes | Yes |

The 2025 files contain `price` columns in listings and calendar, but every
value is null. The 2026 listings file introduces explicit quote-context fields
and 17,784 usable quote labels, while its calendar removes `price`.

Training on one target definition and testing on another would not be temporal
validation. The pipeline therefore refuses to construct that backtest.

## Population continuity

- Older listings: 17,730
- Newer listings: 20,573
- Listing overlap: 13,587
- Older-listing retention: 76.6%
- Newer listings seen previously: 66.0%
- Newer hosts seen previously: 75.7%

The overlap is sufficient for future seen-listing and cold-start cohort
analysis once compatible labels exist.

## Schema drift

The newer listings schema adds the quote check-in, check-out, total, per-night,
and raw quote fields. The newer calendar schema removes `price` and
`adjusted_price`.

These changes are capability drift, not merely harmless column reordering. The
Phase 0 audit evaluates both column presence and non-null values so an empty
legacy price column cannot incorrectly authorize a model.

## Enforced authority

`compare_inside_airbnb_snapshots.py` writes the machine-readable compatibility
decision. Model training embeds that decision into the artifact, and every
prediction now returns:

```json
{
  "deployment_authority": "research_only",
  "temporal_validation_status": "TEMPORAL_PRICE_VALIDATION_BLOCKED"
}
```

This authority flag does not claim that the current host-disjoint evaluation
is invalid. It prevents that evaluation from being misrepresented as evidence
of cross-time production reliability.

Target compatibility alone no longer changes deployment authority. A
compatible later snapshot only moves the workflow to "ready for evaluation".
`inside_airbnb_temporal_validation.py` must then pass every predeclared gate:

- the newer observation range is strictly after the older range;
- at least 2,000 compatible newer quotes are available;
- model MAE improves at least 10% over the older-snapshot market baseline;
- forward-time conformal coverage is at least 85%;
- at least 200 genuinely new-host rows are available;
- model MAE also improves over baseline for that new-host cohort.

Only a passing out-of-time report may recommend `temporally_validated`.

## Evidence needed to unblock

Either of the following is required:

1. a later Sydney snapshot containing the same non-null quote-context target
   as 2026-06-16; or
2. an approved archived dataset with a compatible non-null price target and
   documented semantics.

Until then, availability can be compared across snapshots only as a proxy; it
cannot supply a missing price label.

## Automated discovery monitor

The scheduled GitHub Actions workflow
`.github/workflows/sydney-snapshot-monitor.yml` checks the official index every
Monday at 02:17 UTC. It runs:

```powershell
python inside_airbnb_snapshot_discovery.py --fail-on-action-required
```

`NO_NEWER_SNAPSHOT` is the only passing discovery status. A newly published
Sydney date or an index-parsing failure makes the workflow fail so a maintainer
can review the candidate. The workflow always uploads the JSON discovery
report, but it cannot mutate the registry, download data, run Phase 0, or
upgrade deployment authority.

## Reproduction

```powershell
python inside_airbnb_snapshot_discovery.py

python inside_airbnb_phase0.py all `
  --snapshot-date 2025-09-12 `
  --report reports/inside_airbnb/sydney_2025-09-12_phase0_audit.json

python compare_inside_airbnb_snapshots.py
python inside_airbnb_quote_model.py train
```

When discovery reports `NEWER_SNAPSHOT_DISCOVERED`, review and pin its source
URLs, run Phase 0, and build its Silver table. Then run:

```powershell
python inside_airbnb_temporal_validation.py `
  --newer-silver data/silver/inside_airbnb/sydney/snapshot_date=YYYY-MM-DD/listing_quotes.csv

python inside_airbnb_quote_model.py train
```

The compatibility evidence is stored in
`reports/inside_airbnb/sydney_snapshot_target_compatibility.json`. Discovery
evidence is stored in
`reports/inside_airbnb/sydney_snapshot_discovery.json`; a future out-of-time
evaluation will be stored in
`reports/inside_airbnb/sydney_temporal_quote_validation.json`.
