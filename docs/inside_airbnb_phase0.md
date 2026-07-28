# Inside Airbnb Sydney Phase 0 decision

## Decision

The 2026-06-16 Sydney source is suitable for a **quote-level listed-price
MVP**, not the originally proposed listing-date price model.

The current `calendar.csv.gz` contains availability and stay restrictions but
does not contain a `price` column. It therefore cannot support weekday,
weekend, seasonal, or lead-time price modelling at listing-date grain.

The viable supervised target is
`listings.price_quote_price_per_night`, together with its explicit quote
context:

- `price_quote_checkin_date`
- `price_quote_checkout_date`
- `last_scraped`
- `source`

This target remains a public quoted price. It is not a realised booking price,
an optimal price, or evidence of revenue impact.

## Verified snapshot

| Dataset | Verified result |
| --- | ---: |
| Listings | 20,573 unique listings |
| Listings with a complete quote | 17,784 |
| Calendar | 7,509,145 rows; exactly 365 rows per listing |
| Calendar daily price | Not present |
| Reviews summary | 801,398 dated reviews |
| Neighbourhoods | 38 CSV rows and 38 matching GeoJSON features |

The download-page snapshot label is 2026-06-16, but listing-level
`last_scraped` values range from 2026-06-17 through 2026-06-29. All temporal
features must therefore use each row's `last_scraped` as its `as_of_date`;
the directory snapshot label must not be treated as a universal observation
timestamp.

The complete machine-readable result, including file hashes, schemas, checks,
warnings, and the capability blocker, is in
`reports/inside_airbnb/sydney_2026-06-16_phase0_audit.json`.

## Data contract

### Listings

- Primary key: `id`
- Observation timestamp: `last_scraped`
- Calendar observation timestamp: `calendar_last_scraped`
- Candidate target: `price_quote_price_per_night`
- Required quote context: check-in date, check-out date, and observation time
- Training eligibility: all target and quote-context fields parse successfully
- Missing and partial quotes are retained in Silver data but excluded from
  supervised training
- `host_id` may be retained as a grouping key, but host names, profile URLs,
  listing descriptions, and other unnecessary identity fields must be dropped

### Calendar

- Primary key: `listing_id + date`
- Foreign key: `listing_id -> listings.id`
- Expected coverage: 365 rows for each listing
- Supported use: future public availability and stay-restriction analysis
- Unsupported use: daily listed-price modelling
- `available=f` must not be described as a verified booking

### Reviews

- Required fields: `listing_id`, `date`
- Source: summary `reviews.csv`; detailed review text is not downloaded
- Temporal rule: `review.date <= listings.last_scraped` for the same listing
- Supported use: review recency and velocity proxies
- Unsupported use: verified demand or occupancy

### Neighbourhoods

- The CSV and GeoJSON neighbourhood sets must match exactly
- Every listing's `neighbourhood_cleansed` must be covered
- Coordinates are approximate public locations and must not be represented as
  exact property addresses

## Reproduction

Run a one-time download and audit:

```powershell
python inside_airbnb_phase0.py all
```

Subsequent audit runs do not access the network:

```powershell
python inside_airbnb_phase0.py audit
python -m unittest discover -s tests -v
```

Raw downloads and their local manifest are stored under
`data/raw/inside_airbnb/` and ignored by Git. The registry fixes the source
URLs and required schema in `config/inside_airbnb_snapshots.json`; the tracked
audit report preserves the actual source hashes and schemas.

## MVP boundary

The next implementation stage should:

1. build a leakage-safe Silver listing-quote table;
2. add quote check-in lead time, stay length, neighbourhood, room type, and
   property attributes;
3. compare against neighbourhood-by-room-type median baselines;
4. use host/listing grouping and later snapshots for honest evaluation;
5. add conformal intervals and an evidence sufficiency gate;
6. keep calendar availability and review velocity as separately reported
   proxies until their incremental value is validated.

The 17,784 usable quote labels range from AUD 1.74 to AUD 52,850.00. The MVP
must define an auditable outlier policy and report results both overall and by
market segment rather than silently deleting extreme observations.

## Source obligations

Inside Airbnb asks users to download only what they need, avoid repeated
scraping, avoid republishing raw data, and cite the source. The source page
labels the downloads as CC BY 4.0. See:

- <https://insideairbnb.com/get-the-data/>
- <https://insideairbnb.com/data-policies/>
- <https://insideairbnb.com/data-assumptions/>
