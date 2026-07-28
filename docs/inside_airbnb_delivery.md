# Inside Airbnb delivery and release controls

## One-command entry point

`run_inside_airbnb_pipeline.py` provides three explicit modes:

```powershell
# Offline compile and fixture tests only. No raw-data download or live discovery.
python run_inside_airbnb_pipeline.py ci

# Rebuild all research outputs from already pinned local raw data.
python run_inside_airbnb_pipeline.py research

# Run live discovery and Phase 0 download/audit before the research workflow.
python run_inside_airbnb_pipeline.py refresh
```

`refresh` is the only mode allowed to perform network discovery or download
raw files. `ci` and `research` operate without downloading raw sources.

## Research workflow

The research mode runs, in order:

1. snapshot-target compatibility;
2. governed Silver construction;
3. modern EDA;
4. feature ablation;
5. primary model training;
6. upper-tail challenger;
7. premium Silver and premium/two-stage challenger;
8. interval challenger;
9. held-out error analysis and model card;
10. example prediction;
11. compile and unit tests;
12. research and production release gates;
13. reproducibility Manifest.

Every challenger and diagnostic command is marked as artifact-preserving. The
orchestrator hashes the primary model before and after each protected command
and fails immediately if the hash changes.

## Reproducibility Manifest

The generated
`reports/inside_airbnb/reproducibility_manifest.json` records:

- Python, platform, NumPy, scikit-learn, and joblib versions;
- Git commit and dirty-worktree status;
- fixed random seeds;
- source registry, raw manifests, Silver tables, artifact, prediction, and
  report SHA-256 values;
- report decisions and model authority;
- every pipeline command, status, return code, and duration;
- governance expectations for CI, production, and challenger writes.

The Manifest does not claim that a dirty worktree is reproducible from Git
alone. It records that condition explicitly.

## Release gates

Research release:

```powershell
python inside_airbnb_release_gate.py --target research --enforce
```

Production release:

```powershell
python inside_airbnb_release_gate.py --target production --enforce
```

Both gates require a matching Silver hash, supported artifact version, model
improvement over the market baseline, matching artifact/report authority,
successful compile and unit-test commands, and proof that challengers did not
write the primary artifact.

Production additionally requires:

- `deployment_authority=temporally_validated`;
- a passing strict out-of-time report;
- the temporal report's older Silver SHA-256 to match the training Silver;
- target compatibility status `TEMPORAL_PRICE_VALIDATION_READY`.

Without these conditions, `--enforce` exits non-zero.

## Current verified status

The full local research workflow completed successfully with 15 commands. All
eight artifact-preserving commands retained the exact primary artifact hash.
The current gates are:

- `RESEARCH_RELEASE_ALLOWED`;
- `PRODUCTION_RELEASE_BLOCKED`.

Production is blocked by missing compatible temporal-price evidence, not by a
software or test failure.

## CI

`.github/workflows/ci.yml` installs pinned-compatible dependencies and runs the
offline `ci` mode. It uploads the CI run report and CI reproducibility
Manifest. The workflow never invokes live discovery, Phase 0, or raw-data
download.

## Scheduled snapshot monitor

`.github/workflows/sydney-snapshot-monitor.yml` runs every Monday at 02:17 UTC
and can also be started manually. It performs only the read-only official-index
discovery step and uploads the resulting JSON evidence.

The monitor passes while Sydney's latest official snapshot matches the pinned
registry. It fails with an actionable signal when a newer snapshot appears or
when the official index no longer yields a Sydney date. The evidence upload
still runs after failure. The monitor never changes the registry, downloads raw
files, runs Phase 0, or changes model authority.
