"""Audit held-out quote-model errors without tuning on the governed test set."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

from inside_airbnb_phase0 import (
    ROOT,
    active_snapshot_date,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from inside_airbnb_quote_model import (
    ALPHA,
    DEFAULT_ARTIFACT,
    DEFAULT_SILVER,
    group_split,
    interval_bounds,
    load_silver,
    market_prediction,
    price_metrics,
    quantiles_for_records,
)


_SNAPSHOT = active_snapshot_date()
DEFAULT_REPORT = (
    ROOT / "reports" / "inside_airbnb" / f"sydney_{_SNAPSHOT}_error_analysis.json"
)
DEFAULT_MODEL_CARD = ROOT / "docs" / "inside_airbnb_model_card.md"
DEFAULT_ASSETS = ROOT / "reports" / "inside_airbnb" / "error_assets"
DEFAULT_CHALLENGER_REPORT = (
    ROOT / "reports" / "inside_airbnb" / f"sydney_{_SNAPSHOT}_upper_tail_challenger.json"
)
DEFAULT_PREMIUM_CHALLENGER_REPORT = (
    ROOT / "reports" / "inside_airbnb" / f"sydney_{_SNAPSHOT}_premium_challenger.json"
)
DEFAULT_INTERVAL_CHALLENGER_REPORT = (
    ROOT / "reports" / "inside_airbnb" / f"sydney_{_SNAPSHOT}_interval_challenger.json"
)
MIN_SEGMENT_ROWS = 30
MIN_DIAGNOSTIC_ROWS = 100
CONDITIONAL_COVERAGE_FLOOR = 0.80
MAX_ABSOLUTE_MEDIAN_BIAS_AUD = 50.0


def finite_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    errors = predicted - actual
    absolute = np.abs(errors)
    covered = (actual >= lower) & (actual <= upper)
    model = price_metrics(actual, predicted)
    market = price_metrics(actual, baseline)
    return {
        "rows": len(actual),
        "model": model,
        "market_baseline": market,
        "relative_mae_improvement_vs_market": float(
            1 - model["mae"] / market["mae"]
        ),
        "mean_error_predicted_minus_actual": float(np.mean(errors)),
        "median_error_predicted_minus_actual": float(np.median(errors)),
        "underprediction_rate": float(np.mean(errors < 0)),
        "severe_underprediction_rate": float(
            np.mean(predicted < actual * 0.70)
        ),
        "median_absolute_percentage_error": float(
            np.median(absolute / np.maximum(actual, 1.0))
        ),
        "interval_coverage": float(np.mean(covered)),
        "average_interval_width": float(np.mean(upper - lower)),
        "absolute_error_quantiles": {
            f"p{int(probability * 100):02d}": float(
                np.quantile(absolute, probability)
            )
            for probability in (0.50, 0.75, 0.90, 0.95, 0.99)
        },
    }


def grouped_audit(
    records: list[dict[str, str]],
    test_indices: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    field: str,
    value_for_row: Callable[[dict[str, str], int], str] | None = None,
) -> list[dict[str, Any]]:
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(test_indices):
        row = records[int(index)]
        value = (
            value_for_row(row, position)
            if value_for_row
            else (row.get(field) or "(missing)")
        )
        positions[value].append(position)
    results = []
    for value, selected_positions in positions.items():
        if len(selected_positions) < MIN_SEGMENT_ROWS:
            continue
        selected = np.asarray(selected_positions, dtype=int)
        results.append(
            {
                field: value,
                **finite_metrics(
                    actual[selected],
                    predicted[selected],
                    baseline[selected],
                    lower[selected],
                    upper[selected],
                ),
            }
        )
    return sorted(results, key=lambda row: (-row["rows"], str(row[field])))


def price_band_definitions(training_target: np.ndarray) -> list[dict[str, Any]]:
    probabilities = (0.50, 0.75, 0.90, 0.95, 0.99)
    thresholds = [
        float(np.quantile(training_target, probability))
        for probability in probabilities
    ]
    labels = [
        "up_to_p50",
        "p50_to_p75",
        "p75_to_p90",
        "p90_to_p95",
        "p95_to_p99",
        "above_p99",
    ]
    lower_bounds = [0.0, *thresholds]
    upper_bounds = [*thresholds, math.inf]
    return [
        {
            "label": label,
            "lower_exclusive": lower,
            "upper_inclusive": upper if math.isfinite(upper) else None,
        }
        for label, lower, upper in zip(labels, lower_bounds, upper_bounds)
    ]


def price_band(value: float, definitions: list[dict[str, Any]]) -> str:
    for definition in definitions:
        upper = definition["upper_inclusive"]
        if upper is None or value <= upper:
            return str(definition["label"])
    raise AssertionError("Price band definitions must cover every value")


def capacity_band(row: dict[str, str], _: int) -> str:
    value = float(row["accommodates"])
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    if value <= 6:
        return "5-6"
    return "7+"


def diagnostic_flags(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flags = []
    for dimension, rows in groups.items():
        for row in rows:
            if row["rows"] < MIN_DIAGNOSTIC_ROWS:
                continue
            segment = str(row[dimension])
            if row["interval_coverage"] < CONDITIONAL_COVERAGE_FLOOR:
                flags.append(
                    {
                        "dimension": dimension,
                        "segment": segment,
                        "rows": row["rows"],
                        "flag": "conditional_coverage_below_floor",
                        "observed": row["interval_coverage"],
                        "threshold": CONDITIONAL_COVERAGE_FLOOR,
                    }
                )
            if row["relative_mae_improvement_vs_market"] < 0:
                flags.append(
                    {
                        "dimension": dimension,
                        "segment": segment,
                        "rows": row["rows"],
                        "flag": "model_mae_worse_than_market_baseline",
                        "observed": row["relative_mae_improvement_vs_market"],
                        "threshold": 0.0,
                    }
                )
            median_bias = row["median_error_predicted_minus_actual"]
            if abs(median_bias) > MAX_ABSOLUTE_MEDIAN_BIAS_AUD:
                flags.append(
                    {
                        "dimension": dimension,
                        "segment": segment,
                        "rows": row["rows"],
                        "flag": "absolute_median_bias_above_limit",
                        "observed": median_bias,
                        "threshold": MAX_ABSOLUTE_MEDIAN_BIAS_AUD,
                    }
                )
    return sorted(
        flags,
        key=lambda row: (row["flag"], -row["rows"], row["dimension"], row["segment"]),
    )


def svg_document(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        f"{body}</svg>\n"
    )


def price_band_error_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 920, 450
    left, top, plot_width, plot_height = 90, 85, 760, 270
    maximum = max(row["model"]["mae"] for row in rows)
    bar_width = plot_width / max(len(rows), 1) * 0.55
    body = [
        '<text x="35" y="35" font-family="sans-serif" font-size="22" '
        'font-weight="700" fill="#17252a">Held-out MAE by training-defined price band</text>',
        '<text x="35" y="60" font-family="sans-serif" font-size="13" '
        'fill="#526066">Bands use training-target quantiles; test labels are used only for diagnosis.</text>',
    ]
    for index, row in enumerate(rows):
        x = left + (index + 0.2) * plot_width / len(rows)
        bar_height = plot_height * row["model"]["mae"] / max(maximum, 1)
        y = top + plot_height - bar_height
        body.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="#c9573d"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="12">'
                f'{row["model"]["mae"]:.0f}</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{top + plot_height + 24}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="11">'
                f'{html.escape(row["price_band"])}</text>',
            ]
        )
    body.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#17252a"/>'
    )
    return svg_document(width, height, "".join(body), "Held-out MAE by price band")


def observed_predicted_svg(actual: np.ndarray, predicted: np.ndarray) -> str:
    maximum_points = 2500
    if len(actual) > maximum_points:
        positions = np.linspace(0, len(actual) - 1, maximum_points).astype(int)
        actual = actual[positions]
        predicted = predicted[positions]
    x_values = np.log1p(actual)
    y_values = np.log1p(predicted)
    low = min(float(np.min(x_values)), float(np.min(y_values)))
    high = max(float(np.max(x_values)), float(np.max(y_values)))
    width, height = 680, 650
    left, top, size = 75, 70, 520
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = left + (x_value - low) / max(high - low, 1e-9) * size
        y = top + (high - y_value) / max(high - low, 1e-9) * size
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" '
            'fill="#287271" fill-opacity="0.35"/>'
        )
    body = (
        '<text x="30" y="34" font-family="sans-serif" font-size="22" '
        'font-weight="700" fill="#17252a">Observed versus predicted quote price</text>'
        '<text x="30" y="58" font-family="sans-serif" font-size="13" '
        'fill="#526066">Log scale; diagonal denotes perfect prediction.</text>'
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" '
        'fill="#eef2ef" stroke="#aab5b1"/>'
        f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" '
        'stroke="#d97b29" stroke-width="2"/>'
        + "".join(points)
        + f'<text x="{left + size / 2}" y="{height - 25}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">Observed log1p price</text>'
        f'<text x="18" y="{top + size / 2}" transform="rotate(-90 18 '
        f'{top + size / 2})" text-anchor="middle" font-family="sans-serif" '
        'font-size="13">Predicted log1p price</text>'
    )
    return svg_document(width, height, body, "Observed versus predicted quote price")


def write_model_card(path: Path, report: dict[str, Any], asset_dir: Path) -> None:
    overall = report["overall"]
    flags = report["diagnostic_flags"]
    relative_assets = Path("..") / asset_dir.relative_to(ROOT)
    lines = [
        "# Inside Airbnb Sydney quote-model card",
        "",
        f"Model artifact version: `{report['artifact']['artifact_version']}`. "
        f"Snapshot: `{report['artifact']['snapshot_label']}`.",
        "",
        "## Intended use",
        "",
        "Estimate a public quoted nightly price in AUD for research and portfolio demonstration. "
        "It is not a realised booking price, optimal price, occupancy forecast, or revenue guarantee.",
        "",
        "## Held-out performance",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Rows | {overall['rows']:,} |",
        f"| MAE | AUD {overall['model']['mae']:.2f} |",
        f"| Median absolute error | AUD {overall['model']['median_absolute_error']:.2f} |",
        f"| RMSE | AUD {overall['model']['rmse']:.2f} |",
        f"| MAE improvement vs market median | {overall['relative_mae_improvement_vs_market']:.1%} |",
        f"| 90% interval observed coverage | {overall['interval_coverage']:.1%} |",
        f"| Median absolute percentage error | {overall['median_absolute_percentage_error']:.1%} |",
        "",
        f"![Price-band errors]({(relative_assets / 'price_band_mae.svg').as_posix()})",
        "",
        f"![Observed versus predicted]({(relative_assets / 'observed_vs_predicted.svg').as_posix()})",
        "",
        "## Diagnostic flags",
        "",
    ]
    if flags:
        lines.extend(
            [
                "| Dimension | Segment | Rows | Flag | Observed |",
                "| --- | --- | ---: | --- | ---: |",
            ]
        )
        lines.extend(
            f"| {row['dimension']} | {row['segment']} | {row['rows']:,} | "
            f"{row['flag']} | {row['observed']:.3f} |"
            for row in flags
        )
    else:
        lines.append("No predeclared diagnostic threshold was breached.")
    lines.extend(
        [
            "",
            "## Upper-tail challenger status",
            "",
            (
                f"- Decision: `{report['upper_tail_challenger']['status']}`."
                if report.get("upper_tail_challenger")
                else "- No development-only challenger benchmark is attached."
            ),
            (
                f"- Selected challenger: `{report['upper_tail_challenger']['selected_challenger']}`."
                if report.get("upper_tail_challenger")
                else ""
            ),
            (
                "- The primary artifact was not replaced and the governed test set was not used."
                if report.get("upper_tail_challenger")
                else ""
            ),
            (
                f"- Premium/two-stage decision: `{report['premium_challenger']['status']}`."
                if report.get("premium_challenger")
                else "- No premium/two-stage benchmark is attached."
            ),
            (
                f"- Conditional-interval decision: `{report['interval_challenger']['status']}`."
                if report.get("interval_challenger")
                else "- No conditional-interval benchmark is attached."
            ),
            "",
            "## Evidence and limitations",
            "",
            "- Test hosts do not overlap train or conformal-calibration hosts.",
            "- Price bands are defined from training-target quantiles.",
            "- Slice findings are disclosures, not test-set-driven model tuning rules.",
            "- Conditional coverage is not guaranteed even when marginal coverage is near 90%.",
            "- Coordinates are approximate and reference distances are straight-line distances.",
            "- Availability and review-velocity proxies are excluded from the primary model.",
            f"- Deployment authority remains `{report['artifact']['deployment_authority']}`.",
            "- A compatible later Sydney snapshot is still required for temporal validation.",
            "",
            f"Machine-readable audit: `{report['report_path']}`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_error_analysis(
    silver_path: Path,
    artifact_path: Path,
    report_path: Path,
    model_card_path: Path,
    asset_dir: Path,
) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    records, X, y, groups = load_silver(silver_path)
    development, test_indices = group_split(X, groups, 0.20, 42)
    train_relative, _ = group_split(
        X[development], groups[development], 0.20, 43
    )
    training_indices = development[train_relative]
    if artifact["snapshot_label"] != records[0]["snapshot_label"]:
        raise ValueError("Artifact snapshot does not match Silver snapshot")
    expected_hash = artifact.get("training_silver_sha256")
    actual_hash = sha256_file(silver_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Artifact training Silver hash does not match input")

    predicted_log = artifact["pipeline"].predict(X[test_indices])
    predicted = np.maximum(0.0, np.expm1(predicted_log))
    quantiles = quantiles_for_records(
        records,
        test_indices,
        artifact["global_conformal_quantile"],
        artifact["segment_conformal_quantiles"],
        artifact["room_type_conformal_quantiles"],
    )
    lower, upper = interval_bounds(predicted_log, quantiles)
    baseline = np.asarray(
        [
            market_prediction(artifact["market_baseline"], records[int(index)])[0]
            for index in test_indices
        ],
        dtype=float,
    )
    actual = y[test_indices]
    definitions = price_band_definitions(y[training_indices])
    groups_audit = {
        "room_type": grouped_audit(
            records, test_indices, actual, predicted, baseline, lower, upper, "room_type"
        ),
        "property_type": grouped_audit(
            records,
            test_indices,
            actual,
            predicted,
            baseline,
            lower,
            upper,
            "property_type",
        ),
        "neighbourhood": grouped_audit(
            records,
            test_indices,
            actual,
            predicted,
            baseline,
            lower,
            upper,
            "neighbourhood",
        ),
        "capacity_band": grouped_audit(
            records,
            test_indices,
            actual,
            predicted,
            baseline,
            lower,
            upper,
            "capacity_band",
            capacity_band,
        ),
        "price_band": grouped_audit(
            records,
            test_indices,
            actual,
            predicted,
            baseline,
            lower,
            upper,
            "price_band",
            lambda _row, position: price_band(actual[position], definitions),
        ),
    }
    overall = finite_metrics(actual, predicted, baseline, lower, upper)
    flags = diagnostic_flags(groups_audit)
    challenger_report = (
        json.loads(DEFAULT_CHALLENGER_REPORT.read_text(encoding="utf-8"))
        if DEFAULT_CHALLENGER_REPORT.exists()
        else None
    )
    premium_challenger_report = (
        json.loads(DEFAULT_PREMIUM_CHALLENGER_REPORT.read_text(encoding="utf-8"))
        if DEFAULT_PREMIUM_CHALLENGER_REPORT.exists()
        else None
    )
    interval_challenger_report = (
        json.loads(DEFAULT_INTERVAL_CHALLENGER_REPORT.read_text(encoding="utf-8"))
        if DEFAULT_INTERVAL_CHALLENGER_REPORT.exists()
        else None
    )
    worst_positions = np.argsort(np.abs(predicted - actual))[::-1][:20]
    report = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "report_path": str(report_path.relative_to(ROOT)),
        "artifact": {
            "path": str(artifact_path.relative_to(ROOT)),
            "artifact_version": artifact["artifact_version"],
            "snapshot_label": artifact["snapshot_label"],
            "training_silver_sha256_matches": (
                expected_hash == actual_hash if expected_hash else None
            ),
            "deployment_authority": artifact["deployment_authority"],
        },
        "protocol": {
            "governed_host_disjoint_test": True,
            "test_rows": len(test_indices),
            "test_unique_hosts": len(set(groups[test_indices])),
            "train_test_host_overlap": len(
                set(groups[training_indices]) & set(groups[test_indices])
            ),
            "model_tuning_from_this_report": False,
            "price_band_definitions": definitions,
        },
        "overall": overall,
        "segments": groups_audit,
        "diagnostic_thresholds": {
            "minimum_rows": MIN_DIAGNOSTIC_ROWS,
            "conditional_coverage_floor": CONDITIONAL_COVERAGE_FLOOR,
            "maximum_absolute_median_bias_aud": MAX_ABSOLUTE_MEDIAN_BIAS_AUD,
            "model_must_outperform_market_baseline": True,
        },
        "diagnostic_flags": flags,
        "upper_tail_challenger": (
            {
                "report": str(DEFAULT_CHALLENGER_REPORT.relative_to(ROOT)),
                "status": challenger_report["decision"]["status"],
                "selected_challenger": challenger_report["decision"][
                    "selected_challenger"
                ],
                "governed_test_used": challenger_report["governance"][
                    "governed_test_used"
                ],
                "primary_artifact_written": challenger_report["governance"][
                    "primary_artifact_written"
                ],
            }
            if challenger_report
            else None
        ),
        "premium_challenger": (
            {
                "report": str(
                    DEFAULT_PREMIUM_CHALLENGER_REPORT.relative_to(ROOT)
                ),
                "status": premium_challenger_report["decision"]["status"],
                "selected_challenger": premium_challenger_report["decision"][
                    "selected_challenger"
                ],
                "governed_test_used": premium_challenger_report["governance"][
                    "governed_test_used"
                ],
                "primary_artifact_written": premium_challenger_report[
                    "governance"
                ]["primary_artifact_written"],
            }
            if premium_challenger_report
            else None
        ),
        "interval_challenger": (
            {
                "report": str(
                    DEFAULT_INTERVAL_CHALLENGER_REPORT.relative_to(ROOT)
                ),
                "status": interval_challenger_report["decision"]["status"],
                "qualifies": interval_challenger_report["decision"]["qualifies"],
                "upper_tail_coverage_absolute_improvement": (
                    interval_challenger_report["decision"][
                        "upper_tail_coverage_absolute_improvement"
                    ]
                ),
                "governed_test_used": interval_challenger_report["governance"][
                    "governed_test_used"
                ],
                "primary_artifact_written": interval_challenger_report[
                    "governance"
                ]["primary_artifact_written"],
            }
            if interval_challenger_report
            else None
        ),
        "largest_absolute_errors_anonymised": [
            {
                "actual": float(actual[position]),
                "predicted": float(predicted[position]),
                "absolute_error": float(abs(predicted[position] - actual[position])),
                "room_type": records[int(test_indices[position])]["room_type"],
                "property_type": records[int(test_indices[position])]["property_type"],
                "neighbourhood": records[int(test_indices[position])]["neighbourhood"],
            }
            for position in worst_positions
        ],
        "governance": {
            "listing_or_host_ids_emitted": False,
            "slice_findings_change_model_or_gate": False,
            "next_required_evidence": (
                "Validate any candidate mitigation on development folds and a "
                "future compatible temporal snapshot."
            ),
        },
    }
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "price_band_mae.svg").write_text(
        price_band_error_svg(groups_audit["price_band"]), encoding="utf-8"
    )
    (asset_dir / "observed_vs_predicted.svg").write_text(
        observed_predicted_svg(actual, predicted), encoding="utf-8"
    )
    write_json_atomic(report_path, report)
    write_model_card(model_card_path, report, asset_dir)
    print(f"error MAE {overall['model']['mae']:.2f}; flags {len(flags)}")
    print(f"report    {report_path}")
    print(f"modelcard {model_card_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-card", type=Path, default=DEFAULT_MODEL_CARD)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_error_analysis(
        args.silver,
        args.artifact,
        args.report,
        args.model_card,
        args.assets,
    )


if __name__ == "__main__":
    main()
