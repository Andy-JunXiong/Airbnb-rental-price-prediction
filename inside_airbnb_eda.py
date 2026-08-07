"""Generate a reproducible modern EDA pack from the governed Silver quote table."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from inside_airbnb_phase0 import (
    ROOT,
    active_snapshot_date,
    utc_now,
    write_json_atomic,
)
from inside_airbnb_quote_model import (
    CATEGORICAL_FEATURES,
    DEFAULT_SILVER,
    NUMERIC_FEATURES,
    TARGET,
    finite_float,
)


DEFAULT_JSON = (
    ROOT
    / "reports"
    / "inside_airbnb"
    / f"sydney_{active_snapshot_date()}_modern_eda.json"
)
DEFAULT_MARKDOWN = ROOT / "docs" / "inside_airbnb_modern_eda.md"
DEFAULT_ASSET_DIR = ROOT / "reports" / "inside_airbnb" / "eda_assets"
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def numeric_values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    values = [finite_float(row.get(field)) for row in rows]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=float)


def numeric_summary(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    values = numeric_values(rows, field)
    result: dict[str, Any] = {
        "rows": len(rows),
        "non_missing": int(len(values)),
        "missing": len(rows) - int(len(values)),
        "missing_rate": (len(rows) - len(values)) / len(rows) if rows else None,
    }
    if not len(values):
        return result
    result.update(
        {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            **{
                f"p{int(probability * 100):02d}": float(
                    np.quantile(values, probability)
                )
                for probability in QUANTILES
            },
            "max": float(np.max(values)),
        }
    )
    return result


def categorical_summary(
    rows: list[dict[str, str]], field: str, top_n: int = 20
) -> dict[str, Any]:
    counts = Counter((row.get(field) or "").strip() or "(missing)" for row in rows)
    total = len(rows)
    return {
        "unique_values": len(counts),
        "missing": counts.get("(missing)", 0),
        "top_values": [
            {
                "value": value,
                "rows": count,
                "share": count / total if total else None,
            }
            for value, count in counts.most_common(top_n)
        ],
    }


def paired_correlation(
    rows: list[dict[str, str]], field: str
) -> dict[str, float | int | None]:
    pairs = []
    for row in rows:
        value = finite_float(row.get(field))
        target = finite_float(row.get(TARGET))
        if math.isfinite(value) and math.isfinite(target) and target > 0:
            pairs.append((value, math.log1p(target)))
    if len(pairs) < 3:
        return {"rows": len(pairs), "pearson_with_log1p_target": None}
    matrix = np.asarray(pairs, dtype=float)
    if float(np.std(matrix[:, 0])) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1])
    return {"rows": len(pairs), "pearson_with_log1p_target": correlation}


def segment_price_summary(
    rows: list[dict[str, str]], field: str, minimum_rows: int = 20
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = (row.get(field) or "").strip() or "(missing)"
        target = finite_float(row.get(TARGET))
        if math.isfinite(target):
            grouped[value].append(target)
    result = []
    for value, prices in grouped.items():
        if len(prices) < minimum_rows:
            continue
        array = np.asarray(prices, dtype=float)
        result.append(
            {
                field: value,
                "rows": len(prices),
                "median_price": float(np.median(array)),
                "p25_price": float(np.quantile(array, 0.25)),
                "p75_price": float(np.quantile(array, 0.75)),
            }
        )
    return sorted(result, key=lambda item: (-item["rows"], str(item[field])))


def svg_document(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        f"{body}</svg>\n"
    )


def price_histogram_svg(prices: np.ndarray) -> str:
    cap = float(np.quantile(prices, 0.99))
    clipped = prices[prices <= cap]
    counts, edges = np.histogram(clipped, bins=24, range=(0, cap))
    width, height = 900, 480
    left, top, plot_width, plot_height = 80, 70, 760, 320
    maximum = max(int(np.max(counts)), 1)
    bars = []
    for index, count in enumerate(counts):
        bar_width = plot_width / len(counts) - 2
        bar_height = plot_height * int(count) / maximum
        x = left + index * plot_width / len(counts)
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#287271"/>'
        )
    body = (
        '<text x="40" y="35" font-family="sans-serif" font-size="22" '
        'font-weight="700" fill="#17252a">Quoted nightly price distribution</text>'
        f'<text x="40" y="58" font-family="sans-serif" font-size="13" '
        f'fill="#526066">Displayed through p99 = AUD {cap:,.0f}; '
        f'{len(prices) - len(clipped):,} higher values excluded from the chart.</text>'
        + "".join(bars)
        + f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        'y2="390" stroke="#17252a"/>'
        f'<text x="{left}" y="420" font-family="sans-serif" font-size="12">AUD 0</text>'
        f'<text x="{left + plot_width - 80}" y="420" font-family="sans-serif" '
        f'font-size="12">AUD {cap:,.0f}</text>'
    )
    return svg_document(width, height, body, "Quoted nightly price distribution")


def horizontal_bar_svg(
    rows: list[dict[str, Any]],
    label_field: str,
    value_field: str,
    title: str,
    subtitle: str,
    limit: int = 12,
) -> str:
    selected = rows[:limit]
    width = 900
    row_height = 34
    height = 110 + max(1, len(selected)) * row_height
    left, bar_left, bar_width = 30, 260, 560
    maximum = max((float(row[value_field]) for row in selected), default=1.0)
    items = []
    for index, row in enumerate(selected):
        y = 88 + index * row_height
        value = float(row[value_field])
        label = html.escape(str(row[label_field]))
        items.append(
            f'<text x="{left}" y="{y + 16}" font-family="sans-serif" '
            f'font-size="12" fill="#17252a">{label[:34]}</text>'
            f'<rect x="{bar_left}" y="{y}" width="{bar_width * value / maximum:.1f}" '
            'height="22" rx="3" fill="#d97b29"/>'
            f'<text x="{bar_left + bar_width * value / maximum + 8:.1f}" '
            f'y="{y + 16}" font-family="sans-serif" font-size="12" '
            f'fill="#17252a">{value:,.2f}</text>'
        )
    body = (
        f'<text x="30" y="34" font-family="sans-serif" font-size="22" '
        f'font-weight="700" fill="#17252a">{html.escape(title)}</text>'
        f'<text x="30" y="58" font-family="sans-serif" font-size="13" '
        f'fill="#526066">{html.escape(subtitle)}</text>'
        + "".join(items)
    )
    return svg_document(width, height, body, title)


def spatial_scatter_svg(rows: list[dict[str, str]], maximum_points: int = 3000) -> str:
    points = []
    for row in rows:
        lat = finite_float(row.get("latitude"))
        lon = finite_float(row.get("longitude"))
        price = finite_float(row.get(TARGET))
        if math.isfinite(lat) and math.isfinite(lon) and price > 0:
            points.append((lat, lon, price))
    if len(points) > maximum_points:
        step = len(points) / maximum_points
        points = [points[min(int(index * step), len(points) - 1)] for index in range(maximum_points)]
    array = np.asarray(points, dtype=float)
    width, height = 760, 720
    left, top, plot_width, plot_height = 65, 75, 630, 570
    min_lat, max_lat = float(np.min(array[:, 0])), float(np.max(array[:, 0]))
    min_lon, max_lon = float(np.min(array[:, 1])), float(np.max(array[:, 1]))
    price_low = float(np.quantile(array[:, 2], 0.05))
    price_high = float(np.quantile(array[:, 2], 0.95))
    rendered = []
    for lat, lon, price in points:
        x = left + (lon - min_lon) / max(max_lon - min_lon, 1e-9) * plot_width
        y = top + (max_lat - lat) / max(max_lat - min_lat, 1e-9) * plot_height
        scaled = min(1.0, max(0.0, (math.log1p(price) - math.log1p(price_low)) / max(
            math.log1p(price_high) - math.log1p(price_low), 1e-9
        )))
        red = int(40 + 205 * scaled)
        blue = int(170 - 130 * scaled)
        rendered.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" '
            f'fill="rgb({red},90,{blue})" fill-opacity="0.45"/>'
        )
    body = (
        '<text x="30" y="34" font-family="sans-serif" font-size="22" '
        'font-weight="700" fill="#17252a">Spatial quote-price pattern</text>'
        f'<text x="30" y="58" font-family="sans-serif" font-size="13" '
        f'fill="#526066">{len(points):,} deterministic sample points; colour scales '
        'from lower (blue) to higher (red) log-price.</text>'
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="#eef2ef" stroke="#aab5b1"/>'
        + "".join(rendered)
        + f'<text x="{left}" y="{height - 35}" font-family="sans-serif" '
        f'font-size="12">Longitude {min_lon:.2f} to {max_lon:.2f}</text>'
    )
    return svg_document(width, height, body, "Spatial quote-price pattern")


def write_markdown(path: Path, report: dict[str, Any], asset_dir: Path) -> None:
    target = report["numeric_features"][TARGET]
    room_rows = report["segment_prices"]["room_type"]
    strongest = report["numeric_correlations_with_log_target"][:5]
    relative_assets = Path("..") / asset_dir.relative_to(ROOT)
    lines = [
        "# Inside Airbnb Sydney modern EDA",
        "",
        f"Generated from the governed Silver table. Snapshot: `{report['snapshot_label']}`.",
        "",
        "## Executive findings",
        "",
        f"- {report['rows']['eligible']:,} of {report['rows']['all']:,} rows are training eligible.",
        f"- Median quoted nightly price is AUD {target['p50']:,.0f}; p95 is AUD {target['p95']:,.0f} and p99 is AUD {target['p99']:,.0f}.",
        f"- There are {report['hosts']['unique_hosts']:,} unique hosts; the largest host contributes {report['hosts']['largest_host_rows']:,} listings.",
        "- The target is strongly right-skewed, so modelling and correlation diagnostics use `log1p(price)`.",
        "- Availability and review-velocity columns remain labelled proxies and are excluded from the primary model.",
        "",
        "## Charts",
        "",
        f"![Price distribution]({(relative_assets / 'price_distribution.svg').as_posix()})",
        "",
        f"![Room type median prices]({(relative_assets / 'room_type_median_price.svg').as_posix()})",
        "",
        f"![Numeric correlations]({(relative_assets / 'numeric_correlations.svg').as_posix()})",
        "",
        f"![Spatial pattern]({(relative_assets / 'spatial_price_pattern.svg').as_posix()})",
        "",
        "## Room-type segments",
        "",
        "| Room type | Rows | Median AUD | IQR AUD |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['room_type']} | {row['rows']:,} | {row['median_price']:,.0f} | "
        f"{row['p25_price']:,.0f}–{row['p75_price']:,.0f} |"
        for row in room_rows
    )
    lines.extend(
        [
            "",
            "## Strongest numeric associations",
            "",
            "These are descriptive Pearson correlations with `log1p(price)`, not causal effects.",
            "",
            "| Feature | Complete rows | Correlation |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{row['feature']}` | {row['rows']:,} | "
        f"{row['pearson_with_log1p_target']:.3f} |"
        for row in strongest
        if row["pearson_with_log1p_target"] is not None
    )
    lines.extend(
        [
            "",
            "## Method and boundaries",
            "",
            "- Numeric summaries report explicit missingness and robust quantiles.",
            "- Segment tables suppress groups with fewer than 20 rows.",
            "- Spatial coordinates in Inside Airbnb are approximate; the map is a diagnostic, not parcel-level evidence.",
            "- This report does not use the governed model test split and does not select a production model.",
            "",
            f"Machine-readable details: `{report['json_path']}`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_eda(
    silver_path: Path,
    json_path: Path,
    markdown_path: Path,
    asset_dir: Path,
) -> dict[str, Any]:
    with silver_path.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    eligible = [row for row in all_rows if row["training_eligible"] == "1"]
    if not eligible:
        raise ValueError("EDA requires at least one training-eligible quote")
    numeric_fields = list(dict.fromkeys([TARGET, *NUMERIC_FEATURES]))
    numeric = {field: numeric_summary(eligible, field) for field in numeric_fields}
    correlations = [
        {"feature": field, **paired_correlation(eligible, field)}
        for field in NUMERIC_FEATURES
    ]
    correlations.sort(
        key=lambda row: abs(row["pearson_with_log1p_target"] or 0), reverse=True
    )
    host_counts = Counter(row["host_id"] for row in eligible)
    report: dict[str, Any] = {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "snapshot_label": eligible[0]["snapshot_label"],
        "source": str(silver_path.relative_to(ROOT)),
        "json_path": str(json_path.relative_to(ROOT)),
        "markdown_path": str(markdown_path.relative_to(ROOT)),
        "rows": {"all": len(all_rows), "eligible": len(eligible)},
        "hosts": {
            "unique_hosts": len(host_counts),
            "largest_host_rows": max(host_counts.values()),
            "hosts_with_multiple_listings": sum(
                count > 1 for count in host_counts.values()
            ),
        },
        "numeric_features": numeric,
        "categorical_features": {
            field: categorical_summary(eligible, field)
            for field in CATEGORICAL_FEATURES
        },
        "numeric_correlations_with_log_target": correlations,
        "segment_prices": {
            field: segment_price_summary(eligible, field)
            for field in ("room_type", "neighbourhood", "property_type")
        },
        "governance": {
            "descriptive_not_causal": True,
            "model_test_split_used": False,
            "excluded_primary_model_proxies": [
                "availability_30/60/90/365",
                "review counts and review velocity",
            ],
        },
    }
    asset_dir.mkdir(parents=True, exist_ok=True)
    prices = numeric_values(eligible, TARGET)
    (asset_dir / "price_distribution.svg").write_text(
        price_histogram_svg(prices), encoding="utf-8"
    )
    room_rows = report["segment_prices"]["room_type"]
    room_by_price = sorted(room_rows, key=lambda row: -row["median_price"])
    (asset_dir / "room_type_median_price.svg").write_text(
        horizontal_bar_svg(
            room_by_price,
            "room_type",
            "median_price",
            "Median quoted price by room type",
            "Only segments with at least 20 eligible listings; values are AUD.",
        ),
        encoding="utf-8",
    )
    correlation_rows = [
        {
            "feature": row["feature"],
            "absolute_correlation": abs(row["pearson_with_log1p_target"]),
        }
        for row in correlations
        if row["pearson_with_log1p_target"] is not None
    ]
    (asset_dir / "numeric_correlations.svg").write_text(
        horizontal_bar_svg(
            correlation_rows,
            "feature",
            "absolute_correlation",
            "Numeric association with log quoted price",
            "Absolute Pearson correlation; descriptive and not causal.",
        ),
        encoding="utf-8",
    )
    (asset_dir / "spatial_price_pattern.svg").write_text(
        spatial_scatter_svg(eligible), encoding="utf-8"
    )
    write_json_atomic(json_path, report)
    write_markdown(markdown_path, report, asset_dir)
    print(f"EDA JSON {json_path}")
    print(f"EDA docs {markdown_path}")
    print(f"charts   {asset_dir}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_eda(args.silver, args.json, args.markdown, args.assets)


if __name__ == "__main__":
    main()
