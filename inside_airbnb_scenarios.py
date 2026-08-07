"""Model-based scenario comparison — what-if sensitivity without causal claims.

Usage:
    python inside_airbnb_scenarios.py --input examples/inside_airbnb_quote_request.json

The output shows the base estimate and model-implied differences for each scenario.
Important: these are model-based sensitivities, NOT causal uplift estimates.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import joblib

from inside_airbnb_evidence import assess_evidence
from inside_airbnb_phase0 import ROOT, utc_now, write_json_atomic
from inside_airbnb_quote_model import (
    DEFAULT_ARTIFACT,
    feature_row_from_request,
    predict_request,
)

DEFAULT_SCENARIOS = [
    {
        "name": "Extra bedroom (+1)",
        "changes": {"bedrooms": 1},
        "type": "additive",
    },
    {
        "name": "More amenities (+15)",
        "changes": {"amenities_count": 15},
        "type": "additive",
    },
    {
        "name": "Manly location",
        "changes": {"neighbourhood": "Manly", "latitude": -33.7969, "longitude": 151.2871},
        "type": "replacement",
    },
    {
        "name": "Bondi location",
        "changes": {"neighbourhood": "Bondi", "latitude": -33.8915, "longitude": 151.2767},
        "type": "replacement",
    },
    {
        "name": "Superhost status",
        "changes": {"host_is_superhost": "t"},
        "type": "replacement",
    },
]


def apply_changes(base: dict[str, Any], changes: dict[str, Any], change_type: str) -> dict[str, Any]:
    """Apply scenario changes to the base request payload."""
    scenario = copy.deepcopy(base)
    for key, value in changes.items():
        if change_type == "additive" and key in scenario and isinstance(scenario[key], (int, float)):
            scenario[key] = scenario[key] + value
        else:
            scenario[key] = value
    return scenario


def compare_scenarios(
    artifact: dict[str, Any],
    base_payload: dict[str, Any],
    scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run base prediction and scenario what-if comparisons.

    Returns a report with base prediction and per-scenario deltas.
    """
    scenarios = scenarios or DEFAULT_SCENARIOS

    base_result = predict_request(artifact, base_payload)
    base_price = base_result.get("estimated_price")
    base_interval = base_result.get("prediction_interval")
    base_comparables = base_result.get("comparable_count", 0)

    base_evidence = assess_evidence(
        estimated_price=base_price or 0,
        prediction_interval=(
            (base_interval[0], base_interval[1]) if base_interval else None
        ),
        comparable_count=base_comparables,
        snapshot_age_days=base_result.get("snapshot_age_days", 0),
        deployment_authority=artifact.get("deployment_authority", "research_only"),
        refusal_reasons=base_result.get("refusal_reasons", []),
    )

    comparisons = []
    for scenario in scenarios:
        changed = apply_changes(base_payload, scenario["changes"], scenario["type"])
        scenario_result = predict_request(artifact, changed)
        scenario_price = scenario_result.get("estimated_price")
        scenario_interval = scenario_result.get("prediction_interval")

        delta = None
        if base_price is not None and scenario_price is not None:
            delta = round(scenario_price - base_price, 2)

        comparisons.append({
            "name": scenario["name"],
            "changes": scenario["changes"],
            "type": scenario["type"],
            "estimated_price": scenario_price,
            "prediction_interval": scenario_interval,
            "delta_vs_base": delta,
            "delta_pct": (
                round(100 * delta / base_price, 1)
                if delta is not None and base_price and base_price > 0
                else None
            ),
            "status": scenario_result.get("status", "refused"),
        })

    return {
        "report_version": 1,
        "generated_at_utc": utc_now(),
        "disclaimer": (
            "Model-based scenario sensitivity, NOT estimated causal uplift. "
            "The model is observational and cannot establish that changing a "
            "feature will cause a specific price change."
        ),
        "base": {
            "estimated_price": base_price,
            "prediction_interval": base_interval,
            "evidence_tier": base_evidence["tier"],
            "evidence_tier_label": base_evidence["tier_label"],
            "comparable_count": base_comparables,
        },
        "scenarios": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--input", type=Path, required=True, help="Base quote request JSON")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.artifact)
    base_payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = compare_scenarios(artifact, base_payload)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")

    base = report["base"]
    print(f"\nBase: {base['estimated_price']:,.0f} AUD ({base['evidence_tier']})")
    for s in report["scenarios"]:
        if s["estimated_price"]:
            sign = "+" if (s["delta_vs_base"] or 0) >= 0 else ""
            print(
                f"  {s['name']}: {s['estimated_price']:,.0f} AUD "
                f"({sign}{s['delta_vs_base']:,.0f}, {s['delta_pct']}%)"
            )


if __name__ == "__main__":
    main()
