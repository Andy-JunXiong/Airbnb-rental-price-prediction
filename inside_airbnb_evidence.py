"""Evidence-level policy for model predictions.

Assigns an interpretable evidence tier to every prediction based on
predeclared, verifiable criteria — not black-box confidence scores.

Tiers (from strongest to weakest):
    HIGH   — Well-supported by comparable listings, narrow interval, in-distribution.
    MEDIUM — Usable as a reference point; moderate uncertainty.
    LOW    — Research estimate only; limited supporting evidence.
    REFUSE — Cannot estimate responsibly; see refusal reasons.

This module implements §4 of the serving review (Uncertainty as Product).
"""

from __future__ import annotations

from typing import Any

# Thresholds
MIN_COMPARABLES_HIGH = 100
MIN_COMPARABLES_MEDIUM = 20
MAX_RELATIVE_WIDTH_HIGH = 1.0
MAX_RELATIVE_WIDTH_MEDIUM = 2.0
UPPER_TAIL_PERCENTILE = 0.90


def _relative_interval_width(lower: float, upper: float, price: float) -> float:
    """Return interval width as a multiple of the predicted price."""
    if price <= 0:
        return float("inf")
    return (upper - lower) / price


def _is_upper_tail(
    price: float, training_prices: list[float], quantile: float = UPPER_TAIL_PERCENTILE
) -> bool:
    """Check if a predicted price falls in the upper tail of training prices."""
    if not training_prices:
        return False
    import numpy as np

    threshold = float(np.quantile(training_prices, quantile))
    return price > threshold


def assess_evidence(
    estimated_price: float,
    prediction_interval: tuple[float, float] | None,
    comparable_count: int,
    snapshot_age_days: int,
    deployment_authority: str,
    refusal_reasons: list[str],
    supported_price_range: list[float] | None = None,
    training_price_quantiles: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return an evidence tier with supporting details.

    Args:
        estimated_price: The predicted price (or 0 if refused).
        prediction_interval: (lower, upper) or None if refused.
        comparable_count: Number of comparable listings used.
        snapshot_age_days: Days since the training snapshot.
        deployment_authority: e.g. 'research_only' or 'temporally_validated'.
        refusal_reasons: Reasons the prediction was refused (empty means ok).
        supported_price_range: [p01, p99] of training prices.
        training_price_quantiles: Optional dict with 'p50', 'p90', 'p95', 'p99'.

    Returns:
        dict with keys: tier, tier_label, reasons, recommendation.
    """
    # --- REFUSE ---
    if refusal_reasons:
        return {
            "tier": "REFUSE",
            "tier_label": "Cannot estimate responsibly",
            "reasons": refusal_reasons,
            "recommendation": (
                "Address the refusal reasons before requesting a new prediction."
            ),
            "comparable_count": comparable_count,
            "relative_interval_width": None,
        }

    # --- Compute evidence signals ---
    lower, upper = prediction_interval or (0, 0)
    rel_width = _relative_interval_width(lower, upper, estimated_price)

    signals = {
        "comparable_count": comparable_count,
        "relative_interval_width": round(rel_width, 2),
        "snapshot_age_days": snapshot_age_days,
        "deployment_authority": deployment_authority,
    }

    # --- Tier assignment ---
    tier = "HIGH"
    reasons: list[str] = []

    if comparable_count < MIN_COMPARABLES_MEDIUM:
        tier = "LOW"
        reasons.append(
            f"Fewer than {MIN_COMPARABLES_MEDIUM} comparable listings "
            f"({comparable_count} available)"
        )
    elif comparable_count < MIN_COMPARABLES_HIGH:
        tier = "MEDIUM"
        reasons.append(
            f"Moderate comparables ({comparable_count}, threshold {MIN_COMPARABLES_HIGH})"
        )

    if rel_width > MAX_RELATIVE_WIDTH_MEDIUM:
        tier = "LOW"
        reasons.append(
            f"Very wide prediction interval ({rel_width:.1f}x the predicted price)"
        )
    elif rel_width > MAX_RELATIVE_WIDTH_HIGH and tier == "HIGH":
        tier = "MEDIUM"
        reasons.append(f"Moderate interval width ({rel_width:.1f}x the predicted price)")

    if snapshot_age_days > 90:
        if tier == "HIGH":
            tier = "MEDIUM"
        elif tier == "MEDIUM":
            tier = "LOW"
        reasons.append(f"Training snapshot is {snapshot_age_days} days old")

    # Upper-tail listings get downgraded one tier
    if training_price_quantiles and estimated_price > training_price_quantiles.get(
        "p90", float("inf")
    ):
        reasons.append(
            f"Premium listing (above p90 of training prices); "
            "upper-tail calibration is weaker"
        )
        if tier == "HIGH":
            tier = "MEDIUM"
        elif tier == "MEDIUM":
            tier = "LOW"

    # Research-only authority caps at MEDIUM
    if deployment_authority != "temporally_validated" and tier == "HIGH":
        tier = "MEDIUM"
        reasons.append(
            "Model authority is research_only — temporal validation is pending"
        )

    tier_labels = {
        "HIGH": "Well-supported estimate",
        "MEDIUM": "Use as a reference point",
        "LOW": "Research estimate only — limited supporting evidence",
    }

    recommendations = {
        "HIGH": "The estimate is supported by ample comparable data and a narrow interval.",
        "MEDIUM": "The estimate is usable as a reference but has moderate uncertainty. Consider additional market research for high-stakes decisions.",
        "LOW": "This is a research-grade estimate. Obtain manual market comparables before relying on it for any financial decision.",
    }

    return {
        "tier": tier,
        "tier_label": tier_labels.get(tier, tier),
        "reasons": reasons,
        "recommendation": recommendations.get(tier, ""),
        **signals,
    }
