"""Generate human-readable explanations for quote-model predictions.

Two modes:
- **Template mode** (always available): Uses prediction context, market
  comparisons, and feature-level heuristics to explain a prediction.
- **LLM mode** (requires ollama or openai): Sends structured context to
  a local LLM for more natural, conversational explanations.

The LLM mode is designed for local inference via Ollama (e.g., llama3.2,
qwen2.5) with a fallback to any OpenAI-compatible endpoint. The prompt
constrains the model to factual output — no hallucinated prices or features.
"""

from __future__ import annotations

import json
import math
from typing import Any

from sydney_geography import (
    MAJOR_HUBS,
    REFERENCE_BEACHES,
    SYDNEY_AIRPORT,
    SYDNEY_CBD,
)

# ---------------------------------------------------------------------------
# Template-mode explanation
# ---------------------------------------------------------------------------

AUD_FORMAT = "${:,.0f} AUD"
PCT_FORMAT = "{:.1%}"


def _fmt_aud(value: float) -> str:
    return f"AUD {value:,.0f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _price_band_label(price: float, supported_range: list[float]) -> str:
    low, high = supported_range
    if price < low:
        return "below the typical market range"
    if price <= (low + high) / 3:
        return "budget-friendly"
    if price <= 2 * (low + high) / 3:
        return "mid-range"
    if price <= high:
        return "premium"
    return "luxury-tier (above the 99.5th percentile)"


def _comparable_blurb(count: int, level: str) -> str:
    if count >= 100:
        return f"based on {count} very similar listings"
    if count >= 50:
        return f"based on {count} similar listings"
    if count >= 20:
        return f"based on {count} listings in the same area and room type"
    return "based on broader market data (few direct comparables)"


def _nearest_anchor(latitude: float, longitude: float) -> dict[str, Any]:
    """Find the nearest CBD, airport, beach, and hub with distances."""
    from sydney_geography import haversine_km

    cbd_dist = haversine_km(latitude, longitude, SYDNEY_CBD[1], SYDNEY_CBD[2])
    airport_dist = haversine_km(
        latitude, longitude, SYDNEY_AIRPORT[1], SYDNEY_AIRPORT[2]
    )
    beaches = [
        (name, haversine_km(latitude, longitude, lat, lon))
        for name, lat, lon in REFERENCE_BEACHES
    ]
    nearest_beach = min(beaches, key=lambda x: x[1])
    hubs = [
        (name, haversine_km(latitude, longitude, lat, lon))
        for name, lat, lon in MAJOR_HUBS
    ]
    nearest_hub = min(hubs, key=lambda x: x[1])
    return {
        "cbd_km": round(cbd_dist, 1),
        "airport_km": round(airport_dist, 1),
        "nearest_beach": nearest_beach[0],
        "nearest_beach_km": round(nearest_beach[1], 1),
        "nearest_hub": nearest_hub[0],
        "nearest_hub_km": round(nearest_hub[1], 1),
    }


def explain_prediction_template(
    result: dict[str, Any],
    row: dict[str, Any],
    artifact: dict[str, Any],
) -> str:
    """Generate a structured markdown explanation without an LLM."""

    predicted = float(result["estimated_price"] or 0)
    lower = float((result.get("prediction_interval") or [0, 0])[0])
    upper = float((result.get("prediction_interval") or [0, 0])[1])
    comparable_count = int(result.get("comparable_count", 0))
    comparable_level = str(result.get("comparable_level", ""))

    market = artifact["market_baseline"]
    exact_key = (
        str(row.get("neighbourhood", "")),
        str(row.get("room_type", "")),
    )
    market_median = market["exact"].get(exact_key, {}).get("median")
    if market_median is None:
        market_median = market["global"]
    premium_vs_market = (predicted - market_median) / max(market_median, 1.0)

    price_range = artifact.get("supported_price_range", [50.0, 800.0])
    band = _price_band_label(predicted, price_range)

    lat = float(row.get("latitude", 0) or 0)
    lon = float(row.get("longitude", 0) or 0)
    anchors = _nearest_anchor(lat, lon) if lat and lon else None

    status = result.get("status", "ok")
    if status == "refused":
        reasons = result.get("refusal_reasons", [])
        reason_text = "\n".join(f"- `{r}`" for r in reasons)
        return (
            f"## Prediction Refused\n\n"
            f"The prediction was refused for the following reasons:\n\n"
            f"{reason_text}\n\n"
            f"Please address these issues before requesting a new prediction."
        )

    lines = [
        "## Price Prediction Explanation",
        "",
        f"**Predicted nightly price: {_fmt_aud(predicted)}**",
        f"90% confidence interval: {_fmt_aud(lower)} – {_fmt_aud(upper)}",
        "",
        "### Market Comparison",
        "",
        f"- This listing is priced {_fmt_pct(abs(premium_vs_market))} "
        f"{'above' if premium_vs_market > 0 else 'below'} the "
        f"neighbourhood median of {_fmt_aud(market_median)}.",
        f"- It falls in the **{band}** category.",
        f"- The estimate is {_comparable_blurb(comparable_count, comparable_level)}.",
        "",
    ]

    if anchors:
        lines.extend([
            "### Location Context",
            "",
            f"- {anchors['cbd_km']} km from Sydney CBD",
            f"- {anchors['airport_km']} km from Sydney Airport",
            f"- {anchors['nearest_beach_km']} km from {anchors['nearest_beach']} (nearest reference beach)",
            f"- {anchors['nearest_hub_km']} km from {anchors['nearest_hub']} (nearest transport hub)",
            "",
        ])

    lines.extend([
        "### Listing Details",
        "",
        f"- Room type: **{row.get('room_type', 'unknown')}**",
        f"- Property type: **{row.get('property_type', 'unknown')}**",
        f"- Neighbourhood: **{row.get('neighbourhood', 'unknown')}**",
        f"- Accommodates: {row.get('accommodates', '?')} guests",
        f"- Bedrooms: {row.get('bedrooms', '?')}, Bathrooms: {row.get('bathrooms', '?')}",
        f"- Stay: {row.get('stay_nights', '?')} nights, booked {row.get('quote_lead_days', '?')} days ahead",
        "",
    ])

    # Warnings
    warnings = []
    if result.get("snapshot_staleness_warning"):
        warnings.append(
            f"Snapshot is {result.get('snapshot_age_days', '?')} days old — consider refreshing."
        )
    if result.get("authority_warning"):
        warnings.append(result["authority_warning"])
    if result.get("evidence_level") == "low":
        warnings.append(
            "Low confidence: limited comparable listings in this area and room type."
        )

    if warnings:
        lines.append("### ⚠️ Warnings")
        lines.append("")
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")

    lines.extend([
        "### Limitations",
        "",
        "- This is a research prototype, not a pricing recommendation.",
        "- The model predicts quoted (listed) price, not realised booking revenue.",
        "- Geographic distances are straight-line, not driving or transit times.",
        "- Conditional coverage is not guaranteed even when marginal coverage is near 90%.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-mode explanation
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM_PROMPT = (
    "You are a helpful data-science assistant explaining an Airbnb price "
    "prediction to a non-technical user. "
    "Be concise and factual. Use only the information provided. "
    "Never invent features, amenities, or statistics that are not in the context. "
    "If the prediction was refused, explain why clearly. "
    "Always include the predicted price and confidence interval. "
    "Keep your response under 200 words."
)

EXPLAIN_TEMPLATE_CONTEXT = """Prediction context:
- Listing: {room_type} {property_type} in {neighbourhood}
- Accommodates: {accommodates} guests, {bedrooms} bedrooms, {bathrooms} bathrooms
- Stay: {stay_nights} nights, booked {lead_days} days in advance
- Location: {cbd_dist} km from Sydney CBD, {beach_dist} km from {beach_name}

Prediction:
- Predicted nightly price: {predicted_aud}
- 90% confidence interval: {lower_aud} – {upper_aud}
- Market median for this area/room type: {market_aud}
- Based on {comparable_count} similar listings

Evidence quality: {evidence_level}
Status: {status}
"""


def llm_client() -> Any | None:
    """Return an OpenAI-compatible client if available, else None."""
    try:
        from openai import OpenAI

        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    except ImportError:
        pass
    return None


def explain_prediction_llm(
    result: dict[str, Any],
    row: dict[str, Any],
    artifact: dict[str, Any],
    model: str = "llama3.2",
) -> str | None:
    """Attempt LLM explanation. Returns None if LLM is unavailable."""
    client = llm_client()
    if client is None:
        try:
            import openai
        except ImportError:
            return None

    predicted = float(result.get("estimated_price") or 0)
    interval = result.get("prediction_interval") or [0, 0]
    lower = float(interval[0])
    upper = float(interval[1])

    market = artifact["market_baseline"]
    exact_key = (
        str(row.get("neighbourhood", "")),
        str(row.get("room_type", "")),
    )
    market_median = market["exact"].get(exact_key, {}).get("median")
    if market_median is None:
        market_median = market["global"]

    lat = float(row.get("latitude", 0) or 0)
    lon = float(row.get("longitude", 0) or 0)
    anchors = _nearest_anchor(lat, lon) if lat and lon else None

    context = EXPLAIN_TEMPLATE_CONTEXT.format(
        room_type=row.get("room_type", "unknown"),
        property_type=row.get("property_type", "unknown"),
        neighbourhood=row.get("neighbourhood", "unknown"),
        accommodates=row.get("accommodates", "?"),
        bedrooms=row.get("bedrooms", "?"),
        bathrooms=row.get("bathrooms", "?"),
        stay_nights=row.get("stay_nights", "?"),
        lead_days=row.get("quote_lead_days", "?"),
        cbd_dist=anchors["cbd_km"] if anchors else "?",
        beach_dist=anchors["nearest_beach_km"] if anchors else "?",
        beach_name=anchors["nearest_beach"] if anchors else "?",
        predicted_aud=_fmt_aud(predicted),
        lower_aud=_fmt_aud(lower),
        upper_aud=_fmt_aud(upper),
        market_aud=_fmt_aud(market_median),
        comparable_count=result.get("comparable_count", 0),
        evidence_level=result.get("evidence_level", "unknown"),
        status=result.get("status", "ok"),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def explain_prediction(
    result: dict[str, Any],
    row: dict[str, Any],
    artifact: dict[str, Any],
    prefer_llm: bool = True,
    llm_model: str = "llama3.2",
) -> dict[str, str]:
    """Generate a prediction explanation in template or LLM mode.

    Returns a dict with 'mode' ('template' or 'llm') and 'text' (markdown).
    """
    if prefer_llm:
        llm_text = explain_prediction_llm(result, row, artifact, llm_model)
        if llm_text:
            return {"mode": "llm", "model": llm_model, "text": llm_text}

    return {"mode": "template", "text": explain_prediction_template(result, row, artifact)}
