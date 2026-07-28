"""Privacy-minimised semantic features for upper-tail listing experiments."""

from __future__ import annotations

import json
import math
import re
from typing import Any


PREMIUM_NUMERIC_FEATURES = [
    "premium_amenities_count",
    "premium_amenity_density",
    "has_pool",
    "has_hot_tub",
    "has_waterfront",
    "has_beach_access",
    "has_water_view",
    "has_on_premises_parking",
    "has_gym",
    "has_sauna",
    "has_indoor_fireplace",
    "has_private_outdoor_space",
    "bathrooms_per_guest",
    "bedrooms_per_guest",
    "beds_per_guest",
    "accommodates_per_bedroom",
    "bedroom_bathroom_interaction",
]
PREMIUM_CATEGORICAL_FEATURES = ["bathroom_privacy", "property_group"]
PREMIUM_FIELDS = PREMIUM_NUMERIC_FEATURES + PREMIUM_CATEGORICAL_FEATURES

WATER_VIEW_TERMS = (
    "ocean view",
    "sea view",
    "harbor view",
    "bay view",
    "beach view",
    "canal view",
    "lake view",
    "river view",
    "marina view",
)


def parse_amenities(value: str | None) -> list[str]:
    try:
        parsed = json.loads((value or "").strip() or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip().lower() for item in parsed if str(item).strip()]


def has_phrase(amenities: list[str], phrase: str) -> int:
    return int(any(phrase in amenity for amenity in amenities))


def has_pool(amenities: list[str]) -> int:
    return int(
        any(
            re.search(r"\bpool\b", amenity)
            and not any(
                excluded in amenity
                for excluded in ("pool table", "pool view", "pool house", "pool speaker")
            )
            for amenity in amenities
        )
    )


def finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def safe_ratio(numerator: Any, denominator: Any) -> float | str:
    top = finite_number(numerator)
    bottom = finite_number(denominator)
    if top is None or bottom is None or bottom <= 0:
        return ""
    return round(top / bottom, 6)


def bathroom_privacy(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return "missing"
    if "shared" in cleaned:
        return "shared"
    if "private" in cleaned:
        return "private"
    return "exclusive_or_unspecified"


def property_group(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    if any(
        term in cleaned
        for term in ("hotel", "aparthotel", "hostel", "resort", "bed and breakfast")
    ):
        return "hospitality"
    if any(
        term in cleaned
        for term in (
            "home",
            "house",
            "townhouse",
            "villa",
            "cottage",
            "bungalow",
            "cabin",
            "vacation home",
            "chalet",
        )
    ):
        return "house"
    if any(
        term in cleaned
        for term in ("rental unit", "condo", "loft", "serviced apartment")
    ):
        return "apartment"
    if any(term in cleaned for term in ("guesthouse", "guest suite")):
        return "guest_space"
    if any(
        term in cleaned
        for term in (
            "boat",
            "tiny home",
            "treehouse",
            "island",
            "farm stay",
            "camper",
            "tent",
            "dome",
            "barn",
            "houseboat",
            "earthen",
        )
    ):
        return "unique_stay"
    return "other"


def premium_features(raw: dict[str, str]) -> dict[str, Any]:
    amenities = parse_amenities(raw.get("amenities"))
    flags = {
        "has_pool": has_pool(amenities),
        "has_hot_tub": has_phrase(amenities, "hot tub"),
        "has_waterfront": has_phrase(amenities, "waterfront"),
        "has_beach_access": has_phrase(amenities, "beach access"),
        "has_water_view": int(
            any(
                term in amenity
                for amenity in amenities
                for term in WATER_VIEW_TERMS
            )
        ),
        "has_on_premises_parking": int(
            any(
                "parking" in amenity and "on premises" in amenity
                for amenity in amenities
            )
        ),
        "has_gym": int(
            any(re.search(r"\bgym\b", amenity) for amenity in amenities)
        ),
        "has_sauna": has_phrase(amenities, "sauna"),
        "has_indoor_fireplace": has_phrase(amenities, "indoor fireplace"),
        "has_private_outdoor_space": int(
            any(
                phrase in amenity
                for amenity in amenities
                for phrase in ("private patio", "private balcony", "private backyard")
            )
        ),
    }
    premium_count = sum(flags.values())
    amenity_total = len(amenities)
    bedrooms = finite_number(raw.get("bedrooms"))
    bathrooms = finite_number(raw.get("bathrooms"))
    interaction: float | str = ""
    if bedrooms is not None and bathrooms is not None:
        interaction = round(bedrooms * bathrooms, 6)
    return {
        "premium_amenities_count": premium_count,
        "premium_amenity_density": (
            round(premium_count / amenity_total, 6) if amenity_total else ""
        ),
        **flags,
        "bathrooms_per_guest": safe_ratio(
            raw.get("bathrooms"), raw.get("accommodates")
        ),
        "bedrooms_per_guest": safe_ratio(
            raw.get("bedrooms"), raw.get("accommodates")
        ),
        "beds_per_guest": safe_ratio(raw.get("beds"), raw.get("accommodates")),
        "accommodates_per_bedroom": safe_ratio(
            raw.get("accommodates"), raw.get("bedrooms")
        ),
        "bedroom_bathroom_interaction": interaction,
        "bathroom_privacy": bathroom_privacy(raw.get("bathrooms_text")),
        "property_group": property_group(raw.get("property_type")),
    }
