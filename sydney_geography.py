"""Transparent, offline geographic features for the Sydney quote model.

The reference points are deliberately small, versioned, and auditable.  They
are approximate straight-line anchors, not claims about walking, driving, or
public-transport travel time.
"""

from __future__ import annotations

import math
from typing import Any


REFERENCE_VERSION = 1
EARTH_RADIUS_KM = 6371.0088

SYDNEY_CBD = ("Sydney CBD", -33.8688, 151.2093)
SYDNEY_AIRPORT = ("Sydney Airport", -33.9399, 151.1753)

REFERENCE_BEACHES = (
    ("Bondi Beach", -33.8915, 151.2767),
    ("Coogee Beach", -33.9205, 151.2550),
    ("Manly Beach", -33.7969, 151.2871),
    ("Cronulla Beach", -34.0550, 151.1540),
    ("Palm Beach", -33.5988, 151.3233),
)

MAJOR_HUBS = (
    ("Central", -33.8830, 151.2065),
    ("Circular Quay", -33.8612, 151.2108),
    ("Chatswood", -33.7972, 151.1802),
    ("Parramatta", -33.8175, 151.0034),
    ("Bondi Junction", -33.8910, 151.2477),
)

GEOGRAPHIC_FEATURES = (
    "distance_to_sydney_cbd_km",
    "distance_to_sydney_airport_km",
    "distance_to_nearest_reference_beach_km",
    "distance_to_nearest_major_hub_km",
)


def _finite_coordinate(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return great-circle distance in kilometres."""

    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(value)))


def nearest_anchor_distance_km(
    latitude: float,
    longitude: float,
    anchors: tuple[tuple[str, float, float], ...],
) -> float:
    return min(
        haversine_km(latitude, longitude, anchor_lat, anchor_lon)
        for _, anchor_lat, anchor_lon in anchors
    )


def geographic_features(latitude: Any, longitude: Any) -> dict[str, float | str]:
    """Create reproducible distance features, or blanks for invalid coordinates."""

    lat = _finite_coordinate(latitude)
    lon = _finite_coordinate(longitude)
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return {name: "" for name in GEOGRAPHIC_FEATURES}
    return {
        "distance_to_sydney_cbd_km": round(
            haversine_km(lat, lon, SYDNEY_CBD[1], SYDNEY_CBD[2]), 6
        ),
        "distance_to_sydney_airport_km": round(
            haversine_km(lat, lon, SYDNEY_AIRPORT[1], SYDNEY_AIRPORT[2]), 6
        ),
        "distance_to_nearest_reference_beach_km": round(
            nearest_anchor_distance_km(lat, lon, REFERENCE_BEACHES), 6
        ),
        "distance_to_nearest_major_hub_km": round(
            nearest_anchor_distance_km(lat, lon, MAJOR_HUBS), 6
        ),
    }


def reference_manifest() -> dict[str, Any]:
    def render(anchor: tuple[str, float, float]) -> dict[str, Any]:
        return {
            "name": anchor[0],
            "latitude": anchor[1],
            "longitude": anchor[2],
        }

    return {
        "version": REFERENCE_VERSION,
        "distance_method": "Haversine great-circle distance",
        "units": "kilometres",
        "interpretation": (
            "Approximate straight-line distance to fixed reference points; "
            "not route distance or travel time."
        ),
        "cbd": render(SYDNEY_CBD),
        "airport": render(SYDNEY_AIRPORT),
        "reference_beaches": [render(anchor) for anchor in REFERENCE_BEACHES],
        "major_hubs": [render(anchor) for anchor in MAJOR_HUBS],
    }
