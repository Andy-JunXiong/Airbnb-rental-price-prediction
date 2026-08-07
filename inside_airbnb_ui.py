"""Streamlit UI for the Airbnb Sydney quote-price prediction model.

Run locally:
    pip install streamlit requests
    streamlit run inside_airbnb_ui.py

Or with the API server already running:
    streamlit run inside_airbnb_ui.py -- --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

try:
    import streamlit as st
except ImportError:
    print("streamlit is required. Install with: pip install streamlit")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
NEIGHBOURHOODS = [
    "Sydney", "Manly", "Bondi", "Surry Hills", "Newtown", "Paddington",
    "Darlinghurst", "Pyrmont", "Glebe", "Balmain", "Randwick", "Coogee",
    "Marrickville", "Camperdown", "Redfern", "Potts Point",
]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
PROPERTY_TYPES = [
    "Entire rental unit", "Entire condominium", "Entire home",
    "Entire townhouse", "Entire loft", "Private room in home",
    "Private room in rental unit", "Shared room in home",
]

# Predefined Sydney landmarks for quick placement
LANDMARKS = {
    "🏙️ Sydney CBD": (-33.8688, 151.2093),
    "🏖️ Manly Beach": (-33.7969, 151.2871),
    "🌊 Bondi Beach": (-33.8915, 151.2767),
    "✈️ Near Airport": (-33.9399, 151.1753),
    "🏛️ Central Station": (-33.8830, 151.2065),
    "🌉 Kirribilli": (-33.8480, 151.2170),
    "🏄 Coogee Beach": (-33.9205, 151.2550),
    "🌴 Palm Beach": (-33.5988, 151.3233),
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Airbnb Sydney Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — Listing parameters
# ---------------------------------------------------------------------------

st.sidebar.title("🏠 Listing Details")
st.sidebar.caption("Fill in the listing parameters and click Predict.")

neighbourhood = st.sidebar.selectbox("Neighbourhood", NEIGHBOURHOODS, index=0)
room_type = st.sidebar.selectbox("Room Type", ROOM_TYPES, index=0)
property_type = st.sidebar.selectbox("Property Type", PROPERTY_TYPES, index=0)

col1, col2 = st.sidebar.columns(2)
accommodates = col1.number_input("Guests", 1, 16, 2)
bedrooms = col2.number_input("Bedrooms", 0, 10, 1)

col3, col4 = st.sidebar.columns(2)
bathrooms = col3.number_input("Bathrooms", 0.0, 10.0, 1.0, 0.5)
beds = col4.number_input("Beds", 0, 20, 1)

col5, col6 = st.sidebar.columns(2)
stay_nights = col5.number_input("Stay (nights)", 1, 90, 2)
lead_days = col6.number_input("Booked ahead (days)", 0, 365, 14)

host_listings = st.sidebar.slider("Host's listing count", 1, 100, 1)
is_superhost = st.sidebar.checkbox("Superhost", value=False)
amenities = st.sidebar.slider("Amenities count", 0, 100, 30)
min_nights = st.sidebar.number_input("Minimum nights", 1, 30, 1)

# ---------------------------------------------------------------------------
# Location picker
# ---------------------------------------------------------------------------

st.sidebar.subheader("📍 Location")

landmark_cols = st.sidebar.columns(4)
selected_landmark = None
for i, (name, coords) in enumerate(LANDMARKS.items()):
    if landmark_cols[i % 4].button(name, key=f"lm_{i}", use_container_width=True):
        selected_landmark = coords

if "lat" not in st.session_state:
    st.session_state.lat = -33.8688
    st.session_state.lon = 151.2093

if selected_landmark:
    st.session_state.lat = selected_landmark[0]
    st.session_state.lon = selected_landmark[1]

lat = st.sidebar.number_input("Latitude", -34.5, -33.4, st.session_state.lat, format="%.4f")
lon = st.sidebar.number_input("Longitude", 150.5, 151.5, st.session_state.lon, format="%.4f")
st.session_state.lat = lat
st.session_state.lon = lon

predict_clicked = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area — Map + Results
# ---------------------------------------------------------------------------

st.title("🏠 Airbnb Sydney Price Predictor")
st.caption("Estimate the quoted nightly price for a Sydney Airbnb listing. Research prototype — not a pricing recommendation.")

map_col, result_col = st.columns([3, 2])

with map_col:
    points = [{"lat": lat, "lon": lon, "name": "Your listing"}]
    # Add reference landmarks
    for name, (plat, plon) in list(LANDMARKS.items())[:6]:
        points.append({"lat": plat, "lon": plon, "name": name})
    map_df = {"lat": [p["lat"] for p in points], "lon": [p["lon"] for p in points]}
    st.map(map_df, latitude=lat, longitude=lon, zoom=11)

with result_col:
    if predict_clicked:
        with st.spinner("Predicting..."):
            response = call_api(lat, lon)

        if response is None:
            st.error("Could not connect to the prediction API. Is the server running?")
            st.code("python inside_airbnb_serve.py", language="bash")
        elif response.get("status") == "refused":
            st.error("⚠️ Prediction Refused")
            for reason in response.get("refusal_reasons", []):
                st.warning(f"• `{reason}`")
        else:
            price = response["estimated_price"]
            interval = response["prediction_interval"]
            evidence = response.get("evidence_level", "unknown")
            comparables = response.get("comparable_count", 0)

            st.success(f"### {price:,.0f} AUD / night")
            st.metric("90% Confidence Interval", f"{interval['lower']:,.0f} — {interval['upper']:,.0f} AUD")
            st.caption(f"Evidence: **{evidence}** · {comparables} comparable listings · Snapshot: {response.get('snapshot_label', '?')}")

            if response.get("authority_warning"):
                st.info(response["authority_warning"])
            if response.get("snapshot_staleness_warning"):
                st.warning("Snapshot is stale — consider retraining.")

            if response.get("explanation"):
                with st.expander("📖 Full Explanation", expanded=True):
                    st.markdown(response["explanation"]["text"])
    else:
        st.info("👈 Fill in the listing details and click **Predict Price** to see the result.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Research use only. Predicts quoted (listed) price — not realised booking "
    "revenue. Sydney market only. [GitHub](https://github.com/Andy-JunXiong/"
    "Airbnb-rental-price-prediction)"
)

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def call_api(lat: float, lon: float) -> dict[str, Any] | None:
    api_url = st.session_state.get("api_url", DEFAULT_API_URL)
    payload = {
        "as_of_date": "2026-07-28",
        "quote_checkin_date": "2026-08-15",
        "quote_checkout_date": f"2026-08-{15 + st.session_state.get('stay_nights', 2)}",
        "neighbourhood": st.session_state.get("neighbourhood", neighbourhood),
        "property_type": st.session_state.get("property_type", property_type),
        "room_type": st.session_state.get("room_type", room_type),
        "latitude": lat,
        "longitude": lon,
        "accommodates": accommodates,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "beds": beds,
        "amenities_count": amenities,
        "minimum_nights": min_nights,
        "maximum_nights": 365,
        "calculated_host_listings_count": host_listings,
        "host_is_superhost": "t" if is_superhost else "f",
    }
    try:
        resp = requests.post(
            f"{api_url}/predict",
            params={"explain": "true"},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.ConnectionError, requests.Timeout):
        return None
    except requests.HTTPError as e:
        try:
            return e.response.json()
        except Exception:
            return {"status": "refused", "refusal_reasons": [str(e)]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "api_url" not in st.session_state:
        st.session_state.api_url = args.api_url


if __name__ == "__main__":
    main()
