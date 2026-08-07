"""Real API tests using FastAPI TestClient with CI fixture artifact.

Requires: pip install fastapi httpx
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    from inside_airbnb_serve import create_app

    _SERVE_AVAILABLE = True
except ImportError:
    _SERVE_AVAILABLE = False

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "minimal_artifact.joblib"
_SKIP_REASON = "fastapi and httpx required: pip install fastapi httpx"


@unittest.skipUnless(_SERVE_AVAILABLE, _SKIP_REASON)
class ServingHealthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["MODEL_ARTIFACT_PATH"] = str(FIXTURE)
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_health_returns_healthy_when_artifact_loaded(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["artifact_loaded"])
        self.assertEqual(data["snapshot_label"], "ci-fixture")

    def test_model_info_returns_metadata(self) -> None:
        resp = self.client.get("/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["snapshot_label"], "ci-fixture")
        self.assertEqual(data["deployment_authority"], "research_only")
        self.assertIn("numeric", data["features"])
        self.assertIn("categorical", data["features"])
        self.assertEqual(data["model_family"], "sklearn-histgb")

    def test_model_info_has_all_required_fields(self) -> None:
        resp = self.client.get("/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        required = [
            "model_name", "model_version", "model_family",
            "artifact_sha256", "artifact_path",
            "training_silver_sha256", "snapshot_label",
            "training_as_of_date", "deployment_authority",
            "temporal_validation_status", "release_gate_status",
            "features", "limitations", "environment",
        ]
        for field in required:
            self.assertIn(field, data, f"Missing field: {field}")

    def test_dashboard_returns_html(self) -> None:
        resp = self.client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("Control Tower", resp.text)


@unittest.skipUnless(_SERVE_AVAILABLE, _SKIP_REASON)
class ServingPredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["MODEL_ARTIFACT_PATH"] = str(FIXTURE)
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def _valid_request(self) -> dict:
        return {
            "as_of_date": "2026-07-28",
            "quote_checkin_date": "2026-08-15",
            "quote_checkout_date": "2026-08-17",
            "neighbourhood": "Sydney",
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "latitude": -33.8688,
            "longitude": 151.2093,
            "accommodates": 2,
            "bathrooms": 1,
            "bedrooms": 1,
            "beds": 1,
            "amenities_count": 30,
            "minimum_nights": 1,
            "maximum_nights": 365,
            "calculated_host_listings_count": 1,
            "host_is_superhost": "f",
        }

    def test_predict_returns_valid_response(self) -> None:
        resp = self.client.post("/predict", json=self._valid_request())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["status"], ("ok", "refused"))
        self.assertEqual(data["currency"], "AUD")
        self.assertIn("snapshot_label", data)
        self.assertEqual(data["snapshot_label"], "ci-fixture")
        self.assertEqual(data["deployment_authority"], "research_only")

    def test_predict_ok_has_price_and_interval(self) -> None:
        resp = self.client.post("/predict", json=self._valid_request())
        data = resp.json()
        if data["status"] == "ok":
            self.assertIsNotNone(data["estimated_price"])
            self.assertGreater(data["estimated_price"], 0)
            self.assertIsNotNone(data["prediction_interval"])
            self.assertLess(
                data["prediction_interval"]["lower"],
                data["prediction_interval"]["upper"],
            )

    def test_predict_with_explain_includes_explanation(self) -> None:
        resp = self.client.post(
            "/predict?explain=true", json=self._valid_request()
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        if data["status"] == "ok":
            self.assertIsNotNone(data.get("explanation"))
            self.assertIn("text", data["explanation"])
            self.assertIn("mode", data["explanation"])

    def test_predict_invalid_input_returns_422(self) -> None:
        resp = self.client.post("/predict", json={"bad": "input"})
        self.assertEqual(resp.status_code, 422)

    def test_predict_unseen_neighbourhood_still_returns_200(self) -> None:
        req = {**self._valid_request(), "neighbourhood": "UnknownPlace"}
        resp = self.client.post("/predict", json=req)
        # Model should still attempt prediction; may be refused at evidence gate
        self.assertIn(resp.status_code, (200, 200))

    def test_predict_response_has_evidence_fields(self) -> None:
        resp = self.client.post("/predict", json=self._valid_request())
        data = resp.json()
        self.assertIn("evidence_level", data)
        self.assertIn("comparable_count", data)
        self.assertIn("comparable_level", data)
        self.assertIn("snapshot_staleness_warning", data)
        self.assertIn("authority_warning", data)
        self.assertIn("refusal_reasons", data)


@unittest.skipUnless(_SERVE_AVAILABLE, _SKIP_REASON)
class ServingMissingArtifactTest(unittest.TestCase):
    def test_health_returns_not_ready_without_artifact(self) -> None:
        app = create_app(Path("/nonexistent/artifact.joblib"))
        client = TestClient(app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "not_ready")
        self.assertFalse(data["artifact_loaded"])

    def test_model_info_returns_503_without_artifact(self) -> None:
        app = create_app(Path("/nonexistent/artifact.joblib"))
        client = TestClient(app)
        resp = client.get("/model-info")
        self.assertEqual(resp.status_code, 503)

    def test_predict_returns_503_without_artifact(self) -> None:
        app = create_app(Path("/nonexistent/artifact.joblib"))
        client = TestClient(app)
        resp = client.post("/predict", json={
            "as_of_date": "2026-07-28",
            "quote_checkin_date": "2026-08-15",
            "quote_checkout_date": "2026-08-17",
            "neighbourhood": "Sydney",
            "property_type": "Entire rental unit",
            "room_type": "Entire home/apt",
            "latitude": -33.8688,
            "longitude": 151.2093,
            "accommodates": 2,
            "bathrooms": 1, "bedrooms": 1, "beds": 1,
            "amenities_count": 30,
            "minimum_nights": 1, "maximum_nights": 365,
            "calculated_host_listings_count": 1,
            "host_is_superhost": "f",
        })
        self.assertEqual(resp.status_code, 503)


if __name__ == "__main__":
    unittest.main()
