"""Endpoint contract tests that do not require trained model artifacts.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest
from unittest.mock import patch
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from app import app
from model_service import ACTIONS, FEATURE_NAMES

# Real scikit-learn pipeline with a deterministic prediction for API tests.
_X = pd.DataFrame([[i + 1] * len(FEATURE_NAMES) for i in range(4)], columns=FEATURE_NAMES)
TEST_PIPELINE = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DummyClassifier(strategy="constant", constant="Allow")),
])
TEST_PIPELINE.fit(_X, ACTIONS)
TEST_MODEL = {
    "dataset_name": "UCI Internet Firewall Data",
    "feature_names": FEATURE_NAMES,
    "pipeline": TEST_PIPELINE,
}
VALID = {
    "source_port": 51426,
    "destination_port": 443,
    "nat_source_port": 3125,
    "nat_destination_port": 443,
    "total_bytes": 15947,
    "bytes_sent": 5526,
    "bytes_received": 10421,
    "total_packets": 38,
    "connection_duration": 622,
    "forward_packets": 16,
    "backward_packets": 22,
}


class PredictionAPITests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def send(self, payload):
        with patch("app.load_saved_model", return_value=TEST_MODEL):
            return self.client.post("/api/predict", json=payload)

    def test_prediction_with_valid_measurements(self):
        response = self.send(VALID)
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["action"], "Allow")
        self.assertEqual(result["confidence"], 100.0)

    def test_port_boundaries(self):
        for port in (0, 65535):
            with self.subTest(port=port):
                values = dict(VALID, source_port=port)
                self.assertEqual(self.send(values).status_code, 200)
        for port in (-1, 65536, 42.5):
            with self.subTest(port=port):
                self.assertEqual(self.send(dict(VALID, source_port=port)).status_code, 400)

    def test_bool_rejected_for_every_numeric_field(self):
        for field in FEATURE_NAMES:
            with self.subTest(field=field):
                self.assertEqual(self.send(dict(VALID, **{field: True})).status_code, 400)
                self.assertEqual(self.send(dict(VALID, **{field: False})).status_code, 400)

    def test_fractional_bytes_and_packet_counts_are_invalid(self):
        for field in ("total_bytes", "bytes_sent", "bytes_received", "total_packets",
                      "forward_packets", "backward_packets"):
            with self.subTest(field=field):
                response = self.send(dict(VALID, **{field: 0.5}))
                self.assertEqual(response.status_code, 400)
                self.assertIn("whole number", response.get_json()["error"])

    def test_duration_may_be_fractional(self):
        self.assertEqual(self.send(dict(VALID, connection_duration=0.5)).status_code, 200)

    def test_invalid_missing_nan_and_nonfinite(self):
        for invalid in (None, "not a number", "nan", "inf", "-inf"):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.send(dict(VALID, source_port=invalid)).status_code, 400)
        missing = {k: v for k, v in VALID.items() if k != "source_port"}
        self.assertEqual(self.send(missing).status_code, 400)

    def test_consistency_rules_and_positive_packets(self):
        cases = [
            dict(VALID, total_packets=0),
            dict(VALID, total_packets=1),
            dict(VALID, total_bytes=100),
            dict(VALID, bytes_received=-1),
        ]
        for data in cases:
            with self.subTest(data=data):
                self.assertEqual(self.send(data).status_code, 400)

    def test_malformed_payload_rejected(self):
        self.assertEqual(self.send(["invalid"]).status_code, 400)
        with patch("app.load_saved_model", return_value=TEST_MODEL):
            response = self.client.post("/api/predict", data="{not json",
                                        content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_missing_model_returns_service_unavailable(self):
        with patch("app.load_saved_model", return_value=None):
            response = self.client.post("/api/predict", json=VALID)
        self.assertEqual(response.status_code, 503)

    def test_homepage_and_static_resources(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("NetSentry", body)
        self.assertIn("> ANALYTICS</div>", body)
        self.assertIn('id="traffic-form"', body)
        for path in ("/static/app.js", "/static/style.css"):
            asset = self.client.get(path)
            try:
                self.assertEqual(asset.status_code, 200)
            finally:
                asset.close()


if __name__ == "__main__":
    unittest.main()
