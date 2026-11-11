"""Endpoint contract tests that do not require trained model artifacts.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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

    def test_generate_returns_real_form_fields_without_action(self):
        example = tuple(float(VALID[name]) for name in FEATURE_NAMES)
        with patch("app._sample_rows", return_value=(example,)):
            response = self.client.get("/api/sample")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["source"], "log2.csv")
        self.assertEqual(payload["split"], "test")
        self.assertEqual(payload["features"], VALID)
        self.assertNotIn("action", payload)

    def test_generate_pool_matches_training_script_test_partition(self):
        """Generated records must match the held-out X_test, not X_train."""
        from app import _sample_rows, DATASET_PATH
        from data_pipeline import load_dataset

        rows, _, _ = load_dataset(DATASET_PATH)
        X = pd.DataFrame([row["features"] for row in rows], columns=FEATURE_NAMES)
        y = pd.Series([row["action"] for row in rows], name="action")
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )
        expected_test = set(X_test.itertuples(index=False, name=None))
        training = set(X_train.itertuples(index=False, name=None))
        samples = set(_sample_rows())

        self.assertEqual(len(_sample_rows()), len(X_test))
        self.assertEqual(samples, expected_test)
        self.assertFalse(samples & training)

    def test_generate_gracefully_handles_missing_dataset(self):
        with patch("app._sample_rows", side_effect=FileNotFoundError("hidden path")), \
             patch.object(app.logger, "exception"):
            response = self.client.get("/api/sample")
        self.assertEqual(response.status_code, 503)
        self.assertIn("unavailable", response.get_json()["error"])
        self.assertNotIn("hidden path", response.get_data(as_text=True))

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

    def test_unreadable_model_returns_json_service_unavailable(self):
        # Simulate an unreadable/corrupt saved model, without touching real artifacts.
        with patch("app.load_saved_model", side_effect=OSError("private filesystem path")), \
             patch.object(app.logger, "exception") as log_error:
            health = self.client.get("/api/health")
            prediction = self.client.post("/api/predict", json=VALID)

        self.assertEqual(health.status_code, 503)
        self.assertTrue(health.is_json)
        self.assertEqual(health.get_json()["status"], "model_error")
        self.assertFalse(health.get_json()["model_ready"])
        self.assertIn("retrain", health.get_json()["error"])
        self.assertNotIn("private filesystem path", health.get_data(as_text=True))

        self.assertEqual(prediction.status_code, 503)
        self.assertTrue(prediction.is_json)
        self.assertIn("retrain", prediction.get_json()["error"])
        self.assertNotIn("private filesystem path", prediction.get_data(as_text=True))
        self.assertEqual(log_error.call_count, 2)

    def test_invalid_model_metadata_returns_json_service_unavailable(self):
        with patch("app.load_saved_model", side_effect=ValueError("bad metadata")), \
             patch.object(app.logger, "exception"):
            response = self.client.post("/api/predict", json=VALID)
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.is_json)
        self.assertNotIn("bad metadata", response.get_data(as_text=True))

    def test_health_reports_missing_model(self):
        with patch("app.load_saved_model", return_value=None):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["model_ready"])
        self.assertEqual(response.get_json()["status"], "model_missing")

    def test_homepage_and_static_resources(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("NetSentry", body)
        self.assertIn('id="input-title">Traffic input</h2>', body)
        self.assertIn('Firewall Network Classifier</h1>', body)
        self.assertIn('Model: Random Forest (scikit-learn)', body)
        self.assertIn('id="history-page-info"', body)
        self.assertIn('id="traffic-form"', body)
        self.assertIn('id="generate-btn"', body)
        self.assertNotIn('class="topline"', body)
        self.assertIn('class="dataset-note">UCI INTERNET FIREWALL DATA</span>', body)
        self.assertEqual(body.count('UCI INTERNET FIREWALL DATA'), 1)
        self.assertIn('type="button" class="btn btn-secondary" id="generate-btn"', body)
        self.assertIn('id="result-panel"', body)
        self.assertIn('id="training-results"', body)
        self.assertIn('id="prediction-history"', body)
        self.assertEqual(body.count('type="number"'), 11)
        self.assertNotIn('id="sidebar"', body)
        self.assertNotIn('class="site-header"', body)
        self.assertIn('class="classifier-heading"', body)
        self.assertNotIn('class="page-header"', body)
        self.assertIn('id="history-details"', body)
        self.assertIn('id="details-fields"', body)
        self.assertIn('<th>Date</th><th>Action</th><th>Confidence</th><th>Input details</th>', body)
        self.assertNotIn('<th>Source</th>', body)
        self.assertNotIn('>ANALYTICS<', body)
        self.assertNotIn('class="metric-table"', body)
        self.assertNotIn('Performance by firewall action', body)
        self.assertNotIn('data-page=', body)
        for path in ("/static/app.js", "/static/style.css"):
            asset = self.client.get(path)
            try:
                self.assertEqual(asset.status_code, 200)
            finally:
                asset.close()

    def test_minimal_single_screen_styles_have_no_gradients(self):
        stylesheet = (Path(__file__).resolve().parents[1] / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn("dashboard-grid", stylesheet)
        self.assertNotIn("gradient(", stylesheet)
        self.assertIn("background: var(--blue);", stylesheet)
        self.assertNotIn(".action-allow {", stylesheet)
        frontend = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('class="history-action"', frontend)
        self.assertIn('data-history-index', frontend)
        self.assertIn('detailsDialog.showModal()', frontend)
        self.assertIn('...values,', frontend)
        self.assertIn('(Math.trunc(Number(value) * 100) / 100).toFixed(2)', frontend)
        self.assertIn('actionLabel.dataset.action = result.action.toLowerCase();', frontend)
        self.assertIn('.result-action[data-action="allow"]', stylesheet)
        self.assertIn('.result-action[data-action="deny"]', stylesheet)
        self.assertIn("'Not recorded'", frontend)
        self.assertIn('--background: #fff;', stylesheet)

    def test_homepage_displays_saved_training_metrics(self):
        scores = {name: {"precision": .8, "recall": .7, "f1_score": .75, "support": 10}
                  for name in ACTIONS}
        report = {
            "dataset": "UCI Internet Firewall Data",
            "accuracy": .95, "macro_f1": .86,
            "train_split_size": 800, "test_split_size": 200,
            "cross_validation": {
                "mean_accuracy": .91, "mean_training_accuracy": .94,
                "training_validation_gap_percentage_points": 3.0, "folds": 5,
            },
            "per_class": scores,
        }
        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder) / "evaluation_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch("app.REPORT_PATH", report_path):
                page = self.client.get("/").get_data(as_text=True)
        self.assertIn("95.00%", page)
        self.assertIn("0.8600", page)
        self.assertIn("91.00%", page)
        self.assertIn("3.00 pp", page)
        self.assertNotIn('<th>Recall</th>', page)
        self.assertNotIn('action-pill action-reset', page)

    def test_homepage_handles_missing_training_report(self):
        with tempfile.TemporaryDirectory() as folder:
            missing_path = Path(folder) / "not-created.json"
            with patch("app.REPORT_PATH", missing_path):
                response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("No training report", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
