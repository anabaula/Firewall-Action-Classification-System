"""Regression checks for the deployed scikit-learn prediction pipeline."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from model_service import ACTIONS, DATASET_NAME, FEATURE_NAMES, load_model, predict
from train_model import make_pipeline


class ModelServiceTests(unittest.TestCase):
    def fixture(self):
        frame = pd.DataFrame(
            [[float(i)] * len(FEATURE_NAMES) for i in range(8)],
            columns=FEATURE_NAMES,
        )
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="constant", constant="Allow")),
        ])
        pipeline.fit(frame, ACTIONS * 2)
        return {
            "pipeline": pipeline,
            "feature_names": FEATURE_NAMES,
            "dataset_name": DATASET_NAME,
        }

    def test_colab_matching_random_forest_defaults(self):
        pipeline = make_pipeline()
        classifier = pipeline.named_steps["model"]
        self.assertEqual(classifier.n_estimators, 100)
        self.assertEqual(classifier.class_weight, "balanced")
        self.assertEqual(classifier.random_state, 42)
        self.assertEqual(pipeline.named_steps["imputer"].strategy, "median")

    def test_predict_uses_sklearn_probabilities(self):
        bundle = self.fixture()
        label, confidence, probabilities = predict(bundle, [5.] * 11)
        self.assertEqual(label, "Allow")
        self.assertEqual(confidence, 100.)
        self.assertEqual(set(probabilities), set(ACTIONS))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_invalid_feature_count(self):
        with self.assertRaisesRegex(ValueError, "Expected 11"):
            predict(self.fixture(), [1, 2, 3])

    def test_joblib_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "model.joblib"
            joblib.dump(self.fixture(), filename)
            model = load_model(filename)
            self.assertEqual(model["feature_names"], FEATURE_NAMES)
            self.assertEqual(predict(model, [5.] * 11)[0], "Allow")

    def test_model_rejects_incompatible_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "bad.joblib"
            model = self.fixture()
            model["feature_names"] = ["incorrect"]
            joblib.dump(model, filename)
            with self.assertRaisesRegex(ValueError, "incorrect features"):
                load_model(filename)


if __name__ == "__main__":
    unittest.main()
