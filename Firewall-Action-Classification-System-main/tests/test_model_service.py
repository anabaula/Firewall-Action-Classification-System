"""Regression checks for the deployed scikit-learn prediction pipeline."""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from model_service import ACTIONS, DATASET_NAME, FEATURE_NAMES, load_model, predict
from train_model import make_calibrated_pipeline, make_pipeline


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

    def test_regularized_random_forest_and_calibration_configuration(self):
        forest = make_pipeline(min_samples_leaf=2, max_depth=24)
        self.assertEqual(forest.named_steps["model"].min_samples_leaf, 2)
        self.assertEqual(forest.named_steps["model"].max_depth, 24)
        calibrated = make_calibrated_pipeline(
            min_samples_leaf=2, max_depth=24, calibration_folds=3,
        )
        self.assertEqual(calibrated.method, "sigmoid")
        self.assertEqual(calibrated.cv, 3)
        self.assertFalse(calibrated.ensemble)
        self.assertEqual(calibrated.estimator.named_steps["model"].n_estimators, 100)

    def test_predict_uses_sklearn_probabilities(self):
        bundle = self.fixture()
        label, confidence, probabilities = predict(bundle, [5.] * 11)
        self.assertEqual(label, "Allow")
        self.assertEqual(confidence, 100.)
        self.assertEqual(set(probabilities), set(ACTIONS))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_confidence_does_not_round_up_to_100(self):
        bundle = self.fixture()
        raw_probability = 0.99999123456
        with patch.object(bundle["pipeline"], "predict_proba", return_value=[
            [raw_probability, 1 - raw_probability, 0., 0.]
        ]):
            label, confidence, probabilities = predict(bundle, [5.] * 11)
        self.assertEqual(label, "Allow")
        self.assertAlmostEqual(confidence, 99.999123456)
        self.assertLess(confidence, 100.)
        self.assertEqual(probabilities["Allow"], raw_probability)

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
