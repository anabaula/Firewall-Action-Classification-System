"""Load the trained NetSentry model and classify network traffic.

The machine learning is done in train_model.py. This file only loads the
saved model, receives the 11 numeric inputs, and returns a prediction.
Only open a Joblib model produced by trusted project code.
"""

import joblib
import pandas as pd

# The order of these 11 inputs must match model training and the website.
FEATURE_NAMES = [
    "source_port", "destination_port", "nat_source_port", "nat_destination_port",
    "total_bytes", "bytes_sent", "bytes_received", "total_packets",
    "connection_duration", "forward_packets", "backward_packets",
]

ACTIONS = ["Allow", "Deny", "Drop", "Reset"]
DATASET_NAME = "UCI Internet Firewall Data"


def load_model(path):
    """Load the saved model from the training script."""
    bundle = joblib.load(path)

    # Make sure the model matches the 11 inputs and four actions.
    if not isinstance(bundle, dict) or bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model metadata is missing or contains incorrect features.")

    model = bundle.get("pipeline")
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("Saved file does not contain a fitted sklearn probability model.")
    if set(model.classes_) != set(ACTIONS):
        raise ValueError("Saved model classes do not match supported firewall actions.")

    return bundle


def predict(bundle, features):
    """Predict one action and return its unrounded confidence percentage."""
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} traffic measurements.")

    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    model = bundle["pipeline"]

    action = str(model.predict(X)[0])
    scores = model.predict_proba(X)[0]

    probabilities = {}
    for label, score in zip(model.classes_, scores):
        probabilities[str(label)] = float(score)

    # The browser formats the percentage; do not round it in Python.
    confidence = probabilities[action] * 100
    return action, confidence, probabilities
