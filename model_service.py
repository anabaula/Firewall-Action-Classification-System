"""Scikit-learn model loading and inference for NetSentry.

Only load locally trained trusted .joblib files: pickle/joblib should never
be loaded from untrusted uploads.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd

FEATURE_NAMES = [
    "source_port", "destination_port", "nat_source_port", "nat_destination_port",
    "total_bytes", "bytes_sent", "bytes_received", "total_packets",
    "connection_duration", "forward_packets", "backward_packets",
]
ACTIONS = ["Allow", "Deny", "Drop", "Reset"]
DATASET_NAME = "UCI Internet Firewall Data"


@lru_cache(maxsize=2)
def _load_cached(path: str, stamp: int) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model metadata is missing or contains incorrect features.")
    pipeline = bundle.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "predict_proba"):
        raise ValueError("Saved file does not contain a fitted sklearn Pipeline.")
    if set(pipeline.classes_) != set(ACTIONS):
        raise ValueError("Saved model classes do not match supported firewall actions.")
    return bundle


def load_model(path: str | Path) -> dict:
    path = Path(path).resolve()
    return _load_cached(str(path), path.stat().st_mtime_ns)


def predict(bundle: dict, features: list[float]) -> tuple[str, float, dict[str, float]]:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} traffic measurements.")
    frame = pd.DataFrame([features], columns=FEATURE_NAMES)
    model = bundle["pipeline"]
    action = str(model.predict(frame)[0])
    probability_vector = model.predict_proba(frame)[0]
    probabilities = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probability_vector)
    }
    return action, round(probabilities[action] * 100, 1), probabilities
