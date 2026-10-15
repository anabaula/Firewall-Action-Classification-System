"""Pure-Python Random Forest for the public UCI Internet Firewall Data dataset.

Standard-library-only implementation to avoid compiled NumPy/SciPy dependencies.
Educational use; never use it to directly enforce production firewall decisions.
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Any

# Features are derived directly from the UCI Internet Firewall Data columns.
FEATURE_NAMES = [
    "source_port", "destination_port", "nat_source_port", "nat_destination_port",
    "total_bytes", "bytes_sent", "bytes_received", "total_packets",
    "connection_duration", "forward_packets", "backward_packets",
]
ACTIONS = ["Allow", "Deny", "Drop", "Reset"]


def _gini(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def _leaf(labels: list[str], indices: list[int]) -> dict[str, Any]:
    counts = {action: 0 for action in ACTIONS}
    for idx in indices:
        counts[labels[idx]] += 1
    total = max(1, len(indices))
    return {"probs": {action: counts[action] / total for action in ACTIONS}}


def _candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if len(unique) < 2:
        return []
    if len(unique) <= 20:
        positions = range(1, len(unique))
    else:
        candidate_positions = set()
        for step in range(1, 13):
            pos = round(step * (len(unique) - 1) / 13)
            if 1 <= pos < len(unique):
                candidate_positions.add(pos)
        positions = sorted(candidate_positions)
    return [(unique[pos - 1] + unique[pos]) / 2.0 for pos in positions]


def _build_tree(X: list[list[float]], labels: list[str], indices: list[int], rng: random.Random,
                depth: int, max_depth: int, min_split: int, min_leaf: int) -> dict[str, Any]:
    current_leaf = _leaf(labels, indices)
    distinct_labels = {labels[idx] for idx in indices}
    if depth >= max_depth or len(indices) < min_split or len(distinct_labels) <= 1:
        return current_leaf

    parent_counts = {action: 0 for action in ACTIONS}
    for idx in indices:
        parent_counts[labels[idx]] += 1
    parent_impurity = _gini(parent_counts)

    feature_count = len(FEATURE_NAMES)
    subset_count = max(1, int(math.sqrt(feature_count)))
    feature_subset = rng.sample(range(feature_count), subset_count)
    best_score = float("inf")
    best_feature = None
    best_threshold = None
    best_left = None
    best_right = None

    for feature in feature_subset:
        vals = [X[idx][feature] for idx in indices]
        for threshold in _candidate_thresholds(vals):
            left, right = [], []
            left_counts = {action: 0 for action in ACTIONS}
            right_counts = {action: 0 for action in ACTIONS}
            for idx in indices:
                if X[idx][feature] <= threshold:
                    left.append(idx)
                    left_counts[labels[idx]] += 1
                else:
                    right.append(idx)
                    right_counts[labels[idx]] += 1
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            score = len(left) * _gini(left_counts) + len(right) * _gini(right_counts)
            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
                best_left = left
                best_right = right

    if best_feature is None or best_left is None or best_right is None:
        return current_leaf
    if best_score >= len(indices) * parent_impurity - 1e-12:
        return current_leaf
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": _build_tree(X, labels, best_left, rng, depth + 1, max_depth, min_split, min_leaf),
        "right": _build_tree(X, labels, best_right, rng, depth + 1, max_depth, min_split, min_leaf),
    }


def train_forest(rows: list[dict[str, Any]], tree_count: int = 13, max_depth: int = 6, seed: int = 42,
                 dataset_name: str = "UCI Internet Firewall Data") -> dict[str, Any]:
    if not rows:
        raise ValueError("No training rows were provided.")
    rng = random.Random(seed)
    X = [list(map(float, row["features"])) for row in rows]
    labels = [str(row["action"]) for row in rows]
    trees = []
    n = len(rows)
    for tree_index in range(tree_count):
        bootstrap = [rng.randrange(n) for _ in range(n)]
        tree_rng = random.Random(seed + 1009 * (tree_index + 1))
        trees.append(_build_tree(X, labels, bootstrap, tree_rng, 0, max_depth, 20, 5))
        print(f"Trained tree {tree_index + 1}/{tree_count}", end="\r", flush=True)
    print(" " * 36, end="\r")
    return {
        "format_version": 2,
        "algorithm": "Random Forest Classifier (pure Python educational implementation)",
        "dataset_name": dataset_name,
        "features": FEATURE_NAMES,
        "classes": ACTIONS,
        "trees": trees,
        "demo_synthetic_data": False,
        "warning": "Trained on the public UCI Internet Firewall Data dataset. Reset-Both is mapped to Reset. The rare Reset class and data limitations must be disclosed; educational project only, not for production firewall decisions.",
    }


def _predict_tree(tree: dict[str, Any], features: list[float]) -> dict[str, float]:
    node = tree
    while "probs" not in node:
        if features[node["feature"]] <= node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    return node["probs"]


def predict_proba(model: dict[str, Any], features: list[float]) -> dict[str, float]:
    if not model.get("trees"):
        raise ValueError("The saved model contains no trees. Run python train_model.py again.")
    totals = {action: 0.0 for action in model.get("classes", ACTIONS)}
    for tree in model["trees"]:
        probabilities = _predict_tree(tree, features)
        for action in totals:
            totals[action] += float(probabilities.get(action, 0.0))
    return {action: score / len(model["trees"]) for action, score in totals.items()}


def predict(model: dict[str, Any], features: list[float]) -> tuple[str, float, dict[str, float]]:
    probs = predict_proba(model, features)
    action = max(probs, key=probs.get)
    return action, round(probs[action] * 100.0, 1), probs


def save_model(model: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")


def load_model(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
