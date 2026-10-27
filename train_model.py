"""Train NetSentry with scikit-learn, following the classmate's Colab approach.

Validated UCI data -> 80/20 stratified split -> median imputer and
100-tree RandomForestClassifier(class_weight="balanced") -> held-out evaluation.
Five-fold CV uses training data only; held-out test data is never trained on.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from data_pipeline import load_dataset
from model_service import ACTIONS, DATASET_NAME, FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DATA_PATH = BASE_DIR / "data" / "log2_cleaned.csv"
ORIGINAL_DATA_PATH = BASE_DIR / "data" / "log2.csv"
MODEL_PATH = BASE_DIR / "model" / "firewall_random_forest.joblib"
REPORT_PATH = BASE_DIR / "data" / "evaluation_report.json"
QUALITY_PATH = BASE_DIR / "data" / "data_quality_report.json"
DATASET_DOI = "10.24432/C5131M"


def make_pipeline(trees: int = 100) -> Pipeline:
    """Same sklearn model settings as the reference notebook."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=trees, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )),
    ])


def evaluate(y_true, predictions, probabilities, model_classes) -> dict:
    """Metrics on the untouched holdout; confidences are NOT calibrated."""
    detail = classification_report(
        y_true, predictions, labels=ACTIONS, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=ACTIONS)
    confusion = {
        actual: {predicted: int(matrix[i, j]) for j, predicted in enumerate(ACTIONS)}
        for i, actual in enumerate(ACTIONS)
    }
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.asarray(y_true)
    class_to_index = {str(c): i for i, c in enumerate(model_classes)}
    true_one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(truth):
        true_one_hot[row, class_to_index[str(label)]] = 1.
    multiclass_brier = float(np.mean(np.sum(
        (probabilities - true_one_hot) ** 2, axis=1,
    )))
    confidence = probabilities.max(axis=1)
    correct = (np.asarray(predictions) == truth)
    ece = 0.
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + .1
        membership = (confidence >= lower) & (
            confidence <= upper if upper >= 1 else confidence < upper
        )
        if membership.any():
            ece += float(membership.mean() * abs(
                confidence[membership].mean() - correct[membership].mean()
            ))
    return {
        "test_size": int(len(truth)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=ACTIONS, average="macro", zero_division=0)),
        "per_class": {
            action: {
                "precision": float(detail[action]["precision"]),
                "recall": float(detail[action]["recall"]),
                "f1_score": float(detail[action]["f1-score"]),
                "support": int(detail[action]["support"]),
            }
            for action in ACTIONS
        },
        "confusion_matrix": confusion,
        "calibration_diagnostics": {
            "ece_10_bins": ece,
            "multiclass_brier_score": multiclass_brier,
            "probabilities_calibrated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=CLEANED_DATA_PATH if CLEANED_DATA_PATH.is_file() else ORIGINAL_DATA_PATH,
        help="UCI-style CSV (defaults to classmate's cleaned CSV).",
    )
    parser.add_argument("--validate-only", action="store_true", help="Only audit dataset.")
    parser.add_argument("--trees", type=int, default=100, help="Forest trees (Colab: 100).")
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Training-only stratified cross-validation folds (0 to skip; Colab: 5).",
    )
    args = parser.parse_args()
    if args.trees < 1 or args.cv_folds not in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        parser.error("--trees must be positive; --cv-folds must be 0 or 2–10.")

    rows, counts, quality = load_dataset(args.dataset)
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(
        f"Source: {args.dataset.name}; original rows: {quality['raw_rows']:,}; "
        f"retained: {len(rows):,}; duplicate conflicts quarantined: "
        f"{quality['conflicting_feature_rows_removed']:,}"
    )
    if args.validate_only:
        print(f"Data quality report: {QUALITY_PATH}")
        return

    X = pd.DataFrame(
        [row["features"] for row in rows], columns=FEATURE_NAMES, dtype=float,
    )
    y = pd.Series([row["action"] for row in rows], name="action")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    # This check is important because classmate's CSV contains conflicting
    # labels on identical feature vectors, resolved BEFORE the split.
    train_keys = {tuple(row) for row in X_train.itertuples(index=False, name=None)}
    test_keys = {tuple(row) for row in X_test.itertuples(index=False, name=None)}
    if train_keys & test_keys:
        raise ValueError("Identical feature vectors appear in both splits.")
    print(f"Train: {len(X_train):,}; held-out test: {len(X_test):,}")
    pipeline = make_pipeline(args.trees)
    cv = None
    if args.cv_folds:
        print(f"Evaluating {args.cv_folds}-fold stratified CV on training data only...")
        splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=splitter,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=1, error_score="raise",
        )
        cv = {
            "folds": args.cv_folds,
            "accuracy_scores": [float(x) for x in scores["test_accuracy"]],
            "mean_accuracy": float(np.mean(scores["test_accuracy"])),
            "macro_f1_scores": [float(x) for x in scores["test_macro_f1"]],
            "mean_macro_f1": float(np.mean(scores["test_macro_f1"])),
        }
        print(
            f"Training CV accuracy: {cv['mean_accuracy']:.4%}, "
            f"macro F1: {cv['mean_macro_f1']:.4f}"
        )
    print("Training the scikit-learn Random Forest pipeline...")
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    metrics = evaluate(y_test, predicted, probabilities, pipeline.classes_)
    report = {
        "dataset": DATASET_NAME,
        "dataset_doi": DATASET_DOI,
        "dataset_file": args.dataset.name,
        "implementation": "scikit-learn Pipeline(SimpleImputer, RandomForestClassifier)",
        "model_parameters": {
            "n_estimators": args.trees,
            "random_state": 42,
            "class_weight": "balanced",
            "imputer_strategy": "median",
        },
        "quality_report": quality,
        "train_split_size": len(X_train),
        "test_split_size": len(X_test),
        "training_class_counts": dict(Counter(y_train)),
        "test_class_counts": dict(Counter(y_test)),
        "cross_validation": cv,
        **metrics,
        "warning": (
            "Not calibrated probabilities; accuracy and CV are evaluated on the "
            "provided historical CSV only. Rare Reset support is small. "
            "Classmate's cleaned CSV provenance is not fully reproducible."
        ),
    }
    # Save fitted pipeline and ordered feature names together. Joblib is
    # trusted local data only; never load arbitrary uploaded pickle files.
    bundle = {
        "pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "dataset_name": DATASET_NAME,
        "dataset_file": args.dataset.name,
        "training_class_counts": report["training_class_counts"],
        "warning": report["warning"],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Held-out accuracy: {metrics['accuracy']:.4%}")
    print(f"Held-out macro F1: {metrics['macro_f1']:.4f}")
    for action in ACTIONS:
        c = metrics["per_class"][action]
        print(f"  {action:5s} precision={c['precision']:.3f}, "
              f"recall={c['recall']:.3f}, F1={c['f1_score']:.3f}, support={c['support']}")
    print(f"Uncalibrated ECE (10 bins): {metrics['calibration_diagnostics']['ece_10_bins']:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Reports: {REPORT_PATH} and {QUALITY_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
