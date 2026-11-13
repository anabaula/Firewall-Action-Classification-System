import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
ORIGINAL_DATA_PATH = BASE_DIR / "data" / "log2.csv"
MODEL_PATH = BASE_DIR / "model" / "firewall_random_forest.joblib"
REPORT_PATH = BASE_DIR / "data" / "evaluation_report.json"
QUALITY_PATH = BASE_DIR / "data" / "data_quality_report.json"
DATASET_DOI = "10.24432/C5131M"


def make_pipeline(trees=100, min_samples_leaf=1, max_depth=None):
    """Create our two-step scikit-learn pipeline.

    1. Fill missing values using the training-fold median.
    2. Train a Random Forest. Trees do not require feature scaling.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=trees,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            n_jobs=-1,
        )),
    ])



def make_calibrated_pipeline(trees=100, min_samples_leaf=1,
                             max_depth=None, calibration_folds=3):
    """Improve confidence scores using training-only sigmoid calibration.

    The Random Forest still predicts the action; calibration adjusts its
    probabilities. The reserved 20% test data is NOT used here.
    """
    pipeline = make_pipeline(
        trees, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
    )
    return CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=calibration_folds,
        ensemble=False,
    )



def evaluate(y_true, predictions, probabilities, model_classes, calibrated=False):
    """Calculate the same evaluation metrics used in our training report.

    Accuracy = correct predictions / all predictions.
    Precision, recall and F1 are reported separately for all four actions.
    Confidence diagnostics show whether predicted probabilities are reliable.
    """
    # STEP 1: Classification scores for each action.
    report = classification_report(
        y_true, predictions, labels=ACTIONS, output_dict=True, zero_division=0,
    )
    accuracy = float(accuracy_score(y_true, predictions))
    macro_f1 = float(f1_score(
        y_true, predictions, labels=ACTIONS, average="macro", zero_division=0,
    ))

    class_scores = {}
    for action in ACTIONS:
        score = report[action]
        class_scores[action] = {
            "precision": float(score["precision"]),
            "recall": float(score["recall"]),
            "f1_score": float(score["f1-score"]),
            "support": int(score["support"]),
        }

    # STEP 2: Actual-versus-predicted confusion matrix.
    matrix = confusion_matrix(y_true, predictions, labels=ACTIONS)
    confusion = {}
    for actual_index, actual_action in enumerate(ACTIONS):
        confusion[actual_action] = {}
        for predicted_index, predicted_action in enumerate(ACTIONS):
            confusion[actual_action][predicted_action] = int(
                matrix[actual_index, predicted_index]
            )

    # STEP 3: Confidence scores. These are probabilities, not guarantees.
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.asarray(y_true)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(predictions) == truth

    # The Brier score compares the probabilities with the actual classes.
    true_one_hot = np.zeros_like(probabilities)
    class_index = {str(name): index for index, name in enumerate(model_classes)}
    for row_number, actual_label in enumerate(truth):
        true_one_hot[row_number, class_index[str(actual_label)]] = 1.0
    brier_score = float(np.mean(np.sum(
        (probabilities - true_one_hot) ** 2, axis=1,
    )))

    # Expected calibration error: compare confidence with correctness
    # in ten groups (0-10%, 10-20%, ..., 90-100%).
    calibration_error = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        if upper >= 1:
            in_group = (confidence >= lower) & (confidence <= upper)
        else:
            in_group = (confidence >= lower) & (confidence < upper)

        if in_group.any():
            fraction = in_group.mean()
            average_confidence = confidence[in_group].mean()
            actual_accuracy = correct[in_group].mean()
            calibration_error += float(
                fraction * abs(average_confidence - actual_accuracy)
            )

    diagnostics = {
        "ece_10_bins": calibration_error,
        "multiclass_brier_score": brier_score,
        "probabilities_calibrated": bool(calibrated),
        "exact_100_confidence_count": int(np.sum(confidence == 1)),
        "exact_100_confidence_wrong": int(np.sum((confidence == 1) & ~correct)),
        "median_confidence": float(np.median(confidence)),
    }
    return {
        "test_size": int(len(truth)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": class_scores,
        "confusion_matrix": confusion,
        "calibration_diagnostics": diagnostics,
    }



REGULARIZATION_OPTIONS = [
    ("unrestricted", {"min_samples_leaf": 1, "max_depth": None}),
    ("leaf_2_depth_24", {"min_samples_leaf": 2, "max_depth": 24}),
    ("leaf_3_depth_18", {"min_samples_leaf": 3, "max_depth": 18}),
]


def choose_parameters(X_train, y_train, trees, calibration_folds):
    """Compare a few forest settings without touching the final test set.

    We first compare three tree configurations on training-only validation.
    Then we calibrate the unrestricted forest and the best regularized one.
    The best calibrated macro F1 wins (accuracy and Brier break ties).
    """
    # Hold 20% of the TRAINING partition for model selection.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=43, stratify=y_train,
    )

    # Part A: Which settings work best before calibration?
    comparisons = []
    for name, settings in REGULARIZATION_OPTIONS:
        model = make_pipeline(trees, **settings)
        model.fit(X_fit, y_fit)
        results = evaluate(
            y_val, model.predict(X_val), model.predict_proba(X_val),
            model.classes_,
        )

        comparisons.append({
            "candidate": name,
            "parameters": settings,
            "validation_accuracy": results["accuracy"],
            "validation_macro_f1": results["macro_f1"],
            "validation_reset_recall": results["per_class"]["Reset"]["recall"],
            "exact_100_count": results["calibration_diagnostics"]["exact_100_confidence_count"],
        })
        print(
            f"  {name}: macro F1={results['macro_f1']:.4f}, "
            f"Reset recall={results['per_class']['Reset']['recall']:.3f}",
            flush=True,
        )

    # Always compare the unrestricted forest with the BEST regularized one.
    unrestricted = comparisons[0]
    best_regularized = max(
        comparisons[1:],
        key=lambda result: (
            result["validation_macro_f1"],
            result["validation_accuracy"],
        ),
    )

    # Part B: Check which calibrated candidate gives better probabilities.
    calibrated_comparison = []
    for candidate in (unrestricted, best_regularized):
        model = make_calibrated_pipeline(
            trees, **candidate["parameters"], calibration_folds=calibration_folds,
        )
        model.fit(X_fit, y_fit)
        results = evaluate(
            y_val, model.predict(X_val), model.predict_proba(X_val),
            model.classes_, calibrated=True,
        )
        calibrated_comparison.append({
            "candidate": candidate["candidate"],
            "parameters": candidate["parameters"],
            "validation_accuracy": results["accuracy"],
            "validation_macro_f1": results["macro_f1"],
            "validation_reset_recall": results["per_class"]["Reset"]["recall"],
            "validation_brier_score": results["calibration_diagnostics"]["multiclass_brier_score"],
            "exact_100_count": results["calibration_diagnostics"]["exact_100_confidence_count"],
        })
        print(
            f"  calibrated {candidate['candidate']}: "
            f"macro F1={results['macro_f1']:.4f}, "
            f"Reset recall={results['per_class']['Reset']['recall']:.3f}",
            flush=True,
        )

    best = max(
        calibrated_comparison,
        key=lambda result: (
            result["validation_macro_f1"],
            result["validation_accuracy"],
            -result["validation_brier_score"],
        ),
    )

    selection_report = {
        "procedure": (
            "Training-only stratified 20% validation (random_state=43); "
            "select calibrated macro-F1, break ties by accuracy and Brier"
        ),
        "validation_size": len(y_val),
        "regularization_candidates": comparisons,
        "calibrated_finalists": calibrated_comparison,
        "selected_candidate": best["candidate"],
    }
    return best["parameters"], selection_report



def main():
    # Classroom-style settings: 100 trees, 5 validation folds and 3 calibration folds.
    dataset_path = ORIGINAL_DATA_PATH
    trees = 100
    cv_folds = 5
    calibration_folds = 3

    # Step 1: Read and clean the raw CSV. Keep a report of excluded rows.
    rows, _counts, quality = load_dataset(dataset_path)
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(
        f"Source: {dataset_path.name}; original rows: {quality['raw_rows']:,}; "
        f"retained: {len(rows):,}; duplicate conflicts quarantined: "
        f"{quality['conflicting_feature_rows_removed']:,}"
    )

    # Step 2: Separate input measurements (X) from action labels (y).
    X = pd.DataFrame(
        [row["features"] for row in rows], columns=FEATURE_NAMES, dtype=float,
    )
    y = pd.Series([row["action"] for row in rows], name="action")
    # Step 3: Keep 20% of records aside for independent testing.
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
    print("Comparing tree regularization and calibration on training-only validation...")
    selected_params, selection = choose_parameters(
        X_train, y_train, trees, calibration_folds,
    )
    print(f"Selected {selection['selected_candidate']}: {selected_params}", flush=True)
    pipeline = make_pipeline(trees, **selected_params)
    # Step 4: Cross-validate the selected base forest on training data only.
    cv = None
    if cv_folds:
        print(f"Evaluating {cv_folds}-fold stratified CV on training data only...")
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=splitter,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            n_jobs=1, error_score="raise", return_train_score=True,
        )
        training_accuracy = float(np.mean(scores["train_accuracy"]))
        validation_accuracy = float(np.mean(scores["test_accuracy"]))
        accuracy_gap = 100 * (training_accuracy - validation_accuracy)

        cv = {
            "folds": cv_folds,
            "accuracy_scores": [float(x) for x in scores["test_accuracy"]],
            "mean_accuracy": validation_accuracy,
            "macro_f1_scores": [float(x) for x in scores["test_macro_f1"]],
            "mean_macro_f1": float(np.mean(scores["test_macro_f1"])),
            "training_accuracy_scores": [float(x) for x in scores["train_accuracy"]],
            "mean_training_accuracy": training_accuracy,
            "training_validation_gap_percentage_points": accuracy_gap,
            "validation_accuracy_at_least_85_percent": validation_accuracy >= 0.85,
            "training_validation_gap_at_most_5_percentage_points": accuracy_gap <= 5,
        }
        print(
            f"Training CV accuracy: {cv['mean_accuracy']:.4%}, "
            f"macro F1: {cv['mean_macro_f1']:.4f}"
        )
    # Step 5: Calibrate with internal training folds, then evaluate on X_test.
    print("Fitting training-only cross-validated sigmoid calibration...", flush=True)
    calibrated_pipeline = make_calibrated_pipeline(
        trees, **selected_params, calibration_folds=calibration_folds,
    )
    calibrated_pipeline.fit(X_train, y_train)
    predicted = calibrated_pipeline.predict(X_test)
    probabilities = calibrated_pipeline.predict_proba(X_test)
    metrics = evaluate(y_test, predicted, probabilities,
                       calibrated_pipeline.classes_, calibrated=True)
    # The final base forest from ensemble=False is also trained on X_train.
    raw_model = calibrated_pipeline.calibrated_classifiers_[0].estimator
    raw_metrics = evaluate(y_test, raw_model.predict(X_test),
                           raw_model.predict_proba(X_test), raw_model.classes_)
    report = {
        "dataset": DATASET_NAME,
        "dataset_doi": DATASET_DOI,
        "dataset_file": dataset_path.name,
        "implementation": "CalibratedClassifierCV(sigmoid, Pipeline(SimpleImputer, RandomForestClassifier))",
        "model_selection": selection,
        "calibration": {
            "method": "sigmoid", "cv_folds": calibration_folds,
            "training_only": True, "ensemble": False,
            "note": "Calibration uses only the 80% training partition.",
        },
        "model_parameters": {
            "n_estimators": trees,
            "random_state": 42,
            "class_weight": "balanced",
            "imputer_strategy": "median",
            **selected_params,
        },
        "quality_report": quality,
        "train_split_size": len(X_train),
        "test_split_size": len(X_test),
        "training_class_counts": y_train.value_counts().to_dict(),
        "test_class_counts": y_test.value_counts().to_dict(),
        "cross_validation": cv,
        "uncalibrated_holdout": {
            "accuracy": raw_metrics["accuracy"],
            "macro_f1": raw_metrics["macro_f1"],
            "per_class": raw_metrics["per_class"],
            "calibration_diagnostics": raw_metrics["calibration_diagnostics"],
        },
        **metrics,
        "warning": (
            "Sigmoid-calibrated confidence is not a guarantee of correctness. "
            "Reset has very few examples. Performance on historical UCI traffic "
            "does not establish reliability on new network environments."
        ),
    }
    # Step 6: Save the fitted model, its feature names, and evaluation reports.
    # Joblib files are only safe to load when created by trusted code.
    bundle = {
        "pipeline": calibrated_pipeline,
        "feature_names": FEATURE_NAMES,
        "dataset_name": DATASET_NAME,
        "dataset_file": dataset_path.name,
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
    print(f"Uncalibrated ECE (10 bins): {raw_metrics['calibration_diagnostics']['ece_10_bins']:.4f}")
    print(f"Calibrated ECE (10 bins): {metrics['calibration_diagnostics']['ece_10_bins']:.4f}")
    print(f"Exact 100% confidence count: raw="
          f"{raw_metrics['calibration_diagnostics']['exact_100_confidence_count']}, "
          f"calibrated={metrics['calibration_diagnostics']['exact_100_confidence_count']}")
    print(f"Model: {MODEL_PATH}")
    print(f"Reports: {REPORT_PATH} and {QUALITY_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
