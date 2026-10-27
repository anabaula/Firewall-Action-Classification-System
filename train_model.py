"""Train and evaluate the educational firewall Random Forest with leakage controls.

Default input: classmate-provided data/log2_cleaned.csv (if present), otherwise
original UCI data/log2.csv. The supplied notebook is retained for provenance,
but its scikit-learn model is not compatible with this Flask app's JSON loader.

Stages: validate/clean -> stratified 64/16/20 -> compare two sampling policies
on validation macro-F1 -> retrain on development set -> one held-out test.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_pipeline import (
    capped_resampling,
    load_dataset,
    original_distribution_sample,
    split_train_validation_test,
)
from forest_model import ACTIONS, predict, save_model, train_forest

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DATA_PATH = BASE_DIR / "data" / "log2_cleaned.csv"
ORIGINAL_DATA_PATH = BASE_DIR / "data" / "log2.csv"
MODEL_PATH = BASE_DIR / "model" / "firewall_random_forest.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation_report.json"
QUALITY_PATH = BASE_DIR / "data" / "data_quality_report.json"
DATASET_DOI = "10.24432/C5131M"


def evaluation_metrics(rows: list[dict], model: dict) -> dict:
    if not rows:
        raise ValueError("Evaluation records are empty.")
    confusion = {
        actual: {predicted: 0 for predicted in ACTIONS} for actual in ACTIONS
    }
    for row in rows:
        predicted, _, _ = predict(model, row["features"])
        confusion[row["action"]][predicted] += 1
    per_class = {}
    f1_scores = []
    correct = 0
    for action in ACTIONS:
        tp = confusion[action][action]
        correct += tp
        support = sum(confusion[action].values())
        predicted_count = sum(confusion[actual][action] for actual in ACTIONS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[action] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
        if support:
            f1_scores.append(f1)
    return {
        "test_size": len(rows),
        "accuracy": correct / len(rows),
        "macro_f1_including_classes_present_in_test": (
            sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def select_training_rows(
    rows: list[dict],
    strategy: str,
    max_total: int,
    target_per_class: int,
    seed: int,
) -> list[dict]:
    if strategy == "proportional_sample":
        return original_distribution_sample(rows, max_total=max_total, seed=seed)
    if strategy == "capped_oversampling":
        return capped_resampling(
            rows, target_per_class=target_per_class,
            max_repeat_factor=3, seed=seed
        )
    raise ValueError(f"Unknown resampling strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=CLEANED_DATA_PATH if CLEANED_DATA_PATH.is_file() else ORIGINAL_DATA_PATH,
        help="Input UCI-compatible CSV; defaults to data/log2_cleaned.csv if present.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Audit and clean dataset without training; write data_quality_report.json.",
    )
    parser.add_argument(
        "--trees", type=int, default=13,
        help="Trees in every candidate and final Random Forest (default: 13).",
    )
    parser.add_argument(
        "--max-total", type=int, default=2400,
        help="Maximum training observations for prevalence-proportional candidate.",
    )
    parser.add_argument(
        "--target-per-class", type=int, default=600,
        help="Target per class for capped-resampling candidate (repeats at most 3x).",
    )
    args = parser.parse_args()
    if args.trees < 1 or args.max_total < len(ACTIONS) or args.target_per_class < 1:
        parser.error("trees, max-total and target-per-class must be positive.")

    rows, counts, quality = load_dataset(args.dataset)
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"Dataset: {args.dataset} | raw rows: {quality['raw_rows']:,} "
          f"| cleaned rows: {len(rows):,}")
    print(
        f"Invalid rows: {quality['excluded_invalid_rows']:,} | "
        f"exact duplicates removed: {quality['exact_duplicate_rows_removed']:,} | "
        f"conflicting feature groups: {quality['conflicting_feature_groups']:,} "
        f"({quality['conflicting_feature_rows_removed']:,} rows removed)"
    )
    print("Class counts after cleaning:", dict(counts))
    if args.validate_only:
        print(f"Quality report saved: {QUALITY_PATH}")
        return

    train, validation, test = split_train_validation_test(rows, seed=42)
    print(
        f"Split: training={len(train):,}, validation={len(validation):,}, "
        f"held-out test={len(test):,}"
    )
    print("The untouched test set is excluded from all sampling and model selection.")

    validation_candidates = {}
    strategies = ("proportional_sample", "capped_oversampling")
    for strategy in strategies:
        sample = select_training_rows(
            train, strategy, args.max_total, args.target_per_class, seed=42
        )
        print(f"\nCandidate: {strategy} | training rows: {len(sample):,} "
              f"| class counts: {dict(Counter(r['action'] for r in sample))}")
        candidate = train_forest(sample, tree_count=args.trees, max_depth=6, seed=42)
        metrics = evaluation_metrics(validation, candidate)
        validation_candidates[strategy] = {
            "training_rows": len(sample),
            "training_class_counts": dict(Counter(r["action"] for r in sample)),
            "metrics": metrics,
        }
        print(
            f"Validation macro-F1={metrics['macro_f1_including_classes_present_in_test']:.4f}; "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    # Validation macro-F1 is primary. Tie favours prevalence-proportional
    # sampling because it repeats fewer rare-class records.
    selected = max(
        strategies,
        key=lambda strategy: (
            validation_candidates[strategy]["metrics"][
                "macro_f1_including_classes_present_in_test"
            ],
            strategy == "proportional_sample",
        ),
    )
    print(f"\nSelected on validation only: {selected}")
    development = train + validation
    final_sample = select_training_rows(
        development, selected, args.max_total, args.target_per_class, seed=142
    )
    model = train_forest(final_sample, tree_count=args.trees, max_depth=6, seed=142)
    model.update({
        "training_source": f"UCI Internet Firewall Data DOI {DATASET_DOI}",
        "dataset_file": args.dataset.name,
        "dataset_rows_raw": quality["raw_rows"],
        "dataset_rows_after_cleaning": len(rows),
        "development_rows_before_sampling": len(development),
        "training_rows_after_sampling": len(final_sample),
        "training_class_counts": dict(Counter(r["action"] for r in final_sample)),
        "test_rows": len(test),
        "resampling_strategy": selected,
        "warning": (
            "Educational classifier on historical UCI firewall records, "
            "not a live enforcement engine. Reset-Both is shown as Reset. "
            "Class probabilities are uncalibrated, especially after resampling."
        ),
    })
    test_metrics = evaluation_metrics(test, model)
    report = {
        "dataset": "UCI Internet Firewall Data",
        "dataset_doi": DATASET_DOI,
        "dataset_file": args.dataset.name,
        "quality_report": quality,
        "train_split_size": len(train),
        "validation_split_size": len(validation),
        "test_split_size": len(test),
        "validation_candidates": validation_candidates,
        "selected_resampling_strategy": selected,
        "model_trees": args.trees,
        "development_rows_before_sampling": len(development),
        "final_training_rows": len(final_sample),
        "final_training_class_counts": model["training_class_counts"],
        "test_set_kept_original_class_proportions": True,
        "test_metrics": test_metrics,
        **test_metrics,
        "warning": (
            "Test data were excluded from model selection, sampling and training. "
            "A single stratified holdout does not prove cross-network generalization. "
            "Reset metrics have small support, and predicted probabilities are "
            "uncalibrated. Cleaned CSV provenance is not independently verified."
        ),
    }
    save_model(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nHELD-OUT TEST (never used to select the sampling policy):")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1_including_classes_present_in_test']:.4f}")
    for action in ACTIONS:
        scores = test_metrics["per_class"][action]
        print(
            f"{action:5} support={scores['support']:5d} "
            f"precision={scores['precision']:.3f} "
            f"recall={scores['recall']:.3f} F1={scores['f1_score']:.3f}"
        )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation report: {REPORT_PATH}")
    print(f"Saved data quality report: {QUALITY_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
