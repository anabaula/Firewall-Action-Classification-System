"""Train a pure-Python Random Forest on the real UCI Internet Firewall Data.

Before running, download the UCI dataset archive and place its log2.csv file at:
    data/log2.csv

No NumPy, SciPy, pandas, scikit-learn, or joblib is required. The rare Reset-Both
class is extremely underrepresented in the source data; this script oversamples
that class in the TRAINING set only and reports metrics against a held-out split
that retains the source class distribution.
"""
from __future__ import annotations
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from forest_model import ACTIONS, FEATURE_NAMES, predict, save_model, train_forest

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "log2.csv"
MODEL_PATH = BASE_DIR / "model" / "firewall_random_forest.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation_report.json"

# Maps the UCI source columns to the input feature fields used by this website.
CSV_TO_FEATURE = {
    "source_port": "source_port",
    "destination_port": "destination_port",
    "nat_source_port": "nat_source_port",
    "nat_destination_port": "nat_destination_port",
    "bytes": "total_bytes",
    "bytes_sent": "bytes_sent",
    "bytes_received": "bytes_received",
    "packets": "total_packets",
    "elapsed_time_sec": "connection_duration",
    "pkts_sent": "forward_packets",
    "pkts_received": "backward_packets",
}


def norm_header(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_action(value: str) -> str | None:
    label = value.strip().lower().replace("_", "-").replace(" ", "-")
    if label in {"allow", "deny", "drop"}:
        return label.title()
    if label in {"reset-both", "reset", "reset-client", "reset-server"}:
        return "Reset"
    # Some mirrors/documentation have inconsistently called the deny category "action".
    if label == "action":
        return "Deny"
    return None


def load_dataset(path: Path) -> tuple[list[dict], Counter, Counter]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\nDownload the UCI Internet Firewall Data ZIP, extract log2.csv, "
            f"and place it in the project's data folder as data\\log2.csv."
        )
    rows = []
    bad_rows = 0
    source_counts = Counter()
    required = set(CSV_TO_FEATURE) | {"action"}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The CSV appears to have no header row. Use the original UCI log2.csv file.")
        header_map = {norm_header(h): h for h in reader.fieldnames}
        # UCI calls the time field 'Elapsed Time (sec)', normalized to elapsed_time_sec.
        missing = sorted(required - set(header_map))
        if missing:
            raise ValueError(
                "The CSV columns do not match the expected UCI dataset. Missing: " + ", ".join(missing) +
                ". Make sure you extracted the original log2.csv, not a different file."
            )
        for raw in reader:
            action = normalize_action(raw[header_map["action"]] or "")
            if action not in ACTIONS:
                bad_rows += 1
                continue
            try:
                features = []
                for feature_name in FEATURE_NAMES:
                    # Translate the UI/model feature name back to its UCI column name.
                    source_name = next(k for k, v in CSV_TO_FEATURE.items() if v == feature_name)
                    raw_value = (raw[header_map[source_name]] or "").strip().replace(",", "")
                    number = float(raw_value)
                    if not math.isfinite(number):
                        raise ValueError("non-finite number")
                    features.append(number)
            except (TypeError, ValueError, KeyError, StopIteration):
                bad_rows += 1
                continue
            rows.append({"features": features, "action": action})
            source_counts[action] += 1
    if not rows:
        raise ValueError("No usable records were found in log2.csv.")
    return rows, source_counts, Counter({"bad_or_ignored_rows": bad_rows})


def stratified_split(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["action"]].append(row)
    train, test = [], []
    for action in ACTIONS:
        group = grouped[action][:]
        rng.shuffle(group)
        if len(group) < 2:
            train.extend(group)
            continue
        test_count = max(1, round(len(group) * 0.20))
        test.extend(group[:test_count])
        train.extend(group[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def balance_training_only(train_rows: list[dict], target_per_class: int = 600, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in train_rows:
        grouped[row["action"]].append(row)
    balanced = []
    for action in ACTIONS:
        group = grouped[action]
        if not group:
            print(f"WARNING: no training examples for {action}; model cannot learn this class.")
            continue
        if len(group) >= target_per_class:
            chosen = rng.sample(group, target_per_class)
        else:
            chosen = group[:] + [rng.choice(group) for _ in range(target_per_class - len(group))]
        balanced.extend({"features": row["features"][:], "action": action} for row in chosen)
    rng.shuffle(balanced)
    return balanced


def evaluation_metrics(test_rows: list[dict], model: dict) -> dict:
    confusion = {actual: {predicted: 0 for predicted in ACTIONS} for actual in ACTIONS}
    correct = 0
    for row in test_rows:
        predicted, _, _ = predict(model, row["features"])
        confusion[row["action"]][predicted] += 1
        correct += predicted == row["action"]
    per_class = {}
    f1_values = []
    for action in ACTIONS:
        tp = confusion[action][action]
        support = sum(confusion[action].values())
        predicted_count = sum(confusion[actual][action] for actual in ACTIONS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[action] = {"precision": precision, "recall": recall, "f1_score": f1, "support": support}
        if support:
            f1_values.append(f1)
    total = max(1, len(test_rows))
    return {
        "test_size": len(test_rows),
        "accuracy": correct / total,
        "macro_f1_including_classes_present_in_test": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "test_set_kept_original_class_proportions": True,
    }


def main() -> None:
    rows, source_counts, skipped = load_dataset(DATA_PATH)
    print("Dataset: UCI Internet Firewall Data (real firewall log records)")
    print(f"Usable rows: {len(rows):,}; rows skipped/ignored: {skipped['bad_or_ignored_rows']:,}")
    print("Original class distribution:")
    for action in ACTIONS:
        print(f"  {action:5}: {source_counts[action]:,}")
    if source_counts["Reset"] < 20:
        print("WARNING: Reset examples are very rare. Treat Reset metrics as uncertain.")

    train_rows, test_rows = stratified_split(rows)
    balanced_rows = balance_training_only(train_rows, target_per_class=600)
    print(f"Training split before balancing: {len(train_rows):,}")
    print(f"Training examples after class balancing/oversampling: {len(balanced_rows):,}")
    print(f"Held-out test records (not balanced): {len(test_rows):,}")
    print("Training pure-Python Random Forest (13 trees)...")
    model = train_forest(balanced_rows, tree_count=13, max_depth=6, seed=42)
    model["training_source"] = "UCI Machine Learning Repository: Internet Firewall Data, DOI 10.24432/C5131M"
    model["source_rows_used"] = len(rows)
    model["train_original_rows"] = len(train_rows)
    model["balanced_training_rows"] = len(balanced_rows)
    model["test_rows"] = len(test_rows)
    model["original_class_counts"] = dict(source_counts)
    model["warning"] = (
        "Trained on UCI Internet Firewall Data. Training data was balanced by undersampling majority classes and "
        "oversampling the rare Reset class. Evaluation uses an unbalanced stratified holdout. Reset performance may be unstable. "
        "Educational decision-support prototype only; not for production firewall decisions."
    )
    metrics = evaluation_metrics(test_rows, model)
    report = {
        "dataset": "UCI Internet Firewall Data",
        "dataset_doi": "10.24432/C5131M",
        "source_rows_used": len(rows),
        "skipped_or_ignored_rows": skipped["bad_or_ignored_rows"],
        "original_class_counts": dict(source_counts),
        "train_original_rows": len(train_rows),
        "balanced_training_rows": len(balanced_rows),
        **metrics,
        "warning": "This is a held-out score on this public dataset, not a guarantee of performance on other networks. Reset is rare in the source data.",
    }
    save_model(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nEvaluation on held-out records from the real dataset:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1_including_classes_present_in_test']:.3f}")
    print("Per-class metrics (precision / recall / F1 / support):")
    for action in ACTIONS:
        row = metrics["per_class"][action]
        print(f"  {action:5}: {row['precision']:.3f} / {row['recall']:.3f} / {row['f1_score']:.3f} / {row['support']}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation report: {REPORT_PATH}")
    print("NOTE: these metrics apply only to the held-out UCI dataset split, not to live firewall traffic.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
