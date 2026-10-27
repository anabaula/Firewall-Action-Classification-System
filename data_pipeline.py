"""Auditable, dependency-free preprocessing for UCI Internet Firewall Data.

Only observable invalid records are removed. No feature scaling, imputation,
or class resampling is fitted on evaluation data. Duplicate feature vectors with
contradictory labels are quarantined rather than given arbitrary labels.
"""
from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from model_service import ACTIONS, FEATURE_NAMES

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
SOURCE_BY_FEATURE = {feature: column for column, feature in CSV_TO_FEATURE.items()}
DISCRETE_FEATURES = {
    "source_port", "destination_port", "nat_source_port", "nat_destination_port",
    "total_bytes", "bytes_sent", "bytes_received", "total_packets",
    "forward_packets", "backward_packets",
}
PORT_FEATURES = {
    "source_port", "destination_port", "nat_source_port", "nat_destination_port",
}


def norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_action(value: str) -> str | None:
    label = re.sub(r"[-_\s]+", "-", value.strip().lower())
    return {
        "allow": "Allow", "deny": "Deny", "drop": "Drop",
        "reset-both": "Reset", "reset": "Reset",
    }.get(label)


def _validated_record(raw: dict, header_map: dict[str, str]) -> dict:
    if None in raw:
        raise ValueError("malformed_csv_row")
    raw_action = raw.get(header_map["action"])
    if raw_action is None or not raw_action.strip():
        raise ValueError("missing_action")
    action = normalize_action(raw_action)
    if action is None:
        raise ValueError("unknown_action")

    features = []
    for name in FEATURE_NAMES:
        text = raw.get(header_map[SOURCE_BY_FEATURE[name]])
        if text is None or not text.strip():
            raise ValueError("missing_feature")
        try:
            number = float(text.strip().replace(",", ""))
        except ValueError as exc:
            raise ValueError("nonnumeric_feature") from exc
        if not math.isfinite(number):
            raise ValueError("nonfinite_feature")
        if number < 0:
            raise ValueError("negative_feature")
        if name in DISCRETE_FEATURES and not number.is_integer():
            raise ValueError("noninteger_count_or_port")
        if name in PORT_FEATURES and number > 65535:
            raise ValueError("port_out_of_range")
        features.append(int(number) if name in DISCRETE_FEATURES else number)

    values = dict(zip(FEATURE_NAMES, features))
    if values["total_packets"] < 1:
        raise ValueError("nonpositive_packet_count")
    if values["forward_packets"] + values["backward_packets"] > values["total_packets"]:
        raise ValueError("inconsistent_packet_count")
    if values["bytes_sent"] + values["bytes_received"] > values["total_bytes"] * 1.02 + 1e-9:
        raise ValueError("inconsistent_byte_count")
    return {"features": features, "action": action}


def load_dataset(path: Path) -> tuple[list[dict], Counter, dict]:
    """Read, validate, deduplicate and report row exclusions.

    Identical feature vectors with different labels are excluded entirely.
    This avoids arbitrary relabeling and exact-feature overlap across splits.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Supply data/log2_cleaned.csv, "
            "or the original UCI data/log2.csv."
        )
    accepted = []
    excluded = Counter()
    initial_labels = Counter()
    raw_count = 0
    required = set(CSV_TO_FEATURE) | {"action"}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Dataset has no header.")
        normalized = [norm_header(h) for h in reader.fieldnames]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Dataset has duplicate column names after normalization.")
        header_map = dict(zip(normalized, reader.fieldnames))
        missing = sorted(required - set(header_map))
        if missing:
            raise ValueError("Missing expected columns: " + ", ".join(missing))
        for raw in reader:
            raw_count += 1
            try:
                record = _validated_record(raw, header_map)
            except ValueError as exc:
                excluded[str(exc)] += 1
                continue
            accepted.append(record)
            initial_labels[record["action"]] += 1

    # Full duplicates and conflicting feature-only duplicates are separate issues.
    unique = []
    seen_pairs = set()
    duplicate_exact = 0
    actions_by_features = defaultdict(set)
    for record in accepted:
        features = tuple(record["features"])
        pair = (features, record["action"])
        if pair in seen_pairs:
            duplicate_exact += 1
            continue
        seen_pairs.add(pair)
        unique.append(record)
        actions_by_features[features].add(record["action"])

    conflicting_features = {
        key for key, labels in actions_by_features.items() if len(labels) > 1
    }
    conflict_count = sum(
        tuple(record["features"]) in conflicting_features for record in unique
    )
    rows = [
        record for record in unique
        if tuple(record["features"]) not in conflicting_features
    ]
    counts = Counter(record["action"] for record in rows)
    if not rows:
        raise ValueError("No valid, unambiguous records remain after cleaning.")

    quality = {
        "source_file": path.name,
        "raw_rows": raw_count,
        "valid_rows_before_deduplication": len(accepted),
        "source_class_counts_before_deduplication": dict(initial_labels),
        "excluded_invalid_rows": sum(excluded.values()),
        "invalid_reason_counts": dict(excluded),
        "exact_duplicate_rows_removed": duplicate_exact,
        "conflicting_feature_groups": len(conflicting_features),
        "conflicting_feature_rows_removed": conflict_count,
        "rows_after_cleaning": len(rows),
        "class_counts_after_cleaning": dict(counts),
        "policy": (
            "Drop invalid records with counts by reason; remove exact duplicates; "
            "exclude all identical-feature records carrying contradictory labels. "
            "No arbitrary outlier clipping or imputation."
        ),
    }
    return rows, counts, quality


def stratified_split(
    rows: list[dict], test_fraction: float = 0.20, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Reproducible class-stratified split, for already deduplicated rows."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1.")
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["action"]].append(row)
    train, test = [], []
    for action in ACTIONS:
        group = grouped[action][:]
        if len(group) < 5:
            raise ValueError(
                f"At least five independent {action} records are needed "
                "to form training, validation and test splits."
            )
        rng.shuffle(group)
        count = max(1, min(len(group) - 1, round(len(group) * test_fraction)))
        test.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def split_train_validation_test(
    rows: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """64/16/20 stratified split; zero identical-feature leakage by assertion."""
    development, test = stratified_split(rows, 0.20, seed)
    train, validation = stratified_split(development, 0.20, seed + 1)
    partitions = [
        {tuple(row["features"]) for row in segment}
        for segment in (train, validation, test)
    ]
    if (partitions[0] & partitions[1] or partitions[0] & partitions[2]
            or partitions[1] & partitions[2]):
        raise ValueError(
            "Feature-identical records cross train/validation/test splits. "
            "Deduplicate and resolve conflicts before splitting."
        )
    return train, validation, test


def original_distribution_sample(
    rows: list[dict], max_total: int = 2400, seed: int = 42
) -> list[dict]:
    """Bounded stratified subsample preserving approximate natural prevalence."""
    if max_total < len(ACTIONS):
        raise ValueError("max_total is too small to represent each action.")
    rng = random.Random(seed)
    groups = {action: [row for row in rows if row["action"] == action] for action in ACTIONS}
    if any(not group for group in groups.values()):
        raise ValueError("Every target class needs development records.")
    if len(rows) <= max_total:
        sample = rows[:]
        rng.shuffle(sample)
        return sample

    allocations = {
        action: max(1, min(len(group), round(max_total * len(group) / len(rows))))
        for action, group in groups.items()
    }
    while sum(allocations.values()) > max_total:
        action = max((a for a in ACTIONS if allocations[a] > 1), key=lambda a: allocations[a])
        allocations[action] -= 1
    while sum(allocations.values()) < max_total:
        available = [a for a in ACTIONS if allocations[a] < len(groups[a])]
        if not available:
            break
        action = max(available, key=lambda a: len(groups[a]) - allocations[a])
        allocations[action] += 1
    sample = [
        row
        for action in ACTIONS
        for row in rng.sample(groups[action], allocations[action])
    ]
    rng.shuffle(sample)
    return sample


def capped_resampling(
    rows: list[dict],
    target_per_class: int = 600,
    max_repeat_factor: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Undersample major classes and cap minority duplication.

    Minority records are repeated no more than max_repeat_factor times overall,
    unlike unlimited equal-class oversampling.
    """
    if target_per_class < 1 or max_repeat_factor < 1:
        raise ValueError("Resampling parameters must be positive.")
    rng = random.Random(seed)
    grouped = {action: [row for row in rows if row["action"] == action] for action in ACTIONS}
    sample = []
    for action in ACTIONS:
        group = grouped[action]
        if not group:
            raise ValueError(f"No training examples for {action}.")
        target = min(target_per_class, len(group) * max_repeat_factor)
        if len(group) >= target:
            selected = rng.sample(group, target)
        else:
            selected = group[:] * (target // len(group))
            selected += rng.sample(group, target % len(group))
        sample.extend(
            {"features": row["features"][:], "action": row["action"]}
            for row in selected
        )
    rng.shuffle(sample)
    return sample
