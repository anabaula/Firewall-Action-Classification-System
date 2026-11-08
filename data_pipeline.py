"""Pandas-based cleaning of the UCI Internet Firewall Data.

Following the course workflow: inspect -> clean -> identify X/y -> split ->
fit preprocessing INSIDE a scikit-learn Pipeline in train_model.py.

Only invalid observations and ambiguous labels are excluded here.
Missing numeric measurements are left as NaN for training-fold-only imputation.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model_service import ACTIONS, FEATURE_NAMES

COLUMN_MAP = {
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
PORTS = FEATURE_NAMES[:4]
WHOLE_NUMBERS = [feature for feature in FEATURE_NAMES if feature != "connection_duration"]


def norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_action(value: str) -> str | None:
    label = re.sub(r"[-_\s]+", "-", str(value).strip().lower())
    return {"allow": "Allow", "deny": "Deny", "drop": "Drop",
            "reset-both": "Reset", "reset": "Reset"}.get(label)


def load_dataset(path: str | Path) -> tuple[list[dict], Counter, dict]:
    """Return validated records and a reproducible audit of cleaning decisions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset missing: {path}. Supply data/log2.csv.")

    # 1. Read the raw CSV and check that its columns are correct.
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    names = [norm_header(column) for column in raw.columns]
    if len(names) != len(set(names)):
        raise ValueError("Dataset has duplicate column names after normalization.")
    raw.columns = names
    required = set(COLUMN_MAP) | {"action"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("Missing expected columns: " + ", ".join(missing))

    # 2. Convert the target labels and numerical measurements.
    data = pd.DataFrame(index=raw.index)
    labels = raw["action"].fillna("").str.strip()
    data["action"] = labels.map(normalize_action)
    reasons = pd.Series("", index=raw.index, dtype="object")

    def reject(condition: pd.Series, reason: str) -> None:
        reasons.loc[condition.fillna(False) & reasons.eq("")] = reason

    reject(labels.eq(""), "missing_action")
    reject(data["action"].isna(), "unknown_action")

    for source, feature in COLUMN_MAP.items():
        text = raw[source].str.strip().str.replace(",", "", regex=False)
        # A true blank is a missing measurement, not a fabricated zero.
        empty = text.isna() | text.eq("")
        numeric = pd.to_numeric(text.mask(empty), errors="coerce")
        reject(~empty & numeric.isna(), "nonnumeric_feature")
        reject(np.isinf(numeric), "nonfinite_feature")
        data[feature] = numeric

    # 3. Check for impossible values and inconsistent traffic totals.
    # Record why a row is rejected instead of silently deleting it.
    numbers = data[FEATURE_NAMES]
    reject((numbers < 0).any(axis=1), "negative_feature")
    reject((numbers[WHOLE_NUMBERS].notna() &
            numbers[WHOLE_NUMBERS].mod(1).ne(0)).any(axis=1), "noninteger_count_or_port")
    reject((numbers[PORTS] > 65535).any(axis=1), "port_out_of_range")
    reject(numbers["total_packets"].notna() & numbers["total_packets"].le(0),
           "nonpositive_packet_count")
    packet_columns = ["forward_packets", "backward_packets", "total_packets"]
    has_packet_values = numbers[packet_columns].notna().all(axis=1)
    packets_sent = numbers["forward_packets"] + numbers["backward_packets"]
    reject(has_packet_values & (packets_sent > numbers["total_packets"]),
           "inconsistent_packet_count")

    byte_columns = ["bytes_sent", "bytes_received", "total_bytes"]
    has_byte_values = numbers[byte_columns].notna().all(axis=1)
    bytes_transferred = numbers["bytes_sent"] + numbers["bytes_received"]
    reject(has_byte_values & (bytes_transferred > numbers["total_bytes"] * 1.02 + 1e-9),
           "inconsistent_byte_count")

    # 4. Remove exact duplicates and contradictory labels.
    accepted = data.loc[reasons.eq("")].copy()
    source_labels = Counter(accepted["action"])
    null_counts = accepted[FEATURE_NAMES].isna().sum().astype(int).to_dict()
    unique = accepted.drop_duplicates(subset=FEATURE_NAMES + ["action"]).copy()
    # An identical traffic record with two different answers cannot be
    # learned reliably; quarantine all such conflicts before splitting.
    labels_per_traffic = unique.groupby(FEATURE_NAMES, dropna=False)["action"].transform("nunique")
    conflicting = labels_per_traffic > 1
    conflict_groups = unique.loc[conflicting, FEATURE_NAMES].drop_duplicates()
    clean = unique.loc[~conflicting].copy()
    if clean.empty:
        raise ValueError("No valid, unambiguous records remain after cleaning.")
    counts = Counter(clean["action"])
    if set(counts) != set(ACTIONS):
        raise ValueError("Dataset must contain all four firewall action classes.")

    # 5. Return clean records plus a report of every change made.
    records = [
        {"features": [float(row[feature]) for feature in FEATURE_NAMES],
         "action": str(row["action"])}
        for _, row in clean.iterrows()
    ]
    quality = {
        "source_file": path.name,
        "raw_rows": int(len(raw)),
        "valid_rows_before_deduplication": int(len(accepted)),
        "source_class_counts_before_deduplication": dict(source_labels),
        "excluded_invalid_rows": int(reasons.ne("").sum()),
        "invalid_reason_counts": dict(Counter(reasons.loc[reasons.ne("")])),
        "missing_feature_counts_before_imputation": null_counts,
        "exact_duplicate_rows_removed": int(len(accepted) - len(unique)),
        "conflicting_feature_groups": int(len(conflict_groups)),
        "conflicting_feature_rows_removed": int(conflicting.sum()),
        "rows_after_cleaning": int(len(clean)),
        "class_counts_after_cleaning": dict(counts),
        "policy": ("Pandas validation, drop exact duplicates, quarantine conflicting labels; "
                   "fit SimpleImputer only on each training fold. No arbitrary outlier removal.")
    }
    return records, counts, quality


def split_train_validation_test(rows: list[dict], seed: int = 42):
    """Stratified 64/16/20 split for experiments requiring a validation set."""
    development, test = train_test_split(rows, test_size=0.2, random_state=seed,
                                         stratify=[r["action"] for r in rows])
    train, validation = train_test_split(
        development, test_size=0.2, random_state=seed + 1,
        stratify=[r["action"] for r in development])
    return train, validation, test
