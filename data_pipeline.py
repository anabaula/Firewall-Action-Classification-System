"""Step-by-step data cleaning for the UCI Internet Firewall dataset.

Read CSV -> check values -> remove duplicates/conflicts -> return clean rows.
The missing-value median is learned LATER inside train_model.py's Pipeline.
That way, test records never influence the training preprocessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model_service import ACTIONS, FEATURE_NAMES


# The original CSV uses different column names from our 11 website inputs.
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
WHOLE_NUMBERS = [name for name in FEATURE_NAMES if name != "connection_duration"]
ACTION_LABELS = {
    "allow": "Allow",
    "deny": "Deny",
    "drop": "Drop",
    "reset-both": "Reset",
    "reset": "Reset",
}


def norm_header(value):
    """Convert a CSV heading to a simple lowercase column name."""
    name = str(value).strip().lower()
    for character in " ()-/!.":
        name = name.replace(character, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")



def normalize_action(value):
    """Use the same four action names throughout the project."""
    action = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return ACTION_LABELS.get(action)



def mark_invalid(reasons, condition, message):
    """Remember only the FIRST reason a record is invalid."""
    new_errors = condition.fillna(False) & reasons.eq("")
    reasons.loc[new_errors] = message


def load_dataset(path):
    """Return (clean_records, action_counts, cleaning_report).

    Each record has a list of 11 numeric features and one action label.
    The report also explains exactly which rows were removed and why.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset missing: {path}. Supply data/log2.csv.")

    # STEP 1: Load the CSV with pandas and check its headers.
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    raw.columns = [norm_header(name) for name in raw.columns]
    if raw.columns.duplicated().any():
        raise ValueError("Dataset has duplicate column names after normalization.")

    required_columns = set(COLUMN_MAP) | {"action"}
    missing_columns = sorted(required_columns - set(raw.columns))
    if missing_columns:
        raise ValueError("Missing expected columns: " + ", ".join(missing_columns))

    # STEP 2: Convert labels and numbers. Blank numeric cells stay NaN.
    data = pd.DataFrame(index=raw.index)
    labels = raw["action"].fillna("").str.strip()
    data["action"] = labels.map(normalize_action)
    reasons = pd.Series("", index=raw.index, dtype="object")

    mark_invalid(reasons, labels.eq(""), "missing_action")
    mark_invalid(reasons, data["action"].isna(), "unknown_action")

    for csv_column, feature_name in COLUMN_MAP.items():
        values = raw[csv_column].str.strip().str.replace(",", "", regex=False)
        blank = values.isna() | values.eq("")
        numbers = pd.to_numeric(values.mask(blank), errors="coerce")

        mark_invalid(reasons, ~blank & numbers.isna(), "nonnumeric_feature")
        mark_invalid(reasons, np.isinf(numbers), "nonfinite_feature")
        data[feature_name] = numbers

    # STEP 3: Reject impossible network measurements.
    numbers = data[FEATURE_NAMES]
    mark_invalid(reasons, (numbers < 0).any(axis=1), "negative_feature")

    fraction = numbers[WHOLE_NUMBERS].mod(1).ne(0)
    has_value = numbers[WHOLE_NUMBERS].notna()
    mark_invalid(reasons, (has_value & fraction).any(axis=1),
                 "noninteger_count_or_port")

    mark_invalid(reasons, (numbers[PORTS] > 65535).any(axis=1),
                 "port_out_of_range")
    invalid_packets = numbers["total_packets"].notna() & numbers["total_packets"].le(0)
    mark_invalid(reasons, invalid_packets, "nonpositive_packet_count")

    sent_and_received = numbers["forward_packets"] + numbers["backward_packets"]
    mark_invalid(reasons, sent_and_received > numbers["total_packets"],
                 "inconsistent_packet_count")

    total_bytes = numbers["bytes_sent"] + numbers["bytes_received"]
    allowed_bytes = numbers["total_bytes"] * 1.02 + 1e-9
    mark_invalid(reasons, total_bytes > allowed_bytes, "inconsistent_byte_count")

    # STEP 4: Remove duplicates and contradictory answers.
    accepted = data.loc[reasons.eq("")].copy()
    original_counts = accepted["action"].value_counts().to_dict()
    missing_values = accepted[FEATURE_NAMES].isna().sum().astype(int).to_dict()

    # A repeated traffic record with the SAME label only needs one copy.
    unique = accepted.drop_duplicates(subset=FEATURE_NAMES + ["action"]).copy()

    # If identical inputs have DIFFERENT labels, remove the conflicting group.
    action_variants = unique.groupby(FEATURE_NAMES, dropna=False)["action"].transform("nunique")
    conflict = action_variants > 1
    conflict_groups = unique.loc[conflict, FEATURE_NAMES].drop_duplicates()
    clean = unique.loc[~conflict].copy()

    if clean.empty:
        raise ValueError("No valid, unambiguous records remain after cleaning.")

    counts = clean["action"].value_counts().to_dict()
    if set(counts) != set(ACTIONS):
        raise ValueError("Dataset must contain all four firewall action classes.")

    # STEP 5: Return data in the same format expected by the website.
    records = []
    for _, row in clean.iterrows():
        features = [float(row[name]) for name in FEATURE_NAMES]
        records.append({"features": features, "action": str(row["action"])})

    quality = {
        "source_file": path.name,
        "raw_rows": int(len(raw)),
        "valid_rows_before_deduplication": int(len(accepted)),
        "source_class_counts_before_deduplication": dict(original_counts),
        "excluded_invalid_rows": int(reasons.ne("").sum()),
        "invalid_reason_counts": reasons.loc[reasons.ne("")].value_counts().to_dict(),
        "missing_feature_counts_before_imputation": missing_values,
        "exact_duplicate_rows_removed": int(len(accepted) - len(unique)),
        "conflicting_feature_groups": int(len(conflict_groups)),
        "conflicting_feature_rows_removed": int(conflict.sum()),
        "rows_after_cleaning": int(len(clean)),
        "class_counts_after_cleaning": dict(counts),
        "policy": (
            "Pandas validation, drop exact duplicates, quarantine conflicting labels; "
            "fit SimpleImputer only on each training fold. No arbitrary outlier removal."
        ),
    }
    return records, counts, quality


def split_train_validation_test(rows, seed=42):
    """Optional classroom split: 64% training, 16% validation, 20% testing."""
    development, test = train_test_split(
        rows, test_size=0.2, random_state=seed,
        stratify=[row["action"] for row in rows],
    )
    train, validation = train_test_split(
        development, test_size=0.2, random_state=seed + 1,
        stratify=[row["action"] for row in development],
    )
    return train, validation, test
