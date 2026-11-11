"""Tests for the pandas preprocessing workflow taught in class."""
from __future__ import annotations

import csv
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from data_pipeline import load_dataset, norm_header, normalize_action, split_train_validation_test

HEADERS = ["Source Port", "Destination Port", "NAT Source Port",
           "NAT Destination Port", "Action", "Bytes", "Bytes Sent",
           "Bytes Received", "Packets", "Elapsed Time (sec)",
           "pkts_sent", "pkts_received"]


def sample_row(index: int, action: str = "allow") -> dict:
    return {"Source Port": str(1000 + index),
            "Destination Port": "443",
            "NAT Source Port": str(1000 + index),
            "NAT Destination Port": "443",
            "Action": action,
            "Bytes": "300", "Bytes Sent": "100", "Bytes Received": "200",
            "Packets": "3", "Elapsed Time (sec)": "1.5",
            "pkts_sent": "2", "pkts_received": "1"}


class DataPreprocessingTests(unittest.TestCase):
    def run_cleaner(self, rows, headers=HEADERS):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log2.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            return load_dataset(path)

    def test_action_labels_and_column_names(self):
        self.assertEqual(norm_header("Elapsed Time (sec)"), "elapsed_time_sec")
        self.assertEqual(normalize_action("RESET-BOTH"), "Reset")
        self.assertEqual(normalize_action("allow"), "Allow")
        self.assertIsNone(normalize_action("reset-client"))
        self.assertIsNone(normalize_action("action"))

    def test_unknown_action_is_not_guessed(self):
        records, counts, audit = self.run_cleaner(
            [sample_row(1), sample_row(2, "action"),
             sample_row(3, "deny"), sample_row(4, "drop"),
             sample_row(5, "reset-both")])
        self.assertEqual(len(records), 4)
        self.assertEqual(audit["invalid_reason_counts"]["unknown_action"], 1)
        self.assertEqual(counts, Counter({"Allow": 1, "Deny": 1, "Drop": 1, "Reset": 1}))

    def test_missing_numerical_measurements_are_left_for_training_imputer(self):
        missing = sample_row(1)
        missing["Elapsed Time (sec)"] = ""
        rows = [missing, sample_row(2, "deny"), sample_row(3, "drop"),
                sample_row(4, "reset-both")]
        records, _, audit = self.run_cleaner(rows)
        self.assertEqual(len(records), 4)
        self.assertTrue(math.isnan(records[0]["features"][8]))
        self.assertEqual(audit["missing_feature_counts_before_imputation"]["connection_duration"], 1)

    def test_bad_numeric_rows_are_reported_and_excluded(self):
        mutations = [
            ("Source Port", "65536", "port_out_of_range"),
            ("NAT Destination Port", "-1", "negative_feature"),
            ("Packets", "2.2", "noninteger_count_or_port"),
            ("Bytes", "inf", "nonfinite_feature"),
            ("Elapsed Time (sec)", "not numeric", "nonnumeric_feature"),
            ("Packets", "1", "inconsistent_packet_count"),
            ("Bytes", "50", "inconsistent_byte_count"),
            ("Packets", "0", "nonpositive_packet_count")]
        rows = []
        for index, (field, value, _) in enumerate(mutations):
            record = sample_row(index + 1)
            record[field] = value
            rows.append(record)
        rows.extend([sample_row(100, "allow"), sample_row(101, "deny"),
                     sample_row(102, "drop"), sample_row(103, "reset-both")])
        remaining, _, audit = self.run_cleaner(rows)
        self.assertEqual(len(remaining), 4)
        for _, _, reason in mutations:
            self.assertEqual(audit["invalid_reason_counts"][reason], 1)

    def test_duplicate_rows_and_conflicting_labels_are_removed(self):
        original = sample_row(1)
        rows = [original, dict(original), sample_row(1, "deny"),
                sample_row(2, "allow"), sample_row(3, "deny"),
                sample_row(4, "drop"), sample_row(5, "reset-both")]
        clean, counts, audit = self.run_cleaner(rows)
        self.assertEqual(len(clean), 4)
        self.assertEqual(audit["exact_duplicate_rows_removed"], 1)
        self.assertEqual(audit["conflicting_feature_groups"], 1)
        self.assertEqual(audit["conflicting_feature_rows_removed"], 2)
        self.assertEqual(counts["Allow"], 1)

    def test_normalized_duplicate_header_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            self.run_cleaner([dict(sample_row(1), **{"Action!": "allow"})],
                             HEADERS + ["Action!"])

    def test_missing_dataset_gives_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "data/log2.csv"):
            load_dataset(Path("not-a-real-file.csv"))

    def test_reproducible_stratified_train_validation_test_split(self):
        raw = [sample_row(index + group * 1000, label)
               for group, label in enumerate(("allow", "deny", "drop", "reset-both"))
               for index in range(30)]
        clean, counts, _ = self.run_cleaner(raw)
        train, val, test = split_train_validation_test(clean)
        t2, v2, h2 = split_train_validation_test(clean)
        self.assertEqual([train, val, test], [t2, v2, h2])
        self.assertEqual(sum(map(len, (train, val, test))), len(clean))
        for cls in ("Allow", "Deny", "Drop", "Reset"):
            self.assertGreater(Counter(r["action"] for r in val)[cls], 0)
            self.assertGreater(Counter(r["action"] for r in test)[cls], 0)
        keys = [{tuple(r["features"]) for r in part} for part in (train, val, test)]
        self.assertFalse(keys[0] & keys[1])
        self.assertFalse(keys[0] & keys[2])
        self.assertFalse(keys[1] & keys[2])


if __name__ == "__main__":
    unittest.main()
