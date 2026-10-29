"""Regression tests for preprocessing and leakage controls.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from data_pipeline import (
    capped_resampling, load_dataset, normalize_action, norm_header,
    original_distribution_sample, split_train_validation_test,
)

HEADERS = [
    "Source Port", "Destination Port", "NAT Source Port",
    "NAT Destination Port", "Action", "Bytes", "Bytes Sent",
    "Bytes Received", "Packets", "Elapsed Time (sec)",
    "pkts_sent", "pkts_received",
]


def make_row(i: int, action: str = "allow") -> dict:
    return {
        "Source Port": str(1000 + i),
        "Destination Port": "443", "NAT Source Port": str(1000 + i),
        "NAT Destination Port": "443", "Action": action,
        "Bytes": "300", "Bytes Sent": "100", "Bytes Received": "200",
        "Packets": "3", "Elapsed Time (sec)": "1.5",
        "pkts_sent": "2", "pkts_received": "1",
    }


class DataPipelineTests(unittest.TestCase):
    def parse(self, rows, headers=HEADERS):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            return load_dataset(path)

    def test_normalized_labels_do_not_guess_deny(self):
        self.assertEqual(norm_header("Elapsed Time (sec)"), "elapsed_time_sec")
        self.assertEqual(normalize_action(" reset-both "), "Reset")
        self.assertEqual(normalize_action("ALLoW"), "Allow")
        self.assertIsNone(normalize_action("action"))
        self.assertIsNone(normalize_action("reset-client"))

    def test_invalid_action_is_rejected_and_reported(self):
        good = make_row(1)
        ambiguous = make_row(2, "action")
        rows, counts, audit = self.parse([good, ambiguous])
        self.assertEqual(counts, Counter({"Allow": 1}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["invalid_reason_counts"]["unknown_action"], 1)

    def test_missing_values_are_not_silently_imputed(self):
        incomplete = make_row(1)
        incomplete["Packets"] = ""
        rows, _, audit = self.parse([incomplete, make_row(2)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["invalid_reason_counts"]["missing_feature"], 1)

    def test_reject_invalid_numeric_ranges_and_consistency(self):
        mutations = [
            ("Source Port", "65536", "port_out_of_range"),
            ("NAT Destination Port", "-1", "negative_feature"),
            ("Packets", "2.2", "noninteger_count_or_port"),
            ("Bytes", "inf", "nonfinite_feature"),
            ("Elapsed Time (sec)", "not a number", "nonnumeric_feature"),
            ("Packets", "1", "inconsistent_packet_count"),
            ("Bytes", "50", "inconsistent_byte_count"),
            ("Packets", "0", "nonpositive_packet_count"),
        ]
        invalid = []
        for i, (field, value, _) in enumerate(mutations):
            row = make_row(i + 1)
            row[field] = value
            invalid.append(row)
        rows, _, audit = self.parse(invalid + [make_row(25)])
        self.assertEqual(len(rows), 1)
        for _, _, reason in mutations:
            self.assertEqual(audit["invalid_reason_counts"][reason], 1)

    def test_exact_duplicates_and_conflicts_removed_before_split(self):
        first = make_row(1)
        exact = dict(first)
        conflicting = make_row(1, "deny")
        other = make_row(2)
        rows, counts, audit = self.parse([first, exact, conflicting, other])
        self.assertEqual(len(rows), 1)
        self.assertEqual(counts, Counter({"Allow": 1}))
        self.assertEqual(audit["exact_duplicate_rows_removed"], 1)
        self.assertEqual(audit["conflicting_feature_groups"], 1)
        self.assertEqual(audit["conflicting_feature_rows_removed"], 2)

    def test_reject_duplicate_columns_after_normalization(self):
        extra = HEADERS + ["Action!"]
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            self.parse([dict(make_row(1), **{"Action!": "allow"})], headers=extra)

    def test_split_is_stratified_disjoint_and_reproducible(self):
        raw = [
            make_row(index + offset * 1000, action)
            for offset, action in enumerate(("allow", "deny", "drop", "reset-both"))
            for index in range(30)
        ]
        rows, counts, _ = self.parse(raw)
        train, validation, test = split_train_validation_test(rows)
        train2, validation2, test2 = split_train_validation_test(rows)
        self.assertEqual(
            [[tuple(r["features"]) for r in partition] for partition in (train, validation, test)],
            [[tuple(r["features"]) for r in partition] for partition in (train2, validation2, test2)],
        )
        self.assertEqual(len(train) + len(validation) + len(test), len(rows))
        for action in ("Allow", "Deny", "Drop", "Reset"):
            self.assertEqual(counts[action], 30)
            self.assertGreater(Counter(r["action"] for r in validation)[action], 0)
            self.assertGreater(Counter(r["action"] for r in test)[action], 0)
        groups = [{tuple(row["features"]) for row in segment}
                  for segment in (train, validation, test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_oversampling_has_strict_repeat_cap(self):
        original = [
            {"action": action, "features": [i, 0]}
            for action, size in (("Allow", 30), ("Deny", 15), ("Drop", 10), ("Reset", 2))
            for i in range(size)
        ]
        augmented = capped_resampling(
            original, target_per_class=10, max_repeat_factor=3
        )
        by_action = Counter(r["action"] for r in augmented)
        self.assertEqual(by_action["Reset"], 6)
        self.assertEqual(by_action["Allow"], 10)
        seen = Counter((r["action"], tuple(r["features"])) for r in augmented)
        self.assertTrue(all(count <= 3 for count in seen.values()))

    def test_proportional_sampling_bounded(self):
        original = [
            {"action": action, "features": [i, 0]}
            for action, size in (("Allow", 90), ("Deny", 40), ("Drop", 20), ("Reset", 5))
            for i in range(size)
        ]
        sampled = original_distribution_sample(original, max_total=50)
        self.assertEqual(len(sampled), 50)
        self.assertEqual(len({id(x) for x in sampled}), 50)
        self.assertEqual(set(row["action"] for row in sampled),
                         {"Allow", "Deny", "Drop", "Reset"})


if __name__ == "__main__":
    unittest.main()
