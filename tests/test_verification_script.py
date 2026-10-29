"""Windows verifier must report failure when required project files are absent."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "verify_project.bat"
REQUIRED = [
    "app.py", "model_service.py", "data_pipeline.py", "train_model.py",
    "tests/test_preprocessing.py", "notebooks/DataCleaning.ipynb",
    "requirements.txt", "templates/index.html", "static/app.js",
    "static/style.css", "data/log2_cleaned.csv",
    "model/firewall_random_forest.joblib",
    "data/evaluation_report.json", "data/data_quality_report.json",
    ".venv/Scripts/python.exe",
]


@unittest.skipUnless(sys.platform == "win32", "Windows .bat verification only")
class ProjectVerifierTests(unittest.TestCase):
    def run_copy(self, create_files=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / SCRIPT.name
            shutil.copy2(SCRIPT, script)
            for name in create_files:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            return subprocess.run(
                ["cmd.exe", "/d", "/c", str(script)],
                input="\n", capture_output=True, text=True, timeout=15,
                cwd=root,
            )

    def test_missing_files_return_nonzero_exit_status(self):
        result = self.run_copy()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[MISSING]", result.stdout)
        self.assertIn("FAILED:", result.stdout)

    def test_complete_file_set_returns_success(self):
        result = self.run_copy(REQUIRED)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Verification passed:", result.stdout)
        self.assertNotIn("[MISSING]", result.stdout)

    def test_missing_dataset_alone_returns_failure(self):
        present = [name for name in REQUIRED if name != "data/log2_cleaned.csv"]
        result = self.run_copy(present)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("data\\log2_cleaned.csv or data\\log2.csv", result.stdout)


if __name__ == "__main__":
    unittest.main()
