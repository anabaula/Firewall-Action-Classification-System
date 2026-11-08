"""Keep the teaching notebook valid and free of machine-specific CSV paths."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "DataCleaning.ipynb"


class NotebookTests(unittest.TestCase):
    def test_notebook_is_valid_json_and_python_compiles(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 15)
        for index, cell in enumerate(notebook["cells"]):
            self.assertIn(cell["cell_type"], ("code", "markdown"))
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")

    def test_relative_dataset_and_pipeline_before_cross_validation(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('pd.read_csv(csv_file)', code)
        self.assertIn('Path("data/log2.csv")', code)
        self.assertIn("SimpleImputer", code)
        self.assertIn("StratifiedKFold", code)
        self.assertIn("train_test_split", code)
        self.assertLess(code.index("models ="), code.index("cross_validate("))
        self.assertNotIn("Users\\", code)
        self.assertNotIn("Downloads\\", code)
        self.assertNotIn("log2_cleaned.csv", code)


if __name__ == "__main__":
    unittest.main()
