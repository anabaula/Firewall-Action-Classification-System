# Firewall Action Classification System

Educational Flask application predicting **Allow, Deny, Drop, Reset** from
historical **UCI Internet Firewall Data** records using a dependency-free,
pure-Python Random Forest implementation.

> This system does not configure or operate an actual firewall. Its input
> includes whole-connection traffic measurements; it is a retrospective
> classification demo, not a proven real-time filtering system.

## Project data and classmate contribution

- `data/log2_cleaned.csv`: cleaned subset supplied by a classmate (included).
- `notebooks/DataCleaning.ipynb`: classmate's exploratory notebook (included
  as reference; **not executed** by the Flask app).
- `data/log2.csv`: optional original UCI CSV, if separately acquired.
- `data_pipeline.py`: authoritative cleaning/validation and splitting used
  by the website's model trainer.
- `train_model.py`: chooses a training strategy using validation data, then
  fits the model and evaluates it once on the held-out test set.

By default, `train_model.py` uses `data/log2_cleaned.csv` when it exists and
falls back to `data/log2.csv`. To run against the original file explicitly,
use `python train_model.py --dataset data/log2.csv`.

**Provenance limitation:** The classmate's CSV has 57,170 rows versus 65,532
records reported for the original UCI dataset. The notebook opens an
already-cleaned CSV; it does not demonstrate every step used to remove rows
from the original. Treat this as a supplied subset, not a fully reproducible
cleaning transformation. Original dataset: https://archive.ics.uci.edu/dataset/542/internet%2Bfirewall%2Bdata
(doi:10.24432/C5131M).

## Setup and usage (Windows)

Requirements: Python 3.10+ and Flask. No compiled NumPy, pandas, SciPy,
scikit-learn or joblib dependency is required by the Flask website or
the pure-Python trainer.

1. In the project folder, double-click `setup_project.bat` to create
   `.venv` and install Flask. Or run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. Run the data audit **without training**:
   ```powershell
   .\.venv\Scripts\python.exe train_model.py --validate-only
   ```
   It writes `data/data_quality_report.json`.
3. Train and evaluate:
   ```powershell
   .\.venv\Scripts\python.exe train_model.py
   ```
   Or double-click `train_model.bat`. Outputs:
   `model/firewall_random_forest.json`,
   `data/evaluation_report.json`, and `data/data_quality_report.json`.
4. Double-click `start_website.bat`, or run:
   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```
   Open http://127.0.0.1:5000. Keep the terminal open while using the site.

Test preprocessing separately using the **standard library** (no extra
packages required):

```powershell
py -m unittest discover -s tests -v
```

`verify_project.bat` checks that supporting files are present, but is
not a substitute for running tests or inspecting the evaluation report.

## Implemented preprocessing protocol

1. Normalize and verify CSV column names against the expected schema.
2. Normalize labels conservatively (`reset-both` -> `Reset`). Unknown
   labels are **excluded and counted**, never guessed as Deny.
3. Reject missing, nonnumeric, nonfinite, negative, fractional count/port,
   out-of-range port, impossible packet-total, and inconsistent byte-total
   observations. Each exclusion reason is counted in a JSON report. No
   median imputation is necessary for this supplied subset; missing records
   are explicitly excluded. No indiscriminate outlier clipping is applied.
4. Remove exact duplicates, then quarantine every identical feature vector
   associated with conflicting labels. This prevents those exact vectors
   from appearing on opposite sides of the split.
5. Make a reproducible, stratified **64% training / 16% validation / 20% test**
   split, with an assertion against exact-feature overlap.
6. Compare (a) a bounded prevalence-proportional sample against (b) capped
   class oversampling. Oversampling is applied **only to training data**;
   minority records are repeated at most 3 times for a candidate. Selection
   uses **validation macro-F1**, never test performance.
7. Refit the chosen strategy using the combined training and validation
   records, then evaluate once on the **untouched, unbalanced test split**.
   Report accuracy, macro-F1, precision/recall/F1/support by class, and a
   confusion matrix. The final probabilities are not calibrated.

The current supplied CSV contains 57,170 records. The validation audit
found **0 invalid rows, 0 exact row duplicates**, and **9 feature vectors
with contradictory labels (18 rows removed)**, leaving **57,152 records**.
Actual results are saved to the generated JSON reports.

**Further research limitations:** There are very few Reset records, making
per-class estimates unstable. Records may have hidden session/site/time
relationships not represented in these 11 features; removal of identical
feature vectors does not guarantee independence between sessions. Testing
against data from a different network or period would be needed before any
real-world generalization claim.

## Notebook compatibility

The classmate's notebook uses pandas, seaborn, statsmodels and scikit-learn.
Its first cell has an absolute path from a different Windows machine, so
update that cell to use `data/log2_cleaned.csv` before executing the notebook.
Its `.pkl` scikit-learn model is **not** compatible with
`forest_model.py` or Flask's JSON model loader. The notebook is for
exploration, not part of the application's training pipeline.

## Project layout

```text
app.py                         Flask website and prediction API
forest_model.py                Pure-Python Random Forest
data_pipeline.py               Data validation, deduplication and splitting
train_model.py                 Validation-based model selection and evaluation
requirements.txt              Flask (website dependency)
data/log2_cleaned.csv          Classmate-provided cleaned subset
data/log2.csv                  Optional original UCI CSV (not included)
data/data_quality_report.json  Generated cleaning audit
data/evaluation_report.json    Generated final performance report
model/firewall_random_forest.json  Generated deployable JSON model
notebooks/DataCleaning.ipynb   Classmate research notebook
tests/test_preprocessing.py    Regression tests
templates/index.html           Web interface
static/app.js                  Browser-local prediction history
static/style.css               Styling
setup_project.bat              Windows venv setup
train_model.bat                Windows training command
start_website.bat              Windows Flask launcher
verify_project.bat             Windows file checks
```
