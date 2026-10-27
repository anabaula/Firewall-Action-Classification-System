# NetSentry — Firewall Action Classification

NetSentry is an educational Flask website that classifies **historical firewall
connection records** as Allow, Deny, Drop, or Reset. It uses an actual
**scikit-learn Pipeline** containing `SimpleImputer` and
`RandomForestClassifier`, not a custom Random Forest.

This is a retrospective ML demonstration. It does not apply firewall rules,
detect all attacks, or prove reliability on new networks. Full-connection
measurements (e.g. packet totals and duration) are used as inputs.

## Dataset and classmate's Google Colab notebook

- `data/log2_cleaned.csv` — classmate-supplied CSV, **57,170** records.
- `notebooks/DataCleaning.ipynb` — original classmate notebook retained as a
  reference; nothing in the website executes the notebook.
- `data_pipeline.py` — audited CSV parsing, schema/range checks, exact duplicate
  handling and quarantine of identical features with conflicting labels.
- `train_model.py` — 80/20 stratified split and scikit-learn training.
- `model_service.py` — safe application integration for the locally trained
  scikit-learn pipeline (trusted local `.joblib` files only).

The original public UCI Internet Firewall Data contains 65,532 records.
The supplied cleaned CSV starts with 57,170 records, and the notebook does
not reproduce every prior deletion. NetSentry's additional validation removes
18 contradictory-label records (nine identical-feature pairs), leaving
57,152 records. These safeguards mean **results will not be identical** to
the notebook's results, even though the model settings match.

UCI source: https://archive.ics.uci.edu/dataset/542/internet%2Bfirewall%2Bdata
(DOI:10.24432/C5131M).

## Train and launch on Windows

Requires Python 3.10+ and Windows for the included batch launchers.

1. Double-click `setup_project.bat`. This creates `.venv` and installs
   Flask, pandas, scikit-learn and joblib.
2. Double-click `train_model.bat`, or run:
   ```powershell
   .\.venv\Scripts\python.exe train_model.py
   ```
   This creates `model/firewall_random_forest.joblib`,
   `data/data_quality_report.json`, and `data/evaluation_report.json`.
3. Double-click `start_website.bat`, or run:
   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```
4. Open http://127.0.0.1:5000 and choose **Classify traffic**.

After a fresh Git clone, **retrain the model**: model artifacts, reports
and `.venv` are excluded from Git. The cleaned CSV *is* included.

## Training methodology

1. Normalize required column names and label names (`reset-both` → `Reset`).
2. Reject and count malformed, missing, nonnumeric, nonfinite, negative and
   contradictory numeric observations; reject invalid port ranges. Remove
   exact duplicates and quarantine conflicting feature/label pairs **before**
   splitting. An audit is stored in `data/data_quality_report.json`.
3. Use `sklearn.model_selection.train_test_split` with
   `test_size=0.2`, `stratify=y`, `random_state=42`. Assert that no
   identical feature vector is shared across train/test.
4. Fit the **same core configuration as the classmate's Colab notebook**:
   ```python
   Pipeline([
       ("imputer", SimpleImputer(strategy="median")),
       ("model", RandomForestClassifier(
           n_estimators=100,
           random_state=42,
           class_weight="balanced",
           n_jobs=-1,
       )),
   ])
   ```
   The imputer is fitted **on training folds only** and is unused for the
   fully observed supplied data. Scaling is unnecessary for Random Forests.
5. Run **five-fold stratified cross-validation on training records only**,
   reporting accuracy and macro-F1 (optional: `--cv-folds 0` to skip).
   Fit the final model to the training split; evaluate only once on untouched
   test records. Preserve original class proportions in testing.
6. Report test accuracy, macro-F1, per-class precision/recall/F1/support,
   confusion matrix, 10-bin Expected Calibration Error and multiclass
   Brier score. **The predicted probabilities are not calibrated.**

To train on the optional original UCI dataset:
`python train_model.py --dataset data/log2.csv`.
To audit without training:
`python train_model.py --validate-only`.

## Project limitations and confidence scores

Reset is extremely rare in this dataset. Even a high overall accuracy can
conceal poor Reset recall; always inspect the per-class report. Scikit-learn
`predict_proba()` returns model scores, not guaranteed real-world correctness.
The historical dataset does not justify claims of real-time firewall enforcement
or performance on unfamiliar networks.

The supplied Colab notebook also uses scikit-learn but trains on **57,170**
rather than **57,152** records. Identical training settings do not guarantee
identical predictions; identical data, splitting, cleaning and fitted model
would be needed for that.

Never load model `.joblib` or `.pkl` files from untrusted sources.
The website only loads the locally trained model produced by this project.

## Automated verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests check preprocessing and Flask prediction endpoint validation.
The UI and all 11 traffic inputs remain unchanged by the ML migration.

## Project structure

```text
app.py                             Flask website and prediction API
model_service.py                   scikit-learn model loading / predictions
data_pipeline.py                   CSV data quality and leakage safeguards
train_model.py                     scikit-learn training / CV / evaluation
requirements.txt                   Python requirements
data/log2_cleaned.csv              Classmate-provided cleaned dataset
data/log2.csv                      Optional original UCI CSV (not included)
data/data_quality_report.json      Generated data audit (not committed)
data/evaluation_report.json        Generated metrics (not committed)
model/firewall_random_forest.joblib Generated sklearn pipeline (not committed)
notebooks/DataCleaning.ipynb       Classmate's original exploratory notebook
tests/test_api.py                  Flask API tests
tests/test_preprocessing.py        Data-quality tests
tests/test_model_service.py        Model integration tests
templates/index.html               Website HTML
static/style.css                   Website CSS
static/app.js                      Website interaction and local history
setup_project.bat                  Creates local environment and dependencies
train_model.bat                    Trains and evaluates sklearn model
start_website.bat                  Starts local website
verify_project.bat                 Checks required files
```
