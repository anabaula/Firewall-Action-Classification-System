# NetSentry — Firewall Action Classification System

NetSentry is a **CST 9/L student project** that predicts the action recorded for a network connection using historical firewall data. It is a Flask website with a scikit-learn **Random Forest classifier**.

It predicts **Allow**, **Deny**, **Drop**, or **Reset** (called **Reset-Both** in the source dataset). **NetSentry does not inspect live traffic, enforce firewall rules, or protect a real network.**

## What you can do

- Enter 11 network measurements (ports, bytes, packets, and elapsed time).
- Select **Generate** to fill the form with a record from the **held-out test set**, never the training set.
- Select **Classify** to predict an action and view the model's confidence percentage.
- View training results and browser-stored prediction history, including input details.
- Use the light/dark UI theme.

## How the machine learning works

The Python code is organized into readable steps similar to the class lessons:

1. **Read and clean:** `data_pipeline.py` loads `data/log2.csv` with `pd.read_csv()`. Pandas checks invalid traffic values and removes exact duplicates and feature rows with contradictory action labels. Class totals are counted with `value_counts()`.
2. **Choose X and y:** `X` contains 11 numeric traffic measurements; `y` contains the known firewall action.
3. **Split:** `train_test_split(test_size=0.2, random_state=42, stratify=y)` reserves 20% for independent testing.
4. **Create a pipeline:** `SimpleImputer(strategy="median")` handles missing numeric values within training folds, then `RandomForestClassifier` learns to classify traffic. The default is **100 trees**, `class_weight="balanced"`, and `random_state=42`.
5. **Validate:** Model settings are compared using a training-only validation split. The selected base model is evaluated with **5-fold stratified cross-validation** using `cross_validate()` (a more detailed variant of the `cross_val_score()` method taught in class).
6. **Calibrate:** `CalibratedClassifierCV(method="sigmoid", cv=3)` improves the interpretation of probability/confidence estimates using **training data only**.
7. **Evaluate and save:** The calibrated classifier is tested on the untouched 20% test set and saved to `model/firewall_random_forest.joblib`. Reports are written as JSON.
8. **Predict on the website:** `model_service.py` loads the saved model when needed, and `app.py` serves the web interface. The confidence value is not rounded in Python; the UI displays it to two decimal places.

We do **not** use one-hot encoding because the input features are numerical. Random Forest is tree-based and does **not** require `StandardScaler`.

The core workflow follows the **data preprocessing, pipeline, and cross-validation** lecture PDFs. Random Forest and sigmoid probability calibration extend those examples to support this four-class application. See [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md) for a defense-friendly walkthrough.

## Run on Windows

Install Python, open the project folder, then run these files in order:

1. `setup_project.bat` — install the Python requirements.
2. `train_model.bat` — train the classifier and create the saved model and reports.
3. `start_website.bat` — start the local Flask website.

Open **http://127.0.0.1:5000**.

Use `run_tests.bat` to run automated tests. Use `verify_project.bat` to check required project files. Do not load unknown Joblib files from untrusted sources.

## Dataset and results

Source: [UCI Internet Firewall Data](https://archive.ics.uci.edu/dataset/542/internet+firewall+data), DOI: **10.24432/C5131M**.

| Stage | Records |
| --- | ---: |
| Original CSV | 65,532 |
| Exact duplicates removed | 8,362 |
| Conflicting-label rows removed | 18 |
| After cleaning | 57,152 |
| Training (80%) | 45,721 |
| Held-out test (20%) | 11,431 |

The **saved evaluation report** records:

| Measure | Saved result |
| --- | ---: |
| Held-out accuracy | 99.8775% |
| Held-out macro F1 | 0.9047 |
| 5-fold CV mean accuracy (base model) | 99.8075% |
| 5-fold CV macro F1 (base model) | 0.8581 |
| Reset recall (held-out) | 45.45% (5 out of 11) |
| Exact 100% confidence predictions after calibration | 0 |

These are **historical saved evaluation figures**, not a guarantee that a later training run produces identical scores. The rare Reset action has only 54 cleaned records, so high overall accuracy does not mean every class performs equally well. Model confidence is a probability estimate, not a security guarantee.

## Project file guide

| File | Purpose |
| --- | --- |
| `data_pipeline.py` | Pandas data loading and cleaning |
| `train_model.py` | Model selection, 5-fold validation, calibration, evaluation and model saving |
| `model_service.py` | Load saved model and predict actions/confidence |
| `app.py` | Flask routes and traffic input validation |
| `templates/index.html`, `static/` | Website layout and browser functionality |
| `data/log2.csv` | Original UCI dataset |
| `notebooks/DataCleaning.ipynb` | Classroom-style notebook and comparisons |
| `MODEL_TRAINING_GUIDE.md` | Student study guide explaining each training step |
| `docs/NetSentry_CST9L_Project_Documentation.docx` | Formal CST 9L project documentation |
| `tests/` | Regression tests |

**For CST 9/L educational demonstration only.**
