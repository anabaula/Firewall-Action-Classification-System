# NetSentry — Firewall Action Classification

NetSentry is an educational Flask website that classifies **historical network connection records** as Allow, Deny, Drop, or Reset. It uses supervised **classification**, not numerical linear regression. It does not enforce firewall rules or detect every attack.

## Start here: what each Python file does

| File | Easy explanation |
|---|---|
| `notebooks/DataCleaning.ipynb` | **Learn the process.** Open in Colab or VS Code and run each cell in order. It can work with only an uploaded `log2.csv` in Colab. |
| `data_pipeline.py` | **Clean the raw dataset.** Check invalid values, duplicate rows and contradictory labels. |
| `train_model.py` | **Teach and calibrate the model.** Split records, compare regularization using training-only validation, calibrate the Random Forest, and save accuracy reports. |
| `model_service.py` | **Use the trained model.** Load the local `.joblib` file and make predictions. |
| `app.py` | **Run the website.** Accept the form values, check them, and send them to the model. |
| `static/app.js` | **Make the website interactive.** Change pages, submit forms, and store prediction history in the browser. |

**Recommended reading order:** `DataCleaning.ipynb` → `data_pipeline.py` → `train_model.py` → `model_service.py` → `app.py` → `static/app.js`.

The website files are separate from the methods taught in the PDFs; Flask and JavaScript are needed to make the model accessible through a website.

## Website: one page only

The NetSentry website is a **minimal, flat-color desktop layout** showing four sections without page scrolling at common laptop sizes. On narrow/mobile screens the sections stack and can scroll for readability:

1. **Network traffic input:** the 11 original UCI traffic measurements. **Generate** fills the form with a randomly selected complete, cleaned row **only from the held-out 20% test split** of `data/log2.csv` (no synthetic values). The split matches `train_model.py` (`random_state=42`, stratified, 80/20). It does not classify automatically or reveal the row's action label. Use **Classify** to predict, or **Reset** to clear the inputs.
2. **Classification result:** predicted action and model confidence after submitting the form.
3. **Post-training results:** held-out test accuracy, macro F1, cross-validation accuracy, training–validation gap loaded from `data/evaluation_report.json`.
4. **Prediction history:** date, action, confidence, and a Details button showing the complete saved input record in a popup. Results stay in local browser history (up to 100) and paginate three at a time.

The interface uses a white background, black text, and gray borders, with blue reserved for the Classify button. **Firewall Network Classifier** and **Random Forest (scikit-learn)** appear inside the Traffic Input panel. On desktop, the classification result sits below the input, post-training results are at the upper right, and prediction history is below those results. History shows only Date, Action, Confidence, and a Details button that opens a window with all 11 saved inputs. Older browser records may show **Not recorded** for inputs that weren't saved previously. History action labels are uncolored. There are no gradients, sidebar, old logo banner, extra dashboard or per-action performance table. The full per-action scores remain available in `data/evaluation_report.json` for reports. Open `http://127.0.0.1:5000` after running `start_website.bat`. If training results are unavailable, run `train_model.bat` and refresh the page. The website still predicts using the saved calibrated Random Forest `.joblib` model, not the evaluation report. The reported 5-fold CV accuracy is for the selected **underlying Random Forest** before calibration; the displayed test accuracy and confidence refer to the **deployed calibrated model**.

## Dataset — original UCI file

- **Official dataset:** UCI Internet Firewall Data, DOI **10.24432/C5131M**.
- **Source:** https://archive.ics.uci.edu/dataset/542/internet+firewall+data
- **Raw project file:** `data/log2.csv` — 65,532 records with 11 numeric input features plus an `Action` label.
- The original CSV is kept intact; cleaning is reproducible and its decisions are written to `data/data_quality_report.json`.
- Do not use the previous classmate-derived `log2_cleaned.csv`. It was replaced by the original UCI source.

## Professor's lectures and notebook

`notebooks/DataCleaning.ipynb` is a valid Jupyter notebook and a step-by-step companion to the lecture PDFs:

1. Inspect raw data with `pd.read_csv`, `head`, `info`, `describe`, `isna`, and duplicates.
2. Check missing values, outliers, duplicate records, and contradictory target labels.
3. Separate the 11 numeric features (**X**) from the multiclass action target (**y**).
4. Use **stratified** 80/20 train/test splitting with `random_state=42`.
5. Build an **sklearn Pipeline** first, fitting the imputer and any scaler *only inside training folds*.
6. Compare Logistic Regression, linear SVM, and Random Forest with **five-fold StratifiedKFold cross-validation** on the training set.
7. Examine both **accuracy and macro-F1**, train–validation gap, per-class precision/recall/F1, and confusion matrix. Test records are never used for model selection.

The notebook reads the dataset through **project-relative paths**, whether launched from the project root or `notebooks`. No personal Windows paths or Google Drive paths are used. Launch it with Jupyter or VS Code (Python notebook support).

The learning materials illustrate Logistic Regression and SVM; NetSentry retains **Random Forest** as its deployable classifier because the target has four categories and the existing web interface needs a model with `predict_proba()`. Scaling is appropriate for the linear models but **not required for tree-based Random Forest**. Linear regression would be inappropriate because Allow/Deny/Drop/Reset are labels, not continuous numbers.

### Classifier comparison on the original UCI dataset

The notebook was executed end-to-end using 5-fold stratified cross-validation on the 45,721 training records (the 11,431 held-out records were not used for model selection):

| Candidate | Mean CV accuracy | Mean CV macro-F1 |
|---|---:|---:|
| **Random Forest** | **99.8075%** | **0.8581** |
| Linear SVM | 98.8517% | 0.7414 |
| Logistic Regression | 93.3247% | 0.6927 |

Random Forest scored highest on macro-F1, so the website continues to deploy it. Its mean CV training–validation accuracy gap was **0.1914 percentage points**. The **original uncalibrated notebook baseline** held-out score was **99.8688% accuracy** and **0.8955 macro-F1**. The deployed **sigmoid-calibrated Random Forest** now scores **99.8775% accuracy** and **0.9047 macro-F1**, with no exactly-100% confidence outputs among the 11,431 held-out records. Calibration improves confidence metrics on this dataset, but it does not resolve the scarce Reset examples. All scores are specific to this historical dataset.

## Cleaning and leakage prevention

`data_pipeline.py` uses **pandas** to inspect, normalize and validate the CSV.

- Reject invalid label strings and invalid numerical values (including infinity, negative ports/counts, out-of-range ports, inconsistent packet/byte sums).
- Count every excluded record and the reason in the audit.
- Retain true missing numeric values as `NaN` so `SimpleImputer(strategy="median")` can learn only from a training split/fold. The supplied original CSV has **no missing values**.
- Remove exact duplicate records and quarantine identical feature vectors with contradictory action labels **before splitting**. No artificial outlier trimming or full-dataset scaling.
- This cleaning removes 8,362 exact duplicates and 18 conflicting-label records from the 65,532 raw observations, leaving **57,152** records.

The `Action` label `reset-both` is displayed as `Reset`. There are no categorical **input** columns; consequently `OneHotEncoder`/`ColumnTransformer` are not needed for this particular CSV.

## Train and launch (Windows)

Use Python 3.10+ and a virtual environment.

1. `setup_project.bat` — creates `.venv` and installs `requirements.txt`.
2. `train_model.bat` — trains from **data/log2.csv**, creates the model and JSON reports.
3. `verify_project.bat` — verifies expected files exist.
4. `start_website.bat` — starts Flask locally at http://127.0.0.1:5000.

Equivalent commands, from the project root:

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py
```

Training preserves scikit-learn `Pipeline(SimpleImputer, RandomForestClassifier)`, `class_weight="balanced"`, 100 trees, seed 42, and an untouched stratified 20% test split. Within the 80% training partition, the script compares unrestricted, leaf-size-2/depth-24, and leaf-size-3/depth-18 forests; then compares training-only calibrated finalists by validation macro-F1 (to account for the rare Reset class). The selected Random Forest receives 3-fold cross-validated **sigmoid probability calibration**, fitted entirely on training data. The selected base model still receives 5-fold training-only accuracy/F1 cross-validation, and the holdout is used only for final evaluation. The deployed `model/firewall_random_forest.joblib` contains the fitted calibrated Random Forest and its original feature order.

Training prints and saves `data/evaluation_report.json`, including candidate regularization and calibration comparisons, base-model 5-fold cross-validation, held-out accuracy and per-class scores, the confusion matrix, and **both raw and calibrated confidence diagnostics** (ECE, Brier, and exact-100% count). The calibration is not guaranteed to improve accuracy or Reset recall; check the report before relying on scores. Never use the held-out test set to select hyperparameters.

> **Limits:** Reset is extremely rare. High overall accuracy can conceal poor Reset recall. Sigmoid calibration reduces raw-forest overconfidence on this data but cannot guarantee that reported probability equals real-world correctness. The original notebook remains an uncalibrated Random Forest comparison; deployment adds calibration. Results apply to historical UCI data and do not guarantee real-world firewall decisions. Load trusted local `.joblib` files only.

## Important project files

| File | Responsibility |
|---|---|
| `data/log2.csv` | Original UCI raw records — reproducible source |
| `notebooks/DataCleaning.ipynb` | Professor-style exploratory and model-comparison notebook |
| `data_pipeline.py` | Shared pandas cleaning, label normalization and audit |
| `train_model.py` | Full training, validation, evaluation and joblib saving |
| `model_service.py` | Load trusted scikit-learn model and predict |
| `app.py` | Flask website and traffic-input validation, **not part of the ML training algorithm** |
| `templates/index.html` and `static/` | Website UI |
| `tests/` | Dataset, model, API and Windows-verifier tests |

Flask itself is application code, not a preprocessing concept covered by the PDFs. Its presence is necessary for NetSentry's existing website.
