# NetSentry — Firewall Action Classification System

NetSentry is a school machine learning project that **predicts what action a firewall might take** for a network connection.

You enter network traffic details, and the website predicts one of four actions:

- **Allow** — let the connection through.
- **Deny** — block the connection.
- **Drop** — discard the connection's traffic.
- **Reset** — reset the connection (called "Reset-Both" in the original dataset).

NetSentry learns from past firewall records. **It does not control a real firewall or protect your network.**

## What can the website do?

- **Enter traffic details:** Fill in 11 network measurements, such as ports, bytes, packets, and connection duration.
- **Generate an example:** Fill the form with a real record from the **test dataset**, not the training dataset.
- **Classify:** Predict an action and show the model's confidence.
- **View results:** See the model's test accuracy and other scores after training.
- **View history:** See earlier predictions and open their input details. History is stored in your browser.

## How does it work?

1. **Read the data.** Python and pandas load the original firewall records from `data/log2.csv` using `pd.read_csv()`.
2. **Clean the data.** The project checks invalid values, removes duplicate records, and handles conflicting labels.
3. **Split the data.** It uses **80% for training** and keeps **20% for testing**.
4. **Train the model.** scikit-learn prepares missing values and trains a **Random Forest classifier**. It also uses cross-validation to check performance.
5. **Make a prediction.** The website sends your inputs to the saved model and displays the predicted action and confidence.

The project uses an **sklearn Pipeline** so missing-value handling is learned from training data, not the test data. It uses five-fold stratified cross-validation, and calibrates the model's confidence scores using training data only.

All 11 inputs are numeric, so one-hot encoding is unnecessary. Random Forest also does not require feature scaling. The included notebook shows how this relates to the preprocessing and model-comparison methods taught in class.

## How to run it (Windows)

**You need:** Python installed on your computer.

From the project folder, double-click these files **in order**:

1. **`setup_project.bat`** — sets up Python and installs the required libraries.
2. **`train_model.bat`** — cleans the included dataset, trains the model, and saves the results. Run this after a fresh download or whenever you want to retrain.
3. **`start_website.bat`** — starts the website.

Then open **http://127.0.0.1:5000** in your browser. Keep the website's command window open while you use it.

To check the project, run `run_tests.bat`. You can also run `verify_project.bat` to check that the required files exist.

**If the website says the model is missing:** Run `train_model.bat` first. The trained model is generated on your computer and is not included in GitHub downloads.

## Dataset and results

The data comes from the [UCI Internet Firewall Data](https://archive.ics.uci.edu/dataset/542/internet+firewall+data).

| Dataset stage | Number of records |
| --- | ---: |
| Original data | 65,532 |
| After cleaning | 57,152 |
| Training set | 45,721 |
| Test set | 11,431 |

The saved evaluation report shows about **99.88% test accuracy** on this historical dataset. However, some actions (especially **Reset**) have very few examples, so the model can still make mistakes. **Model confidence is an estimate, not a guarantee.**

## Main project files

| File | What it does |
| --- | --- |
| `data/log2.csv` | Original firewall dataset |
| `data_pipeline.py` | Reads and cleans the dataset |
| `train_model.py` | Trains, tests, and saves the model |
| `model_service.py` | Loads the trained model for predictions |
| `app.py` | Runs the Flask website |
| `templates/` and `static/` | Website design and interactions |
| `notebooks/DataCleaning.ipynb` | Step-by-step notebook for learning the data processing and model comparison |
| `tests/` | Automated checks for the project |

This project was built for **learning and demonstrating machine learning classification**, not for making real firewall security decisions.
