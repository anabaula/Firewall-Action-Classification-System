# Firewall Action Classification System — Clean Setup

This is a clean, single-folder version of the project. The ZIP contains files directly at its root, so after **Extract All** there should not be another project folder nested inside it.

## Requirements
- Windows 10/11
- Python installed (Python 3.10 or newer recommended)
- The original UCI Internet Firewall Data file `log2.csv`

## 1. Extract and open the project
1. Download `Firewall_Action_Classifier_CLEAN.zip`.
2. Right-click it → **Extract All**.
3. Open the extracted folder in VS Code with **File → Open Folder**.
4. The Explorer should show `app.py`, `train_model.py`, `forest_model.py`, `requirements.txt`, `data`, `model`, `static`, and `templates` immediately at the top level.

## 2. Put the real dataset in place
Copy the original UCI `log2.csv` file to `data/log2.csv`.
Do not create a blank CSV or rename a ZIP to CSV. The source dataset is available at:
https://archive.ics.uci.edu/dataset/542/internet%2Bfirewall%2Bdata

If `log2.csv` is in your old project, copy it from the old project's `data` folder into this clean project's `data` folder.

## 3. Set up Python (one time only)
Double-click `setup_project.bat`, or run these commands in the VS Code terminal from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This version uses Flask and a pure-Python Random Forest implementation; it does not require NumPy, SciPy, pandas, scikit-learn, or joblib. This avoids the previously encountered blocked SciPy DLL issue.

## 4. Train the model (when needed)
Make sure `data/log2.csv` exists, then double-click `train_model.bat`, or run:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The script creates:
- `model/firewall_random_forest.json`
- `data/evaluation_report.json`

It reports accuracy, macro F1, and per-class precision/recall/F1. The Reset category is extremely rare in the original dataset, so interpret that class's results carefully.

## 5. Start the website
Double-click `start_website.bat`, or run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Keep the terminal open, then visit http://127.0.0.1:5000. Press Ctrl+C in the terminal to stop Flask.

## 6. Reopening VS Code later
1. Open this same extracted project folder in VS Code.
2. You do not need to recreate `.venv` or reinstall Flask unless you delete the environment.
3. You do not need to retrain each time. Run `start_website.bat` to start the saved model.
4. Run `train_model.bat` only when intentionally retraining.

## Folder layout
```text
Firewall_Action_Classifier_CLEAN/
├── app.py
├── forest_model.py
├── train_model.py
├── requirements.txt
├── setup_project.bat
├── train_model.bat
├── start_website.bat
├── verify_project.bat
├── .gitignore
├── data/
│   ├── .gitkeep
│   └── log2.csv                 # you copy this from the UCI download
├── model/
│   └── .gitkeep
├── templates/
│   └── index.html
└── static/
    ├── app.js
    └── style.css
```

## Important limitations
This project is for academic demonstration. It predicts labels found in historical UCI firewall records; it does not connect to, configure, or block traffic on a real firewall. Confidence is the model's highest class probability, not the model's test accuracy. Do not use it to make production security decisions.
