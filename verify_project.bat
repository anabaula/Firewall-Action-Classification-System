@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Verify Firewall Action Classification Project
echo Project root: %CD%
echo ================================================
call :check "app.py" required
call :check "model_service.py" required
call :check "data_pipeline.py" required
call :check "train_model.py" required
call :check "tests\test_preprocessing.py" required
call :check "notebooks\DataCleaning.ipynb" reference-notebook
call :check "requirements.txt" required
call :check "templates\index.html" required
call :check "static\app.js" required
call :check "static\style.css" required
if exist "data\log2_cleaned.csv" (
  echo [OK]      data\log2_cleaned.csv  [classmate dataset]
) else if exist "data\log2.csv" (
  echo [OK]      data\log2.csv  [original UCI dataset]
) else (
  echo [MISSING] data\log2_cleaned.csv or data\log2.csv
)
call :check "model\firewall_random_forest.joblib" trained-model
call :check "data\evaluation_report.json" evaluation-report
call :check "data\data_quality_report.json" quality-report
call :check ".venv\Scripts\python.exe" python-environment
echo.
echo Verification finished. Missing generated files require setup or training.
pause
exit /b 0
:check
if exist "%~1" (
  echo [OK]      %~1
) else (
  echo [MISSING] %~1  [%~2]
)
exit /b
