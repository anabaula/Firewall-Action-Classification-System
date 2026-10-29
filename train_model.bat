@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Train Random Forest from validated UCI-style data
echo ================================================
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment not found.
  echo First double-click setup_project.bat.
  pause
  exit /b 1
)
if not exist "data\log2_cleaned.csv" if not exist "data\log2.csv" (
  echo ERROR: No input dataset was found.
  echo Provide data\log2_cleaned.csv or data\log2.csv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 (
  echo.
  echo ERROR: Data validation or model training failed. Review the output.
  pause
  exit /b 1
)
echo.
echo SUCCESS: model, evaluation report, and quality report generated.
pause
