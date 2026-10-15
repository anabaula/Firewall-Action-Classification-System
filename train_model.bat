@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Train Random Forest from UCI log2.csv
echo ================================================
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment not found.
  echo First double-click setup_project.bat.
  pause
  exit /b 1
)
if not exist "data\log2.csv" (
  echo ERROR: data\log2.csv was not found.
  echo Copy the original UCI log2.csv into this project's data folder.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 (
  echo.
  echo Training failed. Copy the error text and ask for help.
  pause
  exit /b 1
)
echo.
echo SUCCESS: model and evaluation report were generated.
pause
