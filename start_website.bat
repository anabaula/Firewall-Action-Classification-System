@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 (
  echo ERROR: Could not switch to the project folder.
  pause
  exit /b 1
)

echo ================================================
echo Start Firewall Action Classification Website
echo ================================================

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Project virtual environment was not found.
  echo Double-click setup_project.bat first.
  pause
  exit /b 1
)

if not exist "app.py" (
  echo ERROR: app.py was not found in the project folder.
  echo Restore app.py to this folder, then run this file again.
  pause
  exit /b 1
)

if not exist "model\firewall_random_forest.joblib" (
  echo ERROR: The trained model was not found.
  echo Run train_model.bat to generate model\firewall_random_forest.joblib.
  echo Confirm data\log2_cleaned.csv or data\log2.csv is present.
  pause
  exit /b 1
)

echo Flask will start at http://127.0.0.1:5000
echo Keep this window open while using the website.
echo.
".venv\Scripts\python.exe" app.py
set "FLASK_EXIT_CODE=%ERRORLEVEL%"
if not "%FLASK_EXIT_CODE%"=="0" (
  echo.
  echo ERROR: Flask exited with code %FLASK_EXIT_CODE%.
  echo Review the error message above, then press any key to close this window.
) else (
  echo.
  echo Flask has stopped. Press any key to close this window.
)
pause
exit /b %FLASK_EXIT_CODE%
