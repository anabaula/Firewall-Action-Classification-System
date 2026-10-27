@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Firewall Action Classifier - Initial Setup
echo ================================================
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python Launcher 'py' was not found.
  echo Install Python from https://www.python.org/downloads/ then reopen this file.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -m venv .venv
  if errorlevel 1 goto failed
) else (
  echo Virtual environment already exists; keeping it.
)
echo Installing requirements...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo SUCCESS: setup is ready.
echo Next, run train_model.bat using the included data\log2_cleaned.csv.
pause
exit /b 0
:failed
echo.
echo ERROR: Setup did not finish. Read the message above and ask for help.
pause
exit /b 1
