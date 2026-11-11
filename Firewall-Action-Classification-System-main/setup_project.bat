@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Firewall Action Classifier - Initial Setup
echo ================================================
set "PYTHON_CMD=py"
where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python from https://www.python.org/downloads/ then reopen this file.
    pause
    exit /b 1
  )
  set "PYTHON_CMD=python"
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto failed
) else (
  echo Virtual environment already exists; keeping it.
)
echo Installing requirements...
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if errorlevel 1 goto failed
echo.
echo SUCCESS: setup is ready.
echo Next, run train_model.bat using the included data\log2.csv.
pause
exit /b 0
:failed
echo.
echo ERROR: Setup did not finish. Read the message above and ask for help.
pause
exit /b 1
