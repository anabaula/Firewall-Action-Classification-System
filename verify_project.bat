@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo Verify Project Files
 echo Project root: %CD%
echo ================================================
call :check "app.py" required
call :check "forest_model.py" required
call :check "train_model.py" required
call :check "requirements.txt" required
call :check "templates\index.html" required
call :check "static\app.js" required
call :check "static\style.css" required
call :check "data\log2.csv" dataset
call :check "model\firewall_random_forest.json" trained-model
call :check "data\evaluation_report.json" evaluation-report
call :check ".venv\Scripts\python.exe" python-environment
echo.
echo Verification finished. Items marked MISSING may be created in later setup/training steps.
pause
exit /b 0
:check
if exist "%~1" (
  echo [OK]      %~1
) else (
  echo [MISSING] %~1  (%~2)
)
exit /b
