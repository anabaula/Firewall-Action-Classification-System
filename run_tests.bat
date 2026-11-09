@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment not found. Run setup_project.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest %*
exit /b %ERRORLEVEL%
