@echo off
setlocal
cd /d "%~dp0"
set "METABALL_BIND_HOST=127.0.0.1"
set "METABALL_API_PORT=8780"
set "METABALL_ALLOWED_ORIGINS=*"

where py >nul 2>nul
if errorlevel 1 (
  echo Python 3 is required. Install it from https://www.python.org/downloads/windows/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || goto :error
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :error
)

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8780'"
echo Eigenfluid local inference: http://127.0.0.1:8780
echo Keep this window open while using the application.
".venv\Scripts\python.exe" backend\inference_server.py
exit /b %errorlevel%

:error
echo Local setup failed. Check the message above.
pause
exit /b 1
