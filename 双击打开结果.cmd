@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0start_observatory.ps1"
if errorlevel 1 (
  echo.
  echo Failed to start Eigenfluid Metaball Observatory.
  echo Please confirm that Python and NumPy are installed.
  echo Error details are stored in runtime_logs.
  pause
)
endlocal
