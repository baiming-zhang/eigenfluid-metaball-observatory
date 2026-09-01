$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = (Get-Command python -ErrorAction Stop).Source
$env:METABALL_PREVIEW_GRID = if ($env:METABALL_PREVIEW_GRID) { $env:METABALL_PREVIEW_GRID } else { "64" }
$logDir = Join-Path $root "runtime_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$pidFile = Join-Path $logDir "server.pid"

# Restart only a server previously launched by this package so the static UI
# and inference response contract always stay on the same version.
if (Test-Path -LiteralPath $pidFile) {
    $ownedPid = [int](Get-Content -LiteralPath $pidFile)
    Stop-Process -Id $ownedPid -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 250
}

$existing = Get-NetTCPConnection -LocalPort 8780 -State Listen -ErrorAction SilentlyContinue
if (-not $existing) {
    $process = Start-Process -FilePath $python -ArgumentList @("backend\inference_server.py") `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "server.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "server.stderr.log")
    Set-Content -LiteralPath $pidFile -Value $process.Id
}

Start-Sleep -Milliseconds 800
Start-Process "http://127.0.0.1:8780"
Write-Host "Eigenfluid Metaball Observatory: http://127.0.0.1:8780"
