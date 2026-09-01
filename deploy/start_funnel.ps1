$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$env:METABALL_BIND_HOST = "127.0.0.1"
$env:METABALL_API_PORT = "8780"
$env:METABALL_PREVIEW_GRID = "64"
$env:METABALL_MAX_CONCURRENT = "2"
$env:METABALL_RATE_LIMIT = "30"

$listener = Get-NetTCPConnection -LocalPort 8780 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    exit 0
}

Push-Location $root
try {
    & "C:\ProgramData\anaconda3\python.exe" backend\inference_server.py
} finally {
    Pop-Location
}
