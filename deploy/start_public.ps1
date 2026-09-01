$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:METABALL_BIND_HOST = "0.0.0.0"
$env:METABALL_API_PORT = if ($env:METABALL_API_PORT) { $env:METABALL_API_PORT } else { "8780" }
$env:METABALL_PREVIEW_GRID = if ($env:METABALL_PREVIEW_GRID) { $env:METABALL_PREVIEW_GRID } else { "64" }
$env:METABALL_MAX_CONCURRENT = if ($env:METABALL_MAX_CONCURRENT) { $env:METABALL_MAX_CONCURRENT } else { "2" }
$env:METABALL_RATE_LIMIT = if ($env:METABALL_RATE_LIMIT) { $env:METABALL_RATE_LIMIT } else { "30" }

Push-Location $root
try {
    npm install --no-package-lock
    npm run build
    python -m pip install -r requirements.txt
    python backend/inference_server.py
} finally {
    Pop-Location
}
