param(
    [string]$ArchiveName = "eigenfluid-local-inference-windows.zip"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StageRoot = Join-Path $Repo "local_bundle"
$Stage = Join-Path $StageRoot "eigenfluid-local-inference"
$Artifacts = Join-Path $Repo "artifacts"
$Archive = Join-Path $Artifacts $ArchiveName

if (Test-Path -LiteralPath $Stage) {
    $resolved = (Resolve-Path -LiteralPath $Stage).Path
    if (-not $resolved.StartsWith($StageRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove an unexpected staging path: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

New-Item -ItemType Directory -Path $Stage -Force | Out-Null
$previousLocalBuild = $env:NEXT_PUBLIC_LOCAL_BUILD
try {
    $env:NEXT_PUBLIC_LOCAL_BUILD = "1"
    Push-Location $Repo
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Local frontend build failed." }
    Pop-Location

    Copy-Item -LiteralPath (Join-Path $Repo "out") -Destination (Join-Path $Stage "out") -Recurse
    New-Item -ItemType Directory -Path (Join-Path $Stage "backend") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $Repo "backend\inference_server.py") -Destination (Join-Path $Stage "backend\inference_server.py")
    foreach ($method in @("potential", "velocity", "vorticity")) {
        $target = Join-Path $Stage "models\$method"
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $Repo "models\$method\epoch_2000.bin") -Destination (Join-Path $target "epoch_2000.bin")
    }
    Copy-Item -LiteralPath (Join-Path $Repo "deploy\local\START_LOCAL.bat") -Destination (Join-Path $Stage "START_LOCAL.bat")
    Copy-Item -LiteralPath (Join-Path $Repo "deploy\local\requirements.txt") -Destination (Join-Path $Stage "requirements.txt")
    Copy-Item -LiteralPath (Join-Path $Repo "deploy\local\README_LOCAL.txt") -Destination (Join-Path $Stage "README_LOCAL.txt")

    New-Item -ItemType Directory -Path $Artifacts -Force | Out-Null
    if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
    Compress-Archive -LiteralPath $Stage -DestinationPath $Archive -CompressionLevel Fastest
    Write-Host "Local bundle created: $Archive"
}
finally {
    $env:NEXT_PUBLIC_LOCAL_BUILD = $previousLocalBuild
}
