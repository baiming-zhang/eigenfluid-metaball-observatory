$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root "runtime_logs\server.pid"
if (Test-Path $pidFile) {
    $serverPid = [int](Get-Content -LiteralPath $pidFile)
    Stop-Process -Id $serverPid -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
}
