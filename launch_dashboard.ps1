# PowerShell launcher for the local MouseDroid dashboard (E2E dev mode).
#
# Usage:
#   # First copy the dev overlay template (only need to do this once):
#   Copy-Item config\dev_dashboard.yaml.example dev_dashboard.yaml
#   # Then run from this worktree:
#   .\launch_dashboard.ps1
#
# Then open in a browser:
#   http://127.0.0.1:8080/camera     -> live MJPEG camera feed
#   http://127.0.0.1:8080/lidar      -> LiDAR polar view
#   http://127.0.0.1:8080/api/v1/status   -> orchestrator status JSON
#   http://127.0.0.1:8080/api/v1/sensors  -> live sensor JSON
#   http://127.0.0.1:8080/metrics    -> Prometheus scrape endpoint
#
# Stop with Ctrl+C.

$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$PY = if ($env:MOUSEDROID_PY) { $env:MOUSEDROID_PY } else { "C:\Program Files\Python311\python.exe" }
$REPO = $PSScriptRoot

Set-Location $REPO

if (-not (Test-Path "dev_dashboard.yaml")) {
    Write-Host "dev_dashboard.yaml not found in $REPO — copying template from config/..."
    Copy-Item "config\dev_dashboard.yaml.example" "dev_dashboard.yaml"
}

Write-Host "Launching MouseDroid dashboard (mock hardware, telemetry server at 127.0.0.1:8080)..."
Write-Host "Once you see 'telemetry_server_started', open:"
Write-Host "  http://127.0.0.1:8080/camera"
Write-Host "  http://127.0.0.1:8080/lidar"
Write-Host "Stop with Ctrl+C."
Write-Host ""

& $PY -m mousedroid.main --config config\default.yaml dev_dashboard.yaml
