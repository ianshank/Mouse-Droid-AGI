[CmdletBinding()]
param(
    [string]$Config = "config/local_training.yaml",
    [string]$Phases,
    [string]$Resume,
    [switch]$Upload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$python = "C:\Program Files\Python311\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Configured Python interpreter not found at '$python'."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoSrc = Join-Path $repoRoot "src"
$previousPythonPath = $env:PYTHONPATH
$hadPythonPath = Test-Path Env:PYTHONPATH
$exitCode = 0

$args = @(
    "-m",
    "training.run_pipeline",
    "--config",
    $Config
)

if ($Phases) {
    $args += @("--phases", $Phases)
}

if ($Resume) {
    $args += @("--resume", $Resume)
}

if ($Upload) {
    $args += "--upload"
}

Push-Location $repoRoot
try {
    $env:PYTHONPATH = $repoSrc
    & $python @args
    $exitCode = $LASTEXITCODE
}
finally {
    if ($hadPythonPath) {
        $env:PYTHONPATH = $previousPythonPath
    }
    else {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    Pop-Location
}

exit $exitCode