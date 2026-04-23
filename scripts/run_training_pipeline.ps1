[CmdletBinding()]
param(
    [string]$Config = "config/local_training.yaml",
    [string]$Phases,
    [string]$Resume,
    [switch]$Upload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve the Python interpreter in this order so the script is portable
# across developer machines and CI runners:
#   1) $env:MOUSEDROID_PYTHON (explicit override)
#   2) First 'python' on PATH (Get-Command)
#   3) First 'python3' on PATH
# Hardcoded install paths (e.g. C:\Program Files\Python311\python.exe) are
# intentionally avoided.
$python = $env:MOUSEDROID_PYTHON
if (-not $python) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if ($pythonCmd) {
        $python = $pythonCmd.Source
    }
}
if (-not $python -or -not (Test-Path -LiteralPath $python)) {
    throw "No Python interpreter found. Set `$env:MOUSEDROID_PYTHON or add 'python' to PATH."
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