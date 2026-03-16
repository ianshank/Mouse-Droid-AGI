#Requires -Version 5.1
<#
.SYNOPSIS
    MouseDroid Jetson deployment from Windows — no WSL, no rsync required.

.DESCRIPTION
    Deploys MouseDroid to the Jetson Orin Nano using only Windows-native tools:
      - tar.exe  (built into Windows 10/11) for project code transfer
      - ssh.exe  (OpenSSH for Windows)  for remote execution
      - scp.exe  for small file copies (config)
    Docker handles all large content on the Jetson side:
      - Image built on-device from Dockerfile.jetson
      - Weights pulled from HuggingFace directly on the Jetson
      - Phi-3 Mini GGUF downloaded on-device (--with-llm mode)

.PARAMETER Host
    Jetson IP or hostname. Defaults to 192.168.55.1 (USB-C gadget, fastest).
    Falls back to mousedroid.local (ethernet) if USB unreachable.

.PARAMETER User
    SSH username on the Jetson. Defaults to 'ian'.

.PARAMETER SshKey
    Path to SSH private key. Defaults to ~/.ssh/id_ed25519.

.PARAMETER Mode
    Deployment mode:
      full      — code transfer + Docker build + weights pull + service start
      code-only — code transfer + Docker image rebuild only (no weight sync)
      weights   — pull weights from HuggingFace on Jetson only
      with-llm  — full + Phi-3 Mini GGUF download on Jetson (slow first run)

.EXAMPLE
    .\scripts\deploy_jetson_windows.ps1
    .\scripts\deploy_jetson_windows.ps1 -Mode with-llm
    .\scripts\deploy_jetson_windows.ps1 -Host 192.168.4.29 -Mode weights

#>
[CmdletBinding()]
param(
    [string]$JetsonHost     = "",
    [string]$User           = $env:MOUSEDROID_REMOTE_USER,
    [string]$SshKey         = "",
    [ValidateSet("full", "code-only", "weights", "with-llm")]
    [string]$Mode           = "full",
    [string]$HfRepo         = "ianshank/mousedroid-weights",
    [string]$RemoteSrc      = "/opt/mousedroid",
    [string]$RemoteConfig   = "/etc/mousedroid",
    [string]$RemoteWeights  = "/opt/mousedroid/weights",
    [string]$RemoteModels   = "/home/ian/models"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
if (-not $User)    { $User    = "ian" }
if (-not $SshKey)  { $SshKey  = Join-Path $env:USERPROFILE ".ssh\id_ed25519" }

$ProjectDir = Split-Path $PSScriptRoot -Parent
$DeployStart = Get-Date

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Section([string]$msg) {
    Write-Host ""
    Write-Host "=== $msg === [$((Get-Date).ToString('HH:mm:ss'))]" -ForegroundColor Cyan
}

function Write-Step([string]$msg) {
    Write-Host "--- $msg ---" -ForegroundColor Gray
}

function die([string]$msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

function Invoke-SSH([string]$cmd, [switch]$sudo, [switch]$NoExit) {
    $realCmd = if ($sudo) { "sudo -- bash -c '$cmd'" } else { $cmd }
    $sshTarget = "$User@$JetsonHost"
    $result = & ssh.exe -o StrictHostKeyChecking=accept-new `
                        -o ConnectTimeout=10 `
                        -o BatchMode=yes `
                        -i $SshKey `
                        $sshTarget $realCmd
    if ($LASTEXITCODE -ne 0 -and -not $NoExit) {
        die "Remote command failed (exit $LASTEXITCODE): $cmd"
    }
    return $result
}

function Invoke-SudoSSH([string]$cmd) {
    $sshTarget = "$User@$JetsonHost"
    $result = & ssh.exe -o StrictHostKeyChecking=accept-new `
                        -o ConnectTimeout=10 `
                        -i $SshKey `
                        $sshTarget "sudo bash -c `"$cmd`""
    if ($LASTEXITCODE -ne 0) {
        die "Remote sudo command failed (exit $LASTEXITCODE): $cmd"
    }
    return $result
}

# ---------------------------------------------------------------------------
# Step 0: Resolve host
# ---------------------------------------------------------------------------
function Resolve-JetsonHost {
    Write-Section "Resolving Jetson host"

    if ($JetsonHost -ne "") {
        Write-Step "Using provided host: $JetsonHost"
        return
    }

    # Try USB gadget first (fastest, <1ms)
    Write-Step "Trying USB gadget (192.168.55.1)..."
    $ping = Test-Connection -ComputerName 192.168.55.1 -Count 1 -Quiet -ErrorAction SilentlyContinue
    if ($ping) {
        $script:JetsonHost = "192.168.55.1"
        Write-Step "USB gadget reachable: $JetsonHost"
        return
    }

    # Fall back to ethernet mDNS
    Write-Step "USB not reachable. Trying mousedroid.local (ethernet)..."
    $ping2 = Test-Connection -ComputerName mousedroid.local -Count 1 -Quiet -ErrorAction SilentlyContinue
    if ($ping2) {
        $script:JetsonHost = "mousedroid.local"
        Write-Step "Ethernet reachable: $JetsonHost"
        return
    }

    die "Cannot reach Jetson on 192.168.55.1 or mousedroid.local. Check USB-C/ethernet connection."
}

# ---------------------------------------------------------------------------
# Step 1: Verify SSH
# ---------------------------------------------------------------------------
function Test-SSHConnectivity {
    Write-Section "Verifying SSH connectivity"
    Write-Step "Testing ${User}@${JetsonHost} with key ${SshKey}..."
    $sshTarget = "$User@$JetsonHost"

    $result = & ssh.exe -o StrictHostKeyChecking=accept-new `
                        -o ConnectTimeout=8 `
                        -o BatchMode=yes `
                        -i $SshKey `
                        $sshTarget "echo SSH_OK; hostname; uname -m" 2>&1

    if ($LASTEXITCODE -ne 0) {
        die "SSH failed. Ensure ~/.ssh/id_ed25519.pub is in ~ian/.ssh/authorized_keys on the Jetson."
    }
    Write-Step "SSH verified: $($result -join ' | ')"
}

# ---------------------------------------------------------------------------
# Step 2: Transfer project code via git archive | ssh
# ---------------------------------------------------------------------------
function Send-ProjectCode {
    Write-Section "Transferring project code (git archive | ssh)"

    # Ensure destination exists, owned by $User — clean slate
    Write-Step "Preparing $RemoteSrc on Jetson (clean)..."
    $sshTarget = "$User@$JetsonHost"
    & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
        $sshTarget "sudo rm -rf $RemoteSrc && sudo mkdir -p $RemoteSrc && sudo chown -R ${User}:${User} $RemoteSrc"
    if ($LASTEXITCODE -ne 0) { die "Failed to prepare remote directory." }

    # git archive HEAD — only committed tracked files, no locked/temp/test-cache files
    Write-Step "Archiving HEAD and streaming to ${JetsonHost}:${RemoteSrc}/"
    $sshArgs = "-o StrictHostKeyChecking=accept-new -o BatchMode=yes -i `"$SshKey`" ${User}@${JetsonHost} `"tar -xzf - -C $RemoteSrc`""
    $cmd = "git -C `"$ProjectDir`" archive --format=tar.gz HEAD | ssh.exe $sshArgs"
    & cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) { die "Code transfer failed." }

    # Fix Windows CRLF -> LF (git archive on Windows may add \r\n)
    Write-Step "Fixing line endings (CRLF -> LF)..."
    & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
        $sshTarget "find $RemoteSrc -type f \( -name '*.sh' -o -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' -o -name '*.cfg' -o -name '*.md' -o -name '*.txt' -o -name '*.service' -o -name 'Dockerfile*' -o -name '.dockerignore' \) -exec sed -i 's/\r$//' {} +"

    Write-Step "Code transfer complete."
}

# ---------------------------------------------------------------------------
# Step 3: Deploy config files
# ---------------------------------------------------------------------------
function Send-Config {
    Write-Section "Deploying configuration"
    $sshTarget = "$User@$JetsonHost"

    & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
        $sshTarget "sudo mkdir -p $RemoteConfig && sudo chown ${User}:${User} $RemoteConfig"

    $configDir = Join-Path $ProjectDir "config"
    Get-ChildItem $configDir -Filter "*.yaml" | ForEach-Object {
        Write-Step "  -> $($_.Name)"
        & scp.exe -o StrictHostKeyChecking=accept-new -i $SshKey `
            $_.FullName "${User}@${JetsonHost}:/tmp/$($_.Name)"
        & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
            $sshTarget "sudo cp /tmp/$($_.Name) $RemoteConfig/$($_.Name) && rm /tmp/$($_.Name)"
    }
    Write-Step "Config deployment complete."
}

# ---------------------------------------------------------------------------
# Step 4: Docker build + start on Jetson
# ---------------------------------------------------------------------------
function Invoke-DockerDeploy {
    Write-Section "Docker deploy on Jetson"
    Write-Step "Running docker_deploy.sh remotely (L4T base pull ~10 GB on first run)..."
    $sshTarget = "$User@$JetsonHost"

    # Stream output live via ssh without BatchMode so we see progress
    & ssh.exe -o StrictHostKeyChecking=accept-new -i $SshKey `
        $sshTarget "sudo bash $RemoteSrc/scripts/docker_deploy.sh 2>&1"

    if ($LASTEXITCODE -ne 0) { die "docker_deploy.sh failed on Jetson." }
    Write-Step "Docker deploy complete."
}

# ---------------------------------------------------------------------------
# Helper: write a bash script to a temp file, scp it, run it, clean up.
# Avoids all PowerShell here-string / quoting issues with embedded Python.
# ---------------------------------------------------------------------------
function Invoke-RemoteBashScript([string[]]$ScriptLines, [string]$ScriptName) {
    $tempFile = Join-Path ([System.IO.Path]::GetTempPath()) $ScriptName
    $ScriptLines -join "`n" | Set-Content -Path $tempFile -Encoding UTF8
    & scp.exe -q -o StrictHostKeyChecking=accept-new -i $SshKey `
        $tempFile "${User}@${JetsonHost}:/tmp/$ScriptName"
    Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    $sshTarget = "$User@$JetsonHost"
    & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
        $sshTarget "bash /tmp/$ScriptName; rm -f /tmp/$ScriptName"
    if ($LASTEXITCODE -ne 0) { die "Remote script '$ScriptName' failed (exit $LASTEXITCODE)." }
}

# ---------------------------------------------------------------------------
# Step 5: Pull weights from HuggingFace on Jetson
# ---------------------------------------------------------------------------
function Invoke-WeightsPull {
    Write-Section "Pulling weights from HuggingFace on Jetson"
    Write-Step "Repo: $HfRepo  ->  $RemoteWeights"

    $lines = @(
        '#!/bin/bash',
        'set -e',
        'python3 -c "import huggingface_hub" 2>/dev/null || pip3 install -q huggingface-hub',
        "mkdir -p $RemoteWeights",
        "python3 - <<'PYEOF'",
        'from huggingface_hub import snapshot_download',
        "snapshot_download(repo_id='$HfRepo', local_dir='$RemoteWeights', local_dir_use_symlinks=False)",
        "print('weights_download_complete')",
        'PYEOF'
    )
    Invoke-RemoteBashScript -ScriptLines $lines -ScriptName 'md_pull_weights.sh'
    Write-Step "Weights pulled successfully."
}

# ---------------------------------------------------------------------------
# Step 6: Provision LLM model (--with-llm mode)
# ---------------------------------------------------------------------------
function Invoke-LLMProvision {
    Write-Section "Provisioning LLM model (Phi-3 Mini 4K GGUF)"
    $modelFile = "$RemoteModels/Phi-3-mini-4k-instruct-q4.gguf"
    Write-Step "Checking for existing model at $modelFile..."

    $sshTarget = "$User@$JetsonHost"
    $exists = & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
        $sshTarget "test -f $modelFile && echo EXISTS || echo MISSING" 2>&1

    if ($exists -match 'EXISTS') {
        Write-Step "LLM model already present - skipping download."
        return
    }

    Write-Step "Downloading Phi-3-mini-4k-instruct-q4.gguf from HuggingFace (~2.4 GB)..."
    $lines = @(
        '#!/bin/bash',
        'set -e',
        "mkdir -p $RemoteModels",
        "python3 - <<'PYEOF'",
        'from huggingface_hub import hf_hub_download',
        "hf_hub_download('microsoft/Phi-3-mini-4k-instruct-gguf', 'Phi-3-mini-4k-instruct-q4.gguf', local_dir='$RemoteModels')",
        "print('llm_download_complete')",
        'PYEOF'
    )
    Invoke-RemoteBashScript -ScriptLines $lines -ScriptName 'md_pull_llm.sh'
    Write-Step "LLM model provisioned."
}

# ---------------------------------------------------------------------------
# Step 7: Smoke test
# ---------------------------------------------------------------------------
function Invoke-SmokeTest {
    Write-Section "Running smoke test"
    Write-Step "Testing mousedroid import inside container..."
    $sshTarget = "$User@$JetsonHost"

    $importCheck = & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
        $sshTarget "docker exec mousedroid python3 -c 'import mousedroid; print(mousedroid.__version__)'" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  mousedroid import OK: $importCheck" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: mousedroid import check failed - $importCheck" -ForegroundColor Yellow
    }

    $llmCheck = & ssh.exe -o StrictHostKeyChecking=accept-new -o BatchMode=yes -i $SshKey `
        $sshTarget "docker exec mousedroid python3 -c 'from mousedroid.llm_gateway.gateway import LLMGateway; print(\"LLMGateway OK\")'" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  LLMGateway import OK" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: LLMGateway check failed - $llmCheck" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
function Write-Summary {
    Write-Section "Deployment Summary"
    Write-Host "  Host:     ${User}@${JetsonHost}"
    Write-Host "  Mode:     $Mode"
    Write-Host "  Source:   $ProjectDir"
    Write-Host "  Started:  $($DeployStart.ToString('HH:mm:ss'))"
    Write-Host "  Finished: $((Get-Date).ToString('HH:mm:ss'))"
    Write-Host ""
    Write-Host "=== Deployment complete ===" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Resolve-JetsonHost
Test-SSHConnectivity

switch ($Mode) {
    "full" {
        Send-ProjectCode
        Send-Config
        Invoke-DockerDeploy
        Invoke-WeightsPull
        Invoke-SmokeTest
    }
    "code-only" {
        Send-ProjectCode
        Send-Config
        Invoke-DockerDeploy
        Invoke-SmokeTest
    }
    "weights" {
        Invoke-WeightsPull
    }
    "with-llm" {
        Send-ProjectCode
        Send-Config
        Invoke-DockerDeploy
        Invoke-WeightsPull
        Invoke-LLMProvision
        Invoke-SmokeTest
    }
}

Write-Summary
