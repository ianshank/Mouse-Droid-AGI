#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI -- HuggingFace Weight Downloader
# =============================================================================
# Downloads pre-trained model weights from HuggingFace Hub.
# Idempotent: skips repos whose local directories already contain files.
#
# Usage:
#   ./scripts/download_weights.sh [OPTIONS]
#
# Options:
#   --weights-dir DIR   Target directory for downloads (default: weights/)
#   --repo REPO         Download only this repo (can be repeated)
#   --force             Re-download even if files exist locally
#   --help              Show this help message
#
# Examples:
#   ./scripts/download_weights.sh
#   ./scripts/download_weights.sh --weights-dir /data/models
#   ./scripts/download_weights.sh --repo ianshank/mousedroid-weights
#   ./scripts/download_weights.sh --force
# =============================================================================

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
WEIGHTS_DIR="weights/"
FORCE=false
SELECTED_REPOS=()

# All HuggingFace repos containing MouseDroid model weights
ALL_REPOS=(
    "ianshank/mousedroid-weights"
    "ianshank/mousedroid-dual-stream-rssm"
)

# --- Argument parsing --------------------------------------------------------
show_help() {
    sed -n '2,/^# =====/{ /^# =====/d; s/^# //; s/^#$//; p }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights-dir)
            WEIGHTS_DIR="$2"
            shift 2
            ;;
        --repo)
            SELECTED_REPOS+=("$2")
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# Use selected repos or all repos
if [[ ${#SELECTED_REPOS[@]} -gt 0 ]]; then
    REPOS=("${SELECTED_REPOS[@]}")
else
    REPOS=("${ALL_REPOS[@]}")
fi

# --- Preflight checks -------------------------------------------------------
HF_CLI_AVAILABLE=false
PYTHON_AVAILABLE=false

if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI_AVAILABLE=true
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_AVAILABLE=true
elif command -v python >/dev/null 2>&1; then
    PYTHON_AVAILABLE=true
fi

if [[ "$HF_CLI_AVAILABLE" == "false" && "$PYTHON_AVAILABLE" == "false" ]]; then
    echo "ERROR: Neither huggingface-cli nor python3 found."
    echo "Install huggingface-hub with:  pip install huggingface-hub"
    exit 1
fi

if [[ "$HF_CLI_AVAILABLE" == "false" ]]; then
    echo "WARNING: huggingface-cli not found, falling back to Python download."
fi

# --- Helper functions --------------------------------------------------------
download_with_hf_cli() {
    local repo="$1"
    local local_dir="$2"

    huggingface-cli download "$repo" \
        --local-dir "$local_dir" \
        --repo-type model
}

download_with_python() {
    local repo="$1"
    local local_dir="$2"

    local python_cmd="python3"
    if ! command -v python3 >/dev/null 2>&1; then
        python_cmd="python"
    fi

    "$python_cmd" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$repo',
    local_dir='$local_dir',
    repo_type='model',
)
print('Download complete.')
"
}

validate_download() {
    local local_dir="$1"
    local repo="$2"

    if [[ ! -d "$local_dir" ]]; then
        echo "  FAIL: Directory $local_dir was not created"
        return 1
    fi

    local file_count
    file_count=$(find "$local_dir" -type f | wc -l)
    if [[ "$file_count" -eq 0 ]]; then
        echo "  FAIL: No files downloaded for $repo"
        return 1
    fi

    echo "  OK: $file_count file(s) in $local_dir"
    return 0
}

# --- Download ----------------------------------------------------------------
echo "MouseDroidAGI Weight Downloader"
echo "================================"
echo "Target directory: $WEIGHTS_DIR"
echo "Repos to download: ${#REPOS[@]}"
echo ""

mkdir -p "$WEIGHTS_DIR"

SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

for repo in "${REPOS[@]}"; do
    # Derive local subdirectory from repo name (owner--repo)
    local_dir="${WEIGHTS_DIR}/${repo//\//__}"

    # Idempotency: skip if directory exists and has files (unless --force)
    if [[ "$FORCE" == "false" ]] && [[ -d "$local_dir" ]] && [[ "$(ls -A "$local_dir" 2>/dev/null)" ]]; then
        echo "SKIP: $repo (already at $local_dir)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo "Downloading: $repo -> $local_dir ..."

    # Try huggingface-cli first, fall back to Python
    download_ok=false
    if [[ "$HF_CLI_AVAILABLE" == "true" ]]; then
        if download_with_hf_cli "$repo" "$local_dir"; then
            download_ok=true
        else
            echo "  WARNING: huggingface-cli failed, trying Python fallback..."
        fi
    fi

    if [[ "$download_ok" == "false" ]] && [[ "$PYTHON_AVAILABLE" == "true" ]]; then
        if download_with_python "$repo" "$local_dir"; then
            download_ok=true
        fi
    fi

    if [[ "$download_ok" == "false" ]]; then
        echo "  ERROR: Failed to download $repo"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Validate the download
    if validate_download "$local_dir" "$repo"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# --- Summary -----------------------------------------------------------------
echo ""
echo "================================"
echo "Summary: $SUCCESS_COUNT downloaded, $SKIP_COUNT skipped, $FAIL_COUNT failed"
echo "Weights directory: $WEIGHTS_DIR"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo ""
    echo "Some downloads failed. Check your network connection and HuggingFace authentication."
    echo "For private repos, run: huggingface-cli login"
    exit 1
fi

exit 0
