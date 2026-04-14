#!/usr/bin/env bash
# =============================================================================
# MouseDroidAGI — HuggingFace Weight Downloader
# =============================================================================
# Downloads pre-trained model weights from HuggingFace Hub.
# Idempotent: skips repos that already exist locally.
#
# Usage:
#   ./scripts/download_weights.sh [target_dir]
#
# Arguments:
#   target_dir  Override the default weights directory (default: weights/bdi)
# =============================================================================

set -euo pipefail

# --- Configuration (mirrors CognitiveConfig defaults in schema.py) -----------
WEIGHTS_DIR="${1:-weights/bdi}"
REPOS=(
    "ianshank/mousedroid-weights"
    "ianshank/mousedroid-dual-stream-rssm"
)

# --- Preflight checks -------------------------------------------------------
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found."
    echo "Install it with:  pip install huggingface-hub"
    exit 1
fi

# --- Download ----------------------------------------------------------------
mkdir -p "${WEIGHTS_DIR}"

for repo in "${REPOS[@]}"; do
    # Derive local subdirectory from repo name (owner--repo)
    local_dir="${WEIGHTS_DIR}/${repo//\//__}"

    if [ -d "${local_dir}" ] && [ "$(ls -A "${local_dir}" 2>/dev/null)" ]; then
        echo "SKIP: ${repo} already downloaded to ${local_dir}"
        continue
    fi

    echo "Downloading ${repo} -> ${local_dir} ..."
    huggingface-cli download "${repo}" \
        --local-dir "${local_dir}" \
        --quiet
    echo "  OK: ${repo}"
done

echo ""
echo "All weights downloaded to ${WEIGHTS_DIR}"
