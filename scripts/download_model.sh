#!/usr/bin/env bash
# Download Llama-3 GGUF model for MouseDroid LLM Gateway.
#
# Config-driven via environment variables:
#   MODEL_URL      — download URL (default from config)
#   MODEL_PATH     — destination path (default: /opt/mousedroid/models/)
#   MODEL_CHECKSUM — expected SHA-256 (empty = skip verification)
#   MAX_RETRIES    — download retry attempts (default: 3)
#
# Usage:
#   ./scripts/download_model.sh
#   MODEL_URL="https://..." MODEL_PATH="/tmp/model.gguf" ./scripts/download_model.sh

set -euo pipefail

# --- Configuration (override via environment) ---
DEFAULT_URL="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
MODEL_URL="${MODEL_URL:-$DEFAULT_URL}"
MODEL_PATH="${MODEL_PATH:-/opt/mousedroid/models/llama-3-8b-instruct.Q4_K_M.gguf}"
MODEL_CHECKSUM="${MODEL_CHECKSUM:-}"
MAX_RETRIES="${MAX_RETRIES:-3}"

# --- Helpers ---
log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }
log_warn() { echo "[WARN] $*"; }

# --- Verify checksum ---
verify_checksum() {
    local file="$1"
    local expected="$2"

    if [ -z "$expected" ]; then
        log_warn "No checksum provided — skipping verification."
        return 0
    fi

    log_info "Verifying SHA-256 checksum..."
    local actual
    if command -v sha256sum &>/dev/null; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif command -v shasum &>/dev/null; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        log_warn "No sha256sum or shasum found — skipping verification."
        return 0
    fi

    if [ "$actual" != "$expected" ]; then
        log_error "Checksum mismatch!"
        log_error "  Expected: $expected"
        log_error "  Actual:   $actual"
        rm -f "$file"
        return 1
    fi

    log_info "Checksum verified successfully."
    return 0
}

# --- Download with retry ---
download_with_retry() {
    local url="$1"
    local dest="$2"
    local retries="$3"
    local attempt=1

    while [ "$attempt" -le "$retries" ]; do
        log_info "Download attempt $attempt/$retries..."

        if command -v wget &>/dev/null; then
            if wget --progress=bar:force -c -O "$dest.tmp" "$url" 2>&1; then
                mv "$dest.tmp" "$dest"
                return 0
            fi
        elif command -v curl &>/dev/null; then
            if curl -L --progress-bar -C - -o "$dest.tmp" "$url"; then
                mv "$dest.tmp" "$dest"
                return 0
            fi
        else
            log_error "Neither wget nor curl found. Install one and retry."
            return 1
        fi

        log_warn "Attempt $attempt failed. Retrying in $((attempt * 5))s..."
        sleep $((attempt * 5))
        attempt=$((attempt + 1))
    done

    log_error "Download failed after $retries attempts."
    rm -f "$dest.tmp"
    return 1
}

# --- Main ---
main() {
    log_info "MouseDroid LLM Model Downloader"
    log_info "================================"
    log_info "URL:      $MODEL_URL"
    log_info "Path:     $MODEL_PATH"
    log_info "Checksum: ${MODEL_CHECKSUM:-<not set>}"
    log_info "Retries:  $MAX_RETRIES"
    echo

    # Check if model already exists
    if [ -f "$MODEL_PATH" ]; then
        if verify_checksum "$MODEL_PATH" "$MODEL_CHECKSUM"; then
            log_info "Model already exists and is valid. Nothing to do."
            exit 0
        fi
        log_warn "Existing model failed checksum — re-downloading."
    fi

    # Create destination directory
    local dest_dir
    dest_dir=$(dirname "$MODEL_PATH")
    if [ ! -d "$dest_dir" ]; then
        log_info "Creating directory: $dest_dir"
        mkdir -p "$dest_dir"
    fi

    # Check available disk space (need ~5GB for Q4_K_M)
    local available_mb
    if command -v df &>/dev/null; then
        available_mb=$(df -m "$dest_dir" | awk 'NR==2 {print $4}')
        if [ "$available_mb" -lt 5000 ] 2>/dev/null; then
            log_warn "Low disk space: ${available_mb}MB available. Model requires ~5GB."
        fi
    fi

    # Download
    log_info "Starting download..."
    if ! download_with_retry "$MODEL_URL" "$MODEL_PATH" "$MAX_RETRIES"; then
        log_error "Failed to download model."
        exit 1
    fi

    # Verify
    if ! verify_checksum "$MODEL_PATH" "$MODEL_CHECKSUM"; then
        log_error "Downloaded model failed checksum verification."
        exit 1
    fi

    log_info "Model downloaded successfully to: $MODEL_PATH"
    log_info "File size: $(du -h "$MODEL_PATH" | awk '{print $1}')"
}

main "$@"
