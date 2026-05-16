#!/usr/bin/env bash
# =============================================================================
# scripts/cloud_train.sh — Vertex AI entry point for the Tier C1 cloud trainer.
# =============================================================================
# Invoked as the Dockerfile.cloud ENTRYPOINT. Reads the operator-provided
# config from ${MOUSEDROID_CLOUD_TRAIN_CONFIG} (default
# /etc/mousedroid/cloud_train.yaml), runs ``train_offline_rl`` against the
# LMDB shards Vertex AI mounted under /gcs/<bucket>/, and pushes the
# resulting policy artifact + sha256.txt to HuggingFace Hub.
#
# Idempotency: ``train_offline_rl`` checks for a ``shard_consumed_marker``
# in GCS before processing each shard so duplicate uploads from a Jetson
# restart do NOT double-train. See ADR-010.
# =============================================================================

set -euo pipefail

readonly CONFIG_PATH="${MOUSEDROID_CLOUD_TRAIN_CONFIG:-/etc/mousedroid/cloud_train.yaml}"

# Vertex AI injects credentials via GOOGLE_APPLICATION_CREDENTIALS or via
# Workload Identity; both are picked up automatically by google-cloud-storage.
# HF Hub auth comes from HUGGINGFACE_HUB_TOKEN (a Vertex AI secret).
: "${HUGGINGFACE_HUB_TOKEN:?HUGGINGFACE_HUB_TOKEN required for --push-to-hf}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "cloud_train_error: missing config at ${CONFIG_PATH}" >&2
    exit 1
fi

echo "cloud_train_starting config=${CONFIG_PATH}"

exec python -m training.train_offline_rl \
    --config "${CONFIG_PATH}" \
    --push-to-hf \
    "$@"
