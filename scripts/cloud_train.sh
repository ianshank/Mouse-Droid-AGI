#!/usr/bin/env bash
# =============================================================================
# scripts/cloud_train.sh — Vertex AI entry point for the Tier C1 cloud trainer.
# =============================================================================
# Invoked as the Dockerfile.cloud ENTRYPOINT. Reads the operator-provided
# config from ${MOUSEDROID_CLOUD_TRAIN_CONFIG} (default
# /etc/mousedroid/cloud_train.yaml) and runs ``train_offline_rl`` against
# the LMDB shards Vertex AI mounted under /gcs/<bucket>/.
#
# What this script DOES today (Tier C1):
#   * Runs the offline-RL trainer.
#   * Honours the per-job GCS idempotency marker (a job that's already
#     produced its marker short-circuits without retraining; on success
#     the marker is written so subsequent re-invocations skip).
#
# What this script does NOT do yet (deferred follow-up, see ADR-010 +
# Tier C plan §"Out-of-Scope Items"):
#   * Push the resulting artifact to HuggingFace Hub. The upload module
#     (``training/upload_weights.py``) does not yet exist; once it lands
#     this script will be extended to run the upload after a successful
#     train + write a ``sha256.txt`` manifest. The Vertex AI WeightUpdatePoller
#     on the Jetson is already wired to verify SHA-256 against that manifest.
#
# Idempotency: ``train_offline_rl`` checks for the per-job
# ``shard_consumed_marker`` in GCS at startup and writes it on success.
# See ADR-010.
# =============================================================================

set -euo pipefail

readonly CONFIG_PATH="${MOUSEDROID_CLOUD_TRAIN_CONFIG:-/etc/mousedroid/cloud_train.yaml}"

# Vertex AI injects GCP credentials via GOOGLE_APPLICATION_CREDENTIALS or
# Workload Identity; both are picked up automatically by google-cloud-storage.
# HUGGINGFACE_HUB_TOKEN is NOT required for Tier C1 because no upload step
# runs yet — it'll become a hard requirement once the upload module lands.

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "cloud_train_error: missing config at ${CONFIG_PATH}" >&2
    exit 1
fi

echo "cloud_train_starting config=${CONFIG_PATH}"

exec python -m training.train_offline_rl \
    --config "${CONFIG_PATH}" \
    "$@"
