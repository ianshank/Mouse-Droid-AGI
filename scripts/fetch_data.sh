#!/usr/bin/env bash
# =============================================================================
# MouseDroid -- Training Data Fetcher (bdi_annotations.npz)
# =============================================================================
# The BDI annotation array is a *generated* artifact (~26 MB). It is NOT
# committed to git (binary blobs bloat clone size and the file is reproducible).
# This helper either regenerates it from source (authoritative) or pulls a
# published Hugging Face mirror (fast path).
#
# Usage:
#   bash scripts/fetch_data.sh              # regenerate via the pretraining pipeline (default)
#   bash scripts/fetch_data.sh --from-hf    # download from the HF dataset mirror
#   bash scripts/fetch_data.sh --verbose    # xtrace every command (or DEBUG=1)
#   bash scripts/fetch_data.sh --help
#
# Environment overrides (no hardcoded values):
#   CONFIG      Config YAML for regeneration      (default: config/mock_hardware.yaml)
#   HF_DATASET  HF dataset repo id for --from-hf  (default: ianshank/mouse-droid-bdi-annotations)
#   DATA_DIR    HF download dir (--from-hf only)  (default: training/data)
#   DEBUG       Set to 1 to xtrace every command  (default: 0)
# =============================================================================
set -euo pipefail

CONFIG="${CONFIG:-config/mock_hardware.yaml}"
HF_DATASET="${HF_DATASET:-ianshank/mouse-droid-bdi-annotations}"
DATA_DIR="${DATA_DIR:-training/data}"
MODE="regenerate"

while [ $# -gt 0 ]; do
  case "$1" in
    --from-hf) MODE="hf" ;;
    --regenerate) MODE="regenerate" ;;
    -v | --verbose) set -x ;;
    -h | --help)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1 (try --help)" >&2
      exit 2
      ;;
  esac
  shift
done

# Debugging: DEBUG=1 (env) or --verbose turns on an xtrace of every command.
if [ "${DEBUG:-0}" = "1" ]; then set -x; fi

# Resolve to the repo root so relative paths work from any CWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Resolve the artefact path per mode. Regeneration writes to the config's
# ``training.data_dir`` (NOT the DATA_DIR override, which only governs the HF
# download location), so read it back from CONFIG to check the real output path.
if [ "${MODE}" = "hf" ]; then
  TARGET="${DATA_DIR%/}/bdi_annotations.npz"
else
  # Pass CONFIG as an argv value (never interpolated into Python source — a crafted
  # path could otherwise break out of the string literal), and let a load/import
  # error propagate under `set -e` rather than masking it with a wrong-path fallback.
  REGEN_DIR="$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path

from mousedroid.config.loader import load_settings

print(load_settings(Path(sys.argv[1])).training.data_dir)
PY
  )"
  TARGET="${REGEN_DIR%/}/bdi_annotations.npz"
fi

if [ -f "${TARGET}" ]; then
  echo "[fetch_data] ${TARGET} already present — nothing to do (delete it to force a refresh)."
  exit 0
fi

mkdir -p "$(dirname "${TARGET}")"

if [ "${MODE}" = "hf" ]; then
  echo "[fetch_data] Downloading ${HF_DATASET} -> ${TARGET}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${HF_DATASET}" bdi_annotations.npz \
      --repo-type dataset --local-dir "${DATA_DIR}"
  elif python -c "import huggingface_hub" >/dev/null 2>&1; then
    python - "${HF_DATASET}" "${DATA_DIR}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
path = hf_hub_download(
    repo_id=repo_id,
    filename="bdi_annotations.npz",
    repo_type="dataset",
    local_dir=local_dir,
)
print(f"[fetch_data] downloaded: {path}")
PY
  else
    echo "[fetch_data] ERROR: huggingface_hub not installed. Run: pip install huggingface_hub" >&2
    echo "[fetch_data] ...or regenerate instead: bash scripts/fetch_data.sh" >&2
    exit 1
  fi
else
  echo "[fetch_data] Regenerating annotations via the pretraining pipeline (phase 0)"
  echo "[fetch_data]   config: ${CONFIG} -> ${TARGET}"
  python -m training.run_pipeline --config "${CONFIG}" --phases 0
fi

if [ -f "${TARGET}" ]; then
  echo "[fetch_data] OK: ${TARGET} ($(du -h "${TARGET}" | cut -f1))"
else
  echo "[fetch_data] WARNING: expected ${TARGET} was not produced — check the output above." >&2
  exit 1
fi
