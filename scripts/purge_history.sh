#!/usr/bin/env bash
# =============================================================================
# MouseDroid -- Git History Purge (bdi_annotations.npz + docs/3D_printing_files)
# =============================================================================
# DESTRUCTIVE + IRREVERSIBLE. Rewrites git history to permanently remove the
# large CAD/data blobs, shrinking the clone from ~28 MB (~25.8 MB of it the
# .npz) to ~2 MB.
#
#   *** READ docs/runbooks/history-purge.md BEFORE RUNNING THIS. ***
#
# Preconditions the operator MUST satisfy first (this script CANNOT verify them):
#   1. bdi_annotations.npz is mirrored to the Hugging Face dataset.
#   2. The STL/FCStd files are uploaded to the `hardware-v6` GitHub Release.
# Losing these before the purge = permanent data loss.
#
# What it does — in a throwaway fresh clone, NEVER your working repo:
#   fresh clone -> git filter-repo (drop the paths) -> re-pin
#   deployments/jetson-image.json to the commit-map image of the deployed SHA ->
#   verify config-compat -> (optionally) force-push the default branch.
#
# Usage:
#   bash scripts/purge_history.sh            # DRY RUN (default): clone, purge, verify; NO push
#   bash scripts/purge_history.sh --push     # do it for real and force-push the default branch
#   bash scripts/purge_history.sh --help
#
# Env overrides (no hardcoded values):
#   ORIGIN_URL     remote to clone/push   (default: origin of the current repo)
#   DEFAULT_BRANCH branch to rewrite      (default: claude/markdown-implementation-plan-aVJ2l)
#   PINNED_SHA     deployed SHA to re-pin (default: read from deployments/jetson-image.json)
#   IMAGE_JSON     re-pin target file     (default: deployments/jetson-image.json)
#   WORKDIR        scratch dir            (default: mktemp -d)
# =============================================================================
set -euo pipefail

PUSH=0
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    --dry-run) PUSH=0 ;;
    -h | --help)
      sed -n '2,37p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $a (try --help)" >&2
      exit 2
      ;;
  esac
done

command -v git >/dev/null || {
  echo "git not found" >&2
  exit 1
}
if ! git filter-repo --version >/dev/null 2>&1; then
  echo "ERROR: git-filter-repo is not installed. Run: pip install git-filter-repo" >&2
  exit 1
fi

SRC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGIN_URL="${ORIGIN_URL:-$(git -C "${SRC_REPO}" remote get-url origin)}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-claude/markdown-implementation-plan-aVJ2l}"
IMAGE_JSON="${IMAGE_JSON:-deployments/jetson-image.json}"
WORKDIR="${WORKDIR:-$(mktemp -d)}"
PATHS=(training/data/bdi_annotations.npz docs/3D_printing_files)
CLONE="${WORKDIR}/mouse-droid"

# Default PINNED_SHA from the deployment record unless the operator overrode it.
if [ -z "${PINNED_SHA:-}" ]; then
  PINNED_SHA="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sha'])" \
    "${SRC_REPO}/${IMAGE_JSON}")"
fi

echo "=== history purge ==="
echo "  origin:        ${ORIGIN_URL}"
echo "  branch:        ${DEFAULT_BRANCH}"
echo "  paths to drop: ${PATHS[*]}"
echo "  re-pin SHA:    ${PINNED_SHA}"
echo "  workdir:       ${CLONE}"
echo "  mode:          $([ "${PUSH}" = 1 ] && echo 'EXECUTE + FORCE-PUSH' || echo 'DRY RUN (no push)')"
echo ""

echo "[1/5] Fresh clone (all refs)"
git clone --no-local "${ORIGIN_URL}" "${CLONE}"
cd "${CLONE}"

echo "[2/5] filter-repo: dropping the blobs from ALL history"
FR_ARGS=()
for p in "${PATHS[@]}"; do FR_ARGS+=(--path "${p}"); done
git filter-repo --force "${FR_ARGS[@]}" --invert-paths

echo "[3/5] Re-pin ${IMAGE_JSON} to the commit-map image of ${PINNED_SHA} (NOT HEAD)"
MAP=".git/filter-repo/commit-map"
[ -f "${MAP}" ] || {
  echo "no commit-map at ${MAP}" >&2
  exit 1
}
NEW_SHA="$(awk -v old="${PINNED_SHA}" '$1==old {print $2}' "${MAP}")"
if [ -z "${NEW_SHA}" ] || printf '%s' "${NEW_SHA}" | grep -qE '^0+$'; then
  echo "ERROR: ${PINNED_SHA} maps to a pruned/empty commit ('${NEW_SHA:-<none>}')." >&2
  echo "The purge dropped the pinned commit. Pick another schema-equivalent, reachable" >&2
  echo "SHA and re-run with PINNED_SHA=<sha> (see docs/runbooks/history-purge.md)." >&2
  exit 1
fi
echo "       ${PINNED_SHA} -> ${NEW_SHA}"
git checkout "${DEFAULT_BRANCH}"
python - "${IMAGE_JSON}" "${NEW_SHA}" <<'PY'
import json, sys

path, new = sys.argv[1], sys.argv[2]
with open(path) as fh:
    data = json.load(fh)
old = data.get("sha")
data["sha"] = new
with open(path, "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
print(f"[re-pin] {path}: {old} -> {new}")
PY
git add "${IMAGE_JSON}"
git commit -m "chore(deploy): re-pin jetson image SHA after history purge"

echo "[4/5] Verify config-compat against the re-pinned SHA (schema must LOAD, not just be reachable)"
python scripts/check_config_compat.py --platform jetson \
  --changed-files config/jetson_production.yaml

echo "[5/5] Result"
git count-objects -vH | grep -E 'count:|size-pack:' || true
if [ "${PUSH}" = 1 ]; then
  echo ">>> FORCE-PUSHING ${DEFAULT_BRANCH} + tags to ${ORIGIN_URL}"
  echo ">>> (filter-repo removed 'origin'; re-adding it now.)"
  git remote add origin "${ORIGIN_URL}" 2>/dev/null || git remote set-url origin "${ORIGIN_URL}"
  git push --force origin "${DEFAULT_BRANCH}"
  git push --force --tags origin
  echo "Done. Every existing clone/PR/fork is now stale and must re-clone."
else
  echo "DRY RUN complete — nothing was pushed. Rewritten repo at: ${CLONE}"
  echo "Inspect it, then re-run with --push (or push manually from ${CLONE})."
fi
