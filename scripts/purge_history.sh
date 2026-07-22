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
# What it does — in a throwaway bare *mirror* clone, NEVER your working repo:
#   mirror clone (ALL refs) -> git filter-repo (drop the blobs from every ref) ->
#   re-pin deployments/jetson-image.json to the commit-map image of the deployed
#   SHA (in a worktree) -> verify config-compat -> (optionally) force-push EVERY
#   rewritten branch + tags. All branches are pushed because a blob left reachable
#   from any un-rewritten branch defeats the purge.
#
# Usage:
#   bash scripts/purge_history.sh            # DRY RUN (default): clone, purge, verify; NO push
#   bash scripts/purge_history.sh --push     # do it for real and force-push all rewritten refs
#   bash scripts/purge_history.sh --verbose  # xtrace every command (or DEBUG=1)
#   bash scripts/purge_history.sh --help
#
# Env overrides (no hardcoded values):
#   ORIGIN_URL     remote to clone/push   (default: origin of the current repo)
#   DEFAULT_BRANCH branch to re-pin       (default: the TARGET remote's detected default branch)
#   PINNED_SHA     deployed SHA to re-pin (default: read from deployments/jetson-image.json)
#   IMAGE_JSON     re-pin target file     (default: deployments/jetson-image.json)
#   WORKDIR        scratch dir            (default: mktemp -d)
#   DEBUG          set to 1 to xtrace every command (or pass --verbose)
# =============================================================================
set -euo pipefail

PUSH=0
for a in "$@"; do
  case "$a" in
    --push) PUSH=1 ;;
    --dry-run) PUSH=0 ;;
    -v | --verbose) set -x ;;
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

# Debugging: DEBUG=1 (env) or --verbose turns on an xtrace of every command —
# valuable when triaging a failed rewrite on the operator's machine.
if [ "${DEBUG:-0}" = "1" ]; then set -x; fi

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
IMAGE_JSON="${IMAGE_JSON:-deployments/jetson-image.json}"
WORKDIR="${WORKDIR:-$(mktemp -d)}"
CLONE="${WORKDIR}/mouse-droid.git" # bare mirror (all refs)
WT="${WORKDIR}/repin"              # worktree for the re-pin commit + config-compat verify

# Purge the generated data blob + the CAD *binaries* only — NOT the whole
# docs/3D_printing_files/ dir, which also holds the pointer README the reframe adds.
PURGE_DISPLAY="training/data/bdi_annotations.npz docs/3D_printing_files/*.{stl,FCStd}"

# Default branch to re-pin: resolved from the TARGET remote (ORIGIN_URL), not the
# source checkout — an ORIGIN_URL override must re-pin the target's default, not
# this repo's. An explicit DEFAULT_BRANCH env always wins; otherwise it is filled
# from the mirror's HEAD after cloning.
DEFAULT_BRANCH="${DEFAULT_BRANCH:-}"

# Default PINNED_SHA from the deployment record unless the operator overrode it.
if [ -z "${PINNED_SHA:-}" ]; then
  PINNED_SHA="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sha'])" \
    "${SRC_REPO}/${IMAGE_JSON}")"
fi

echo "=== history purge ==="
echo "  origin:        ${ORIGIN_URL}"
echo "  branch:        ${DEFAULT_BRANCH:-<detect from target remote>}"
echo "  paths to drop: ${PURGE_DISPLAY}"
echo "  re-pin SHA:    ${PINNED_SHA}"
echo "  workdir:       ${CLONE}"
echo "  mode:          $([ "${PUSH}" = 1 ] && echo 'EXECUTE + FORCE-PUSH (all refs)' || echo 'DRY RUN (no push)')"
echo ""

echo "[1/6] Bare mirror clone (ALL refs)"
git clone --mirror "${ORIGIN_URL}" "${CLONE}"
cd "${CLONE}"

# Resolve the target remote's default branch (mirror HEAD reflects it).
if [ -z "${DEFAULT_BRANCH}" ]; then
  DEFAULT_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || true)"
fi
if [ -z "${DEFAULT_BRANCH}" ]; then
  DEFAULT_BRANCH="$(git ls-remote --symref "${ORIGIN_URL}" HEAD \
    | sed -n 's#^ref:[[:space:]]*refs/heads/\(.*\)[[:space:]]*HEAD$#\1#p' | head -1)"
fi
[ -n "${DEFAULT_BRANCH}" ] || {
  echo "ERROR: could not resolve the default branch; set DEFAULT_BRANCH=<name> and re-run." >&2
  exit 1
}
echo "      default branch: ${DEFAULT_BRANCH}"

echo "[2/6] filter-repo: dropping the data blob + CAD binaries from ALL refs"
# Glob the CAD *binaries* rather than the whole dir so the pointer README.md survives.
git filter-repo --force --invert-paths \
  --path training/data/bdi_annotations.npz \
  --path-glob 'docs/3D_printing_files/*.stl' \
  --path-glob 'docs/3D_printing_files/*.FCStd'

echo "[3/6] Re-pin ${IMAGE_JSON} to the commit-map image of ${PINNED_SHA} (NOT HEAD)"
MAP="$(git rev-parse --git-dir)/filter-repo/commit-map"
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

echo "[4/6] Re-pin in a worktree on ${DEFAULT_BRANCH} + commit"
git worktree add "${WT}" "${DEFAULT_BRANCH}"
python - "${WT}/${IMAGE_JSON}" "${NEW_SHA}" <<'PY'
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
git -C "${WT}" add "${IMAGE_JSON}"
git -C "${WT}" commit -m "chore(deploy): re-pin jetson image SHA after history purge"

echo "[5/6] Verify config-compat against the re-pinned SHA (schema must LOAD, not just be reachable)"
(cd "${WT}" && python scripts/check_config_compat.py --platform jetson \
  --changed-files config/jetson_production.yaml)

echo "[6/6] Result"
git count-objects -vH | grep -E 'count:|size-pack:' || true
if [ "${PUSH}" = 1 ]; then
  echo ">>> FORCE-PUSHING every rewritten branch + tags to ${ORIGIN_URL}"
  echo ">>> (filter-repo removed 'origin'; re-adding it now.)"
  git remote add origin "${ORIGIN_URL}" 2>/dev/null || git remote set-url origin "${ORIGIN_URL}"
  # --all publishes every rewritten branch (a blob reachable from any un-rewritten
  # branch would defeat the purge); --tags publishes rewritten tags. Neither deletes.
  git push --force --all origin
  git push --force --tags origin
  echo "Done. Every existing clone/PR/fork is now stale and must re-clone."
else
  echo "DRY RUN complete — nothing was pushed. Rewritten mirror at: ${CLONE}"
  echo "Inspect it, then re-run with --push (pushes all rewritten branches + tags)."
fi
