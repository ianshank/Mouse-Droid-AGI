#!/usr/bin/env bash
# Prove a regression pin can actually fail.
#
# A test that cannot fail is not a pin -- it is decoration that reads like
# safety. .claude/skills/regression-pair-scaffold/SKILL.md states the
# discipline ("confirm the pair can fail by temporarily reverting the change
# under test") but leaves it a manual cp/edit/run/restore dance, which is
# error-prone and easy to skip under time pressure. This mechanises it.
#
# What it does, in order:
#   1. snapshot the named paths
#   2. restore them from a git revision that predates the change
#   3. run the named tests -- they MUST fail (this is the proof)
#   4. restore the snapshot unconditionally, even on interrupt
#   5. run the named tests again -- they MUST pass
#
# Usage:
#   bash scripts/prove_pin_fails.sh --from <git-ref> \
#        --paths "<path> [<path> ...]" --tests "<pytest target> [...]"
#
# Example -- prove the F-029 egress pin detects a reverted default:
#   bash scripts/prove_pin_fails.sh --from origin/main \
#     --paths "src/mousedroid/config/schema/gcp_cloud.py" \
#     --tests "tests/regression/test_gcp_egress_defaults_aqa.py"
#
# Exit codes: 0 proof succeeded; 1 the pin did NOT fail (it is not load-bearing);
# 2 invocation error; 3 restore failed (working tree needs manual attention).
set -uo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

FROM_REF=""; PATHS=""; TESTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)  FROM_REF="${2:-}"; shift 2 ;;
    --paths) PATHS="${2:-}";    shift 2 ;;
    --tests) TESTS="${2:-}";    shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${FROM_REF}" || -z "${PATHS}" || -z "${TESTS}" ]]; then
  echo "usage: $0 --from <git-ref> --paths \"<paths>\" --tests \"<pytest targets>\"" >&2
  exit 2
fi

if ! git rev-parse --verify --quiet "${FROM_REF}^{commit}" >/dev/null; then
  echo "PROVE-PIN FAIL: --from ref '${FROM_REF}' does not resolve" >&2
  exit 2
fi

# Refuse to run against a dirty copy of the target paths: the snapshot/restore
# below would silently discard uncommitted work.
# shellcheck disable=SC2086
if ! git diff --quiet -- ${PATHS} || ! git diff --cached --quiet -- ${PATHS}; then
  echo "PROVE-PIN FAIL: uncommitted changes in --paths; commit or stash first" >&2
  exit 2
fi

SNAPSHOT="$(mktemp -d)"

# Map a --paths entry to its slot under $SNAPSHOT, mirroring the source tree.
#
# Keying on $(basename) instead would COLLIDE for two entries sharing a
# filename ("a/config.py b/config.py"): the second snapshot overwrites the
# first, and restore then writes b's content over a. That is silent data loss
# in a tool whose entire purpose is safe restore. Mirroring the relative path
# is collision-free by construction.
#
# Leading "./" and "/" are stripped so an absolute path lands inside the
# snapshot rather than at the filesystem root, and ".." segments are
# neutralised so no argument can write outside $SNAPSHOT.
snap_path() {
    local rel="$1"
    rel="${rel#./}"
    while [[ "${rel}" == /* ]]; do rel="${rel#/}"; done
    rel="${rel//..\//__up__/}"
    printf '%s/%s' "${SNAPSHOT}" "${rel}"
}

restore() {
    local rc=$?
    local snap
    # shellcheck disable=SC2086
    for f in ${PATHS}; do
        snap="$(snap_path "$f")"
        if [[ -f "${snap}" ]]; then
            cp "${snap}" "$f" || { echo "PROVE-PIN: RESTORE FAILED for $f" >&2; exit 3; }
        fi
    done
    # `git checkout <ref> -- <paths>` updates the INDEX as well as the working
    # tree, so restoring file contents alone leaves the revert staged. Without
    # this the script reports a clean restore while `git status` shows MM --
    # and the next commit would silently ship the reverted source.
    # shellcheck disable=SC2086
    git restore --staged -- ${PATHS} 2>/dev/null || git reset -q HEAD -- ${PATHS} 2>/dev/null || true
    rm -rf "${SNAPSHOT}"
    return "${rc}"
}
trap restore EXIT INT TERM

# shellcheck disable=SC2086
for f in ${PATHS}; do
    dest="$(snap_path "$f")"
    mkdir -p "$(dirname "${dest}")"
    cp "$f" "${dest}"
done

echo "=== Reverting ${PATHS} to ${FROM_REF} ==="
# shellcheck disable=SC2086
if ! git checkout "${FROM_REF}" -- ${PATHS}; then
    echo "PROVE-PIN FAIL: could not check out those paths at ${FROM_REF}" >&2
    exit 2
fi

echo "=== Running the pin against the reverted source (expecting FAILURE) ==="
set +e
# shellcheck disable=SC2086
"$PY_BIN" -m pytest ${TESTS} --import-mode=importlib --no-cov -q
reverted_rc=$?
set -e

# Explicit `if` rather than `assert` -- PYTHONOPTIMIZE=1 strips asserts.
if [[ "${reverted_rc}" -eq 0 ]]; then
    echo "" >&2
    echo "PROVE-PIN FAIL: the tests PASSED against the reverted source." >&2
    echo "  The pin does not detect the change it claims to protect -- it is" >&2
    echo "  decoration, not a gate. Tighten the assertions before merging." >&2
    exit 1
fi

echo "=== Pin failed as required (rc=${reverted_rc}); restoring ==="
# shellcheck disable=SC2086
for f in ${PATHS}; do cp "$(snap_path "$f")" "$f"; done

echo "=== Re-running against restored source (expecting PASS) ==="
# shellcheck disable=SC2086
if ! "$PY_BIN" -m pytest ${TESTS} --import-mode=importlib --no-cov -q; then
    echo "PROVE-PIN FAIL: tests do not pass against the restored source" >&2
    exit 1
fi

echo "PROVE-PIN OK: the pin fails without the change and passes with it"
