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
#   1. reject anything it cannot restore, BEFORE touching the tree
#   2. snapshot the named paths (a failed snapshot aborts; nothing is reverted)
#   3. restore them from a git revision that predates the change
#   4. run the named tests -- they MUST fail with pytest's "tests failed" code
#   5. restore the snapshot unconditionally, even on interrupt
#   6. run the named tests again -- they MUST pass
#
# Usage:
#   bash scripts/prove_pin_fails.sh --from <git-ref> \
#        --paths "<file> [<file> ...]" --tests "<pytest target> [...]"
#
# Example -- prove the F-029 egress pin detects a reverted default:
#   bash scripts/prove_pin_fails.sh --from origin/main \
#     --paths "src/mousedroid/config/schema/gcp_cloud.py" \
#     --tests "tests/regression/test_gcp_egress_defaults_aqa.py"
#
# --paths takes FILES ONLY. See "Directories" below.
#
# Exit codes: 0 proof succeeded; 1 the pin did NOT fail (it is not load-bearing);
# 2 invocation error, or the reverted run produced no verdict (nothing was left
# reverted); 3 restore failed (working tree needs manual attention).
set -uo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# pytest's documented exit codes. Only TESTS_FAILED is evidence that the pin
# detected the revert: INTERNAL_ERROR / USAGE_ERROR / NO_TESTS_COLLECTED are
# all non-zero and all mean the run produced no verdict at all. Treating "any
# non-zero" as proof would let a collection error at the base ref masquerade as
# a load-bearing pin -- the precise illusion this tool exists to dispel.
readonly PYTEST_OK=0
readonly PYTEST_TESTS_FAILED=1

FROM_REF=""; PATHS=""; TESTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)  FROM_REF="${2:-}"; shift 2 ;;
    --paths) PATHS="${2:-}";    shift 2 ;;
    --tests) TESTS="${2:-}";    shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${FROM_REF}" || -z "${PATHS}" || -z "${TESTS}" ]]; then
  echo "usage: $0 --from <git-ref> --paths \"<files>\" --tests \"<pytest targets>\"" >&2
  exit 2
fi

if ! git rev-parse --verify --quiet "${FROM_REF}^{commit}" >/dev/null; then
  echo "PROVE-PIN FAIL: --from ref '${FROM_REF}' does not resolve" >&2
  exit 2
fi

# Directories: refused, deliberately.
#
# Snapshot/restore over a directory is not a copy problem, it is a set problem:
# `git checkout <ref> -- <dir>` can ADD files that exist at the base ref and not
# at HEAD, and restoring the snapshot over the top would leave those behind --
# a tree that matches neither revision. Refusing up front is honest; silently
# half-restoring is how a safe-restore tool loses your work.
#
# This check runs BEFORE anything is written or reverted, so a rejected
# invocation leaves the tree byte-identical.
# shellcheck disable=SC2086
for f in ${PATHS}; do
    if [[ -d "$f" ]]; then
        echo "PROVE-PIN FAIL: --paths takes files, not directories ('$f')." >&2
        echo "  Pass the individual files the pin depends on." >&2
        exit 2
    fi
    if [[ ! -f "$f" ]]; then
        echo "PROVE-PIN FAIL: --paths entry '$f' is not an existing file" >&2
        exit 2
    fi
done

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
# in a tool whose entire purpose is safe restore.
#
# Mirroring the relative path gives distinct files distinct slots. The map is
# not injective over argument *spellings* -- "a/x.py", "./a/x.py" and "/a/x.py"
# all land in the same slot -- but those spellings name the same file, so the
# collision is harmless. Leading "./" and "/" are stripped so an absolute path
# lands inside the snapshot rather than at the filesystem root.
#
# Escaping $SNAPSHOT is prevented by TWO things, and it is worth being exact
# about which does what: the substitution below rewrites "../" segments, while
# a bare ".." or a trailing ".." (which the substitution does NOT touch, since
# it has no following slash) is rejected earlier by the file-only precondition
# -- both are directories. Neither guard is sufficient alone.
snap_path() {
    local rel="$1"
    rel="${rel#./}"
    while [[ "${rel}" == /* ]]; do rel="${rel#/}"; done
    rel="${rel//..\//__up__/}"
    printf '%s/%s' "${SNAPSHOT}" "${rel}"
}

# Until the snapshot is complete there is nothing to restore -- only a temp dir
# to remove. The full restore trap replaces this one once every path is safely
# captured, so an abort during snapshotting can never leave a reverted tree.
trap 'rm -rf "${SNAPSHOT}"' EXIT INT TERM

# shellcheck disable=SC2086
for f in ${PATHS}; do
    dest="$(snap_path "$f")"
    mkdir -p "$(dirname "${dest}")" || { echo "PROVE-PIN FAIL: cannot create a snapshot slot for $f" >&2; exit 2; }
    # An unchecked cp here is how the directory case used to corrupt the tree:
    # the snapshot silently did not happen, the revert did, and restore found
    # nothing to put back.
    cp "$f" "${dest}" || { echo "PROVE-PIN FAIL: could not snapshot $f; nothing was reverted" >&2; exit 2; }
done

_RESTORED=0
restore() {
    if [[ "${_RESTORED}" -eq 1 ]]; then return 0; fi
    _RESTORED=1
    local snap
    # shellcheck disable=SC2086
    for f in ${PATHS}; do
        snap="$(snap_path "$f")"
        if [[ ! -f "${snap}" ]]; then
            # Unreachable unless the snapshot was tampered with mid-run. Loud,
            # because the alternative is leaving $f at the base ref while
            # reporting success -- the failure mode exit 3 exists to name.
            echo "PROVE-PIN: RESTORE FAILED -- no snapshot for $f; it may still be reverted" >&2
            exit 3
        fi
        cp "${snap}" "$f" || { echo "PROVE-PIN: RESTORE FAILED for $f" >&2; exit 3; }
    done
    # `git checkout <ref> -- <paths>` updates the INDEX as well as the working
    # tree, so restoring file contents alone leaves the revert staged. Without
    # this the script reports a clean restore while `git status` shows MM --
    # and the next commit would silently ship the reverted source.
    # shellcheck disable=SC2086
    git restore --staged -- ${PATHS} 2>/dev/null || git reset -q HEAD -- ${PATHS} 2>/dev/null || true
    rm -rf "${SNAPSHOT}"
}

_on_exit()   { local rc=$?; restore; return "${rc}"; }
# A signal handler that unwinds state must EXIT. Returning from it hands
# control back to the line after the interrupted command, so the script would
# carry on with its snapshot already deleted -- and a Ctrl-C during the
# reverted-source test run would leave the tree at the base ref.
_on_signal() { restore; exit 130; }
trap _on_exit EXIT
trap _on_signal INT TERM

echo "=== Reverting ${PATHS} to ${FROM_REF} ==="
# shellcheck disable=SC2086
if ! git checkout "${FROM_REF}" -- ${PATHS}; then
    echo "PROVE-PIN FAIL: could not check out those paths at ${FROM_REF}" >&2
    exit 2
fi

echo "=== Running the pin against the reverted source (expecting FAILURE) ==="
# shellcheck disable=SC2086
"$PY_BIN" -m pytest ${TESTS} --import-mode=importlib --no-cov -q
reverted_rc=$?

# Explicit `if` rather than `assert` -- PYTHONOPTIMIZE=1 strips asserts.
if [[ "${reverted_rc}" -eq "${PYTEST_OK}" ]]; then
    echo "" >&2
    echo "PROVE-PIN FAIL: the tests PASSED against the reverted source." >&2
    echo "  The pin does not detect the change it claims to protect -- it is" >&2
    echo "  decoration, not a gate. Tighten the assertions before merging." >&2
    exit 1
fi

if [[ "${reverted_rc}" -ne "${PYTEST_TESTS_FAILED}" ]]; then
    echo "" >&2
    echo "PROVE-PIN FAIL: the reverted run produced no verdict (pytest rc=${reverted_rc})." >&2
    echo "  Only rc=1 (tests failed) is evidence the pin fired. A collection" >&2
    echo "  error, usage error, or empty selection is non-zero for reasons that" >&2
    echo "  have nothing to do with the pin. The tree is being restored." >&2
    exit 2
fi

echo "=== Pin failed as required (rc=${reverted_rc}); restoring ==="
# shellcheck disable=SC2086
for f in ${PATHS}; do
    cp "$(snap_path "$f")" "$f" || { echo "PROVE-PIN: RESTORE FAILED for $f" >&2; exit 3; }
done

echo "=== Re-running against restored source (expecting PASS) ==="
# shellcheck disable=SC2086
if ! "$PY_BIN" -m pytest ${TESTS} --import-mode=importlib --no-cov -q; then
    echo "PROVE-PIN FAIL: tests do not pass against the restored source" >&2
    exit 1
fi

echo "PROVE-PIN OK: the pin fails without the change and passes with it"
