#!/usr/bin/env bash
# =============================================================================
# MouseDroid — rover WIP guard (F-051, campaign step B1)
# =============================================================================
# Refuse to let a destructive sync run over uncommitted rover-local work, and
# preserve that work two ways before anything is deleted:
#
#   1. an ARCHIVE taken off the rover (a whitespace-insensitive diff plus the
#      untracked files themselves), because a branch on a rover that loses its
#      microSD is not a backup, and
#   2. a ``rover/wip-<date>`` BRANCH commit, so the operator can keep working
#      from the rover's own git history.
#
# ``git clean`` is NEVER used here, and this script never deletes anything in
# the target tree. Its whole purpose is to stand in front of
# ``rsync -avz --delete``.
#
# Usage:
#   bash scripts/rover_wip_guard.sh inspect  <target-dir>
#   bash scripts/rover_wip_guard.sh preserve <target-dir> --archive <file>|-
#
# ``preserve --archive -`` writes a gzipped tar to STDOUT, which is how
# deploy_remote.sh streams the archive straight off the rover without it ever
# touching rover disk. Every human-readable message therefore goes to STDERR.
#
# Exit codes (deploy_remote.sh branches on these, so they are a contract):
#   0  clean — safe to sync (inspect), or preservation completed (preserve)
#   2  usage error
#   3  dirty — uncommitted rover-local work is present, preservation required
#   4  cannot determine — not a git work tree, or git refused to answer.
#      Treated as a refusal, never as "clean": the known root-ownership drift
#      on the rover makes `git status` fail, and reading that as "nothing to
#      preserve" is exactly how work gets deleted.
#   5  preservation failed
#
# Environment variables (all optional):
#   MOUSEDROID_ROVER_WIP_BRANCH  Full branch name (default: rover/wip-<UTC date>)
#   MOUSEDROID_ROVER_WIP_DATE    Date stamp for the default branch name
#                                (default: `date -u +%Y%m%d`)
# =============================================================================
# USAGE-END  (the marker `usage()` stops at — keep it directly below the header)
set -euo pipefail

EXIT_DIRTY=3
EXIT_UNKNOWN=4
EXIT_PRESERVE_FAILED=5

log()  { echo "[wip-guard] $*" >&2; }
fail() { echo "[wip-guard] ERROR: $*" >&2; }

usage() {
    sed -n '2,/^# USAGE-END/{ /^# USAGE-END/d; /^# ===/d; s/^# \{0,1\}//p; }' "$0" >&2
}

# ---------------------------------------------------------------------------
# Resolve the git work tree that encloses the sync target
# ---------------------------------------------------------------------------
# The rsync destination is not necessarily the checkout root (the rover syncs
# into a subdirectory of the bind-mounted checkout), so walk up to the
# enclosing work tree rather than assuming the two are the same path.
resolve_toplevel() {
    local target="$1"
    if [[ ! -d "${target}" ]]; then
        log "target does not exist yet: ${target} (nothing to preserve)"
        return 1
    fi
    local top
    if ! top="$(git -C "${target}" rev-parse --show-toplevel 2>&1)"; then
        fail "not a git work tree, or git refused to answer, for ${target}"
        fail "git said: ${top}"
        fail "REFUSING: cannot prove the target holds no uncommitted work."
        fail "Fix ownership/permissions first (see the campaign plan's B2 step)"
        fail "or point the sync at a real checkout; do not work around this."
        return 1
    fi
    printf '%s\n' "${top}"
}

# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
cmd_inspect() {
    local target="$1"
    local top
    if ! top="$(resolve_toplevel "${target}")"; then
        # A missing directory is genuinely nothing to preserve; an
        # unreadable/non-git one is a refusal. resolve_toplevel already said
        # which, so distinguish on existence here.
        if [[ ! -d "${target}" ]]; then
            return 0
        fi
        return "${EXIT_UNKNOWN}"
    fi

    local status
    if ! status="$(git -C "${top}" status --porcelain=v1 2>&1)"; then
        fail "git status failed in ${top}: ${status}"
        return "${EXIT_UNKNOWN}"
    fi

    if [[ -z "${status}" ]]; then
        log "target is clean: ${top}"
        return 0
    fi

    local n
    n="$(printf '%s\n' "${status}" | wc -l | tr -d '[:space:]')"
    fail "DIRTY TARGET: ${n} uncommitted change(s) under ${top}"
    printf '%s\n' "${status}" | sed 's/^/    /' >&2
    fail "rsync --delete would destroy this. Preserve it first."
    return "${EXIT_DIRTY}"
}

# ---------------------------------------------------------------------------
# preserve
# ---------------------------------------------------------------------------
# Archive FIRST, mutate second. The archive is the artifact that survives the
# rover; the branch is a convenience on top of it.
build_archive() {
    local top="$1"
    local dest="$2"
    local staging
    staging="$(mktemp -d)"
    # shellcheck disable=SC2064  # expand now: $staging must be captured here
    trap "rm -rf -- '${staging}'" RETURN

    git -C "${top}" status --porcelain=v1 > "${staging}/status.txt"
    {
        printf 'toplevel=%s\n' "${top}"
        printf 'branch=%s\n' "$(git -C "${top}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
        printf 'head=%s\n' "$(git -C "${top}" rev-parse HEAD 2>/dev/null || echo 'unborn')"
        printf 'archived_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } > "${staging}/head.txt"

    # Whitespace-insensitive, per the campaign plan's B1 step: rover-local
    # edits routinely differ from the PC checkout in line endings and
    # indentation, and a diff dominated by that noise is one nobody reads.
    if git -C "${top}" rev-parse --verify --quiet HEAD >/dev/null; then
        git -C "${top}" diff --ignore-all-space HEAD \
            > "${staging}/diff-ignore-whitespace.patch"
    else
        : > "${staging}/diff-ignore-whitespace.patch"
    fi

    # Untracked files are the half a diff cannot carry and the half
    # rsync --delete removes, so archive their contents, not just their names.
    git -C "${top}" ls-files --others --exclude-standard -z > "${staging}/untracked.z"
    git -C "${top}" ls-files --others --exclude-standard > "${staging}/untracked.txt"
    if [[ -s "${staging}/untracked.z" ]]; then
        tar --null -C "${top}" -T "${staging}/untracked.z" -cf "${staging}/untracked.tar"
    else
        tar -C "${staging}" -cf "${staging}/untracked.tar" -T /dev/null
    fi
    rm -f "${staging}/untracked.z"

    if [[ "${dest}" == "-" ]]; then
        tar -C "${staging}" -czf - .
    else
        tar -C "${staging}" -czf "${dest}" .
        log "archive written: ${dest}"
    fi
}

wip_branch_name() {
    local top="$1"
    if [[ -n "${MOUSEDROID_ROVER_WIP_BRANCH:-}" ]]; then
        printf '%s\n' "${MOUSEDROID_ROVER_WIP_BRANCH}"
        return 0
    fi
    local stamp
    stamp="${MOUSEDROID_ROVER_WIP_DATE:-$(date -u '+%Y%m%d')}"
    local name="rover/wip-${stamp}"
    # Never reuse an existing branch: a second sync on the same day must not
    # land on top of the first rescue.
    if git -C "${top}" rev-parse --verify --quiet "refs/heads/${name}" >/dev/null; then
        name="${name}-$(date -u '+%H%M%S')"
    fi
    printf '%s\n' "${name}"
}

cmd_preserve() {
    local target="$1"
    local dest="$2"
    local top
    if ! top="$(resolve_toplevel "${target}")"; then
        if [[ ! -d "${target}" ]]; then
            fail "nothing to preserve: ${target} does not exist"
            return "${EXIT_PRESERVE_FAILED}"
        fi
        return "${EXIT_UNKNOWN}"
    fi

    if ! build_archive "${top}" "${dest}"; then
        fail "archive step failed — NOT creating a branch, NOT proceeding"
        return "${EXIT_PRESERVE_FAILED}"
    fi

    local branch
    branch="$(wip_branch_name "${top}")"
    log "committing rover-local state to ${branch}"
    if ! git -C "${top}" checkout -b "${branch}" >&2; then
        fail "could not create ${branch} in ${top}"
        return "${EXIT_PRESERVE_FAILED}"
    fi
    if ! git -C "${top}" add -A >&2; then
        fail "git add failed in ${top}"
        return "${EXIT_PRESERVE_FAILED}"
    fi
    if ! git -C "${top}" commit \
            -m "rover WIP preserved before sync (${branch})" >&2; then
        fail "git commit failed in ${top}"
        return "${EXIT_PRESERVE_FAILED}"
    fi
    log "preserved on ${branch}; the working tree is now clean"
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local command="${1:-}"
    case "${command}" in
        inspect)
            if [[ $# -ne 2 ]]; then
                fail "inspect takes exactly one target directory"
                usage
                return 2
            fi
            cmd_inspect "$2"
            ;;
        preserve)
            if [[ $# -ne 4 || "$3" != "--archive" ]]; then
                fail "preserve takes <target-dir> --archive <file>|-"
                usage
                return 2
            fi
            cmd_preserve "$2" "$4"
            ;;
        --help|-h)
            usage
            return 0
            ;;
        *)
            fail "unknown command: ${command:-<none>}"
            usage
            return 2
            ;;
    esac
}

main "$@"
