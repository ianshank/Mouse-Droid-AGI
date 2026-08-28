#!/usr/bin/env bash
# =============================================================================
# MouseDroid -- Create the annotated tags that keep gate-critical SHAs reachable
# =============================================================================
# The mirror image of scripts/archive_stale_branches.sh. That script REFUSES to
# delete a branch while it is the last thing keeping a pinned SHA reachable;
# this one removes the need for that refusal by tagging the SHA directly, so
# the branch becomes free to archive.
#
# Why this exists: deployments/jetson-image.json's own notes assert its SHA "is
# kept reachable by the annotated tag 'deployments/jetson-image-<short>', NOT by
# any branch." That was aspirational -- zero tags exist on the remote. The SHA
# is not an ancestor of the default branch either, so its only reachability is
# a handful of stale feature branches, which is exactly what the sibling script
# is built to delete. Until a real tag exists, the documented protection is
# fiction and the CI config-schema-compat gate is one branch cleanup away from
# dying repo-wide.
#
# Two pin families, both DERIVED (never hardcoded), matching the sources
# archive_stale_branches.sh already protects:
#   a. deployments/jetson-image.json -- the deploy pin the config-compat gate
#      worktrees out.        -> tag: $DEPLOY_NS-<sha>
#   b. features.yaml -- every `implemented_in` pin the nightly
#      `validate.py --strict-git` resolves.   -> tag: $FEATURE_NS/<sha>
#
# Tags are named by the FULL 40-char SHA, not per-feature: F-028 and F-029
# share a pin, so SHA-based naming means one tag covers both with no duplicate
# bookkeeping and no name to keep in sync when a feature is renamed.
#
# A SHA already reachable from ANY tag the remote publishes is left alone, even
# under a different name -- the goal is reachability, not a naming convention.
# Only a REMOTE tag counts: a purely local tag makes a pin look safe here while
# the remote would still lose the commit (the same rule archive_stale_branches.sh
# applies, for the same reason).
#
# Usage:
#   bash scripts/repin_tags.sh           # DRY RUN (default): print the plan
#   bash scripts/repin_tags.sh --push    # create + push the tags for real
#   bash scripts/repin_tags.sh --help
#
# Env overrides (no hardcoded values):
#   REMOTE       remote to operate on                  (default: origin)
#   DEPLOY_NS    tag prefix for the deploy-image pin   (default: deployments/jetson-image)
#   FEATURE_NS   tag namespace for features.yaml pins  (default: features)
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:-origin}"
DEPLOY_NS="${DEPLOY_NS:-deployments/jetson-image}"
FEATURE_NS="${FEATURE_NS:-features}"
PUSH=0

while [ $# -gt 0 ]; do
    case "$1" in
        --push) PUSH=1 ;;
        --help | -h)
            sed -n '2,45p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

# Every value extracted below is re-validated through this before it reaches a
# tag name or any git argument. The features.yaml extraction is already
# structurally constrained (its sed pattern matches 40 hex chars or nothing),
# but the deploy-pin extraction is NOT: python prints whatever the JSON value
# is -- a branch name, a number, a list, stray whitespace -- and that value
# would otherwise flow straight into `git tag <name> <committish>`.
# `git cat-file -e` is not a substitute: it accepts far more than 40-hex
# (branch names, HEAD, "main~3"), so a malformed pin could resolve to an
# entirely different commit and be tagged as though it were the pinned one.
_is_full_sha() {
    case "${1:-}" in
        *[!0-9a-f]*) return 1 ;;
    esac
    [ "${#1}" -eq 40 ]
}

echo "=== Fetching $REMOTE ==="
git fetch "$REMOTE" --tags --prune --quiet

# Collect the pins. Kept as two parallel lists rather than one, because the two
# families get different tag prefixes.
deploy_sha=""
if [ -f deployments/jetson-image.json ]; then
    # `|| true` so a malformed/absent key produces our own diagnostic below
    # rather than a bare python traceback and an abort at the assignment.
    deploy_sha=$(python3 -c 'import json;print(json.load(open("deployments/jetson-image.json"))["sha"])' 2>/dev/null || true)
    if [ -z "$deploy_sha" ]; then
        echo "ERROR: could not read [\"sha\"] from deployments/jetson-image.json" >&2
        exit 1
    fi
    if ! _is_full_sha "$deploy_sha"; then
        # Loud, not skipped: a deploy pin that is not a full SHA means the file
        # no longer says what every consumer assumes it says.
        echo "ERROR: deployments/jetson-image.json [\"sha\"] is not a 40-char lowercase hex SHA: '$deploy_sha'" >&2
        exit 1
    fi
fi

feature_pins=""
if [ -f features.yaml ]; then
    # sed rather than a YAML parse, matching archive_stale_branches.sh: this
    # script must keep working in a bare environment with no PyYAML, because a
    # protection that silently stops applying is worse than none -- it still
    # prints a plan that looks complete.
    feature_pins=$(sed -n 's/.*implemented_in:[[:space:]]*"\?\([0-9a-f]\{40\}\)"\?.*/\1/p' features.yaml |
        sort -u | tr '\n' ' ')
fi

# Only tags the REMOTE actually publishes protect anything. Fetched once.
# `|| true`: an empty tag list is the normal starting state here, not an error.
remote_tags=$(git ls-remote --tags "$REMOTE" 2>/dev/null |
    sed 's|.*refs/tags/||; s|\^{}$||' | sort -u || true)

# Returns 0 and echoes the covering tag name when $1 is reachable from some
# remote tag; returns 1 otherwise.
_covering_remote_tag() {
    local sha="$1" t
    for t in $(git tag --contains "$sha" 2>/dev/null || true); do
        if printf '%s\n' "$remote_tags" | grep -qxF "$t"; then
            printf '%s' "$t"
            return 0
        fi
    done
    return 1
}

to_create=""   # newline-separated "<tag>\t<sha>" pairs
covered=0
missing=0

_consider() {
    local sha="$1" tag="$2" label="$3" covering

    if ! _is_full_sha "$sha"; then
        echo "ERROR: $label pin is not a 40-char lowercase hex SHA: '$sha'" >&2
        exit 1
    fi
    if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
        # Nonzero exit rather than a silent skip: an unresolvable pin means
        # either the commit is already lost or the clone is too shallow to
        # tell -- both need a human, and a "0 tags to create" summary would
        # read as success.
        echo "ERROR: $label pin $sha does not resolve to a commit in this clone" >&2
        echo "       (fetch the branch that carries it, or unshallow, then re-run)" >&2
        missing=$((missing + 1))
        return 0
    fi
    if covering=$(_covering_remote_tag "$sha"); then
        printf '  already reachable from remote tag %-46s %s\n' "'$covering'" "$sha"
        covered=$((covered + 1))
        return 0
    fi
    to_create="${to_create}${tag}	${sha}
"
}

echo
echo "=== Pins ==="
if [ -n "$deploy_sha" ]; then
    _consider "$deploy_sha" "${DEPLOY_NS}-${deploy_sha}" "deployments/jetson-image.json"
fi
for pin in $feature_pins; do
    _consider "$pin" "${FEATURE_NS}/${pin}" "features.yaml"
done

if [ "$missing" -ne 0 ]; then
    echo >&2
    echo "FAILED: $missing pin(s) do not resolve; refusing to report a partial plan as success." >&2
    exit 1
fi

count=$(printf '%s' "$to_create" | grep -c . || true)
echo
echo "=== $count tag(s) to create ($covered pin(s) already covered) ==="
printf '%s' "$to_create" | while IFS='	' read -r tag sha; do
    [ -n "$tag" ] || continue
    printf '  %-60s -> %s\n' "$tag" "$sha"
done

if [ "$count" -eq 0 ]; then
    echo "nothing to do."
    exit 0
fi

if [ "$PUSH" -ne 1 ]; then
    echo
    echo "DRY RUN -- nothing was changed. Re-run with --push to create + push these tags."
    exit 0
fi

echo
echo "=== Creating annotated tags ==="
# Annotated (-a), not lightweight: an annotated tag is its own object with a
# message, so the reason a SHA is protected travels with the repo rather than
# living only in this script's scrollback.
printf '%s' "$to_create" | while IFS='	' read -r tag sha; do
    [ -n "$tag" ] || continue
    git tag -a -f "$tag" "$sha" \
        -m "Keep $sha reachable: pinned by a MouseDroid gate (see scripts/repin_tags.sh)." >/dev/null
    echo "  tagged $tag"
done

echo
echo "=== Pushing tags to $REMOTE ==="
# Quoted per-ref pushes in a loop rather than one unquoted $(...) expansion:
# tag names here are derived from file contents, and an unquoted expansion
# would word-split anything unexpected into separate refspecs.
printf '%s' "$to_create" | while IFS='	' read -r tag sha; do
    [ -n "$tag" ] || continue
    git push "$REMOTE" "refs/tags/$tag"
done

echo
echo "Done. Verify with:"
echo "  git ls-remote --tags $REMOTE"
echo "  bash scripts/archive_stale_branches.sh   # pins should now report 'carriers not needed'"
