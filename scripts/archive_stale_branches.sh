#!/usr/bin/env bash
# =============================================================================
# MouseDroid -- Archive + delete stale remote branches
# =============================================================================
# DESTRUCTIVE (reversible). Archives each stale branch as a tag, then deletes
# the branch. The tag keeps every commit reachable and GC-safe, so any branch
# can be restored with:
#
#     git push origin "archive/<name>^{commit}:refs/heads/<name>"
#
# Why this exists: the repo carries ~88 remote branches, and 74 of them descend
# from a pre-history-rewrite root that shares NO ancestry with the current
# default branch. `git branch --merged` is therefore structurally meaningless
# for them -- it reports "not merged" regardless of whether the work shipped --
# so staleness is decided by lineage, not by merge status.
#
# What it protects, all derived rather than hardcoded:
#   1. The default branch and anything sharing its lineage (still reviewable
#      via a normal diff, so it gets human triage, not bulk deletion).
#   2. dependabot/** -- open dependency PRs.
#   3. archive/** -- already archived.
#   4. THE PIN CARRIERS: whichever branches currently keep the
#      deployments/jetson-image.json SHA reachable. The config-compat CI gate
#      worktrees that commit out; if the last branch containing it is deleted
#      it becomes unreachable and the gate dies repo-wide. Once that SHA is
#      reachable from a tag, this protection lifts on its own.
#
# Usage:
#   bash scripts/archive_stale_branches.sh           # DRY RUN (default): print the plan
#   bash scripts/archive_stale_branches.sh --push    # archive + delete for real
#   bash scripts/archive_stale_branches.sh --help
#
# Env overrides (no hardcoded values):
#   REMOTE        remote to operate on        (default: origin)
#   ARCHIVE_NS    tag namespace for archives  (default: archive)
#   KEEP_EXTRA    space-separated extra branch names to protect (default: empty)
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:-origin}"
ARCHIVE_NS="${ARCHIVE_NS:-archive}"
KEEP_EXTRA="${KEEP_EXTRA:-}"
PUSH=0

while [ $# -gt 0 ]; do
    case "$1" in
        --push) PUSH=1 ;;
        --help | -h)
            sed -n '2,38p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

echo "=== Fetching $REMOTE ==="
git fetch "$REMOTE" --prune --quiet

# `|| true` matters: under `set -e` a failing command substitution aborts the
# script at the assignment, so the explicit check below would never run. A fresh
# or pruned clone frequently has no local refs/remotes/<remote>/HEAD, hence the
# fallback to the remote's own answer.
default_branch=$(git symbolic-ref --quiet --short "refs/remotes/$REMOTE/HEAD" 2>/dev/null |
    sed "s|^$REMOTE/||" || true)
if [ -z "$default_branch" ]; then
    default_branch=$(git remote show "$REMOTE" 2>/dev/null |
        awk '/HEAD branch:/ {print $NF}' || true)
fi
if [ -z "$default_branch" ] || [ "$default_branch" = "(unknown)" ]; then
    echo "ERROR: cannot resolve $REMOTE's default branch; run 'git remote set-head $REMOTE -a'" >&2
    exit 1
fi
echo "default branch: $default_branch"

# The default branch's own root commit. A branch reachable from this root shares
# the post-rewrite lineage and is excluded from bulk deletion.
new_root=$(git rev-list --max-parents=0 "$REMOTE/$default_branch" | tail -1)
echo "default-branch root: $new_root"

# Branches that alone keep the deployed-image schema SHA reachable.
pinned_sha=""
if [ -f deployments/jetson-image.json ]; then
    pinned_sha=$(python3 -c 'import json;print(json.load(open("deployments/jetson-image.json"))["sha"])')
fi
pin_carriers=""
if [ -n "$pinned_sha" ] && git cat-file -e "$pinned_sha" 2>/dev/null; then
    # Only a tag that exists ON THE REMOTE protects the pin. A purely local tag
    # must NOT count: it makes the pin look safe here while the remote would
    # still lose the commit on deletion. So intersect the local "tags containing
    # this commit" set with the tag names the remote actually publishes.
    remote_tags=$(git ls-remote --tags "$REMOTE" 2>/dev/null |
        sed 's|.*refs/tags/||; s|\^{}$||' | sort -u || true)
    protecting_tag=""
    for t in $(git tag --contains "$pinned_sha" 2>/dev/null || true); do
        if printf '%s\n' "$remote_tags" | grep -qxF "$t"; then
            protecting_tag="$t"
            break
        fi
    done
    if [ -z "$protecting_tag" ]; then
        pin_carriers=$(git branch -r --contains "$pinned_sha" --format='%(refname:short)' |
            sed "s|^$REMOTE/||" | tr '\n' ' ')
        echo "pin $pinned_sha is NOT reachable from any REMOTE tag -- protecting its carriers"
    else
        echo "pin $pinned_sha is reachable from remote tag '$protecting_tag' -- carriers not needed"
    fi
fi

archive=""
skipped=0
for b in $(git branch -r --format='%(refname:short)' | sed "s|^$REMOTE/||" | grep -v '^HEAD$'); do
    case "$b" in
        "$default_branch" | dependabot/* | "$ARCHIVE_NS"/*) continue ;;
    esac
    case " $KEEP_EXTRA " in *" $b "*) continue ;; esac
    case " $pin_carriers " in
        *" $b "*)
            echo "  PROTECTED (holds the config-compat pin): $b"
            skipped=$((skipped + 1))
            continue
            ;;
    esac
    # Shares the default branch's lineage -> reviewable by normal diff -> triage, not bulk delete.
    if git merge-base --is-ancestor "$new_root" "$REMOTE/$b" 2>/dev/null; then
        continue
    fi
    archive="$archive $b"
done

count=$(echo "$archive" | wc -w)
echo
echo "=== $count stale branches to archive+delete ($skipped protected) ==="
for b in $archive; do
    printf '  %-55s %s\n' "$b" "$(git log -1 --format='%cs' "$REMOTE/$b")"
done

if [ "$count" -eq 0 ]; then
    echo "nothing to do."
    exit 0
fi

if [ "$PUSH" -ne 1 ]; then
    echo
    echo "DRY RUN -- nothing was changed. Re-run with --push to archive + delete."
    exit 0
fi

echo
echo "=== Archiving as tags ==="
for b in $archive; do
    git tag -f "$ARCHIVE_NS/$b" "$REMOTE/$b" >/dev/null
    echo "  tagged $ARCHIVE_NS/$b"
done
# shellcheck disable=SC2086
git push "$REMOTE" $(for b in $archive; do echo "refs/tags/$ARCHIVE_NS/$b"; done)

echo
echo "=== Deleting branches (archived above) ==="
# shellcheck disable=SC2086
git push "$REMOTE" --delete $archive

echo
echo "Done. Restore any branch with:"
echo "  git push $REMOTE \"$ARCHIVE_NS/<name>^{commit}:refs/heads/<name>\""
