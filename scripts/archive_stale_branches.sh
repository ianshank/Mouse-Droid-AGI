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
#   4. THE PIN CARRIERS: whichever branches currently keep a gate-critical SHA
#      reachable. Two sources, both derived:
#        a. deployments/jetson-image.json -- the config-compat CI gate worktrees
#           that commit out; if the last branch containing it is deleted the
#           gate dies repo-wide.
#        b. features.yaml -- every `implemented_in` pin, which the nightly
#           `validate.py --strict-git` resolves. A feature closed out on a
#           branch and then squash-merged leaves its pin on a commit that
#           exists nowhere else, so deleting the branch breaks provenance for
#           that feature. Sourcing only (a) was a real gap: the same failure,
#           one file over.
#      Once a SHA is reachable from a REMOTE tag, its protection lifts on its own.
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

# Every SHA a gate needs to stay reachable, collected from both pin files.
pinned_shas=""
if [ -f deployments/jetson-image.json ]; then
    pinned_shas=$(python3 -c 'import json;print(json.load(open("deployments/jetson-image.json"))["sha"])')
fi
if [ -f features.yaml ]; then
    # Extracted with sed rather than a YAML parse so this script keeps working
    # in a bare environment with no PyYAML -- a protection that silently stops
    # applying because an import failed is worse than no protection, since it
    # still prints a plan that looks complete. Verified to match a yaml.safe_load
    # of the same file exactly. Every pin is collected, not just `done` ones:
    # over-protecting a branch only skips a deletion, while under-protecting
    # loses a commit permanently.
    feature_pins=$(sed -n 's/.*implemented_in:[[:space:]]*"\?\([0-9a-f]\{40\}\)"\?.*/\1/p' features.yaml |
        sort -u | tr '\n' ' ')
    pinned_shas="$pinned_shas $feature_pins"
fi

# Only a tag that exists ON THE REMOTE protects a pin. A purely local tag must
# NOT count: it makes the pin look safe here while the remote would still lose
# the commit on deletion. So intersect the local "tags containing this commit"
# set with the tag names the remote actually publishes. Fetched once, outside
# the loop.
remote_tags=$(git ls-remote --tags "$REMOTE" 2>/dev/null |
    sed 's|.*refs/tags/||; s|\^{}$||' | sort -u || true)

pin_carriers=""
for pinned_sha in $pinned_shas; do
    git cat-file -e "$pinned_sha" 2>/dev/null || continue
    protecting_tag=""
    for t in $(git tag --contains "$pinned_sha" 2>/dev/null || true); do
        if printf '%s\n' "$remote_tags" | grep -qxF "$t"; then
            protecting_tag="$t"
            break
        fi
    done
    if [ -z "$protecting_tag" ]; then
        carriers=$(git branch -r --contains "$pinned_sha" --format='%(refname:short)' |
            sed "s|^$REMOTE/||" | grep -Fvx -e "HEAD" -e "$REMOTE" | tr '\n' ' ')
        pin_carriers="$pin_carriers $carriers"
        echo "pin $pinned_sha is NOT reachable from any REMOTE tag -- protecting its carriers"
    else
        echo "pin $pinned_sha is reachable from remote tag '$protecting_tag' -- carriers not needed"
    fi
done

# Every real remote branch name, with $REMOTE/ stripped -- excluding BOTH
# renderings of the origin/HEAD symref that `git branch -r --format=` can
# produce. `grep -v '^HEAD$'` alone is not enough: when refs/remotes/<remote>/HEAD
# is configured (the normal state after any plain `git clone`), git renders it
# via `%(refname:short)` as the BARE remote name ("origin"), not "origin/HEAD" --
# a documented git quirk, not a repo-specific one. Reproduced empirically: a
# throwaway clone with `--push` walked straight into
# `fatal: Failed to resolve 'origin/origin' as a valid ref` and died at exit
# 128 before deleting anything -- this script's entire purpose, unusable on
# any standard fresh clone. This clone happened not to have origin/HEAD set,
# which is why the bug shipped unnoticed in the round that "fixed" this file.
# `-F` matters too: without it, $REMOTE is a regex, not a literal string, so a
# remote name containing a metacharacter (e.g. "upstream.fork") would match
# and wrongly exclude an unrelated branch differing only at that position.
_remote_branches() {
    git branch -r --format='%(refname:short)' |
        sed "s|^$REMOTE/||" |
        grep -Fvx -e "HEAD" -e "$REMOTE"
}

archive=""
skipped=0
for b in $(_remote_branches); do
    case "$b" in
        "$default_branch" | dependabot/* | "$ARCHIVE_NS"/*) continue ;;
    esac
    case " $KEEP_EXTRA " in *" $b "*) continue ;; esac
    case " $pin_carriers " in
        *" $b "*)
            echo "  PROTECTED (holds a gate-critical pin): $b"
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
