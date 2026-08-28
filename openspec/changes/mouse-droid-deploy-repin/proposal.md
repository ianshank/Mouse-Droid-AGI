# Proposal — Tag-protect gate-critical pinned SHAs

- change_id: mouse-droid-deploy-repin
- project: mouse-droid
- status: in progress
- feature_id: F-035
- epic: Deploy hygiene
- owner: ianshank
- created: 2026-08-28
- basis_commit: 7a84a0e
- rev: A

## Why

`deployments/jetson-image.json` asserted, in its own `notes` field, that its
SHA "is kept reachable by the annotated tag `deployments/jetson-image-032942b`,
NOT by any branch."

Every clause of that is false today, verified directly:

- `git ls-remote --tags origin` returns **zero tags**. The named tag has never
  existed.
- `git merge-base --is-ancestor` confirms the SHA is **not** an ancestor of the
  default branch.
- `git branch -r --contains` returns **10 stale feature branches** — the SHA's
  only reachability is exactly the set `scripts/archive_stale_branches.sh`
  exists to delete.

So the documented protection is fiction, and the CI config-schema-compat gate
(which worktrees that SHA out) is one branch cleanup away from dying
repo-wide. The same exposure applies to every `implemented_in` pin the nightly
`validate.py --strict-git` resolves.

A documented protection that does not exist is worse than a known gap,
because nobody goes looking for it.

## What changes

- **`scripts/repin_tags.sh`** (new) — the remedy half of a pair whose
  diagnostic half (`.claude/skills/pin-reachability-audit/`) already shipped
  and whose deferral half (`archive_stale_branches.sh`'s carrier protection)
  works by *not* cleaning up. Dry-run by default; `--push` creates annotated
  tags on the remote for both pin families.
- **`tests/unit/scripts/test_repin_tags.py`** (new, 13 tests) — real bare-remote
  fixture mirroring `test_archive_stale_branches.py`, including the
  cross-script contract.
- **`scripts/validations/F-035.sh`** (new).
- **`.claude/skills/deploy-repin/SKILL.md`** (new) — the F-036 half that
  backtick-references `scripts/repin_tags.sh`, so it must land in the same PR
  or `validate_skill_commands.py` fails on a dead path reference.
- **`deployments/jetson-image.json`** — the false reachability claim replaced
  with what is actually true, plus how to make the original sentence true.
- `features.yaml` F-035, `openspec/project.md` row, `SKILLS.md` roster entry,
  `CHANGELOG.md`.

## What does NOT change

No tags are created by this PR. `--push` writes to a shared remote and is an
operator action requiring separate explicit confirmation; this change covers
building and dry-running only. The 19 tags the dry run currently plans (1
deploy pin + 18 feature pins) do not exist yet.

## Risk

Low. The script is additive and dry-run by default; nothing in CI invokes it.
The one genuinely irreversible operation (`--push`, since annotated tags are
not meant to be deleted) is gated behind an explicit flag and documented in
the skill as needing per-run confirmation.
