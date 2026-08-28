# Tasks — mouse-droid-deploy-repin (F-035)

## Phase 1 — the script
- [x] 1.1 `scripts/repin_tags.sh`, mirroring `archive_stale_branches.sh`'s
      idiom: `set -euo pipefail`, `--push`/`--help`/unknown-arg-exit-2, env
      overrides (`REMOTE`, `DEPLOY_NS`, `FEATURE_NS`), `|| true` on the
      command substitutions that can legitimately return nothing.
- [x] 1.2 Both pin families derived, never hardcoded: `python3 -c` for the
      JSON pin, `sed` for `features.yaml` (no PyYAML dependency, matching the
      sibling's stated reason).
- [x] 1.3 `_is_full_sha` format gate applied to every extracted value before
      it reaches a tag name or git argument.
- [x] 1.4 Remote-tag intersection for coverage; local tags deliberately do not
      count.
- [x] 1.5 Annotated (`-a`) tags with a message, and per-ref quoted pushes in a
      loop rather than one unquoted expansion.

## Phase 2 — tests
- [x] 2.1 Bare-remote fixture mirroring `test_archive_stale_branches.py`.
- [x] 2.2 Dry-run asserts *both* the reported plan and that nothing changed.
- [x] 2.3 `--push` asserts the tag is annotated (`cat-file -t` == `tag`), not
      merely present.
- [x] 2.4 Idempotence, foreign-tag coverage, `REMOTE` override.
- [x] 2.5 Format-rejection cases assert the **format-check message**, not just
      a nonzero exit. The first version asserted only the exit code and passed
      with `_is_full_sha` deleted, because those values also fail the later
      resolve check — 5 of 6 security tests were vacuous. Re-proven: with the
      tightened assertions, 5 of 5 go red without the check.
- [x] 2.6 The branch-name case (`"sha": "main"`), which establishes its own
      premise by asserting `git cat-file -e` really does resolve it.
- [x] 2.7 Cross-script: `archive_stale_branches.sh` flips from protecting the
      carrier to "carriers not needed" after a `--push`.

## Phase 3 — catalog + docs
- [x] 3.1 `scripts/validations/F-035.sh` (also covers `--help` and unknown-arg,
      which no pytest case drives through the on-disk file).
- [x] 3.2 `features.yaml` F-035 entry.
- [x] 3.3 `openspec/project.md` row.
- [x] 3.4 `.claude/skills/deploy-repin/SKILL.md` + `SKILLS.md` roster row.
- [x] 3.5 `deployments/jetson-image.json` false reachability claim corrected.
- [x] 3.6 `CHANGELOG.md` entry.

## Explicitly deferred
- **Running `--push` for real.** Operator action; needs separate explicit
  confirmation. The 19 tags the dry run plans do not exist yet.
- **Retiring tags for superseded pins.** Accepted permanent litter: annotated
  tags are not meant to be deleted, and the object may outlive the pin.
  `archive_stale_branches.sh` has the same one-directional behaviour.
- **A CI job invoking the script.** Nothing in CI runs it; the dry run is an
  operator/agent tool. Wiring it into `harness.yml`'s nightly as a
  report-only sweep is a candidate for a future F-number, alongside the
  advisory-promotion sweep gap already recorded in F-034's peer review.
