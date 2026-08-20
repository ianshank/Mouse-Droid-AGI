# OpenSpec directory — archival status

This directory holds **imported OpenSpec change documents** for traceability with
workflows that author changes in the [OpenSpec](https://openspec.dev) format. It is
**documentation-only**: no OpenSpec CLI/tooling is installed in this repository, and
nothing in CI validates or consumes this tree.

The repository's **living, authoritative** change mechanism is:

- `features.yaml` — the F-number feature catalog validated by the spec-driven
  harness (`HARNESS_SPEC.md`, `scripts/validate.py`, `.github/workflows/harness.yml`)
- `docs/superpowers/specs/` + `docs/superpowers/plans/` — dated design specs and
  workstream implementation plans
- `docs/architecture/ADR-*.md` — architecture decision records

When an archived change here disagrees with the repo-native artifacts, the
repo-native artifacts win. Each change under `changes/<change-id>/` cross-links its
authoritative counterparts.

## Changes

| change-id | status | F-number | authoritative artifacts |
|---|---|---|---|
| `mouse-droid-alayaworld-memory-distill` | in progress | F-023 | `docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md`, `docs/superpowers/plans/2026-07-23-alayaworld-memory-distill.md`, `docs/architecture/ADR-015-bounded-context-latent-memory.md` |
| `mouse-droid-claude-workforce` | implemented | F-024 | `openspec/changes/mouse-droid-claude-workforce/peer-review.md`, `docs/runbooks/claude-workforce-hooks.md`, `docs/superpowers/plans/2026-07-03-claude-code-foundry.md` (WS-F7b coexistence), `features.yaml` (F-024) |
| `mouse-droid-nemoclaw-integration` | in progress | F-027 | `openspec/changes/mouse-droid-nemoclaw-integration/proposal.md`, `openspec/changes/mouse-droid-nemoclaw-integration/design.md`, `features.yaml` (F-027) |
