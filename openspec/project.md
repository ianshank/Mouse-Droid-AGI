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

| change-id | status | F-number | landed | authoritative artifacts |
|---|---|---|---|---|
| `mouse-droid-alayaworld-memory-distill` | implemented | F-023 | `e730a0a` | `docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md`, `docs/superpowers/plans/2026-07-23-alayaworld-memory-distill.md`, `docs/architecture/ADR-015-bounded-context-latent-memory.md` |
| `mouse-droid-ci-tier-completeness` | implemented | F-028 | `pending` | `openspec/changes/mouse-droid-ci-tier-completeness/peer-review.md`, `tests/regression/test_ci_gate_wiring_aqa.py`, `scripts/validations/F-028.sh`, `features.yaml` (F-028) |
| `mouse-droid-claude-workforce` | implemented | F-024 | `6e89033` | `openspec/changes/mouse-droid-claude-workforce/peer-review.md`, `docs/runbooks/claude-workforce-hooks.md`, `docs/superpowers/plans/2026-07-03-claude-code-foundry.md` (WS-F7b coexistence), `features.yaml` (F-024) |
| `mouse-droid-nemoclaw-integration` | implemented | F-027 | `cb2d724` | `openspec/changes/mouse-droid-nemoclaw-integration/proposal.md`, `openspec/changes/mouse-droid-nemoclaw-integration/design.md`, `features.yaml` (F-027) |
