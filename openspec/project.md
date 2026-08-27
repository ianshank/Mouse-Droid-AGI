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
| `mouse-droid-gcp-observability-wiring` | implemented | F-032 | `69b9aa0` | `src/mousedroid/cloud/protocol.py` (`CloudFirestoreSyncProtocol`, widened `CloudLoggingSinkProtocol`), `src/mousedroid/factory.py` (`build_cloud_logging_sink`, `build_cloud_metrics_exporter`, `build_cloud_firestore_sync`, all five now SDK-detection-guarded via `module_available`), `src/mousedroid/orchestrator/orchestrator.py`, `src/mousedroid/main.py`, `scripts/validations/F-032.sh`, `features.yaml` (F-032) |
| `mouse-droid-autonomous-orchestrator-disposition` | implemented | F-031 | `5d40ee4` | `docs/architecture/ADR-016-autonomous-orchestrator-disposition.md`, `tests/regression/test_import_graph_freeze.py` (`test_no_active_module_imports_autonomous_orchestrator_at_module_scope`, `test_no_production_entrypoint_calls_the_autonomous_orchestrator_builder`), `scripts/validations/F-031.sh`, `features.yaml` (F-031) |
| `mouse-droid-doc-reconciliation` | implemented | F-030 | `14b2b34` | `tests/regression/test_doc_reconciliation_aqa.py`, `tests/regression/test_ci_gate_wiring_aqa.py` (`TestOrphanTierNarrativeAccuracy` + `_tracked_docs`), `tests/regression/test_claude_workforce_aqa.py` (`test_every_skill_directory_is_mentioned_in_the_index`, `test_every_agent_is_listed_in_the_subagent_skills_table`), `scripts/validations/F-030.sh`, `features.yaml` (F-030), `SKILLS.md`, `AGENTS.md` |
| `mouse-droid-alayaworld-memory-distill` | implemented | F-023 | `e730a0a` | `docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md`, `docs/superpowers/plans/2026-07-23-alayaworld-memory-distill.md`, `docs/architecture/ADR-015-bounded-context-latent-memory.md` |
| `mouse-droid-cloud-egress-default-off` | implemented | F-029 | `9bd3dc7` | `openspec/changes/mouse-droid-cloud-egress-default-off/peer-review.md`, `tests/regression/test_gcp_egress_defaults_aqa.py`, `scripts/validations/F-029.sh`, `features.yaml` (F-029) |
| `mouse-droid-ci-tier-completeness` | implemented | F-028 | `9bd3dc7` | `openspec/changes/mouse-droid-ci-tier-completeness/peer-review.md`, `tests/regression/test_ci_gate_wiring_aqa.py`, `scripts/validations/F-028.sh`, `features.yaml` (F-028) |
| `mouse-droid-claude-workforce` | implemented | F-024 | `6e89033` | `openspec/changes/mouse-droid-claude-workforce/peer-review.md`, `docs/runbooks/claude-workforce-hooks.md`, `docs/superpowers/plans/2026-07-03-claude-code-foundry.md` (WS-F7b coexistence), `features.yaml` (F-024) |
| `mouse-droid-nemoclaw-integration` | implemented | F-027 | `cb2d724` | `openspec/changes/mouse-droid-nemoclaw-integration/proposal.md`, `openspec/changes/mouse-droid-nemoclaw-integration/design.md`, `features.yaml` (F-027) |
