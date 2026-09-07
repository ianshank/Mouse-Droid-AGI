# Proposal — Enumerate factory/orchestrator branch-coverage exemptions

- change_id: mouse-droid-coverage-allowlist
- project: mouse-droid
- status: active
- feature_id: F-042
- epic: Hygiene
- owner: ianshank
- created: 2026-09-07
- rev: A

## Why

`scripts/check_branch_coverage.py` `_ALLOWED_DIR_PREFIXES` exempted every
file under `src/mousedroid/factory/` and every `orchestrator/_*` path.
That is ~30% of `src/mousedroid` on the changed-line gate. A new factory
module inherited the exemption without review. Algorithmic factory code
(`on_device_learning.py`, `mcp_harness.py`, `_replay_batch_helpers.py`)
was treated as pure DI.

## What Changes

- Drop `src/mousedroid/factory/` and `src/mousedroid/orchestrator/_` prefixes.
- Enumerate the ADR-017 DI factory modules and orchestrator mixins in
  `_ALLOWED_FILES`.
- Keep gating the three algorithmic factory modules.
- Leave schema / telemetry / validation/runtime prefixes in place.

## Impact

No runtime behaviour. Future factory files fail the coverage gate until
they are classified (exempt vs gated). The hardcoded-value script still
uses directory prefixes.

## Charter

No CHARTER §3 carve-out: CI gate honesty only. Coverage threshold stays 90%.

## Validation

`bash scripts/validations/F-042.sh`
