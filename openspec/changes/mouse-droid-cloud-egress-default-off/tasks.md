# Tasks — Cloud egress defaults OFF

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/ tests/ tools/ && python -m ruff format --check src/ tests/ tools/
python -m mypy src/ --strict --ignore-missing-imports
bash scripts/validations/F-029.sh
```

Task ordering is binding: each task lands green before the next starts.
Deviations from task wording are recorded inline — declared, not silent.

**Phase 1 — Flip the defaults**

- [x] 1.1 `GCPLoggingConfig.enabled` → `False` with a rationale-bearing description.
- [x] 1.2 `GCPMonitoringConfig.enabled` → `False` with a rationale-bearing description.
- [x] 1.3 Confirm a partial block (`GCPConfig(project_id=...)`) opens no channel.

**Phase 2 — The regression pair**

- [x] 2.1 `tests/regression/test_gcp_egress_defaults_aqa.py` — FieldInfo defaults, description substance, the partial-block scenario, and the still-works opt-in.
- [x] 2.2 `tests/regression/test_gcp_egress_defaults_backwards_compat.py` — the twin overlay resolves unchanged, asserted through both the model and the raw YAML.
- [x] 2.3 Prove the pair can fail: revert both defaults to `True`, confirm red, restore.

**Phase 3 — Catalog**

- [x] 3.1 F-029 entry in `features.yaml`, validated against `features.schema.json`.
- [x] 3.2 `scripts/validations/F-029.sh`.
- [x] 3.3 Register in `openspec/project.md`.

## Explicitly deferred (separate changes, do not fold in)

- Wiring `CloudLoggingSink` / `CloudMetricsExporter` / `CloudFirestoreSync` — that
  is F-032, and it is gated on this change landing first.
- Authoring a CHARTER §3 carve-out for cloud egress — the operator elected to
  treat it as covered by the existing "sensing / comms / telemetry" scope line.
  The deviation belongs in F-032's `notes:` field.
- Auditing the remaining `gcp_cloud.py` sub-configs (pubsub, storage, training,
  simulation) for the same default-ON asymmetry.
