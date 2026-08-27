# Tasks — MLflow sqlite tracking default

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/ tests/ tools/ scripts/ && python -m ruff format --check src/ tests/ tools/ scripts/
python -m mypy src/ --strict --ignore-missing-imports
bash scripts/validations/F-034.sh
```

Task ordering is binding: each task lands green before the next starts.

**Phase 1 — Core default flip**

- [x] 1.1 `config/schema/telemetry.py`: `ExperimentLoggerConfig.tracking_uri`
      default → `sqlite:///mlflow.db`; `backend`'s and `tracking_uri`'s own
      docstrings updated to match.
- [x] 1.2 `pyproject.toml`: `[mlflow]` extra gains `sqlalchemy>=2.0,<3` and
      `alembic>=1.13,<2`, with a comment explaining why mlflow-skinny alone
      is insufficient for a `sqlite:///` tracking_uri.
- [x] 1.3 Empirically verified (not assumed) before 1.1/1.2 landed: installed
      mlflow 3.15.2 + the two new deps, confirmed
      `UnsupportedModelRegistryStoreURIException` without them and a full
      real-client run (`create_experiment`/`create_run`/`log_metric`/
      `set_terminated`) succeeding with them.

**Phase 2 — CI coverage**

- [x] 2.1 `.github/workflows/ci.yml`: new advisory `mlflow-extras` job
      mirroring `onnx-world-model-extras`'s exact shape (matrix, checkout/
      setup-python pins, `continue-on-error: true`), naming explicit test
      paths (`test_mlflow_logger.py`, `test_factory_observability.py`,
      `test_offline_rl_observability.py`,
      `test_pipeline_orchestrator_observability.py`) rather than a bare
      directory, per that job's own convention.
- [x] 2.2 `.github/advisory_stages.yaml`: `mlflow-extras` entry added —
      `scripts/check_advisory_promotions.py` would otherwise WARN
      "untracked advisory stage" the moment this job lands.
- [x] 2.3 `CLAUDE.md` + `docs/claude/surfaces/ci-gates.md`: job counts (16→17
      total, 5→6 advisory) and stage lists updated in the same commit —
      both were accurate before this change and would have gone stale the
      moment it landed otherwise.

**Phase 3 — Tests**

- [x] 3.1 `tests/regression/test_f034_mlflow_sqlite_aqa.py` — Type A pins
      (default value, `[mlflow]` extra's parsed requirement names) proven
      via `scripts/prove_pin_fails.sh`-equivalent manual revert/red/
      restore/green (schema default reverted to `file:./mlruns` and
      `pyproject.toml`'s extra reverted to `mlflow-skinny` only; both
      independently confirmed to fail their respective pin, then restored
      byte-identical).
- [x] 3.1a *(added during review)* Type B pins (`_resolve_tracking_uri`
      passthrough for `sqlite:`/`http:`, still-correct `file:` resolution)
      moved to `tests/unit/factory/test_factory_observability.py` — a
      test-engineer review round caught these as behavioural, not
      schema-property, pins, per `.claude/skills/test-tier-mirror/SKILL.md`'s
      own placement rule ("AQA is for schema properties, not behaviour").
      Placed directly (bypassing `build_experiment_logger`/mlflow
      construction) so they need no `pytest.importorskip("mlflow")` and
      always run, complementing the existing
      `test_relative_file_uri_is_resolved_to_absolute` (which proves the
      same `file:` resolution through the full factory path) rather than
      duplicating it.
- [x] 3.2 `tests/regression/test_f034_mlflow_sqlite_backwards_compat.py` —
      Type C: every `config/*.yaml` overlay scanned for any
      `observability:`/`experiment_logger:` reference at all (none found),
      mirroring F-029's twin-overlay precedent but for a stronger
      zero-shipped-configs-affected case.
- [x] 3.3 Re-ran the full 88-test mlflow-touching surface (with `[mlflow]`
      installed) against both the old and new `tracking_uri` default to
      confirm no regression either way.

**Phase 4 — Docs**

- [x] 4.1 `docs/architecture/c4-experiment-logger.md`: tracking-store
      diagram label + operator-usage YAML sample updated to the new
      default.
- [x] 4.2 `docs/runbooks/mlflow-local-ui.md`: intro, "Viewing runs" commands,
      "Common pitfalls" (concurrent-writer semantics genuinely differ
      between sqlite and the file backend — not just a find/replace),
      "Enabling on the rover", and the closing "Required extras" paragraph
      (which incorrectly claimed "no ... SQLAlchemy" before this bundle)
      all updated.
- [x] 4.3 `NEXT_STEPS.md` item 0 removed (resolved) — see `design.md` D-5
      for the verification trail.

**Phase 5 — Catalog**

- [x] 5.1 F-034 entry in `features.yaml` (`status: "in_progress"`,
      `implemented_in: null` until the dedicated closeout PR).
- [x] 5.2 `scripts/validations/F-034.sh`.
- [x] 5.3 Register in `openspec/project.md`.

## Explicitly deferred (separate change, do not fold in)

- Threading `log_step_every_n` from config into the orchestrator's
  offline-RL phases — pre-existing follow-up named in
  `docs/architecture/c4-experiment-logger.md`, unrelated to this bundle.
- Promoting `mlflow-extras` from advisory to blocking — gated on its own
  7-consecutive-green-run count, per `.github/advisory_stages.yaml`'s entry.
