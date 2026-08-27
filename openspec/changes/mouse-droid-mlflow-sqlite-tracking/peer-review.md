# Peer review — MLflow sqlite tracking default

Six independent agent passes (peer-review, config-guardian, logging/debuggability,
test-engineering, doc-reconciliation, automation-opportunities) plus a Copilot
review round on PR #210. The peer-review pass returned **REQUEST_CHANGES** with
three independently-sufficient findings; all are resolved below.

## Verdict table

| Claim as originally written | Verdict |
|---|---|
| mlflow 3.x blocks the file store, so this unblocks a broken path | **REFUTED by our own verification** — 88/88 green on mlflow 3.15.2 against `file:./mlruns`, because `mlflow_logger.py` already sets `MLFLOW_ALLOW_FILE_STORE`. Reframed as completing mlflow's recommended fix, not an emergency. |
| `sqlalchemy` + `alembic` are required for a `sqlite:///` URI | **CONFIRMED** — `mlflow-skinny` alone raises `UnsupportedModelRegistryStoreURIException`; full run lifecycle succeeds with them. |
| No shipped config is affected | **CONFIRMED but incomplete** — true for every config *file* (27 scanned, widened from 17). **The documented env-var opt-in IS affected**; see below. |
| Residual risk is limited to `mlflow-skinny` 2.x pins | **REFUTED** — both halves wrong. Corrected in `proposal.md` and `CHANGELOG.md`. |
| `_resolve_tracking_uri` needed no code change | **REFUTED** — leaving it made the new default the *only* unpinned local store, contradicting the function's own chdir contract. Widened; see D-4. |
| The `mlflow-extras` job verifies the sqlite stack | **REFUTED** — it installed `sqlalchemy`/`alembic` and never opened a sqlite store. Closed with a real integration test. |

## Corrected-design map

| Original intent | Corrected |
|---|---|
| Pin `_resolve_tracking_uri`'s sqlite *passthrough* as correct | Inverted: sqlite is now **pinned** like `file:`, and matched case-insensitively (RFC 3986; mlflow lowercases via `urlparse`). |
| Test: "a chdir cannot strand the database" | **Rejected as tautological — it passed with the fix reverted.** Verified directly that mlflow caches its store per URI string per process, and that two processes launched from different directories split regardless. Replaced with the narrow provable claim (the effective path is absolute and therefore *reportable*), re-proven red without the pin. |
| Claim the flip is "fully inert" | Narrowed to "inert for every shipped config *file*", with the env-var path pinned by its own test so the rosier claim cannot drift back. |
| `tracking_uri: str` accepts any string | Added a validator rejecting the two silent-black-hole values: empty/whitespace, and `sqlite://` (two slashes = SQLAlchemy in-memory, every run discarded at exit). `sqlite:///:memory:` stays allowed as the explicit spelling. |

## What survives review unchanged

- sqlite over keeping the `MLFLOW_ALLOW_FILE_STORE` escape hatch (D-1).
- Placing the default inline in `Field(...)` rather than `constants.py` —
  matches this repo's convention for single-site default paths.
- `mlflow-extras` mirroring `onnx-world-model-extras` byte-for-byte in shape,
  and starting advisory under the same green-run-count gate.
- The legacy `file:` backend remaining fully supported, pinned, and tested.

## Load-bearing pins any implementation must satisfy

1. `ExperimentLoggerConfig().tracking_uri == "sqlite:///mlflow.db"`.
2. The `[mlflow]` extra names `mlflow-skinny`, `sqlalchemy`, and `alembic`.
3. `_resolve_tracking_uri` pins relative `file:` **and** `sqlite:///` paths,
   is idempotent, leaves remote/in-memory URIs alone, and matches
   case-insensitively.
4. Empty/whitespace and `sqlite://` are rejected at schema validation.
5. A real SQLite store round-trips runs, params, metrics, and nested phase runs
   through `build_experiment_logger`.
6. No `config/**` file (27 scanned incl. `*.yml`, subdirs, `*.example`)
   references `observability:`/`experiment_logger`.
7. The env-var opt-in materializes the sqlite default — pinned deliberately,
   because it is the change's real blast radius.

## Defects found in this bundle's own work

1. **Tautological test** — the first chdir test passed with the fix reverted.
   Caught only by running the revert, which is why `prove-pin-fails` is applied
   to behavioural pins and not just Type-A ones.
2. **Cross-test pollution, pre-existing and latent** —
   `test_backend_mlflow_degrades_to_noop_when_extras_missing` used a raw
   `sys.modules.pop()` and evicted `mousedroid...mlflow_logger` alongside
   `mlflow`. The pop was permanent, and re-importing the wrapper rebound a
   *package attribute* that `monkeypatch.undo()` does not restore, so a later
   test's patch silently targeted a different module object. Invisible until
   the new integration test became the first to open a real store afterwards.
   Fixed with `monkeypatch.delitem` scoped to `mlflow*` only.
3. **Windows sqlite slash count** — the runbook's "4 slashes for an absolute
   path" is POSIX-only; Windows uses 3 plus the drive letter. Corrected.
4. **`.dockerignore`** did not receive the `mlflow.db*` entries `.gitignore` got.

## Appendix — open follow-ups (deliberately not folded in)

- `tracking_uri` is a plain `str`, not `SecretStr`, and is logged raw. Dormant
  (the default carries no credentials) but a credentialed remote URI would be
  logged verbatim. Recorded in `features.yaml`'s F-034 `notes`.
- `MLFLOW_ALLOW_FILE_STORE` is still set unconditionally at module import in
  `mlflow_logger.py` — a hidden config source outside the Pydantic schema.
  Retiring it is a separate change.
- `_resolve_tracking_uri` mishandles RFC-8089 authority-form `file://` URIs
  (`file://./mlruns` → filesystem root). Pre-existing, unrelated to the sqlite
  default, and out of this bundle's scope.
- The `security` CI job audits only the base install, so `[mlflow]` (and every
  other extra) escapes `pip-audit`. Pre-existing.
- `.github/advisory_stages.yaml` states a "7-consecutive-green-run" gate that
  `check_advisory_promotions.py` does not implement (it only checks calendar
  days). Pre-existing across all advisory entries.
