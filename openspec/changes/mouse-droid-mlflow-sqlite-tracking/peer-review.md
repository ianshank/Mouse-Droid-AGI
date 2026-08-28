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

## Round 2 — five-agent reconciliation pass (docs, config/build, tests, workforce, security)

A second review swept the branch against the default tip across every surface
the change touches. It found one security regression introduced *by this
bundle*, two live defects in the fix for it, and a set of doc/catalog claims
that had drifted. Findings that changed the code:

1. **P1, CONFIRMED, introduced here — a credentialed `tracking_uri` reached the
   logs in cleartext.** The hardening round added three URI-carrying log sites,
   two of them on failure paths where the URI had never been logged before;
   `tracking_uri` is a plain `str` and the structlog chain has no redaction
   processor. This had been filed in the appendix below as "dormant"; the
   round that added those sites is what made it live, so it is now fixed
   rather than deferred. New `mousedroid.logging.redaction`, applied at all
   four sites including the pre-existing `mlflow_logger_initialised`.
   Revert-proven: without it the integration test fails with the password
   visible in the captured events.
2. **CONFIRMED, found in the fix itself — `error=str(exc)` was still raw.**
   Verified against mlflow 3.15.2 rather than assumed:
   `UnsupportedModelRegistryStoreURIException` — the exception this code path
   actually catches — quotes the offending URI back with its password. (The
   same check refuted the accompanying claim about SQLAlchemy, which already
   masks its own URLs.) Added `redact_uris_in_text` for message bodies.
3. **CONFIRMED, found in the fix itself — a test wrote an 872 KB SQLite
   database into the repository root** and left it there, because it exercised
   the CWD-relative default without a `chdir`. Two tests in the same file also
   made real DNS calls and took minutes; both now stub the client.
4. **CONFIRMED — the validator missed `sqlite:///`.** It rejected `sqlite://`
   but not the three-slash empty-path form, which `PRAGMA database_list`
   confirms is equally in-memory — and which is exactly where an operator
   lands after following the error message's own "use three slashes" advice
   and omitting the filename. The factory already classified it as in-memory,
   so the schema was contradicting the factory.
5. **CONFIRMED — three of five `_resolve_tracking_uri` unit tests passed with
   the production change reverted.** Worst was the idempotence test, whose
   docstring described a four-slash-form check it never made: `f(f(x)) == f(x)`
   is trivially true of the identity function. Now asserts the pinned shape.
6. **CONFIRMED — `return stripped` was load-bearing but untested.** A single
   leading space makes the URI miss every scheme branch in
   `_resolve_tracking_uri` and pass through unpinned.
7. **CONFIRMED, empirically, by two independent agents — `mlruns/` is still an
   active artifact root under the sqlite default.** mlflow's SQLAlchemy store
   keeps its artifact root at `./mlruns` regardless of where the database
   lives. Both ignore files described it as "legacy", and the runbook's
   disk-cleanup and absolute-URI advice were wrong as a result.
8. **CONFIRMED — `features.yaml` still said `_resolve_tracking_uri` was
   unchanged**, contradicting its own `verification` list four lines below.
   `3f5be99` had fixed that contradiction in `CHANGELOG.md` and missed the
   machine-readable catalog.
9. Smaller confirmed fixes: three CLAUDE.md invariant miscitations (#9→#6,
   #3→#2) in the C4 doc and two source docstrings; the schema description
   still carrying the "mlflow 3.x rejects the file store" framing this
   bundle's own verification refuted; `F-034.sh` saying "three" node IDs while
   listing five; a `pip list | grep` CI step that could fail the job under
   `bash -e`; `.dockerignore` patterns being root-anchored where `.gitignore`'s
   match at any depth; a Windows-hostile `f"file:{tmp_path}/mlruns"`; and an
   `os.chdir` + try/finally that should be `monkeypatch.chdir`.

Also noted, not acted on: `tests/integration/test_e2e_5sec_run.py::
TestDeadlineAdherence::test_mean_tick_latency_within_budget` fails under
concurrent CPU load and passes in isolation (3.8 s). Unrelated to this change —
it is a 30 Hz tick-latency test and this bundle touches training-side logging —
but it is load-sensitive enough to be worth hardening separately.

## Appendix — open follow-ups (deliberately not folded in)

- `tracking_uri` remains a plain `str` rather than `SecretStr`. Deliberate:
  the field is usually *not* a secret (`sqlite:///mlflow.db`), and `SecretStr`
  would render every local path as `**********` in config dumps, making
  ordinary debugging worse. Redaction masks precisely the sensitive component
  instead. The one gap this leaves is `resolved_settings.json`, the artifact
  `pipeline_orchestrator` logs via `model_dump`, where a credentialed URI is
  still written unmasked — pre-existing, and best closed with a
  `field_serializer` rather than a type change.
- Six other sites log a URL the same way (`llm_gateway/openai_compatible.py`
  ×4, `comms/wifi_driver.py`, `factory.py`'s `base_url`). The new helper makes
  those a mechanical follow-up; not widened into this bundle.
- The blocking `test` CI job installs `[dev,telemetry,mcp]`, so
  `tests/integration/test_mlflow_sqlite_backend.py` skips there and runs only
  in the advisory `mlflow-extras` job. The new integration tests therefore
  cannot fail the build. Closing this means either installing `[mlflow]` in
  the blocking job or promoting `mlflow-extras` off advisory — a CI-cost
  decision for the operator, not a unilateral one.
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
