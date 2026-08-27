# Design — MLflow sqlite tracking default

## D-1. Why sqlite over the alternatives

Three options were on the table for resolving mlflow 3.x's file-store
rejection:

1. **Keep `file:./mlruns` + rely on `MLFLOW_ALLOW_FILE_STORE=true`.** Already
   the de facto state (`mlflow_logger.py` sets this unconditionally since
   PR #198). Rejected as the long-term default: it's an escape hatch mlflow
   itself frames as a stopgap, not a guarantee — a future mlflow major could
   remove it, and the file backend is explicitly not safe for concurrent
   writers regardless of the env var.
2. **`sqlite:///mlflow.db`.** mlflow's own documented recommended local
   backend. Handles concurrent access far better than the file store
   (serializes at the database-file level rather than having no safety
   story at all), needs no special env var, and is a drop-in local
   replacement — no server process, no extra infrastructure.
3. **A remote tracking server (`http://host:port`).** Out of scope: this
   bundle is about fixing the *local default*, not standing up
   infrastructure. Remote tracking stays available as an explicit operator
   override, unchanged by this bundle.

Chose (2). It is mlflow's own recommended local backend, requires no new
infrastructure, and removes the codebase's dependency on an env-var
workaround for a rejection that could tighten in a future mlflow release.

## D-2. Why `sqlalchemy` + `alembic` specifically, and why both

Confirmed by direct construction test before writing this bundle:
`MlflowClient(tracking_uri="sqlite:///...")` with only `mlflow-skinny`
installed raises `UnsupportedModelRegistryStoreURIException` immediately —
`sqlite` is not in the skinny client's own supported-URI-scheme list without
its optional sqlalchemy-backed store implementation present. Installing
`sqlalchemy>=2.0,<3` alone was insufficient in the same test; `alembic` is
needed too (mlflow's sqlite store runs schema migrations via Alembic on
first connect — confirmed by the `mlflow.store.db.utils: Creating initial
MLflow database tables` / `Updating database tables` log lines emitted
during the successful run). Version bounds mirror this repo's existing
convention for exactly-pinned major-version-safe ranges (cf. the `mcp`
extra's `<2` bound, `mlflow-skinny`'s own `<4` bound) rather than leaving
either dependency unbounded.

## D-3. Why the schema default flips rather than only fixing the doc/runbook

Considered leaving `tracking_uri`'s default alone and only updating the
runbook to *recommend* an explicit `sqlite:///` override. Rejected: that
approach leaves every operator who doesn't read the runbook first on the
legacy `file:` backend, still dependent on the env-var escape hatch this
bundle is trying to retire. A schema-default change is the only form of
this fix where "do nothing extra" resolves to the safer state.

## D-4. Why `_resolve_tracking_uri` was widened to cover `sqlite:///`

**This section reverses an earlier revision of this bundle**, which left the
function untouched and presented that as a win. It was not: the function's
own docstring promises the pin exists "so the resolved path survives
`chdir()` inside trainers", and the new default — a *relative*
`sqlite:///mlflow.db` — was then the one configuration where that contract
did not apply. Shipping the retired default as the only pinned one, and the
new default as unpinned, was backwards.

The function now pins both local-store schemes and matches scheme prefixes
case-insensitively (RFC 3986 makes schemes case-insensitive and mlflow
lowercases via `urlparse`, so a configured `FILE:`/`SQLITE:` still selects
the local store downstream — matching case-sensitively would silently skip
the pin for exactly those URIs). In-memory sqlite URIs (`sqlite://`,
`sqlite:///:memory:`) and remote URIs are excluded: they carry no filesystem
path, and resolving `:memory:` would corrupt it into a bogus relative file.

**Scope, stated precisely because the obvious stronger claim is false and
was verified false rather than assumed.** Pinning does *not* make the
default CWD-independent:

- Two processes launched from different directories still get two different
  databases — each pins against its own CWD. Verified with two subprocesses.
- An in-process `chdir()` is invisible either way, because mlflow caches its
  store per URI *string* per process. Verified directly: after a `chdir`, a
  freshly-constructed client still resolved to the original database.

What pinning genuinely buys is that the effective path is **absolute and
therefore reportable** — which is exactly what the new
`experiment_logger_tracking_uri_resolved` event uses to tell an operator
which database a given run landed in. Without it, two processes writing to
two different databases both log the identical relative string. The
operator-facing cure for the split itself is an absolute `tracking_uri`,
which `docs/runbooks/mlflow-local-ui.md` now recommends explicitly.

This distinction was not academic: the first version of the accompanying
test asserted the false stronger claim ("a chdir cannot strand it") and
**passed with the fix reverted** — a tautology caught only by running the
revert. It was rewritten to assert the narrow, true property, and re-proven
red without the pin.

## D-5. Why `NEXT_STEPS.md` item 0 was removed, not reworded, and how that was verified

Two independent staleness findings, both verified against `git log` before
acting on either:

- The diagnosis's own headline claim ("29 failed, 17 errored... needs a
  decision before merging") no longer reproduces: this bundle's own
  pre-implementation verification ran the full 88-test mlflow-touching
  surface against the *current* tree (mlflow 3.15.2, old `file:./mlruns`
  default, the already-present `MLFLOW_ALLOW_FILE_STORE=true` env var) and
  got 88/88 passing, not 29 failed / 17 errored. `git log` confirms why:
  PR #198 (2026-08-20) added the env var three days after PR #145
  (2026-08-17, the version-bound widening the diagnosis references) — the
  diagnosis and the fix landed in the same rough window, but the doc was
  never updated to reflect that the fix actually worked.
- Independently, the item's own "Open dependabot PR #145 proposes
  widening..." framing was stale on its own terms: `git log -- pyproject.toml`
  shows PR #145 merged as commit `f8b6870` on 2026-08-17 — not open.

Given the diagnosed blocker no longer reproduces and this bundle completes
the underlying fix (D-1/D-3), the item describes a resolved concern, not an
open one. Removed rather than reworded, matching this repo's own convention
("Landed work moves to `CHANGELOG.md`") — `NEXT_STEPS.md` is forward-looking
priorities only, and CHANGELOG.md's F-034 entry carries the resolution
forward as the permanent record. The pre-existing CHANGELOG entry describing
the original diagnosis (a historical record of what was true *at the time
it was written*) is deliberately left untouched — this repo's convention is
append-only history there, not retroactive rewrites when later work
resolves what an earlier entry described.

## D-6. Why the CI job installs the full mlflow-touching surface, not just `test_mlflow_logger.py`

`onnx-world-model-extras`'s own precedent names two test paths, not one —
the pattern is "everything this extra's absence would silently leave
uncovered," not "the one file most obviously named after the extra."
Applied the same standard here: `tests/unit/factory/test_factory_observability.py`,
`tests/integration/test_offline_rl_observability.py`, and
`tests/integration/test_pipeline_orchestrator_observability.py` all
exercise `build_experiment_logger`/`MlflowExperimentLogger` end-to-end and
were silently uncovered by every prior CI job (none installs `[mlflow]`),
same as `test_mlflow_logger.py` itself.
