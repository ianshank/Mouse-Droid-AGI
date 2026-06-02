# AGENTS.md — Behavioural contract for agentic workers

> Companion to `CLAUDE.md`. `CLAUDE.md` is the *project* surface — what
> MouseDroidAGI is + how its code is laid out. `AGENTS.md` is the *worker*
> surface — what an agent (Claude Code, a subagent, an MCP client) MUST do
> when touching this repo.

## Top-level invariants (never weaken these)

1. **Factory-first DI.** Concrete types are imported INSIDE
   `src/mousedroid/factory.py` builders only. Application code typed against
   `@runtime_checkable Protocol` interfaces. Never `from mousedroid.hardware.camera.jetson_csi import JetsonCSICamera`
   in business logic — call `factory.build_camera(cfg)`.
2. **Schema-driven configuration.** Every threshold, dimension, pin number,
   path, and tunable parameter comes from `src/mousedroid/config/schema.py`
   loaded from YAML in `config/`. Operators flip behaviour by editing YAML
   or setting `MOUSEDROID_*__*` env vars — never by editing source.
3. **Structured logging only.** `from mousedroid.logging.setup import get_logger`,
   then `_log.info("event_name", key=value)`. No `print()`. No f-string log
   messages — keys/values are structured fields.
4. **Asyncio everywhere.** All I/O-bound work is `async`. Never `threading.Thread`
   for application logic. Blocking syscalls go through `asyncio.to_thread`.
5. **Strict typing.** `mypy --strict` must pass on every PR. All public
   functions have type annotations + Google-style docstrings.
6. **Backwards compatibility.** New config fields MUST carry Pydantic
   `Field(default=..., description=...)`. Existing YAML must load unchanged
   after a `git pull`. The regression suite (`tests/regression/test_pr*_backwards_compat.py`)
   pins this — if you add a field, add a regression test for its default.
7. **`torch.no_grad()`** on every inference path. `deque(maxlen=N)` (with
   `N` from config) for every sensor ring buffer.
8. **Test-pyramid discipline.** Every behavioural change lands across the
   matching tiers — see "Test surface mirror" in `CLAUDE.md` and the PR #104
   files as the reference shape.

## Workflow expectations

### Before you start

- Read `CLAUDE.md` end-to-end for the project's architectural invariants.
- Skim the most recent `CHANGELOG.md` block — it's the project's
  context-recovery surface across sessions.
- Check `docs/architecture/` for C4 diagrams describing the area you're
  changing.
- If you're an agent acting on behalf of the user, prefer `TaskCreate` /
  `TaskUpdate` for any multi-step work — the user reviews progress via
  that surface.

### While you work

- One in-progress task at a time.
- Don't commit until tests + ruff + mypy pass locally.
- When you touch config schema, run the regression suite
  (`pytest tests/regression -q`) before committing — the default-value
  invariants there exist BECAUSE a previous PR shipped a silent default
  flip.
- Don't update `.git/config`. Don't `git push --force` on `main` or any
  shared branch.

### When you're done

- Commit with a Conventional-Commits prefix (`feat`, `fix`, `test`,
  `docs`, `refactor`, `chore`, `perf`) and the
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- Update `CHANGELOG.md` with a one-paragraph entry for any user-visible
  change.
- Update `docs/architecture/` if you changed an interface boundary, added
  a new external dependency, or moved a responsibility between modules.
- Mark your tasks complete; surface follow-ups via `TaskCreate`.

## Hardware-mode discipline

Two flags decide whether code talks to real hardware:

- **`Settings.mock_hardware: bool`** — global flag. Set by
  `MOUSEDROID_MOCK_HARDWARE` env var. The root `tests/conftest.py`
  forces it `true` for every test; `tests/hardware/conftest.py` reverses
  that override for `@pytest.mark.hardware`-tagged tests.
- **Per-subsystem `enabled: bool`** — finer-grained dev escape hatches
  (`ESP32Config.enabled`, `lidar.enabled`, etc.). Use these when you
  need *some* real hardware live but not all of it — e.g. live camera +
  LiDAR with no ESP32 plugged in (the PR #104 dashboard-verification
  posture).

**Rule of thumb:** if your code needs to behave differently when a piece
of hardware is missing, add a schema-driven `enabled` toggle — don't
sprinkle `try/except` around the driver calls.

## Subagent dispatch

When delegating to a subagent (security-auditor, code-quality, code-reviewer):

- The subagent does NOT inherit your conversation context. Brief it like a
  smart colleague who just walked in: include file paths, the worktree
  root, the python interpreter to use, and the specific question to
  answer.
- Ask for a length-capped report ("under 250 words"). Subagent reports go
  back to you as tool output — keep them concise so they don't blow your
  context budget.
- For research questions, use foreground dispatch. For independent
  long-running checks (security scan + lint), use `run_in_background=true`
  and continue work in parallel.

## Commit-message tone

- Imperative mood: "fix proxy hop-by-hop header strip" not "fixed".
- First line ≤ 72 chars, scope tag in parentheses: `feat(dashboard): ...`.
- Body explains the *why*, not the *what* — the diff shows the what.
- For multi-file PRs, bullet the surfaces touched + the verification
  done. The PR #104 commits are the reference style.

## Red flags — pause and check

- About to add a default value to an existing field? That's a behaviour
  change. Add a regression test.
- About to `os.getenv` something deep in a module? That's a hidden
  configuration source. Surface it as a Pydantic field on `Settings`.
- About to add a `print()` for "just this one debug line"? No — use
  `_log.debug` with structured kwargs.
- About to add `time.sleep()` in async code? That's `await asyncio.sleep()`.
- About to mutate a session-scoped pytest fixture in a test? Use
  `Settings.model_copy(deep=True)` instead.
- About to write `assert isinstance(x, ...)` in any code path that runs
  under the Jetson Docker entrypoint? Stop. `PYTHONOPTIMIZE=1` strips
  asserts. Use `if not isinstance(x, ...): raise RuntimeError(...)`.
- About to add a hardcoded default credential / token / dev-key
  fallback ("just for dev")? Stop. The security-auditor subagent will
  flag it. Require explicit env var or CLI arg; fail loudly if missing.
- About to call `Path.glob(...)` on a directory that might not exist?
  Add an `is_dir()` guard first — `Path.glob` raises `FileNotFoundError`
  on a missing root in Python 3.10/3.11, not an empty iterator.
- About to copy a YAML to `*.bak.<timestamp>` for safety before editing?
  The repo's `.gitignore` covers `*.bak.*` — but if you're editing on
  the Jetson, the file lives outside the repo tree. Either way: don't
  commit the backup.

If any of these apply, the PR will bounce on review — fix before
pushing.

## USB-C smoke gate — adding a new endpoint

When the rover gains a new USB-C device (a second LiDAR, an IMU bridge,
etc.), wire it through the discovery layer rather than hardcoding the
path:

1. Add a `USBCEndpointSpec(name="...", by_id_glob="...", required=...)`
   to the `usbc_discovery.required_endpoints` list in
   `config/jetson_production.yaml`.
2. If a driver needs to override its literal `serial_port` based on this
   endpoint (as `esp32` does), add a sibling helper to
   `factory.py:_resolve_esp32_serial_via_usbc_discovery` following the
   same two-condition contract (only override when discovery is enabled
   AND the literal does not exist on disk).
3. Add a unit test under `tests/unit/diagnostics/test_usbc.py` covering
   PRESENT / MISSING / WARN status transitions.
4. Add a hardware test under `tests/hardware/test_usbc_enumeration.py`
   (skip-gated by `tests/_jetson_hardware.is_jetson_host`) asserting
   the endpoint resolves on a live Jetson.
5. Add the regression assertion to
   `tests/unit/test_jetson_production_overlay.py` so the CI
   `usbc-config-gate` job (`.github/workflows/ci.yml`) catches any
   YAML-vs-driver drift before merge.

## LLM gateway — adding a new backend (PR #107 pattern)

When the rover gains a new LLM backend (a future tier-A reasoning model,
a different cloud provider, a self-hosted vLLM endpoint), wire it
through the existing dispatch + composite layers rather than forking
the factory:

1. Add a concrete gateway class in `src/mousedroid/llm_gateway/<name>_gateway.py`
   that structurally conforms to `LLMGatewayProtocol` (`is_ready`,
   `start`, `translate_mission`, `stop` — `is_degraded` is non-protocol
   but expected by the composite via `getattr`).
2. **Never raise on backend failure** — return a neutral `GoalVector`
   and flip `_degraded = True`. Reset `_degraded` on a successful
   round-trip to enable composite self-heal. Explicit
   `except asyncio.CancelledError: raise` BEFORE the broad
   `except Exception` so cooperative task cancellation propagates
   (e.g. orchestrator e-stop tearing down the mission task).
3. **Lazy SDK import** in `start()` (NEVER in `__init__`) so the
   factory + import-graph tests work without the optional dependency
   installed. Degrade rather than crash when the SDK is absent.
4. **`SecretStr` for credentials.** The API key is read via
   `cfg.api_key.get_secret_value()` ONCE at client construction;
   anywhere else the wrapper masks repr. Never `.get_secret_value()`
   inside a log call or exception message.
5. **Prompt-injection filter pre-egress** for any cloud-hitting
   backend — the rover sends operator NL commands to a third-party
   service, so `RegexInjectionFilter.sanitize()` MUST fire BEFORE the
   API call.
6. Extend `LLMConfig.backend` Literal in `src/mousedroid/config/schema.py`
   with the new backend name; do NOT touch `fallback_backend` Literal
   unless the new backend is local (only local backends are valid as
   failover targets — cloud-to-cloud failover defeats the off-network
   autonomy invariant).
7. Wire dispatch in `_build_single_llm_gateway` in `factory.py`. Log
   the `llm_gateway_built` event with `backend=<name>` for triage.
8. Add unit tests under `tests/unit/llm_gateway/test_<name>_gateway.py`
   (PR #107's `test_anthropic_gateway.py` is the reference shape:
   end-to-end with a faked SDK via `sdk=` test seam — no network).
9. Add a regression test in `tests/unit/config/test_llm_config_<name>.py`
   pinning the Literal extension default + that existing YAML still
   loads byte-identical.
10. Add a factory-dispatch test that asserts the new backend resolves
    to the right class and that any operator-tunable cooldown / timing
    flows through to the composite (see
    `test_fallback_retry_cooldown_s_threaded_through_to_composite`).
