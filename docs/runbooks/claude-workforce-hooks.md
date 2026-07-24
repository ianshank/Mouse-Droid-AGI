# Runbook — Claude Code workforce hooks

Operator guide for the edit-time governance shipped with F-024: what the hooks
do, how to configure them, how to turn them off, and how to debug them.

Related: `openspec/changes/mouse-droid-claude-workforce/` (the change bundle and
its peer review), `docs/runbooks/secret-scanning.md` (the CI-side secret gate).

## What is wired

`.claude/settings.json` declares three hooks. Hook blocks **merge** across
scopes, so these run alongside any personal hooks in `~/.claude/settings.json` —
adding them here does not shadow yours.

| Hook | Event | Blocks? | Purpose |
|---|---|---|---|
| `tools.claude_hooks.secret_scan` | PreToolUse (`Write`/`Edit`/`MultiEdit`/`NotebookEdit`) | Yes | Scans pending content with the repo's own scanner + `.gitleaks.toml` before it reaches disk |
| `tools.claude_hooks.freeze_gate` | PreToolUse (same matcher) | Yes | Denies edits to frozen capability paths until the gate feature lands |
| `tools.claude_hooks.post_edit_check` | PostToolUse | No | Runs `ruff`/`mypy` on the file just edited and reports findings |

Each command is `cd "$CLAUDE_PROJECT_DIR" && python3 -m tools.claude_hooks.<module>`.
The `cd` and the `-m` form are both load-bearing: running the module *file* by
path leaves the repository root off `sys.path`, and the package import fails on
every edit. `tests/regression/test_claude_workforce_aqa.py` pins this.

## Known limitations

* **`python3` on PATH.** The hook commands invoke `python3`. On a Windows shell
  without a `python3` alias the hook fails to start; Claude Code surfaces the
  error and continues (only a deny decision blocks), so edits are not stuck, but
  the gates are silently inactive. Adjust the commands in `.claude/settings.json`
  if you work on such a host.
* **Repository dependencies must be installed.** The hooks import `pydantic` and
  `pyyaml` — both core dependencies, so `pip install -e .` is enough. In a fresh
  clone without an install, each hook exits with an import error, which Claude
  Code reports as a non-blocking warning. The gates are inactive until the
  install completes; the AQA test in CI is the backstop.
* **Config sections `docs`, `worktree` and `evidence` are declared but not yet
  read** by any shipped code. They are the homes reserved for the later phases of
  the change bundle (skills, worktree flow, evidence audit) so thresholds land in
  one place rather than being scattered when those phases arrive.

## Configuration

Everything tunable lives in `.claude/workforce.yaml`, validated by
`tools/claude_hooks/config.py::WorkforceConfig` with `extra="forbid"` — a typo
such as `frozen_path` for `frozen_paths` fails at load instead of silently
disabling a gate. Deleting the file falls back to schema defaults.

Quick check that the config parses:

```bash
python3 -c "from tools.claude_hooks.config import load_config; print(load_config())"
```

### Turning a hook off

Prefer configuration over editing `settings.json`:

```yaml
secret_scan:
    enabled: false      # or freeze.enabled / post_edit.enabled
```

## The freeze gate

While `freeze.feature_key` (default `F-008`) has any status other than `done` in
`features.yaml`, edits to `freeze.frozen_paths` are denied with the rule quoted:
*hardware readiness preempts all in-flight software streams.* When the feature
flips to `done` the gate self-disables — no code change and no redeploy.

Failure posture is split on purpose:

* **Governance failure denies.** A missing, unreadable or malformed catalog, or
  an absent feature key, blocks the edit: the gate cannot prove the freeze
  lifted, and a broken governance input is itself a signal worth stopping on.
* **Environment failure allows.** An unexpected internal error allows the edit
  with a logged warning, because bricking every write in a session is worse than
  a missed gate.

### Overriding

```bash
MOUSEDROID_WORKFORCE_ALLOW_FROZEN=1 claude
```

The override is honoured and always logged (`freeze_gate_override_used`). Use it
for a deliberate, reviewed exception — not as a habit.

## The secret scan

Reuses `gitleaks` and the repository's regex-only `.gitleaks.toml` allowlist, so
there is exactly one secret policy. Pending content is written to a temporary
file outside the repository and scanned in `--no-git` mode.

When the scanner is not installed (or times out), behaviour follows
`secret_scan.strict`:

* `false` (default) — warn and allow, mirroring the advisory CI job;
* `true` — deny, for operators who want the stricter stance.

If a scan fires on a documented placeholder, add that placeholder's literal
regex to `.gitleaks.toml`. **Never allowlist by path** — the incident this gate
exists to prevent started in documentation.

## Debugging

Hook logs go to **stderr**, never stdout: Claude Code parses a hook's stdout as
its decision payload, so a stray log line there would corrupt the decision.

Raise the log level:

```bash
export MOUSEDROID_WORKFORCE_DEBUG=1
```

Drive a hook by hand with a synthetic payload:

```bash
echo '{"tool_name":"Write","tool_input":{"file_path":"src/mousedroid/arm/x.py"}}' \
  | python3 -m tools.claude_hooks.freeze_gate
```

Empty stdout means "no objection" (an explicit `allow` would bypass your normal
permission prompt, so silence is the correct signal). A denial prints a JSON
`hookSpecificOutput` payload carrying the reason.

Structured events worth grepping: `freeze_gate_denied`,
`freeze_gate_self_disabled`, `freeze_gate_override_used`,
`freeze_gate_catalog_unusable`, `secret_scan_denied`,
`secret_scan_unavailable`, `post_edit_check_findings`.

## Local gates

```bash
# Config parses and validates
python3 -c "from tools.claude_hooks.config import load_config; load_config()"

# Hook package: types, tests, coverage (line gate + advisory branch)
MYPYPATH=. mypy tools/claude_hooks/ --strict --ignore-missing-imports --explicit-package-bases
pytest tests/unit/tools/claude_hooks -q -o addopts="" \
    --cov=tools/claude_hooks --cov-branch --cov-report=term-missing

# The PR gate over the whole .claude/ surface
pytest tests/regression/test_claude_workforce_aqa.py -q
```

All of these also run in `bash scripts/ci.sh`.
