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

Each command is `bash "$CLAUDE_PROJECT_DIR/tools/claude_hooks/run_hook.sh" -m
tools.claude_hooks.<module>`. The wrapper resolves an interpreter that can actually
import the hook package (probing `import tools.claude_hooks.config`) instead of
trusting `PATH` order, and chdirs to the project root so `-m` works. Set
`MOUSEDROID_PYTHON` to override it outright.
The `cd` and the `-m` form are both load-bearing: running the module *file* by
path leaves the repository root off `sys.path`, and the package import fails on
every edit. `tests/regression/test_claude_workforce_aqa.py` pins this.

## Known limitations

* **~~`python3` on PATH.~~ Fixed — and it was not hypothetical.** This bullet used
  to say the hook commands invoke `python3`, that the gates would then be
  "silently inactive", and that you should adjust `.claude/settings.json`
  yourself. That is exactly what happened, on Linux rather than Windows: `python3`
  resolved to the system interpreter, which has no `pydantic`, so every hook
  exited 1 — a non-blocking hook *error* — and the F-008 freeze gate and the
  edit-time secret scan were bypassed for months with no visible symptom.
  `tools/claude_hooks/run_hook.sh` now resolves an interpreter by *capability*
  (it probes `import tools.claude_hooks.config`) rather than by `PATH` order,
  preferring a project virtualenv in either layout (`.venv/Scripts/python.exe`,
  then `.venv/bin/python`) and honouring `MOUSEDROID_PYTHON` outright. Two
  regression pins in `tests/regression/test_claude_workforce_aqa.py` keep it that
  way: no wired command may name an interpreter from `PATH`, and the wrapper must
  actually resolve one that can import the hook package when `PATH` is hostile.
* **The no-capable-interpreter path still fails open, by decision.** In a fresh
  clone with no install, the wrapper prints
  `run_hook.sh: no interpreter could import tools.claude_hooks.config … gates
  will fail open` on stderr and then runs the incapable interpreter anyway, so the
  hook exits 1 and the edit proceeds ungated. Failing closed is not a one-liner —
  a PreToolUse deny is exit 0 plus a JSON payload, and exit 2 on a PostToolUse
  hook feeds stderr back to the model instead of blocking — so it needs the
  wrapper to know which event it serves. `pip install -e .` is the fix; the
  blocking `gitleaks` CI job and the F-024 review discipline are the backstop.
  A typo'd `MOUSEDROID_PYTHON` is the one case that now fails *loudly*: the
  wrapper checks it is executable and names the variable if it is not.
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
  | bash "$CLAUDE_PROJECT_DIR/tools/claude_hooks/run_hook.sh" -m tools.claude_hooks.freeze_gate
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

All of these also run in `bash scripts/ci.sh`, and in GitHub CI: the hooks
mypy, workforce coverage, and skill-validator commands via the `local-gates`
job, the AQA regression via the `test` job's regression step.
