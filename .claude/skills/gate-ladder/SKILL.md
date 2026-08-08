---
name: gate-ladder
description: Run the pre-push gate ladder in the order scripts/ci.sh runs it, with the correct extras installed so failures are real rather than phantom
status: active
---

# Gate Ladder

Run the local gates in the same order CI runs them, fail fast on the cheap ones,
and get a signal that matches what the PR will see.

Use this before pushing, and whenever a gate goes red and you need to know
whether the failure is yours.

## Step 0 — install the right extras first

```bash
make install     # = <resolved-python> -m pip install -e ".[dev,telemetry,mcp]"
```

Prefer the `make` target over a bare `pip`: it resolves the interpreter the
same way `scripts/ci.sh` does (`MOUSEDROID_PYTHON`, then the project venv,
then PATH), so the extras land in the environment the gates actually use.

This is not optional and it is not the same as `pip install -e ".[dev]"`.

The `test` job in `.github/workflows/ci.yml` installs `[dev,telemetry,mcp]`.
A bare `[dev]` environment is missing Pillow (`[telemetry]`) and the MCP SDK
(`[mcp]`), which produces dozens of failures and several `mypy` errors that do
not exist on CI and are not defects in your change. Chasing them is pure lost
time. If a gate is red, confirm the extras before reading the traceback.

The one-shot equivalent of everything below is `bash scripts/ci.sh`. Run the
ladder by hand when you want to fail fast; run the script when you want the
full signal including the stages this file summarises.

## Rung 1 — lint and format (seconds)

```bash
make lint format
# i.e. <resolved-python> -m ruff check src/ tests/ tools/ && ruff check scripts/
#      <resolved-python> -m ruff format --check src/ tests/ tools/
```

Invoke ruff through the resolved interpreter (`python -m ruff`), never bare
`ruff` — a stray global install drifts from the pinned version and produces
phantom pass/fail deltas.

`ruff` is version-pinned in `pyproject.toml` to match the workflow. A version
skew makes local lint disagree with CI in both directions — bump both literals
in the same change.

`tools/` is in scope for both commands. `scripts/` is checked but not
format-gated.

## Rung 2 — types (tens of seconds)

```bash
make typecheck
# i.e. the authoritative CI invocations, verbatim:
#   mypy src/ --strict --ignore-missing-imports
#   MYPYPATH=. mypy tools/claude_hooks/ --strict --ignore-missing-imports \
#       --explicit-package-bases
```

Use CI's exact scope (`src/`, not `src/mousedroid/`) and its exact flags. A
narrower scope or a missing `--ignore-missing-imports` makes local results
disagree with the PR in both directions.

The second invocation exists because `tools/` is a namespace package with no
`__init__.py`; `--explicit-package-bases` plus `MYPYPATH=.` is what makes
`tools.claude_hooks` resolve. `tools/` as a whole is not strict-clean — only the
hook package is held to that bar.

## Rung 3 — the cheap standalone validators (seconds)

```bash
python tools/validate_skill_commands.py
python scripts/check_settings_identity.py
python scripts/check_no_hardcoded_values.py
```

`check_no_hardcoded_values.py` needs a git diff base — it gates *changed* lines,
so it is local-only by design and has no CI equivalent.

## Rung 4 — tests, in ci.sh order

```bash
python -m pytest tests/unit tests/property tests/integration \
    -m "not hardware" --import-mode=importlib \
    --cov=src/mousedroid --cov-report=term-missing --cov-fail-under=90

python -m pytest tests/smoke -m "not hardware and not slow" --no-cov
python -m pytest tests/performance/ -m "not hardware"
python -m pytest tests/regression/ -m "not hardware"
python -m pytest tests/e2e/ -m "not hardware"
```

The `-m "not hardware"` filter matters: hardware-marked tests open real GPIO and
serial devices, and on a host that does not own the peripherals — or where a
running daemon already holds them — they fail for reasons unrelated to the diff.

The coverage threshold is repeated in `scripts/ci.sh`, `.github/workflows/ci.yml`
and `pyproject.toml`; a regression test pins them equal, so move all of them
together or don't move any.

## Rung 5 — the gates that need a diff base

```bash
python scripts/check_branch_coverage.py --min 90 \
    --tests tests/unit tests/property tests/integration
python scripts/validate.py --tier fast
```

`check_branch_coverage.py` gates *changed lines* despite its name — branch
coverage is not measured repo-wide. It and the hardcoded-value gate both need a
git diff base, which is why they are local-only.

## Rung 6 — governance coverage

```bash
python -m pytest tests/unit/tools/claude_hooks -o addopts="" \
    --cov=tools/claude_hooks --cov-branch --cov-report=term-missing
```

The repo-wide coverage gate measures `src/mousedroid` only and cannot see
`tools/`, so the hook package needs its own invocation or it ships untested. The
threshold comes from `coverage.tools_line_min` in `.claude/workforce.yaml` — read
it from the config, never hardcode it.

## Reading a failure

| Symptom | Almost always |
|---|---|
| Dozens of failures across unrelated modules | wrong extras — go back to step 0 |
| `ruff` disagrees with CI | pinned version skew between `pyproject.toml` and the workflow |
| Only `@pytest.mark.hardware` tests fail | missing `-m "not hardware"`, or a daemon holds the device |
| Coverage gate red on an untouched file | the changed-lines gate, not the repo gate — look at your diff |
| `config-compat` red | an overlay edit uses a field the pinned image's schema lacks |
