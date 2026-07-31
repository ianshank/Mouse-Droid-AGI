# Contributing to MouseDroid

Thanks for your interest! MouseDroid is an edge-AI / robotics portfolio project
for the Star Wars MSE-6 "mouse droid" running on an NVIDIA Jetson Orin Nano. It
is governed by [`docs/CHARTER.md`](docs/CHARTER.md) (the project constitution) —
please skim it before a substantial change.

## Development setup

```bash
git clone https://github.com/ianshank/mouse-droid-agi
cd mouse-droid-agi
pip install -e ".[dev]"          # add ",arm" for the parked robot-arm platform
bash scripts/ci.sh               # full local gate
```

No hardware is required for development — run in mock mode:

```bash
MOUSEDROID_MOCK_HARDWARE=true mousedroid
```

## Quality gates (CI — `.github/workflows/ci.yml`; advisory stages tracked in `.github/advisory_stages.yaml`)

- **Lint / format:** `ruff==0.16.0` (`ruff check` + `ruff format --check` on `src/`, `tests/`, `tools/`; plus `ruff check scripts/`)
- **Types:** `mypy==2.3.0` `--strict`
- **Tests:** `pytest` with **≥90% line coverage** over `src/mousedroid`
  (`--cov-fail-under=90`), plus regression + e2e + smoke tiers; locally,
  `scripts/check_branch_coverage.py --min 90` additionally gates changed
  *lines* (git branch — not branch coverage, despite the name)
- **Performance tier:** advisory CI job (shared-runner latency is noisy; budgets are Jetson-calibrated)
- **Complexity:** `ruff C901`, `max-complexity = 15` (ADR-014)
- **Config compatibility:** `config-compat` — existing YAML overlays must load unchanged
- **Workflow lint:** `actionlint` on any `.github/workflows/*` change

## Architecture invariants (docs/CHARTER.md §4 — hold across every module)

1. Protocol-based DI — interfaces are `@runtime_checkable Protocol`; concrete types are imported only in factory builders.
2. `src/mousedroid/factory.py` is the single wiring point; every `build_*()` returns a protocol type.
3. No hardcoded values — every threshold/dimension/pin/path comes from Pydantic config (YAML or `MOUSEDROID_*__*` env), never source edits.
4. Structured logging only — `structlog` via `mousedroid.logging.setup.get_logger`; no `print()`.
5. Asyncio everywhere — blocking work goes through `asyncio.to_thread`.
6. `mypy --strict` passes; public functions carry type annotations + Google docstrings.
7. `torch.no_grad()` on every inference path.
8. `deque(maxlen=N)` for every sensor ring buffer (`N` from config).
9. **Backwards compatibility** — new config fields carry a Pydantic default; existing YAML must load unchanged (pinned by `tests/regression/test_pr*_backwards_compat.py`).
10. **Hot-loop purity** — the 30 Hz reactive loop (RSSM → MCTS → ESP32) stays deterministic, LLM-free, and training-free; all deliberation and learning run off-loop.
11. **No secrets or machine fingerprints in version control** — credentials use `SecretStr`; live per-host values live only in `/etc/mousedroid/docker.env`.

## Scope & invariant changes (docs/CHARTER.md §6)

A change that would weaken an invariant, expand scope, or open a new carve-out is a **human decision** —
surface it in the PR/issue for maintainer sign-off; do not implement it unilaterally.

## Workflow

1. Branch from the default branch.
2. Pick work with the spec harness: `python scripts/select_next.py` (honours the feature DAG in `features.yaml`; ADR-012).
3. Add tests in the matching tier (`tests/unit|integration|property|regression|e2e|performance|smoke`). A new config field needs a default **and** a regression/AQA test.
4. Update docs and add a `CHANGELOG.md` entry. `NEXT_STEPS.md` is budget-guarded by `tools/doc_hygiene.py` — link to it, don't restate landed work there.
5. Open a pull request — the body should describe *why*; the diff shows *what*.

## Agentic contributors

Claude Code, subagents, and MCP clients: read [`AGENTS.md`](AGENTS.md) and [`SKILLS.md`](SKILLS.md) —
the behavioural contracts for this repo.

## Adding media (demo clips, GIFs)

Never commit video or large binaries into git — it re-creates the history bloat this project already
removed (see [`docs/runbooks/history-purge.md`](docs/runbooks/history-purge.md)). Host clips as a GitHub
Release asset and embed the URL.

## Reporting security issues

Do **not** open a public issue for a vulnerability — see [`SECURITY.md`](SECURITY.md).
