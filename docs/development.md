# Developer Setup

```bash
git clone https://github.com/ianshank/mouse-droid-agi && cd mouse-droid-agi
pip install -e ".[dev]"          # add ",arm" for the parked robot-arm platform
MOUSEDROID_MOCK_HARDWARE=true mousedroid    # run with no hardware
```

## Local loop

- **Lint / format:** `ruff check src/ tests/ tools/ && ruff format --check src/ tests/ tools/`
- **Types:** `mypy src/ --strict --ignore-missing-imports` (plus
  `MYPYPATH=. mypy tools/claude_hooks/ --strict --ignore-missing-imports --explicit-package-bases`)
- **Edit-time hooks:** editing files in a Claude Code session runs the workforce
  hooks (secret scan + capability freeze gate). See
  `docs/runbooks/claude-workforce-hooks.md` if an edit is unexpectedly blocked.
- **Tests:** `pytest tests/` (or `bash scripts/ci.sh` for the full local gate)
- **Health check:** `mousedroid --health-check --config config/default.yaml`
- **Pick next work:** `python scripts/select_next.py`

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for the contribution process and quality gates, and
[CHARTER.md](CHARTER.md) for the architecture invariants. Full testing strategy: [testing.md](testing.md).
