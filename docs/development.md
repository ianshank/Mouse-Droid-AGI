# Developer Setup

```bash
git clone https://github.com/ianshank/mouse-droid-agi && cd mouse-droid-agi
pip install -e ".[dev]"          # add ",arm" for the parked robot-arm platform
MOUSEDROID_MOCK_HARDWARE=true mousedroid    # run with no hardware
```

## Local loop

- **Lint / format:** `ruff check src/ tests/ && ruff format --check src/ tests/`
- **Types:** `mypy src/ --strict --ignore-missing-imports`
- **Tests:** `pytest tests/` (or `bash scripts/ci.sh` for the full local gate)
- **Health check:** `mousedroid --health-check --config config/default.yaml`
- **Pick next work:** `python scripts/select_next.py`

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for the contribution process and quality gates, and
[CHARTER.md](CHARTER.md) for the architecture invariants. Full testing strategy: [testing.md](testing.md).
