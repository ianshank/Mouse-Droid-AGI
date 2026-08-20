---
description: Run the dedicated workforce-tools coverage gate and report coverage delta
status: active
---

# Coverage Gate

Run the dedicated `--cov=tools/claude_hooks` coverage invocation, gated by the
line threshold from `.claude/workforce.yaml` (`coverage.tools_line_min`).

## Usage

```bash
# Full coverage gate (fails below threshold)
python -m pytest tests/unit/tools/ \
  --cov=tools/claude_hooks \
  --cov-fail-under=$(python -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('.claude/workforce.yaml').read_text())
print(cfg['coverage']['tools_line_min'])
") \
  --cov-report=term-missing

# Advisory branch report (never blocking per rev-B doctrine)
python -m pytest tests/unit/tools/ \
  --cov=tools/claude_hooks \
  --cov-branch \
  --cov-report=term-missing
```

## PR Delta Table

When reviewing a PR, report the coverage delta:

| File | Before | After | Delta |
|------|--------|-------|-------|
| (file) | (%) | (%) | (+/-) |

## Guardrails

- The `tools_line_min` threshold is read from `.claude/workforce.yaml`, never hardcoded
- Branch coverage is advisory only (`tools_branch_min_advisory: 0`)
- The repository-wide gate measures `src/mousedroid` only — this covers `tools/`
