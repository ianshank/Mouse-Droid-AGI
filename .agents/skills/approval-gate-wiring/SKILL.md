---
name: approval-gate-wiring
description: >
  Pattern for adding a new approval gate to the MouseDroid harness.
  Follow this skill when implementing a new safety gate, sandbox policy,
  or any decorator-chain approval component.
---

# Approval Gate Wiring Skill

## When to Use

Use this skill when:
- Adding a new approval gate (safety, policy, rate-limit, etc.)
- Modifying the decorator-chain composition in `src/mousedroid/factory/mcp_harness.py`
- Extending `ApprovalGateProtocol` with a new enforcement layer

## Architecture

Approval gates follow the **Decorator Chain** pattern:

```
inner_gate → SandboxPolicyGate → OpenClawSafetyGate → caller
```

Each gate wraps an `inner: ApprovalGateProtocol` and either:
1. **Rejects** the request (returning `ApprovalDecision(approved=False, ...)`)
2. **Delegates** to `self._inner.decide(request)`

## Checklist for Adding a New Gate

### 1. Create the Gate Class

File: `src/mousedroid/harness/approval/<name>_gate.py`

```python
from __future__ import annotations

from mousedroid.config.schema import <YourConfig>
from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

class YourNewGate:
    name = "your_gate_name"

    def __init__(self, inner: ApprovalGateProtocol, cfg: <YourConfig>) -> None:
        self._inner = inner
        self._cfg = cfg

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        # Your enforcement logic here
        # ...
        return await self._inner.decide(request)
```

### 2. Add Config to Schema

File: `src/mousedroid/config/schema.py`

- All thresholds, limits, and tunable parameters MUST be Pydantic `Field(default=..., description=...)`
- Never hardcode values in the gate class itself

### 3. Update `__init__.py` Exports

File: `src/mousedroid/harness/approval/__init__.py`

Add the new class name to `__all__`.

### 4. Wire in Factory

File: `src/mousedroid/factory/mcp_harness.py` — `build_approval_gate()` function

Wrap the new gate around the existing chain:

```python
gate = build_inner_gate(cfg)
gate = YourNewGate(gate, cfg.your_config)
gate = SandboxPolicyGate(gate, cfg.openclaw.policy)
gate = OpenClawSafetyGate(gate, filter_impl, cfg.openclaw)
```

### 5. Add Tests

Required test files:
- `tests/unit/harness/approval/test_<name>_gate.py` — unit tests
- `tests/regression/test_<name>_config_defaults.py` — pin config defaults

Use the shared `DummyGate` from `tests/unit/harness/approval/conftest.py`.

Required test cases:
- Happy path (passes through to inner gate)
- Each rejection reason
- Edge cases (None values, empty strings)
- Config-driven behavior (custom limits from YAML)

### 6. Regression Test for Config Defaults

```python
def test_your_config_default_field() -> None:
    cfg = YourConfig()
    assert cfg.field_name == expected_default
```

## Deterministic Validation

After completing all steps, run:

```bash
python -m ruff check --fix src/mousedroid/harness/approval/
python -m mypy --strict src/mousedroid/harness/approval/
python -m pytest tests/unit/harness/approval/ -v
python -m pytest tests/regression/ -k "config_defaults" -v
```

All must pass with zero errors.
