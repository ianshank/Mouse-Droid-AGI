# Tests

Tiered pytest suites (see [`../docs/testing.md`](../docs/testing.md) for the full guide):

| Tier | Directory | Scope |
|------|-----------|-------|
| Unit | `unit/` | Single-function behaviour, mocked deps |
| Integration | `integration/` | Multi-module wiring through the factory |
| Property | `property/` | Hypothesis-driven invariants |
| Regression | `regression/` | AQA / backwards-compat / golden |
| E2E | `e2e/` | Full request path |
| Hardware | `hardware/` | `@pytest.mark.hardware`, rover-only |
| Performance | `performance/` | Latency / throughput budgets |
| Smoke | `smoke/` | Sub-second import / parse |

Gate: **85% branch coverage**. Run: `pytest tests/` (auto-loads hardware mocks via the global `conftest.py`).
