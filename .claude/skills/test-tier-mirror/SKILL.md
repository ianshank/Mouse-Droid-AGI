---
name: test-tier-mirror
description: Place new tests in the right tier of the nine-tier mirror, with the correct filename, skip gate, and marker for that tier
status: active
---

# Test Tier Mirror

Decide which tier a new test belongs in, name the file the way that tier names
files, and gate it so it skips cleanly instead of failing on a host that cannot
run it.

Use this whenever a change adds behaviour that needs coverage, and when a review
asks "is this tested at the right level?".

## The nine tiers

| Tier | Directory | Add a test here when |
|---|---|---|
| Unit | `tests/unit/` | one function's behaviour, dependencies mocked |
| Integration | `tests/integration/` | several modules wired together **through the factory** |
| E2E | `tests/e2e/` | a full request path end to end (proxy → server → driver) |
| Regression | `tests/regression/` | a YAML / env / default-value invariant that must not drift |
| AQA | `tests/regression/` (`*_aqa.py`) | schema-field hygiene, protocol conformance |
| Property | `tests/property/` | a Hypothesis-driven invariant over an input space |
| Performance | `tests/performance/` | a latency or throughput budget |
| Sanity | `tests/smoke/` (`*_sanity.py`) | sub-second import + parse smoke |
| Hardware | `tests/hardware/` | needs the real rover; `@pytest.mark.hardware` |

Property and Performance are part of the mirror, not optional extras — both run
in `scripts/ci.sh`.

## Choosing the tier

Two questions settle most cases.

**Does it go through `factory.py`?** If yes it is at least integration. The
factory is the single wiring point, so "does the config actually reach the
driver" is only answerable above the unit tier. A unit test that constructs the
concrete type directly proves the type works, not that the system builds it.

**Would a mock make the test vacuous?** This is the trap that produced the
worst gap in this repo: `esp32.enabled: false` in the production overlay means
a factory build returns `MockESP32Driver`, which implements the protocol
directly and never touches the codec — so every "does the stock command set
work" test was really testing the mock. When the mock bypasses the code under
test, you need a fake at the *transport* boundary instead, and the test belongs
in integration. `tests/integration/test_f025_integration.py` is the reference
shape: a fake serial port, the real driver, the real factory.

An AQA test is the one to reach for when the thing you want to protect is a
*schema property* rather than a behaviour — that every new field carries a
description, that defaults match, that a Protocol's members are actually
callable. Check defaults on `model_fields[...]` (the `FieldInfo`) rather than
via `model_validate`, so a refactor that replaces `Field(...)` with a property
override is caught.

## Skip gates

**Optional extras.** Anything outside the bare `[dev]` install must be gated, or
the module fails instead of skipping on a lean checkout:

```python
pytest.importorskip("PIL", reason="Pillow (mousedroid[telemetry]) encodes the JPEG")
```

Module level when the whole file needs it, function level when one case does.
The suite uses this for `mujoco`, `faiss`, `ncps`, `aiohttp`, `mlflow`, `onnx`,
`llama_cpp` and Pillow. `tests/regression/test_optional_extra_import_gates.py`
enforces it for Pillow specifically.

Note that `importorskip` only catches `ImportError` — a package whose import
raises something else needs an explicit `try`/`except` plus `pytest.skip`.

**Hardware.** Mark the module and let the conftest decide:

```python
pytestmark = pytest.mark.hardware
```

`tests/hardware/conftest.py` reverses the global mock-hardware override and
forces `mock_hardware=True` again on non-Jetson hosts, so a hardware test still
exercises its code path off-rover without touching a device. Use
`tests/_jetson_hardware.is_jetson_host` when a specific case must not run at all
off-rover.

## Writing an assertion that can fail

A test that cannot fail is worse than no test, because it reads as coverage.
Three patterns to avoid, all of which shipped here before being caught:

- `assert driver.inner._connected is True` — set once in `connect()` and never
  cleared, so it cannot fail. Assert through the public protocol instead: do a
  round-trip and check the result.
- `assert isinstance(obj, SomeRuntimeCheckableProtocol)` — `runtime_checkable`
  checks attribute *presence* only, so it passes for a class whose "methods" are
  integers. Keep it as a cheap smoke check, then assert the members are callable
  with the expected arity.
- `assert "continue-on-error" not in job` — pins YAML formatting, not semantics.
  Assert the value: `assert job.get("continue-on-error") is not True`.

Under `PYTHONOPTIMIZE=1` (the Jetson Docker entrypoint) `assert` is stripped
entirely. In `src/` and in inline shell one-liners, use an explicit
`if not ...: raise RuntimeError(...)` — never `assert` — for anything that must
hold in production.

## Filename conventions

Feature-scoped suites carry the identifier in the name so the tier and the
change are both greppable: `test_pr104_integration.py`,
`test_f025_backwards_compat.py`, `test_f025_aqa.py`, `test_pr106_sanity.py`.
Behaviour-scoped tests name the behaviour: `test_safety_monitor.py`.

## Checking your placement

```bash
python -m pytest tests/integration/test_f025_integration.py -q
python scripts/check_branch_coverage.py --min 90 \
    --tests tests/unit tests/property tests/integration
```

The branch-coverage gate scores *changed lines*, so it tells you directly
whether the tier you chose actually reaches the code you wrote. If a new branch
in `src/` stays uncovered after adding a unit test, that is usually the signal
that the behaviour only exists once the factory wires it — move up a tier.
