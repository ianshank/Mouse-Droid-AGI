# Design — Enumerated branch-coverage allowlist

## D-1. Prefixes stay for true package splits

`config/schema/`, `telemetry/metrics/`, `telemetry/server/`, and
`validation/runtime/` remain directory prefixes. Those packages are
still the 1-file-to-many split products whose every module was the old
file.

## D-2. Factory and orchestrator are enumerated files

`_ALLOWED_FILES` lists the pure-DI factory modules and the eight
orchestrator mixin/`_state.py` files. A hypothetical
`factory/new_builder.py` or `orchestrator/_new_mixin.py` is **not**
exempt. `orchestrator/__init__.py` is not listed: the old
`orchestrator/_` prefix matched it accidentally because `__init__.py`
starts with `_`.

## D-3. Algorithmic factory stays on the gate

`factory/on_device_learning.py`, `factory/mcp_harness.py`, and
`factory/_replay_batch_helpers.py` are absent from `_ALLOWED_FILES`.
The evaluate-path test pins that 50% coverage on those files fails
while `factory/health.py` at 50% still passes.

## D-4. Dual lists stay in sync with disk

AQA asserts every `factory/*.py` is either in `_ALLOWED_FILES` or in
the gated trio, and every `orchestrator/_*.py` (excluding `__init__.py`)
is in `_ALLOWED_FILES`. Adding a module without classifying it fails CI.
