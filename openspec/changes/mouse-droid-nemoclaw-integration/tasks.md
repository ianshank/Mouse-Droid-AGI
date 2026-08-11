# Tasks: NemoClaw Integration

## Quality Gate (Per Task)

- `python -m ruff check --fix src/ tests/`
- `python -m ruff format src/ tests/`
- `python -m mypy --strict <new_modules>`
- `python -m pytest tests/unit tests/regression tests/property tests/integration -m "not hardware" --cov=src/mousedroid --cov-branch --cov-fail-under=93 -q`

## Phase 0: Foundations & Baselines

- [x] 0.1 Inventory toolchain and reserve F-IDs in `features.yaml`.
- [x] 0.2 Map existing gating paths (`MissionDispatcher` vs `SkillDelegator`).
- [x] 0.3 Spike: NemoClaw/OpenShell CLI viability on Windows/Jetson.
- [x] 0.4 Define baseline `baselines.yaml` schema for droid-side metrics.
- [x] 0.5 Capture hardware baselines on Orin Nano. (Deferred/Mocked on Windows)

## Phase 1: Governance & Config

- [x] 1.1 Implement F-ID `Agentic-Integration` logic in `features.yaml`.
- [x] 1.2 Implement `baselines.yaml` loading in `Settings`.
- [x] 1.3 Extend `OpenClawConfig` using nested sub-models (`OpenClawMemoryConfig`, etc.).

## Phase 2: Live Memory

- [x] 2.1 Refactor `EpisodicReplay` to support cursor-based deque iteration.
- [x] 2.2 Extend `MemoryResourceProvider` (MCP) to serve episodic cursor queries and semantic retrieves.
- [x] 2.3 Wrap blocking calls (like `SemanticIndex.retrieve()`) with `asyncio.to_thread()`.

## Phase 3: Gating & Sandbox Enforcement (F-027.3)

- [x] 3.1 Extract common gating logic from `SkillDelegator` and `MissionDispatcher` into a shared `ApprovalGateProtocol` envelope.
- [x] 3.2 Implement `SandboxPolicyGate` reading from `openshell` constraints (stubbed to static limits if `openshell` binary is missing).
- [x] 3.3 Wire the new composite gate in `factory.py` so both ingress paths share the exact same evaluation policy.

## Phase 4: Audit & Enforce Cutover (Spike) (F-027.4)

- [x] 4.1 Scaffold NemoClaw CLI wrappers in Python.
- [x] 4.2 Map `SkillSpec.tool_names` to an OpenShell policy generation function.
- [x] 4.3 Audit soak test: run sandbox in log-only mode.
- [x] 4.4 Enforce cutover: Spike concluded; `openshell` binary missing, falling back to static constraints (D8 exit ramp).

## Phase 5: Evaluations & Hardening

- [x] 5.1 Add observability metrics to `MetricsRegistry` (following PR #115 pattern).
- [x] 5.2 Implement baseline-referenced AQA tests.
- [x] 5.3 Integrate with `jetson_full_validation.sh`.
- [x] 5.4 Cross-model peer review (Anthropic vs Gemini).
