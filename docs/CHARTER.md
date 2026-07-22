# Project Charter — MouseDroid

> This charter is the project **constitution**. It sits *above* the day-to-day
> guidance surfaces and states what the project is for, what is in and out of
> scope, and the invariants no change may weaken. The other top-level documents
> are subordinate to it:
>
> | Surface | Role |
> |---|---|
> | **`docs/CHARTER.md`** (this file) | Constitution — vision, scope, invariants, roadmap |
> | `AGENTS.md` | Worker rules — what an agent MUST do when touching the repo |
> | `SKILLS.md` | Capability index — which files to read / commands to run per task |
> | `CLAUDE.md` | Project facts — what the code is and how it is laid out |
>
> Read this before planning work. Keep changes aligned with the scope (§3) and
> invariants (§4). Surface anything that would violate an invariant or expand
> scope for human decision (§6) rather than implementing it unilaterally.

## 1. Vision

A Star Wars MSE-6 ("mouse droid") autonomous-navigation system running on an NVIDIA Jetson
Orin Nano — an edge-AI / robotics engineering project, not a product and not a claim of general
intelligence. The cognitive stack is organised around a "10 Pillars of the Ideal Neural
Network" research framing used as an engineering compass: every pillar is real, unit-tested
code, and the honest axis is integration — seven pillars (world model, cognitive architecture,
memory, continual learning, reward, safety, curiosity) are wired into the 30 Hz runtime loop
(curiosity via the memory subsystem), while three (meta-learning, growth/distillation, scaling)
are implemented and tested but not yet wired in (§5). The same reusable cognitive core also underpins a parked `robot_arm` manipulation
platform, selected by config rather than by forking the code.

## 2. Mission

A 30 Hz sense-plan-act loop (RSSM latent dynamics → MCTS planning → ESP32 motor
control) that stays **deterministic, LLM-free, and training-free**. Deliberation
and learning are strictly off the hot loop:

- A deliberative natural-language → `GoalVector` brain (cloud Claude primary, a
  local GGUF model as the off-network fallback) translates missions and answers
  operator questions **outside** the 30 Hz loop.
- On-device incremental learning refines the rover's own world model *between*
  cloud retraining cycles, at a slow-cadence seam with all torch work offloaded
  via `asyncio.to_thread`.

The rover runs as supervised services on constrained edge hardware, degrading
gracefully when a sensor, radio, or uplink is absent rather than crash-looping.

## 3. Scope

**In scope:** The `orchestrator/` 30 Hz sense-plan-act loop; the cognitive stack — seven pillars
wired into the runtime (`world_model/`, `cognitive/`, `memory/`, `learning/`, `reward/`,
`safety/`, `curiosity/`) plus three implemented-but-not-yet-wired modules (`meta/`, `growth/`,
`scaling/`; §5); the parked `arm/` four-layer hierarchical-reasoning platform (perception →
symbolic planning → world modelling → motor control); the sensing / comms / telemetry /
resilience infrastructure, the unified camera+lidar+fusion dashboard, and the validation / smoke /
preflight harness.

**Out of scope:**

- **Autonomous motion without a human in the loop.** Motor commands that move the
  rover or arm require an explicit, gated authorisation path; the default posture
  is no-motion.
- **Any LLM or training work placed *inside* the 30 Hz hot loop.** Deliberation
  and learning are off-loop by construction (§4, invariant 10).
- **Changing runtime behaviour by editing source.** Operators flip behaviour
  through YAML or `MOUSEDROID_*__*` environment variables (§4, invariant 3).

**Ratified carve-outs (with real gates):**

Each carve-out below permits a bounded expansion of the read-only / no-motion /
off-loop posture, guarded by an explicit config gate that defaults to the safe
value. PR numbers and dates are confirmed against git history.

- **Jetson + USB-C rover smoke validation (#106, 2026-05-31).** Real-hardware
  bring-up is *probe-first*. `ESP32Config.enabled` defaults `True`, but the
  bring-up posture keeps `MOUSEDROID_ESP32__ENABLED=false` until the ESP32
  actually answers (a dead controller with `enabled=True` crash-loops
  `orchestrator.start()`). Motion stays behind the `ESP32Config.smoke_test_allow_motion`
  hard gate (default `False`). USB-C endpoint discovery
  (`USBCDiscoveryConfig.enabled`) defaults `False` so pre-existing YAML loads
  unchanged.

- **Cloud LLM egress (#107, 2026-06-02).** The deliberative path may send
  natural-language commands to `api.anthropic.com`. `LLMConfig.backend` is a
  `Literal["llama_cpp", "openai_compatible", "anthropic"]` defaulting to
  `"llama_cpp"` (byte-identical legacy behaviour); `"anthropic"` opts in. Every
  command is passed through the configured prompt-injection filter (the
  `RegexInjectionFilter` implementation of `PromptInjectionFilterProtocol`,
  `src/mousedroid/security/injection_filter.py`) — its `sanitize()` runs
  **before** egress — the only place rover NL leaves the device. The off-network fallback
  (`LLMConfig.fallback_backend`) is `Literal["none", "llama_cpp",
  "openai_compatible"]` (default `"none"`): `"anthropic"` is deliberately absent,
  so cloud-to-cloud failover is rejected at YAML-parse time and off-network
  autonomy is preserved. The API key is `LLMConfig.api_key: SecretStr | None`
  (default `None`) — never logged (§4, invariant 11).

- **On-device incremental learning (#135, 2026-06-14).** The rover may refine its
  own RSSM world model between cloud retraining cycles.
  `Settings.on_device_learning` is `OnDeviceLearningConfig | None`, default
  `None`; when present, `enabled` and `enable_hot_swap` both default `False`.
  Candidate weights live on a separate SHA-256-stamped slot (`slot_dir`, validated
  against absolute paths and `..` traversal), never overwriting the cloud-pulled
  slot, and auto-revert to the baseline on a reconstruction+KL regression bound.
  **Soak-gated** — kept off on the live rover until a soak gate passes.

## 4. Invariants

These hold across every module and may not be weakened by any change. Invariants
1–9 are the architecture invariants also stated in `CLAUDE.md` and `AGENTS.md`;
10–11 promote two cross-cutting rules to first-class status.

1. **Protocol-based DI.** All interfaces are `@runtime_checkable Protocol`.
   Concrete types are imported only inside factory builders — never in business
   logic.
2. **Factory single wiring point.** `src/mousedroid/factory.py` is the only place
   concrete types are wired; every `build_*()` returns a protocol type.
3. **No hardcoded values.** Every threshold, dimension, pin, path, and tunable
   comes from Pydantic config (`src/mousedroid/config/schema.py`) loaded from YAML
   in `config/`. Operators change behaviour via YAML or `MOUSEDROID_*__*` env
   vars, never by editing source.
4. **Structured logging only.** `structlog` via
   `from mousedroid.logging.setup import get_logger`; `_log.info("event", key=…)`.
   No `print()`, no f-string log messages.
5. **Asyncio everywhere.** All I/O-bound work is `async`; no threading for
   application logic. Blocking syscalls go through `asyncio.to_thread`.
6. **Type safety.** `mypy --strict` passes on every change; public functions carry
   type annotations and Google-style docstrings.
7. **`torch.no_grad()`** on every inference path.
8. **`deque(maxlen=N)`** for every sensor ring buffer, with `N` from config.
9. **Backwards compatibility.** New config fields carry a Pydantic default;
   existing YAML must load unchanged after a `git pull`, pinned by
   `tests/regression/test_pr*_backwards_compat.py`.
10. **Hot-loop purity.** The 30 Hz reactive loop (RSSM → MCTS → ESP32) stays
    deterministic, LLM-free, and training-free. All deliberation (NL translation,
    operator Q&A) and all learning (on-device refinement + gate) run at slow-cadence
    seams *outside* the hot loop, offloaded via `asyncio.to_thread`.
11. **No secrets or machine fingerprints in version control.** Credentials use
    `SecretStr` and are never `.get_secret_value()`-ed into a log or exception.
    Live per-host values live only in `/etc/mousedroid/docker.env` (matched by the
    gitignored `*.env`); `config/docker.env.example` documents the secret surface
    (including the `ANTHROPIC_API_KEY` slot) without holding live values.

**Quality gates are non-negotiable.** Linting (`ruff==0.8.0`), formatting
(`ruff format --check`), strict type-checking (`mypy --strict`), bounded
cyclomatic complexity (`ruff` `C901`, `max-complexity = 15`; ADR-014), and the
85% coverage floor (`--cov-fail-under=85`) stay green. The authoritative pipeline
is `.github/workflows/ci.yml` — actionlint → lint → typecheck → test+coverage →
prometheus/security → docker, across Python 3.10 / 3.11 / 3.12. Decompose a
function that trips the complexity gate; do not re-open a `src/` per-file ignore.

## 5. Long-term Roadmap

The roadmap uses the Physical-AI phase numbering (the repo also carries a legacy
v0.3.0 execution-plan numbering; where they differ, `docs/planning/IMPLEMENTATION_PLAN.md`
is authoritative). Statuses reflect the roadmap docs at time of ratification.

- **M1 — Self-Healing Core Resilience (Phase 2)** ✅ — circuit breaker + retry
  wrappers, a resilient ESP32 driver, and sensor-staleness detection, all
  factory-wired.
- **M2 — Learning Loop (Phase 2.1)** ✅ — behaviour-cloning supervised loss routed
  into the Constitutional-RL PPO / offline-RL path (TD3+BC pattern) via a dedicated
  `bc_optimizer` and a sim/real replay mixer, so real episodes tune the *policy*,
  not just the world model.
- **M3 — VLA + Real-Episode Replay (Phase 3 / Tier A–B)** ✅ — Vision-Language-Action
  inference (`MockVLA` → `DistilledVLAOnnx` with a Hugging Face pull), LMDB
  real-episode replay, and Law-1-gated VLM-derived dense rewards.
- **M4 — Closed-Loop Autonomy + Cloud Retraining (Tier C)** ✅ — a mission-lifecycle
  state machine with a geometric safety projector, cloud retraining feeding a
  SHA-256-verified Jetson OTA weight puller, and an Isaac Lab environment. The
  cloud-Claude deliberative brain, its Prometheus observability, and the operator
  Q&A path landed in and after this tier.
- **M5 — Real-Physics Sim-to-Real Foundation (Phase 5)** ✅ — a MuJoCo skid-steer
  physics simulator (`RoverMuJoCoEnv` behind `RoverEnvProtocol`) replacing the
  NumPy kinematic sim, with the RSSM dynamics core pretrained on its episodes.
- **M6 — On-Device Incremental Learning (Phase 6)** 🔜 ACTIVE — between-cloud-cycle
  RSSM refinement on a gated, integrity-checked weight slot with auto-revert;
  functional, default-OFF, and soak-gated (§3 carve-out).
- **Cognitive-pillar integration** 🔬 — `meta/` (MAML + in-context adaptation), `growth/`
  (knowledge distillation), and `scaling/` (MoE + adaptive compute) are implemented and
  unit-tested (`tests/unit/{meta,growth,scaling}/`) but not yet instantiated by `factory.py` /
  the orchestrator. They are promoted into the 30 Hz runtime loop only when a concrete need and a
  gate exist — not before. (`curiosity/` completed this path and is already wired.)

## 6. How Agents Use This Document

- **Read the charter before planning tasks.** Keep work aligned with scope (§3)
  and invariants (§4).
- **Escalate, don't unilaterally implement.** Any change that would violate an
  invariant, expand scope, or open a new carve-out is a human decision. Surface it
  (with the tradeoff) rather than shipping it — this mirrors the "Red flags —
  pause and check" discipline in `AGENTS.md`. New capabilities are additive and
  opt-in (default-OFF), never silent behaviour changes.
- **Track day-to-day work elsewhere.** Living to-dos and finding-IDs belong in
  `docs/planning/NEXT_STEPS.md`; multi-step agent work is tracked via
  `TaskCreate` / `TaskUpdate`. The charter changes rarely and only by ratification.
- **Follow the subordinate surfaces for detail:** `AGENTS.md` for the worker
  rules and extension playbooks, `SKILLS.md` for the capability-to-command index,
  and `CLAUDE.md` for the project facts and module map.
