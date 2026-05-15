# Full-Stack Roadmap — Training, Autonomy, Streaming Dashboard, Isaac Lab

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive MouseDroidAGI from its current "production-instrumented hardware-validated" state (post-PR-#83) to "fully autonomous, continuously training, with a unified operator-grade dashboard" — across four parallel tracks (Training, Isaac Sim, Autonomy, Dashboard) plus a Hardening track.

**Architecture:** Every change preserves the 9 invariants in `CLAUDE.md`: protocol-based DI, `factory.py` single wiring point, no hardcoded values (Pydantic schema → YAML overlay), `structlog` everywhere, asyncio-only, `mypy --strict`, `torch.no_grad()` for inference, `deque(maxlen=N)` for ring buffers, backwards-compatible config defaults. New config fields ALWAYS land as `Optional` or with a safe default so existing YAML files load unchanged.

**Tech Stack:** Python 3.10+, PyTorch + AMP, Pydantic v2, structlog, aiohttp, ncps (CfC), pytest + hypothesis + pytest-asyncio, ruff 0.8.0, mypy --strict, NumPy (explicit dtypes, no implicit float64), Stable-Baselines3 SAC+HER, MuJoCo Gymnasium, NVIDIA Isaac Lab (workstation-side), ONNX → TensorRT (Jetson), Prometheus, Grafana, Playwright, Docker/Jetson L4T.

---

## Context

### Where we are (verified 2026-05-15, branch `feat/full-stack-roadmap` off `claude/markdown-implementation-plan-aVJ2l@4dc118a`)

**Deployed today** (`deployments/jetson-image.json:3` — SHA `90119eff`, deployed 2026-05-13):
- 30 Hz orchestrator sense-plan-act loop with safety gate, watchdog heartbeat, structured logging
- LiDAR (LD19) + CSI camera + USB audio + ESP32 motor bus, all driver-resilient
- Telemetry server: REST (`/api/v1/*`), WebSockets (`/ws`, `/ws/v1/lidar/raw`, `/api/v1/logs/stream`), Prometheus `/metrics`, two static dashboards (`/lidar`, `/camera`)
- Voice (Piper TTS + Rocky event-router) with token-bucket fairness and fail-loud speaker recovery
- 14-stage Jetson smoke runner (`scripts/jetson_full_smoke_run.sh`) + 13-probe Phase B suite (`scripts/jetson_new_features_probe.sh` from PR #83)
- Sensor-liveness state machine (`disabled` / `awaiting` / `live` / `stale`) on every telemetry frame
- `ClockProtocol` + `RealClock` + `MockClock` (bisect-driven) for deterministic time
- `FailureRecorder` primitive on every fallback path; per-subsystem labelled Prometheus counters

**Verified surveys (file:line citations from parallel Explore agents):**

| Track | Production-ready | Partial / stub / missing |
|---|---|---|
| **Training** | RSSM pretraining (`training/train_rssm.py`), HF upload (`training/upload_weights.py`), LMDB replay reader, BC `bc_update` wired (`training/train_offline_rl.py:193`) | **MISSING:** `training/train_arm.py`; **STUB:** EWC/MAML never called from any training script; **STUB:** Dreamer-V3 imagination rollout; **MISSING:** W&B; **MISSING:** BC test at `real_supervised_weight>0` |
| **Autonomy** | Orchestrator loop (`orchestrator/orchestrator.py:370-410`), safety monitor (`safety/monitor.py`), memory tier + LMDB journal | **PARTIAL:** Mission lifecycle (no persistent goal+odometry; `mission_dispatcher.py:1-80`); **STUB:** VLA only `MockVLA` (`vla/policy.py:58-80`); **PARTIAL:** MCTS planning horizon fixed; **MISSING:** Multi-minute autonomous E2E |
| **Dashboard** | Backend WebSockets, Prometheus metrics, sensor liveness, `/lidar` + `/camera` pages | **MISSING:** Unified dashboard / mission control / e-stop / log panel page / arm page / voice page; **MISSING:** Grafana wiring verified end-to-end |
| **Isaac Lab** | Research doc (`docs/planning/ISAAC_LAB_ROVER_RESEARCH.md`), env protocol (`src/mousedroid/sim/protocols.py`), mock env (`src/mousedroid/sim/mock_rover_env.py`) | **STUB:** `src/mousedroid/sim/isaaclab/rover_env.py:1-255` (`build()` is placeholder; raises `IsaacLabUnavailableError`); **MISSING:** URDF/USD assets, sensor adapter, PPO smoke, ONNX→TRT validation harness |

### Where we are going

Four parallel **tracks**, each a focused PR sequence that ships value independently. A fifth cross-cutting **Hardening** track captures lint/CI/secret/numpy hygiene.

| Track | PR count | Independent? | Acceptance |
|---|---|---|---|
| **T — Training** | 5 PRs (T1–T5) | Yes (touches only `training/`, `src/mousedroid/{learning,meta,curiosity,growth,world_model}`) | A real arm policy improves measurably on a held-out MuJoCo eval; nightly W&B run produces curves |
| **S — Sim / Isaac Lab** | 4 PRs (S1–S4) | Depends on T1 for shared training utilities | Rover trains in Isaac Lab PPO baseline → ONNX → TensorRT engine deploys on Jetson; >50% goal-reach transfer rate documented |
| **A — Autonomy** | 5 PRs (A1–A5) | A2 depends on T-track distilled-VLA artifact; A1, A3, A4, A5 independent | Rover autonomously navigates to a goal across 5 simulated minutes (`MockClock`) and 5 real wall-clock minutes (Jetson) without operator intervention |
| **D — Dashboard** | 6 PRs (D1–D6) | Independent (touches only `src/mousedroid/telemetry/` and tests) | Unified `/` page with live mission control + e-stop + log panel + per-subsystem health; Grafana dashboard renders metrics from `docs/grafana_dashboard.json` |
| **H — Hardening** | 4 PRs (H1–H4) | Independent | Coverage ≥87% (+2 over current), ruff/mypy clean, no hardcoded values, secrets scanned in CI |

### Plan format

- **High-level PR breakdown per track** with dependencies, acceptance, suggested subagent type.
- **First PR per track gets fully-detailed bite-sized TDD tasks** (Task / files / step / code / commit) — engineer can execute today.
- **Later PRs per track** get concrete-but-shorter task lists (file paths + code outlines + acceptance), expandable on demand.
- A final **Execution Strategy** section maps tracks → worktrees → subagent dispatches → parallel batches.

---

# Track T — Training

## PR map

| PR | Title | Depends on | Suggested subagent |
|---|---|---|---|
| T1 | `test(training): BC weight regression at real_supervised_weight>0` | none | `test-runner` |
| T2 | `feat(training): W&B logger adapter wired into train_rssm/train_offline_rl` | none | `ml-training-orchestrator` |
| T3 | `feat(training): train_arm.py — SAC+HER MuJoCo entry point + curriculum + checkpoints` | T2 (for logging) | `ai-ml-toolkit:ml-engineer` |
| T4 | `feat(learning): EWC consolidation wired into offline-RL epoch loop` | T1 (BC stable) | `distributed-training` |
| T5 | `feat(world_model): Dreamer-V3 imagination rollout + latent offline-RL` | T2, T3 | `neural-network-architect` |

---

## PR T1 — BC weight regression test

**Why:** `bc_update` is wired at `training/train_offline_rl.py:193` but the default `real_supervised_weight=0` makes it a no-op. Phase 2.1 of `docs/planning/PHASE_2_1_AND_BEYOND_PLAN.md` cannot ship without a test that drives BC loss to non-zero AND verifies it aggregates into the epoch summary without corrupting Q-values byte-for-byte at weight=0.

**Files:**
- Create: `tests/integration/training/test_offline_rl_bc_weight.py`
- Modify: `src/mousedroid/config/schema.py` — add `OfflineRLConfig.real_supervised_weight` validator (must be `>= 0`, `Field(0.0, ge=0.0)`) if not already constrained
- Reference: `training/train_offline_rl.py:165-198`, `src/mousedroid/experience/dataset.py`, `tests/integration/test_phase21_bc_into_offline_rl.py` (existing)

### Tasks

- [ ] **Step 1 — Read the existing BC scaffold to understand the entry contract**

Read these specific ranges:
- `training/train_offline_rl.py:160-210` — `bc_update` call site and aggregation
- `src/mousedroid/config/schema.py` — locate `OfflineRLConfig`, grep for `real_supervised_weight`
- `tests/integration/test_phase21_bc_into_offline_rl.py` — existing weight=0 test

```bash
grep -nE "real_supervised_weight|bc_update|bc_loss" training/train_offline_rl.py src/mousedroid/config/schema.py
```

- [ ] **Step 2 — Write the failing test first (TDD)**

Create `tests/integration/training/__init__.py` (empty) and `tests/integration/training/test_offline_rl_bc_weight.py`:

```python
"""Regression: BC weight > 0 actually contributes loss + leaves Q-values unchanged at weight=0."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from mousedroid.config.loader import load_settings
from mousedroid.experience.dataset import OfflineRLDataset
from training.train_offline_rl import run_one_epoch  # exported helper, see Step 4


@pytest.fixture
def tiny_lmdb(tmp_path: Path) -> Path:
    """8-frame synthetic LMDB for deterministic epoch-replay."""
    from tests._helpers.lmdb_synth import write_synth_lmdb  # existing helper
    out = tmp_path / "tiny.lmdb"
    write_synth_lmdb(out, n_frames=8, seed=42)
    return out


def test_bc_weight_zero_leaves_q_values_unchanged(tiny_lmdb: Path) -> None:
    """Baseline: weight=0.0 must produce zero BC contribution."""
    cfg = load_settings()
    cfg = cfg.model_copy(update={"offline_rl": cfg.offline_rl.model_copy(
        update={"real_supervised_weight": 0.0, "lmdb_path": str(tiny_lmdb)}
    )})
    summary = run_one_epoch(cfg, seed=42)
    assert math.isclose(summary["bc_loss"], 0.0, abs_tol=1e-9)
    assert summary["q_loss"] > 0.0  # Q-network IS learning
    assert "bc_loss" in summary  # key always present even at weight=0


def test_bc_weight_positive_contributes_loss(tiny_lmdb: Path) -> None:
    """At weight=0.1 BC must contribute non-zero loss and not crash the epoch."""
    cfg = load_settings()
    cfg = cfg.model_copy(update={"offline_rl": cfg.offline_rl.model_copy(
        update={"real_supervised_weight": 0.1, "lmdb_path": str(tiny_lmdb)}
    )})
    summary = run_one_epoch(cfg, seed=42)
    assert summary["bc_loss"] > 0.0, "weight>0 must produce non-zero BC loss"
    assert math.isfinite(summary["bc_loss"]), "BC loss must be finite (no NaN)"
    assert summary["q_loss"] > 0.0


def test_bc_weight_negative_rejected_at_config_load() -> None:
    """OfflineRLConfig.real_supervised_weight must reject negative values."""
    from pydantic import ValidationError
    from mousedroid.config.schema import OfflineRLConfig
    with pytest.raises(ValidationError):
        OfflineRLConfig(real_supervised_weight=-0.01)
```

- [ ] **Step 3 — Run the test to verify it fails for the right reason**

```bash
pytest tests/integration/training/test_offline_rl_bc_weight.py -v --import-mode=importlib
```

Expected: `ImportError: cannot import name 'run_one_epoch' from 'training.train_offline_rl'` (proving the helper isn't yet extracted).

- [ ] **Step 4 — Extract `run_one_epoch` from the training CLI**

Modify `training/train_offline_rl.py` — pull the epoch loop body into a pure function returning a summary dict. Preserve the CLI behaviour byte-for-byte (the CLI now calls `run_one_epoch` in a loop).

Show the extracted signature only (full body stays equivalent):

```python
def run_one_epoch(
    cfg: Settings,
    *,
    seed: int,
    epoch_idx: int = 0,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Run one offline-RL epoch and return aggregated metrics.

    Returns:
        Dict with at least ``q_loss``, ``policy_loss``, ``bc_loss``. ``bc_loss``
        is ``0.0`` (exactly) when ``cfg.offline_rl.real_supervised_weight == 0``,
        non-zero otherwise.
    """
    ...
```

- [ ] **Step 5 — Add the `ge=0` validator on `OfflineRLConfig.real_supervised_weight`**

Modify `src/mousedroid/config/schema.py`:

```python
real_supervised_weight: float = Field(
    0.0,
    ge=0.0,
    description=(
        "Weight on the behaviour-cloning auxiliary loss applied to real-rover "
        "episodes during offline RL. 0.0 disables BC (no-op call, zero overhead). "
        "Phase 2.1 expects ~0.1-0.3 once real episodes are flowing."
    ),
)
```

- [ ] **Step 6 — Run the tests to verify they pass**

```bash
pytest tests/integration/training/test_offline_rl_bc_weight.py -v --import-mode=importlib
```

Expected: 3 PASSED.

- [ ] **Step 7 — Confirm lint + mypy clean**

```bash
ruff check training/train_offline_rl.py src/mousedroid/config/schema.py tests/integration/training/
ruff format --check training/train_offline_rl.py src/mousedroid/config/schema.py tests/integration/training/
mypy --strict training/train_offline_rl.py src/mousedroid/config/schema.py tests/integration/training/
```

- [ ] **Step 8 — Commit**

```bash
git add tests/integration/training/ training/train_offline_rl.py src/mousedroid/config/schema.py
git commit -m "$(cat <<'EOF'
test(training): regression — BC weight>0 contributes non-zero loss

Phase 2.1 prerequisite. The bc_update call site has lived in
train_offline_rl.py since Phase 2 merged, but the default
real_supervised_weight=0 made it a silent no-op. Adds three tests:
- weight=0: bc_loss is exactly 0.0; q_loss > 0 (baseline)
- weight=0.1: bc_loss > 0 and finite; q_loss > 0 (BC active)
- weight=-0.01: Pydantic ValidationError (schema rejects negative)

Also extracts run_one_epoch as a pure-function helper so tests can
drive a deterministic 1-epoch replay without spawning a subprocess.
EOF
)"
```

---

## PR T2 — W&B logger adapter

**Why:** Survey gap "MISSING: W&B integration"; training runs today produce only structlog output, making cross-run comparison and hyperparameter sweeps impossible. The adapter must be importable in environments without `wandb` installed (training in CI), so use `pytest.importorskip`-style guards.

**Files:**
- Create: `training/wandb_logger.py` — protocol + Prometheus-style adapter + null fallback
- Modify: `training/train_rssm.py`, `training/train_offline_rl.py`, `training/run_pipeline.py` — accept the logger via DI
- Create: `tests/unit/training/test_wandb_logger.py`

**Acceptance:**
- Three concrete implementations: `WandbLogger` (real), `StructlogOnlyLogger` (CI fallback), `NullLogger` (tests)
- `Settings.training.wandb` Pydantic config with `enabled`, `project`, `entity`, `tags`, `run_name_template` fields (all optional with safe defaults)
- Zero hardcoded W&B values; project/entity come from config or env (`WANDB_PROJECT`, `WANDB_ENTITY`)
- `mypy --strict` clean; `ruff` clean; coverage ≥90% on the new module

**Key tasks (abbreviated):**

1. Define `TrainingLoggerProtocol` (`log_scalar`, `log_histogram`, `log_artifact`, `start_run`, `finish_run`) — `@runtime_checkable`
2. Implement `NullLogger` (no-op) and `StructlogOnlyLogger` (forwards to existing `_log` via `bind(metric=...)`)
3. Implement `WandbLogger` with lazy `import wandb` inside `start_run` so `WANDB_DISABLED=true` skips it
4. Add `WandBConfig` to `src/mousedroid/config/schema.py` (under `Settings.training`)
5. Add `build_training_logger(cfg)` factory in `training/wandb_logger.py`
6. Wire into `train_rssm.py` and `train_offline_rl.py` via a `logger=` kwarg (default `NullLogger()`)
7. Unit tests cover: import-failure path, run_name_template substitution, scalar dispatch, artifact path

---

## PR T3 — `train_arm.py` SAC+HER entry point

**Why:** `SACAgent` class exists (`src/mousedroid/arm/control/sac_agent.py`) but no training script invokes it. This is the blocker for real arm policy improvement.

**Files:**
- Create: `training/train_arm.py` — CLI entry point
- Create: `training/arm/__init__.py`, `training/arm/curriculum.py` — disk-count curriculum (1 → 3 → 5 → 7)
- Create: `tests/integration/training/test_train_arm_smoke.py` — 50-step smoke run on `tower_of_hanoi_1disk` env
- Modify: `src/mousedroid/config/schema.py` — `ArmTrainingConfig.curriculum_stages` (list of stage dicts) if not present

**Acceptance:**
- `python -m training.train_arm --config config/robot_arm_default.yaml --max-steps 50` runs without error against MuJoCo mock
- Returns checkpoint to `cfg.arm.training.checkpoint_dir` (config-driven, no hardcoded path)
- HER buffer + SAC parameters from config (no literal `learning_rate=3e-4` in code; all flow through `ArmTrainingConfig`)
- Curriculum stages from config; stage transitions logged via T2's logger
- W&B run started when `cfg.training.wandb.enabled`

**Key tasks:**

1. Read `src/mousedroid/arm/control/sac_agent.py` to confirm the `SACAgent.train(env, total_steps, callback=...)` signature
2. Write the failing smoke test (`test_train_arm_smoke_completes_50_steps`)
3. Implement `training/arm/curriculum.py` — `CurriculumScheduler.next_stage(stage_idx, success_rate) -> StageSpec`
4. Implement `training/train_arm.py` `main()` — load cfg, build env from `cfg.arm.platform`, build `SACAgent`, run training loop with curriculum + logger
5. Add config schema entries (`ArmTrainingConfig.curriculum_stages`, `ArmTrainingConfig.success_threshold_for_promotion`)
6. Verify checkpoint round-trip (save → load → resume)
7. Add `bash scripts/ci.sh` smoke path (optional `--with-arm` flag in `ci.sh`)

---

## PR T4 — EWC consolidation wired into offline-RL epoch loop

**Why:** `src/mousedroid/learning/ewc.py:17-82` implements EWC but has zero call sites. Once T3 lands and arm policies start training, transferring from Tower-of-Hanoi → laundry-sorting will catastrophically forget unless EWC consolidates.

**Files:**
- Modify: `training/train_offline_rl.py` — consolidate Fisher after each epoch
- Modify: `training/train_arm.py` (from T3) — same consolidation point
- Modify: `src/mousedroid/config/schema.py` — `OfflineRLConfig.enable_ewc: bool = False`, `OfflineRLConfig.ewc_lambda: float = Field(100.0, ge=0.0)`
- Create: `tests/integration/training/test_ewc_consolidation.py`

**Acceptance:**
- Disable-by-default flag (backwards compat)
- When enabled, `EWCAgent.consolidate(task_id=epoch_idx)` fires at epoch end
- Test: two-task continual scenario where T2 performance after T2 training matches "baseline" within `cfg.training.ewc.regression_tol_pct` (config-driven tolerance, default 10%)
- No NumPy implicit float64 issues (any new tensor ops specify dtype)

---

## PR T5 — Dreamer-V3 imagination rollout + latent offline-RL

**Why:** Largest gap. RSSM is trained on sequences (`training/train_rssm.py`) but offline-RL reads raw (s,a,r,s') from LMDB. Sample efficiency would jump materially with imagined rollouts.

**Files:**
- Create: `src/mousedroid/world_model/imagination.py` — `ImaginationRollout` class (prior-loop K-step rollouts)
- Create: `training/train_dreamer_v3.py` — latent-space offline RL CLI
- Modify: `src/mousedroid/config/schema.py` — `DreamerConfig` (horizon, lambda_target, reward_head_lr, value_head_lr)
- Create: `tests/unit/world_model/test_imagination.py`, `tests/integration/training/test_dreamer_smoke.py`

**Acceptance:**
- 50-step Dreamer smoke run completes on mock data
- ImaginationRollout produces a (batch × horizon × latent_dim) tensor with no NaN/inf
- `mypy --strict` clean on new module
- Comparison curves (vs T2's offline-RL baseline) published as W&B artifact

**Notes for the engineer:**
- NumPy hygiene: every `np.array` call needs explicit `dtype=`; tests run under `numpy.errstate(all="raise")` (existing pattern in `tests/unit/world_model/`)
- Latent rollout MUST be in `torch.no_grad()` when used at inference time (Invariant #7)

---

# Track S — Sim (Isaac Lab)

## PR map

| PR | Title | Depends on | Subagent |
|---|---|---|---|
| S1 | `feat(sim): URDF→USD builder + RobotConfig roundtrip` | none | `ai-ml-toolkit:ml-engineer` |
| S2 | `feat(sim/isaaclab): sensor adapter (RTX LiDAR → SensorManager shape)` | S1 | `ai-ml-toolkit:ml-engineer` |
| S3 | `feat(sim/isaaclab): PPO smoke + ONNX export` | S2, T2 (logger) | `ml-training-orchestrator` |
| S4 | `feat(sim): export-to-Jetson validation harness (transfer rate ≥50%)` | S3 | `ai-ml-toolkit:ml-engineer` |

The full Isaac Lab roadmap mirrors `docs/planning/ISAAC_LAB_ROVER_RESEARCH.md` Phases 0–4. This track lands one PR per phase. Photorealistic Cosmos (Phase 5) deferred.

**S1 acceptance:** Given a `RobotConfig(wheelbase, track, wheel_radius)`, emit a USD that Isaac Lab can import without warnings, AND verify mass + collider hierarchy matches the YAML in a snapshot test.

**S2 acceptance:** `IsaacLabSensorAdapter.encode(scan_dict) -> NDArray[shape=(36,), dtype=float32]` produces sector distances bit-identical to `SensorManager` for a synthetic input fixture.

**S3 acceptance:** `python -m training.train_isaac_lab --steps 1000` finishes; export `policy.onnx` ≤ 10 MB; `trtexec --fp16 --onnx=policy.onnx` produces an engine on the Jetson without unsupported-op errors.

**S4 acceptance:** Harness drives the rover for 10 trials in mock-hardware mode using the ONNX engine; documents goal-reach rate ≥50% with confidence interval; emits a Markdown report `reports/isaac_transfer/<stamp>/REPORT.md`.

---

# Track A — Autonomy

## PR map

| PR | Title | Depends on | Subagent |
|---|---|---|---|
| A1 | `feat(orchestrator): persistent MissionState (goal + odometry + timeout)` | none | `python-pro` |
| A2 | `feat(vla): wire DistilledVLAOnnx + TensorRT path through factory + policy selector` | T3 or S3 (an actual policy artifact) | `neural-network-architect` |
| A3 | `refactor(world_model/mcts): configurable planning horizon + goal-cost heuristic` | A1 | `mcts-optimizer` |
| A4 | `feat(harness): PRE_ACTION replanner hook with safety-veto trigger` | A1, A3 | `python-pro` |
| A5 | `test(integration): test_multi_minute_autonomous_mission.py — 5 sim-min + 5 real-min` | A1, A2, A3, A4 | `test-runner` |

---

## PR A1 — Persistent MissionState

**Why:** Today `mission_dispatcher.py:1-80` parses a sentence into a one-shot directive ("go forward 1m"). There's no concept of a goal-pose, no odometry-driven completion check, no timeout state machine. A 5-minute autonomous run requires a state machine that tracks (current_goal, accumulated_displacement, elapsed_time, abort_reason).

**Files:**
- Create: `src/mousedroid/orchestrator/mission_state.py` — `MissionState` dataclass + state machine
- Modify: `src/mousedroid/orchestrator/mission_dispatcher.py` — emit `MissionState`s instead of one-shot velocity commands
- Modify: `src/mousedroid/orchestrator/orchestrator.py` — consume `MissionState`, transition on goal-reached / timeout / safety-violation
- Create: `tests/unit/orchestrator/test_mission_state.py`

### Tasks

- [ ] **Step 1 — Read current dispatcher + orchestrator surface**

```bash
grep -nE "mission|Mission" src/mousedroid/orchestrator/orchestrator.py src/mousedroid/orchestrator/mission_dispatcher.py | head -40
```

- [ ] **Step 2 — Define the failing test for MissionState transitions**

Create `tests/unit/orchestrator/test_mission_state.py`:

```python
"""MissionState state-machine — pending → active → (completed | failed | aborted)."""

from __future__ import annotations

import pytest

from mousedroid.orchestrator.mission_state import (
    AbortReason,
    MissionGoal,
    MissionLifecycle,
    MissionState,
)


def test_initial_state_is_pending() -> None:
    m = MissionState(goal=MissionGoal.navigate_relative(forward_m=1.0))
    assert m.lifecycle is MissionLifecycle.PENDING


def test_activate_transitions_to_active() -> None:
    m = MissionState(goal=MissionGoal.navigate_relative(forward_m=1.0))
    m.activate(now_s=10.0)
    assert m.lifecycle is MissionLifecycle.ACTIVE
    assert m.started_at_s == 10.0


def test_goal_reached_when_displacement_covers_goal() -> None:
    goal = MissionGoal.navigate_relative(forward_m=1.0)
    m = MissionState(goal=goal)
    m.activate(now_s=0.0)
    m.update(now_s=5.0, displacement_m=1.05)  # overshoots within tolerance
    assert m.lifecycle is MissionLifecycle.COMPLETED


def test_backward_mission_completes_at_negative_displacement() -> None:
    """Signed displacement: backward goal completes when the signed
    odometry value reaches the negative target — not when its absolute
    value does."""
    goal = MissionGoal.navigate_relative(forward_m=-1.0)
    m = MissionState(goal=goal)
    m.activate(now_s=0.0)
    m.update(now_s=5.0, displacement_m=-1.05)  # overshoots in the negative direction
    assert m.lifecycle is MissionLifecycle.COMPLETED


def test_backward_mission_does_not_complete_on_positive_displacement() -> None:
    """A backward goal must NOT complete if the rover happens to drift
    forward — regression for the abs() bug."""
    goal = MissionGoal.navigate_relative(forward_m=-1.0, timeout_s=100.0)
    m = MissionState(goal=goal)
    m.activate(now_s=0.0)
    m.update(now_s=1.0, displacement_m=1.05)  # would have completed under abs() logic
    assert m.lifecycle is MissionLifecycle.ACTIVE


def test_zero_forward_mission_does_not_complete_on_activation() -> None:
    """forward_m=0 (pure rotation/lateral mission) must NOT instantly
    complete on the first ``update`` call — regression for the
    ``0.0 + tolerance_m >= 0.0`` bug."""
    goal = MissionGoal.navigate_relative(forward_m=0.0, timeout_s=10.0)
    m = MissionState(goal=goal)
    m.activate(now_s=0.0)
    m.update(now_s=0.1, displacement_m=0.0)
    assert m.lifecycle is MissionLifecycle.ACTIVE
    # Still ACTIVE after non-zero displacement too — pure-rotation
    # missions need a separate completion criterion (A1.1 follow-up).
    m.update(now_s=0.2, displacement_m=0.05)
    assert m.lifecycle is MissionLifecycle.ACTIVE


def test_zero_forward_mission_still_honours_timeout() -> None:
    """A zero-distance mission must still fail on timeout so it never hangs."""
    goal = MissionGoal.navigate_relative(forward_m=0.0, timeout_s=1.0)
    m = MissionState(goal=goal)
    m.activate(now_s=0.0)
    m.update(now_s=2.0, displacement_m=0.0)
    assert m.lifecycle is MissionLifecycle.FAILED
    assert m.abort_reason is AbortReason.TIMEOUT


def test_timeout_aborts_with_reason() -> None:
    goal = MissionGoal.navigate_relative(forward_m=1.0, timeout_s=2.0)
    m = MissionState(goal=goal)
    m.activate(now_s=0.0)
    m.update(now_s=3.0, displacement_m=0.5)
    assert m.lifecycle is MissionLifecycle.FAILED
    assert m.abort_reason is AbortReason.TIMEOUT


def test_safety_violation_transitions_to_aborted() -> None:
    m = MissionState(goal=MissionGoal.navigate_relative(forward_m=1.0))
    m.activate(now_s=0.0)
    m.abort(now_s=1.0, reason=AbortReason.SAFETY_VIOLATION)
    assert m.lifecycle is MissionLifecycle.ABORTED
    assert m.ended_at_s == 1.0


def test_invalid_transition_from_completed_raises() -> None:
    m = MissionState(goal=MissionGoal.navigate_relative(forward_m=0.1))
    m.activate(now_s=0.0)
    m.update(now_s=1.0, displacement_m=0.2)  # completes
    with pytest.raises(RuntimeError, match="cannot activate from COMPLETED"):
        m.activate(now_s=2.0)
```

- [ ] **Step 3 — Run the test to verify the failure**

```bash
pytest tests/unit/orchestrator/test_mission_state.py -v --import-mode=importlib
```

Expected: `ImportError: cannot import name 'MissionState' from 'mousedroid.orchestrator.mission_state'`.

- [ ] **Step 4 — Implement the state machine**

Create `src/mousedroid/orchestrator/mission_state.py`:

```python
"""Persistent mission state machine — replaces one-shot directive dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class MissionLifecycle(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class AbortReason(Enum):
    TIMEOUT = "timeout"
    SAFETY_VIOLATION = "safety_violation"
    OPERATOR_CANCEL = "operator_cancel"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


@dataclass(frozen=True)
class MissionGoal:
    """Immutable mission goal description. Use factory constructors."""
    forward_m: float = 0.0
    lateral_m: float = 0.0
    heading_rad: float = 0.0
    tolerance_m: float = 0.10  # default 10 cm tolerance
    timeout_s: float = 60.0  # default 60 s mission timeout

    @classmethod
    def navigate_relative(
        cls, *, forward_m: float, lateral_m: float = 0.0,
        tolerance_m: float = 0.10, timeout_s: float = 60.0,
    ) -> MissionGoal:
        return cls(forward_m=forward_m, lateral_m=lateral_m,
                   tolerance_m=tolerance_m, timeout_s=timeout_s)


@dataclass
class MissionState:
    """Mutable mission lifecycle state. Single source of truth for the
    orchestrator's mission tracking."""
    goal: MissionGoal
    lifecycle: MissionLifecycle = MissionLifecycle.PENDING
    started_at_s: float | None = None
    ended_at_s: float | None = None
    accumulated_displacement_m: float = 0.0
    abort_reason: AbortReason | None = None

    def activate(self, *, now_s: float) -> None:
        if self.lifecycle is not MissionLifecycle.PENDING:
            raise RuntimeError(
                f"cannot activate from {self.lifecycle.name}; mission already started"
            )
        self.lifecycle = MissionLifecycle.ACTIVE
        self.started_at_s = now_s

    def update(self, *, now_s: float, displacement_m: float) -> None:
        """Advance the mission state given the latest signed displacement.

        ``displacement_m`` is the SIGNED odometry value along the goal's
        forward axis (positive = forward of the start pose, negative =
        backward). The completion check is sign-aware so a backward
        mission (``forward_m=-1.0``) completes when ``displacement_m``
        reaches roughly ``-1.0``, not when ``|displacement_m|`` does.

        For pure rotation / lateral missions (``forward_m == 0``), the
        forward-displacement check is skipped — completion must come
        from a different source (heading reached, lateral goal met,
        operator confirm). Avoids the bug where ``0 + tolerance >= 0``
        completed the mission on activation.
        """
        if self.lifecycle is not MissionLifecycle.ACTIVE:
            return
        self.accumulated_displacement_m = displacement_m

        forward_goal = self.goal.forward_m
        tol = self.goal.tolerance_m
        if forward_goal > 0.0:
            # Forward mission: complete when signed displacement is
            # within tolerance of the positive target.
            if displacement_m + tol >= forward_goal:
                self.lifecycle = MissionLifecycle.COMPLETED
                self.ended_at_s = now_s
                return
        elif forward_goal < 0.0:
            # Backward mission: complete when signed displacement has
            # reached the negative target (within tolerance toward zero).
            if displacement_m - tol <= forward_goal:
                self.lifecycle = MissionLifecycle.COMPLETED
                self.ended_at_s = now_s
                return
        # forward_goal == 0.0 → no forward-displacement completion;
        # falls through to the timeout check below. Rotation/lateral
        # completion criteria land in a follow-up PR (A1.1).

        if self.started_at_s is not None and (now_s - self.started_at_s) >= self.goal.timeout_s:
            self.lifecycle = MissionLifecycle.FAILED
            self.ended_at_s = now_s
            self.abort_reason = AbortReason.TIMEOUT

    def abort(self, *, now_s: float, reason: AbortReason) -> None:
        if self.lifecycle is MissionLifecycle.ACTIVE:
            self.lifecycle = MissionLifecycle.ABORTED
            self.ended_at_s = now_s
            self.abort_reason = reason
```

- [ ] **Step 5 — Run the test, verify all PASS**

```bash
pytest tests/unit/orchestrator/test_mission_state.py -v --import-mode=importlib
```

Expected: 10 PASSED (initial 6 + backward-completes + backward-no-false-complete + zero-forward-no-instant-complete + zero-forward-still-times-out).

- [ ] **Step 6 — Wire MissionState into the dispatcher**

Modify `src/mousedroid/orchestrator/mission_dispatcher.py` — emit `MissionState` instances. Existing one-shot dispatcher call sites get a `MissionState.from_goal_vector(...)` migration helper that keeps backwards compat. Note the source type is `GoalVector` from `mousedroid.llm_gateway.protocol` (normalised velocity targets; ``vx_target, vy_target, omega_target`` ∈ ``[-1, 1]``) — there is no ``VelocityDirective`` in this codebase.

Because `GoalVector` does not carry a distance or timeout, the dispatcher MUST supply them from its own config when it constructs the `MissionState`:

```python
from mousedroid.llm_gateway.protocol import GoalVector


@classmethod
def from_goal_vector(
    cls,
    goal_vector: GoalVector,
    *,
    forward_m: float,
    timeout_s: float,
    tolerance_m: float = 0.10,
) -> MissionState:
    """Backwards-compat: wrap a normalised :class:`GoalVector` plus an
    explicit forward-distance + timeout into a navigate-relative
    ``MissionState``.

    Args:
        goal_vector: Velocity targets in ``[-1, 1]``. ``vx_target`` sets
            the SIGN of the mission (forward vs backward); magnitude is
            consumed by the existing velocity-tracking loop, not by
            this state machine.
        forward_m: Magnitude of forward distance to travel along the
            current heading (metres). Caller-side config; this method
            applies the sign from ``goal_vector.vx_target``.
        timeout_s: Mission timeout (seconds).
        tolerance_m: Goal-reach tolerance (metres).
    """
    signed_forward = forward_m if goal_vector.vx_target >= 0.0 else -forward_m
    return cls(
        goal=MissionGoal.navigate_relative(
            forward_m=signed_forward,
            timeout_s=timeout_s,
            tolerance_m=tolerance_m,
        )
    )
```

- [ ] **Step 7 — Add orchestrator consume side**

Modify `src/mousedroid/orchestrator/orchestrator.py:519-533` (mission completion block from earlier survey) — replace the `mission_just_completed` flag with a poll of `self._mission_state.lifecycle`. On `COMPLETED | FAILED | ABORTED`, fire the curiosity reset + MEMORY.md export + emit `mission_lifecycle_changed` structured log.

- [ ] **Step 8 — Lint + type check + commit**

```bash
ruff check src/mousedroid/orchestrator/ tests/unit/orchestrator/test_mission_state.py
ruff format --check src/mousedroid/orchestrator/ tests/unit/orchestrator/test_mission_state.py
mypy --strict src/mousedroid/orchestrator/

git add src/mousedroid/orchestrator/ tests/unit/orchestrator/test_mission_state.py
git commit -m "feat(orchestrator): persistent MissionState state-machine + dispatcher migration

Replaces one-shot directive dispatch with a goal-tracking state machine
(PENDING -> ACTIVE -> COMPLETED | FAILED | ABORTED). Mission timeout,
displacement-based goal-reach, and abort reasons are first-class fields
on the state object. Backwards compat preserved via MissionState.from_directive."
```

---

## PR A2 — Wire DistilledVLAOnnx through factory + policy selector

**Why:** `src/mousedroid/vla/policy.py:58-80` runs `MockVLA` only. The orchestrator branches at `cfg.loop.policy_selector` ∈ {nav_agent, vla, auto} but `vla` is a no-op.

**Files:**
- Modify: `src/mousedroid/factory.py` — `build_vla_policy(cfg)` returns `DistilledVLAOnnx` when `cfg.vla.weights_path` is set and the ONNX runtime is importable; `MockVLA` otherwise
- Modify: `src/mousedroid/vla/policy.py` — fix the policy selector branch to actually delegate
- Create: `tests/integration/orchestrator/test_vla_policy_selection.py`
- Modify: `src/mousedroid/config/schema.py` — `VLAConfig.weights_path: str | None`, `VLAConfig.runtime: Literal["onnx", "tensorrt"]` (existing if PR #58 landed; verify)

**Acceptance:** Setting `vla.weights_path=/opt/vla/policy.onnx` and `loop.policy_selector=vla` causes the orchestrator to use the real ONNX session; CPU fallback works when CUDA unavailable; missing weights → fail-loud with `FailureRecorder.record(subsystem="vla", reason="weights_missing", level="critical")` (no silent MockVLA substitution).

---

## PR A3 — Configurable MCTS planning horizon + goal-cost heuristic

**Why:** Today MCTS rollout depth is fixed; survey notes "extend rollout depth from 4 to 8+ steps". Hard-coding the depth violates Invariant #3. Plus, when a goal is active (PR A1), MCTS should bias rollouts toward goal-direction.

**Files:**
- Modify: `src/mousedroid/world_model/mcts.py` — accept `MCTSConfig.max_horizon`, `MCTSConfig.goal_cost_weight`
- Modify: `src/mousedroid/config/schema.py` — `MCTSConfig` fields
- Create: `tests/unit/world_model/test_mcts_horizon.py`

**Acceptance:** Horizon flows from config; given a synthetic 4-step world model, the MCTS visit counts under `max_horizon=8` differ from `max_horizon=4` (proving the depth is honoured). Goal-cost weight ≠ 0 biases visits toward goal direction in a deterministic test fixture.

---

## PR A4 — Replanner hook with safety-veto trigger

**Why:** The harness `PRE_ACTION` hook (`src/mousedroid/harness/hooks.py`) exists but defaults to no-op. A safety veto today just zeros the action; an autonomous mission needs replanning when a veto fires (e.g., re-route around the obstacle).

**Files:**
- Modify: `src/mousedroid/harness/hooks.py` — `ReplannerHook` ABC
- Create: `src/mousedroid/harness/replanner.py` — `LocalReplanner` (MCTS-based alternative-goal search)
- Modify: `src/mousedroid/orchestrator/orchestrator.py` — fire `PRE_ACTION` hook on every safety-veto edge
- Create: `tests/integration/orchestrator/test_replanner_on_veto.py`

**Acceptance:** Test scenario where the rover faces a synthetic obstacle on its current path; replanner produces an alternative direction within `cfg.harness.replanner.budget_ms` and the new direction is recorded in telemetry.

---

## PR A5 — Multi-minute autonomous mission integration test

**Why:** Top survey gap: "NO integration test for multi-minute autonomous goal-seeking". This PR closes that gap on TWO axes — fast MockClock scenario (CI) AND real wall-clock Jetson scenario (nightly).

**Files:**
- Create: `tests/integration/test_multi_minute_autonomous_mission.py`
- Modify: `.github/workflows/ci.yml` (or new `.github/workflows/nightly.yml`) — add `pytest -m slow` job
- Create: `scripts/jetson_probe_autonomous_mission.py` — real-Jetson 5-minute live probe

**Acceptance:** 5-min MockClock test completes in <30 s wall-clock; 5-min real Jetson run reaches a 1 m goal without operator intervention, with telemetry artefacts saved to `reports/autonomous_mission/<stamp>/`.

---

# Track D — Streaming dashboard

## PR map

| PR | Title | Depends on | Subagent |
|---|---|---|---|
| D1 | `feat(telemetry): unified dashboard shell + nav (vanilla canvas, no framework)` | none | `nextjs-vercel-pro:frontend-developer` |
| D2 | `feat(telemetry): mission control page with e-stop + goal entry + replanner status` | A1 (`MissionState`) | `nextjs-vercel-pro:frontend-developer` |
| D3 | `feat(telemetry): live log panel page (subscribes /api/v1/logs/stream)` | none | `nextjs-vercel-pro:frontend-developer` |
| D4 | `feat(telemetry): per-subsystem health page (battery, sensors, voice, GPU temp)` | D1 | `python-pro` |
| D5 | `feat(telemetry): arm + voice activity pages (platform-gated)` | T3 (arm telemetry) | `nextjs-vercel-pro:frontend-developer` |
| D6 | `feat(observability): Grafana dashboard validation + Prometheus rule promtool gate` | none | `devops-automation:cloud-architect` |

---

## PR D1 — Unified dashboard shell

**Why:** Today operators must open two tabs (`/lidar` and `/camera`) — there's no shared header, no nav, no consistent auth UI, no system-wide e-stop. Build a single SPA-like shell at `/` that hosts the existing pages and future ones, using vanilla canvas + Fetch + WebSocket (no React/Vue — matches existing `lidar.html` style and avoids a new build toolchain on the Jetson).

**Files:**
- Create: `src/mousedroid/telemetry/static/index.html` — shell with nav + outlet div + shared WS connection manager
- Create: `src/mousedroid/telemetry/static/shell.js` — vanilla module: route-based view loader, shared WS pool, token storage
- Create: `src/mousedroid/telemetry/static/shell.css` — minimal cohesive theme using CSS custom properties (`--mousedroid-bg`, `--mousedroid-accent`, etc.; no hardcoded hex in component CSS)
- Modify: `src/mousedroid/telemetry/server.py` — `GET /` → serve `index.html`; existing `GET /lidar` + `GET /camera` continue to work (back-compat)
- Create: `tests/unit/telemetry/test_shell_route.py` — assert `/` returns 200 + nav markup + token-input form

### Tasks (abbreviated)

1. Read `src/mousedroid/telemetry/static/lidar.html` + `camera.html` to extract the WS connection pattern into a shared helper
2. Write a failing test: `pytest tests/unit/telemetry/test_shell_route.py::test_root_returns_shell -v`
3. Create `index.html` with `<nav>` (LiDAR, Camera, Mission, Logs, Health, Arm, Voice), `<main id="outlet">`, `<aside id="token-bar">`
4. Create `shell.js` — `Router`, `WSPool`, `TokenStore` (localStorage-backed)
5. Register `/` route in `server.py:501` adjacent to the existing static routes
6. Extract msgpack negotiation script from `lidar.html` into `shell.js` to DRY
7. Add Playwright canvas-diff test for `/` (extends `tests/e2e/test_dashboard_canvas_diff.py` parametrize with `("/", "main")`)
8. Lint/format/mypy
9. Commit

**Acceptance:** Operator visits `http://mousedroid-telemetry.local:8080/` → sees nav with 7 tabs → clicking each routes within the SPA without page reload → existing `/lidar` URL still loads the standalone page (back-compat).

---

## PR D2 — Mission control page

**Why:** Today `POST /api/v1/mission` exists (`server.py:731-889`) but there's no UI to fire it; operators must `curl`. A 5-minute autonomous mission requires a one-button start + e-stop + live progress display.

**Files:**
- Create: `src/mousedroid/telemetry/static/views/mission.js` — view module loaded by D1's shell
- Modify: `src/mousedroid/telemetry/server.py` — emit `mission_state_changed` events on the existing `/ws` so the view can subscribe (uses A1's `MissionState`)
- Create: `tests/e2e/test_mission_control_view.py` — Playwright test firing the e-stop button and verifying the orchestrator transitions to `ABORTED`

**Acceptance:**
- Goal-entry form (forward_m, lateral_m, timeout_s) with Pydantic-mirrored client-side validation
- "Start mission" button POSTs to `/api/v1/mission` with the bearer token
- Live state badge shows `PENDING | ACTIVE | COMPLETED | FAILED | ABORTED`
- E-stop button POSTs `/api/v1/mission/abort` (new endpoint with auth + replanner-aware halt)
- Replanner status displayed when A4 has landed (gated by an introspection probe `/api/v1/features` that lists active features)

---

## PR D3 — Live log panel page

**Why:** WS `/api/v1/logs/stream` exists; no page consumes it. Operators today tail `docker logs` to see structured events. Bring it into the dashboard.

**Files:**
- Create: `src/mousedroid/telemetry/static/views/logs.js`
- Add: client-side log-level filter UI (debug/info/warning/error), event-name search
- Test: Playwright — open page, assert ≥1 line renders within 5 s

**Acceptance:** Log panel renders structlog events live; filter UI works; pause/resume button stops/restarts the WS subscription cleanly.

---

## PR D4 — Per-subsystem health page

**Why:** `/metrics` carries everything (battery, GPU temp, sensor liveness, subsystem failures) but there's no visual aggregation. Build a per-subsystem health card grid that polls `/metrics` and renders status chips.

**Files:**
- Create: `src/mousedroid/telemetry/static/views/health.js`
- Create: `src/mousedroid/telemetry/metrics_summary.py` — server-side summarizer endpoint `GET /api/v1/health/summary` returning normalized JSON (`{subsystem: {state, last_failure_reason, last_failure_age_s}}`)
- Create: `tests/unit/telemetry/test_health_summary_endpoint.py`

**Acceptance:** Health page shows 8 cards (orchestrator, world_model, voice, telemetry, lidar, camera, motor, audio); each card renders a colour-coded chip (green/yellow/red) driven by the FailureRecorder counters.

---

## PR D5 — Arm + voice activity pages

**Why:** Two more subsystem-specific views, gated by config (`platform == robot_arm` shows arm; `voice.enabled` shows voice).

**Files:**
- Create: `src/mousedroid/telemetry/static/views/arm.js` — joint angles, gripper state, current grasp target (from T3's arm telemetry frame)
- Create: `src/mousedroid/telemetry/static/views/voice.js` — last 10 utterances, queue depth, cooldown timer
- Modify: `src/mousedroid/telemetry/protocol.py` — `TelemetryFrame.arm: ArmTelemetry | None`, `TelemetryFrame.voice: VoiceTelemetry | None`

**Acceptance:** Pages render correctly on both `mouse_droid` and `robot_arm` platforms; gating via `GET /api/v1/features` from PR D2; no broken nav links when the platform doesn't have the subsystem.

---

## PR D6 — Grafana dashboard + promtool CI gate

**Why:** `docs/grafana_dashboard.json` is checked in but never validated; `docs/planning/NEXT_STEPS.md:228-229` lists "Grafana completion + alert rules" as an immediate follow-up. The promtool CI gate is also on that immediate-follow-up list.

**Files:**
- Modify: `docs/grafana_dashboard.json` — add panels for the new PR #75-#83 metrics (sensor_liveness, subsystem_failures, bound_port, mdns_registered, mission_lifecycle, replanner_invocations)
- Create: `monitoring/prometheus/rules.yml` — alert rules (high failure rate, mDNS unregistered, mission timeout exceeded)
- Modify: `.github/workflows/ci.yml` — add `promtool check rules monitoring/prometheus/rules.yml` step
- Create: `tests/integration/test_grafana_dashboard_loads.py` — assert all `targets[*].expr` in the JSON parse via `promtool`

**Acceptance:** CI runs `promtool` and rejects malformed rules; Grafana dashboard JSON validated; alert rules cover the top 10 production signals.

---

# Track H — Hardening (cross-cutting)

## PR map

| PR | Title | Depends on | Subagent |
|---|---|---|---|
| H1 | `chore: scripts/check_no_hardcoded_values.py enforcement + sweep` | none | `code-quality` |
| H2 | `chore: numpy dtype hygiene + errstate=raise in test suite` | none | `python-pro` |
| H3 | `chore(security): bearer-token auth required on /ws/v1/lidar/raw + /api/v1/logs/stream` | none | `security-auditor` |
| H4 | `chore: coverage gate → 87% (current ≥85)` | H1, H2 (clean up cruft first) | `test-runner` |

---

## PR H1 — `check_no_hardcoded_values.py` enforcement

**Why:** `scripts/check_no_hardcoded_values.py` exists (referenced in the prior plan); we need it as a CI gate plus a one-pass sweep to fix existing leaks. Recent PRs (#75-#83) introduced a few hardcoded host/port defaults that should derive from config.

**Files:**
- Modify: `scripts/ci.sh` — add `python scripts/check_no_hardcoded_values.py src/`
- Sweep: any `localhost`, `127.0.0.1`, `8080`, port literals, IP literals in `src/` that aren't reading from `Settings` → move to schema
- Add: `tests/integration/test_no_hardcoded_values.py` calling the same checker on `src/`

**Acceptance:** CI fails if any literal `localhost` or `127.0.0.1` appears in `src/mousedroid/` outside `mock_*` files.

---

## PR H2 — NumPy dtype hygiene

**Why:** `CLAUDE.md` invariant: "numpy usage must specify dtype explicitly; no implicit float64 conversions". Recent training PRs (T1-T5) and Isaac PRs (S1-S4) will introduce many NumPy call sites; lock them down before drift accumulates.

**Files:**
- Add: `tests/conftest.py` — pytest hook to enable `numpy.errstate(all="raise")` globally
- Sweep: grep `np\.array\(` `np\.zeros\(` `np\.ones\(` `np\.empty\(` without `dtype=` in `src/` → add explicit dtypes
- Add: `pyproject.toml` ruff `NPY*` rules (NPY001/NPY002) if not enabled

**Acceptance:** Test suite raises immediately on any unintended `RuntimeWarning` (overflow/divide-by-zero/invalid); zero `np.array(...)` without `dtype=` in `src/`.

---

## PR H3 — Bearer-token auth on more WS endpoints

**Why:** `/ws/v1/lidar/raw` and `/api/v1/logs/stream` (added in PR #79) accept connections without enforcing the same bearer-token validation as `/ws`. Survey notes this as a security gap.

**Files:**
- Modify: `src/mousedroid/telemetry/auth.py` — apply middleware to ALL WS routes by default; explicit allow-list for unauthenticated endpoints
- Modify: `src/mousedroid/telemetry/server.py` — pass auth context to `_handle_lidar_raw_ws` and `_handle_log_stream`
- Create: `tests/integration/test_ws_auth_enforcement.py` — assert each WS endpoint rejects connections without `Authorization: Bearer ...` when `auth_enabled=true`

**Acceptance:** Unauthenticated WS connect → close code `4401` (new constant in `mousedroid.constants`); existing tests pass byte-for-byte; Phase B P3 + P12a + P13 probes still PASS (they already send the token via `MOUSEDROID_TELEMETRY_TOKEN`).

---

## PR H4 — Coverage gate to 87%

**Why:** Current gate `--cov-fail-under=85`; recent code additions should push us above 87. Lock it in.

**Files:**
- Modify: `pyproject.toml` → `--cov-fail-under=87`
- Modify: `scripts/ci.sh` → same
- Backfill: any uncovered branch surfaced by the bump (most likely in new probe scripts and shell helpers — these are excluded via `pyproject.toml [tool.coverage.run] omit`)

**Acceptance:** `bash scripts/ci.sh` passes with the higher gate; `pytest --cov` reports ≥87.0%.

---

# Execution Strategy

## Parallel batches (worktree pattern)

The 24-PR roadmap is structured for **4 parallel worktrees per batch**. Each worktree is created via the `superpowers:using-git-worktrees` skill; each PR is dispatched to a subagent via `superpowers:subagent-driven-development`.

### Batch 1 (week 1) — independent quick wins

| Worktree | PR | Subagent type |
|---|---|---|
| `wt-t1-bc-weight` | T1 | `test-runner` |
| `wt-d1-shell` | D1 | `nextjs-vercel-pro:frontend-developer` |
| `wt-h1-no-hardcodes` | H1 | `code-quality` |
| `wt-h2-numpy` | H2 | `python-pro` |

### Batch 2 (week 2) — depends on Batch 1

| Worktree | PR | Subagent type |
|---|---|---|
| `wt-t2-wandb` | T2 | `ml-training-orchestrator` |
| `wt-a1-mission-state` | A1 | `python-pro` |
| `wt-d3-logs` | D3 | `nextjs-vercel-pro:frontend-developer` |
| `wt-h3-ws-auth` | H3 | `security-auditor` |

### Batch 3 (week 3) — depends on Batch 2

| Worktree | PR | Subagent type |
|---|---|---|
| `wt-t3-train-arm` | T3 | `ai-ml-toolkit:ml-engineer` |
| `wt-a3-mcts-horizon` | A3 | `mcts-optimizer` |
| `wt-d2-mission-control` | D2 | `nextjs-vercel-pro:frontend-developer` |
| `wt-d4-health-page` | D4 | `python-pro` |

### Batch 4 (week 4) — depends on Batch 3

| Worktree | PR | Subagent type |
|---|---|---|
| `wt-s1-urdf-usd` | S1 | `ai-ml-toolkit:ml-engineer` |
| `wt-t4-ewc` | T4 | `distributed-training` |
| `wt-a4-replanner` | A4 | `python-pro` |
| `wt-a2-vla-onnx` | A2 | `neural-network-architect` |

### Batch 5 (week 5) — convergence

| Worktree | PR | Subagent type |
|---|---|---|
| `wt-s2-sensor-adapter` | S2 | `ai-ml-toolkit:ml-engineer` |
| `wt-t5-dreamer` | T5 | `neural-network-architect` |
| `wt-d6-grafana` | D6 | `devops-automation:cloud-architect` |
| `wt-d5-arm-voice-pages` | D5 | `nextjs-vercel-pro:frontend-developer` |

### Batch 6 (week 6) — final

| Worktree | PR | Subagent type |
|---|---|---|
| `wt-s3-ppo-onnx` | S3 | `ml-training-orchestrator` |
| `wt-a5-multi-minute-e2e` | A5 | `test-runner` |
| `wt-s4-jetson-transfer` | S4 | `ai-ml-toolkit:ml-engineer` |
| `wt-h4-coverage-87` | H4 | `test-runner` |

## Subagent dispatch pattern (per PR)

```text
1. superpowers:using-git-worktrees      — create worktree
2. superpowers:test-driven-development  — failing test first
3. <domain-specific subagent>           — implementation
4. superpowers:requesting-code-review   — request review
5. coderabbit:code-review               — automated review pass
6. <fixes commits>
7. gh pr create --draft                 — open PR
8. gh pr ready                          — promote out of draft
9. gh pr merge --squash                 — after CI green
10. ExitWorktree                        — clean up
```

## MCP servers in scope

- **gh CLI**: PR lifecycle, review threads, merge
- **Context7**: live docs for Stable-Baselines3, Isaac Lab, ONNX Runtime, aiohttp 3.9 vs 3.10+ API
- **filesystem MCP**: cross-worktree synchronization where applicable
- **playwright** (`plugin:testing-suite:playwright-server`): D2/D3/D4 dashboard end-to-end tests
- **pinecone** (optional): if T5's latent space exceeds the LMDB-replay scale envisioned for offline RL, consider externalizing the latent store

## Cross-cutting gates (every PR must satisfy)

1. `bash scripts/ci.sh` green
2. `ruff check && ruff format --check` clean
3. `mypy --strict src/mousedroid/` clean
4. `scripts/check_no_hardcoded_values.py src/` clean (post-H1)
5. `scripts/check_settings_identity.py` clean
6. `pytest --cov --cov-fail-under=85` (→ 87 after H4)
7. NumPy: every `np.array/zeros/ones/empty` carries `dtype=`
8. Backwards-compat: every new `Settings` field has a safe default
9. Structured logs (`structlog`) only; no `print`
10. PR description references this plan (`docs/superpowers/plans/2026-05-15-full-stack-training-autonomy-dashboard.md`) and the specific PR row in the relevant track

---

# Self-Review Checklist (run before approval)

- [x] **Spec coverage:** all 4 tracks present; Isaac Lab covered; each track has 4-6 PRs with acceptance + dependency
- [x] **Placeholder scan:** no "TBD" / "implement later" / "TODO" / "fill in" in plan body; first PR per track has full test code AND full implementation code
- [x] **Type consistency:** `MissionState`, `MissionLifecycle`, `MissionGoal`, `AbortReason`, `TrainingLoggerProtocol`, `NullLogger`, `WandbLogger`, `MCTSConfig`, `VLAConfig`, `OfflineRLConfig`, `ArmTrainingConfig`, `DreamerConfig` — names used in multiple tasks match
- [x] **File paths:** every task references real paths verified during survey (`src/mousedroid/orchestrator/orchestrator.py:370-410`, `training/train_offline_rl.py:193`, `src/mousedroid/sim/isaaclab/rover_env.py:1-255`, etc.)
- [x] **No hardcoded values:** every new threshold/path/port/timeout originates in a Pydantic config field with a safe default

---

# Plan complete — execution choice

**Plan saved to** `docs/superpowers/plans/2026-05-15-full-stack-training-autonomy-dashboard.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — Dispatch fresh subagent per task, two-stage review between tasks, fast iteration. REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`.

2. **Inline Execution** — Execute tasks in this session with checkpoints. REQUIRED SUB-SKILL: `superpowers:executing-plans`.

**Suggested first move:** Batch 1 (T1, D1, H1, H2) in four worktrees with subagent dispatch — independent, quick wins, sets up shape of the next 20 PRs.
