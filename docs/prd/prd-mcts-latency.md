# PRD: MCTS Search Latency Optimisation

**Epic:** E1 — MCTS Latency  
**Feature Slug:** `mcts-latency`  
**Status:** Draft  
**Date:** 2026-03-14

---

## User Story

> **As** the MouseDroid robot's planning loop,  
> **I want** the MCTS planner to select an action within 50 ms at p50 latency,  
> **So that** the 10 Hz planning cadence (`loop.planning_hz`) is maintained without dropping frames or delaying motor commands.

---

## Background

The current `MCTSPlanner.plan()` runs all `n_simulations_base` (50) iterations unconditionally on every call. Each iteration calls `imagine_step()` (the RSSM world model) up to `n_action_candidates` × `rollout_depth` times. The UCB tuning run recorded a p50 of **219 ms** (4.4× above the 50 ms budget at `planning_hz = 10`).

---

## Acceptance Criteria

### AC-1: Early-Exit Convergence
**Given** the tree value estimate has converged (best child value change < `early_exit_value_threshold` for `early_exit_patience` consecutive simulations),  
**When** `MCTSPlanner.plan()` is called,  
**Then** search must stop early and return the best action without running all `n_simulations_base` iterations.

- `early_exit_value_threshold: float = 0.0` in `MCTSConfig` (0.0 = disabled, backward-compat)
- `early_exit_patience: int = 3` in `MCTSConfig`

### AC-2: Multi-Dimensional Action Diversity
**Given** `n_action_candidates` is 9 and `action_dim` is 3,  
**When** `_generate_candidate_actions()` is called,  
**Then** it must return a `(9, 3)` tensor where each action is a _different_ sample from a stratified distribution across all action dimensions — not a 1-D linspace broadcast.

### AC-3: Time-Budget Adaptive Simulation
**Given** `simulation_budget_ms > 0.0` is set in `MCTSConfig`,  
**When** cumulative search time exceeds `simulation_budget_ms`,  
**Then** `plan()` must stop and return the best action found so far.

- `simulation_budget_ms: float = 0.0` in `MCTSConfig` (0.0 = disabled)
- Clock source must be injectable for deterministic testing

### AC-4: Tree Warm-Start (Optional)
**Given** `reuse_tree: bool = False` is enabled in `MCTSConfig`,  
**When** `plan()` is called on consecutive timesteps with similar `(h, z)` latent states,  
**Then** the root may be initialised from the best subtree of the previous plan instead of a fresh root.

- Default `false` — must not change existing behaviour when disabled
- Must be safe with different `(h, z)` shapes or sudden belief divergence (fallback to fresh root)

### AC-5: Benchmark Gate
**Given** the CI pipeline runs on every PR touching `mcts.py` or `MCTSConfig`,  
**When** `pytest-benchmark` runs `MCTSPlanner.plan()` with CPU-only mocked `imagine_step()`,  
**Then** the p50 latency must be ≤ 50 ms **and** no regression > 10% vs the stored baseline.

---

## Out of Scope

- GPU-accelerated MCTS (future, tracked separately)
- UCB formula changes (UCB-V, PUCT)
- Parallel/async tree search

---

## Success Metrics

| Metric | Current | Target |
|---|---|---|
| `plan()` p50 latency (CPU) | 219 ms | ≤ 50 ms |
| `plan()` p95 latency (CPU) | 265 ms | ≤ 80 ms |
| Mean episode reward | 0.1479 | ≥ 0.20 (diversity fix expected to improve) |
| `pytest-benchmark` regression gate | None | < 10% regression vs baseline |

---

## Open Questions

1. Should `early_exit_patience` be configurable or private?
2. Should `simulation_budget_ms` clock use `time.monotonic` or an injectable `ClockProtocol`?
3. Is tree reuse safe across BDI belief-state resets (e.g., after emergency stops)?
