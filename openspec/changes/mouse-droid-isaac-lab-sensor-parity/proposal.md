# Proposal — Isaac Lab sensor-parity workstation harness

- change_id: mouse-droid-isaac-lab-sensor-parity
- project: mouse-droid
- status: active
- feature_id: F-043
- epic: Isaac Lab workstation
- owner: ianshank
- created: 2026-09-11
- rev: A

## Why

Isaac Lab already has a protocol, factory backend, and env stub, but RSSM
pretraining is gated mujoco-only and Isaac `step` does not emit the
`vx_body_mps` / `omega_rads` info `RoverObsAdapter` needs. Hardcoded device
and prim strings also sit outside schema. This change catalogues F-043–F-049
and lands the first slice: nested schema, sensor/info/DR seams, lifted RSSM
gates. PPO/ONNX hot-load and Cosmos stay deferred.

## What Changes

- Nested `RoverIsaacSimConfig` on `RoverSimConfig` (`src/mousedroid/config/schema/sim.py`).
- Fake-injectable readers + `to_body_action` on `src/mousedroid/sim/isaaclab/rover_env.py`.
- `apply_domain_params` via `src/mousedroid/sim/isaaclab/randomization.py`.
- `src/mousedroid/factory/world_model.py` and
  `src/mousedroid/training/pipeline_orchestrator.py` allow `{mujoco, isaac_lab}`.
- Catalog in `features.yaml`; docs in `HARNESS_SPEC.md`,
  `docs/architecture/c4-rssm-sim-pretraining.md`,
  `docs/architecture/ADR-009-isaac-lab-phase-b.md`.

## Impact

Default `rover.sim.backend: mock` still skips RSSM. No `config/*.yaml` keys
added. Runtime `Settings.harness` stays None. No Prometheus Isaac family.
PPO is not loaded into the 30 Hz loop.

## Charter

No CHARTER §3 carve-out for F-043–F-046/F-048 (off-loop, YAML/env only).
F-047 would need one if it ever hot-loaded PPO.

## Validation

`bash scripts/validations/F-043.sh` (and F-044/F-045/F-046/F-048). Each
script asserts pytest `passed > 0`.
