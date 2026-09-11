# Tasks — Isaac Lab sensor-parity

Quality gate:

```
python -m ruff check src/mousedroid/sim src/mousedroid/factory/world_model.py src/mousedroid/training/pipeline_orchestrator.py tests/regression/test_f043_aqa.py
bash scripts/validations/F-043.sh
```

**Phase 1 — Schema + catalog (F-043)**

- [x] 1.1 Nested `RoverIsaacSimConfig` + shared `battery_voltage_const_v`.
- [x] 1.2 `features.yaml` F-043–F-049; F-043 `in_progress` while coding.
- [x] 1.3 Factory docstring; scene helpers; F-043.sh `passed>0`.

**Phase 2 — Readers + seams (F-044)**

- [x] 2.1 Fake-injectable IMU/pose/lidar readers.
- [x] 2.2 `to_body_action` + `vx_body_mps` / `omega_rads`.

**Phase 3 — Domain randomization (F-045)**

- [x] 3.1 `apply_domain_params` on Isaac; Replicator behind the translator.

**Phase 4 — RSSM gates (F-046)**

- [x] 4.1 Lift `_train_rssm` to `{mujoco, isaac_lab}`; vision stays mujoco.

**Phase 5 — Docs (F-048)**

- [x] 5.1 HARNESS_SPEC / C4 / ADR-009; no Prometheus Isaac family.

**Deferred**

- [ ] F-047 PPO ONNX artifact (never hot-load).
- [ ] F-049 Cosmos / MobilityGen.
