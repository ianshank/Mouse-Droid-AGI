# Proposal — Jetson ONNX runtime evidence + PC-to-rover delivery

- change_id: mouse-droid-jetson-onnx-delivery
- project: mouse-droid
- status: active
- feature_id: F-050, F-051
- epic: Jetson deployment
- owner: ianshank
- created: 2026-09-19
- basis_commit: 18aba56
- rev: C (revision of the external plan `jetson-onnx-runtime-and-pc-delivery`, rev A; three review rounds recorded in `peer-review.md`)

## Why

Rev A of this plan proposed optimizing `DualStreamRSSM.observe_step` with ONNX Runtime
I/O binding, FP16, TensorRT engine caches and CUDA Graph, then promoting it to the rover
through a new release/symlink deployment scheme. Peer review against the tree
(`peer-review.md`) found the direction sound and five load-bearing premises false. Rev C
keeps the architecture and re-sequences the work behind what the repository can actually
prove.

**Round 3 added the fact that should have come first: this decision already has a
precedent in this repository, and it went the other way.**
`docs/analysis/alayaworld-distillation-spike.md:65-75` is a ratified three-part GO rubric for
accelerating this exact world model — a ≥3× p95 primitive speedup on Jetson, ≥0.90 action
agreement with a trained-checkpoint teacher, and a written consumer case — and its verdict for
a sibling optimization was **DEFER**, on the consumer case alone: "This alone justifies DEFER
over ADOPT regardless of accuracy." The Amdahl bound behind that is computed, verified against
the planner, and printed to the operator by `scripts/spike_step_distillation.py:57-61`:
`plan()` makes ~500-650 `imagine_step` calls, rollouts are ~40% of them, so end-to-end planner
gain caps at ~1.25-1.6× *regardless of the primitive speedup*. This change targets
`observe_step` — **one** call per tick, a strictly smaller share, so a strictly worse ceiling.
That number is a desk calculation, and rev C computes it before Phase 2 rather than after a
rover campaign.

### The ceiling, computed — and the gate it closes

`scripts/analyze_observe_step_ceiling.py` derives it from `MCTSConfig` and `LoopConfig` with no
literals, and reproduces the spike's two published constants exactly (`"500-650"`, `"~40%"` at
`scripts/spike_step_distillation.py:59-60`) as its correctness anchor:

| quantity | value |
| --- | --- |
| `imagine_step` calls per `plan()` at `n_simulations_base=50` | **259 - 628** |
| of which the rollout leg | 250 calls = **39.8% - 96.5%** |
| `observe_step` : `imagine_step` calls per tick | **1 : 259** to **1 : 628** |
| equal-per-call-cost share of the tick's RSSM work | **0.159% - 0.385%** |
| end-to-end ceiling at that share | **1.0016x - 1.0039x** |
| share required to reach even 1.25x end-to-end | **>= 20% of tick time**, at any primitive speedup |

The derivation is `total_min = n_action_candidates + n_simulations * rollout_depth`,
`total_max = total_min + max(0, n_simulations - n_action_candidates) * n_action_candidates`. The
`0` lower bound on re-expansions is proved from `mcts.py`: `_ucb1` returns `+inf` for an unvisited
node (`:223-224`) and `_select_child` keeps the *first* strictly-better score (`:238-244`), so the
first `min(budget, n_action_candidates)` simulations each land on a distinct unvisited root child
and never reach the re-expansion at `:318-319`.

**So the gate at task 2.1 closes: this does not justify I/O binding, FP16 and a TensorRT cache.**
Against this repository's own bar — 1.25x, the *low* end of the sibling rollout leg's ceiling —
`observe_step` would have to occupy at least a fifth of every tick. The structural reference point
is two orders of magnitude below that. Per the task's own instruction ("Concluding 'not worth it'
on this calculation is a successful outcome, not a failed one"), rev C therefore lands **Phases
2.4-2.6, 3, 4, 7 and 8 only**, and Phases 5 and 6 — the rover baseline campaign and the
provider-options / I/O-binding / FP16 work — are not built.

Two honest qualifications, both recorded as `unknown_inputs` in the analyzer's report:

1. **The conclusion is conditional, not measured.** No measured `observe_step` share exists in
   this repository — `reports/` and `smoke-reports/` still carry zero `tick_phase` or
   `observe_step` values. The 0.159%-0.385% band is a structural *equal-per-call-cost* reference
   point, not a measurement, and the analyzer exits refusing to conclude unless a share is
   supplied (`--observe-share`). What kills the optimization is not the exact share but the
   distance: the required share is ~52x-126x the reference point.
2. **This is why Phase 3 still lands.** The instrumentation is what would make the number
   measurable rather than derived, and it is worth having on its own terms — the
   `mousedroid_world_model_observe_step_seconds` family has had no production writer and
   `WorldModelObserveStepLatencyHigh` could not fire. Closing the optimization gate does not
   close the observability gap; it is the reason to fix it cheaply instead of expensively.

Four facts reorder everything:

1. **The stage is unmeasured.** `reports/` holds three files and `smoke-reports/` two;
   none carries an `observe_step`, `tick_phase` or `world_model` measurement. The only
   on-device report (`smoke-reports/smoke_report.md`, 2026-05-12) records
   `telemetry down`. Rev A's own risk row — "observe-step optimization does not improve
   the whole loop" — is unfalsifiable today.
2. **The metric that would measure it is dead.** `build_world_model(cfg)`
   (`factory/world_model.py:55`) has no metrics parameter, and `_build_onnx_world_model`
   constructs `DualStreamRSSMOnnx` at `:262-266` without `metrics=`. `observe_step`
   records only `if self._metrics is not None`
   (`dual_stream_rssm_onnx.py:271-279`), so `mousedroid_world_model_observe_step_seconds`
   has no production writer and `WorldModelObserveStepLatencyHigh`
   (`config/prometheus/alerts.yml:388-421`) cannot fire. Where it does fire — in
   `tests/performance/test_observe_step_budget.py` — the timer brackets `session.run`
   only (`:255-258`), excluding both `pack_observation` and the NumPy round-trip that
   I/O binding exists to remove.
3. **The model is not worth optimizing yet.** `build_world_model` returns a
   freshly-constructed module — no `load_state_dict` anywhere in it or in `orchestrator/`,
   and `factory/telemetry.py:315` `build_weight_update_loader` returns `None`
   unconditionally. So the rover runs **random weights**, which makes a numerical-parity gate
   vacuous and the rubric's accuracy criterion unattainable. And production does not even
   build the architecture under optimization: `config/jetson_production.yaml` has no `model:`
   block, so `cfc_hidden_dim` is `0` and `factory/world_model.py:91` falls through to plain
   `RSSM`, not `DualStreamRSSM`. A torch-vs-ONNX A/B would compare two different models.
4. **The path is not enabled on the rover.** No `config/*.yaml` sets a `world_model:`
   block, so `engine` sits at its `"torch"` default. Worse, `engine: onnx_trt` cannot be
   switched on from the overlay ADR-008 documents: `_build_onnx_world_model` raises when
   `cfg.model.cfc_hidden_dim <= 0` (`factory/world_model.py:246-253`), the schema default
   is `0` (`world_model.py:205`), and only `config/jetson_dual_stream.yaml:29` sets `64`.

So rev A would have optimized a code path the rover does not execute, measured by a metric
that never emits, against a baseline that does not exist. Rev C makes instrumentation and
enablement the price of admission, and makes the FP16 / I/O-binding / CUDA-Graph work
conditional on evidence that `world_model` is in fact the dominant tick phase.

Two further gaps justify the change independently of any speedup. ADR-008's own
"Negative" section records that the factory cannot detect a stale `.onnx` against
retrained weights — "wrong weights = wrong inference, silently" — and there is still no
digest check on the download path (`factory/world_model.py` checks only
`model_path.is_file()`), while the OTA weight poller already has the precedent
(`_registry_cloud.py:227`, `inc_cloud_weight_update_sha256_mismatch`). And
`ianshank/mousedroid-dual-stream-rssm` holds only `.gitattributes` and `README.md` —
verified against the Hub — while ADR-008's migration step 2 invites operators to leave
`onnx_path: null` and rely on it. That documented path fails closed at best and silently
at worst.

## What Changes

**Measurement and enablement (F-050, must land first)**

- `src/mousedroid/factory/world_model.py`: `build_world_model` gains a keyword-only
  `metrics: MetricsRegistry | None = None`, threaded to `DualStreamRSSMOnnx`. All four
  **six** call sites updated: `factory/orchestrator.py:121`,
  `factory/on_device_learning.py:102,274,304`, `factory/growth.py:121` and
  `validation/pillars.py:184`. The growth one matters — it is a
  `world_model if world_model is not None else build_world_model(cfg)` fallback, so it can
  construct the ONNX engine without a registry and silently reinstate the dead-metric bug.
  `factory/orchestrator.py` builds the registry before the world model instead of after.
- `src/mousedroid/world_model/dual_stream_rssm_onnx.py`: the existing
  `observe_world_model_observe_step_seconds` timer keeps its ORT-execute scope, and two
  new spans are added around `pack_observation` and the tensor conversions so the
  histogram family finally sums to wall-clock `observe_step`.
- `src/mousedroid/world_model/dual_stream_rssm.py`: the PyTorch engine emits the same
  observe-step histogram, so the two engines are comparable in production and not only in
  a Jetson-gated advisory test.
- `config/jetson_production.yaml`: `model.cfc_hidden_dim` set so `engine: onnx_trt` is
  reachable, or ADR-008 and `scripts/export_dual_stream_rssm_onnx.py:17-21` corrected to
  name `config/jetson_dual_stream.yaml`. Design D-3 picks the second.
- `src/mousedroid/config/schema/world_model.py`: typed ONNX runtime options — execution
  mode, strict-provider policy, device id, engine/timing cache toggles, context-memory
  switch, profile batch. No CUDA-graph switch (design D-12 removes it from scope).
  Precision, workspace size and TRT cache directory are **read from `cfg.jetson`**, not
  duplicated.
- **Opt-in is environment-based, not a tracked overlay.** Rev A proposed
  `config/jetson_onnx_fp16.yaml`; design D-7 forbids a `world_model:` key in any tracked
  YAML, and the two cannot both hold — `check_config_compat.py` would validate such a file
  against the pinned schema where `WorldModelConfig` is a plain `BaseModel`
  (`032942b…:src/mousedroid/config/schema.py:1597`, `extra="ignore"`) and silently drop
  every key. So activation is
  `MOUSEDROID_WORLD_MODEL__ENGINE=onnx_trt` plus the matching `MOUSEDROID_WORLD_MODEL__*`
  variables in `/etc/mousedroid/docker.env`, which is also the F-043 precedent ("Do not add
  isaac keys to config/default.yaml"). The runbook documents the variable set.
- `src/mousedroid/common/onnx_session.py`: `resolve_providers` and `warmup_session` widened
  to carry `(name, options)` provider tuples; a post-construction `session.get_providers()`
  comparison; `sess_options` passed explicitly. `vla/policy.py` and both ORT test stubs
  move with them.
- `src/mousedroid/telemetry/metrics/_registry_onnx_runtime.py` (new module, per ADR-017):
  `onnx_session_build_seconds`, `onnx_copy_seconds`, `onnx_provider_fallback` — registry
  field names without the `_total` suffix that `primitives.py:422-429` appends at render.
- `docs/architecture/ADR-019-onnx-runtime-provider-policy.md` + a row in
  `docs/architecture/adr-log.md`, superseding the ADR-008 clauses it changes.

**Out of scope by the task 2.1 gate.** The bullets above that belong to Phases 5-6 — the typed
ONNX runtime options, the `resolve_providers` widening, `_registry_onnx_runtime.py`,
ADR-019 and the rover baseline campaign — are **not built by this change**, because the computed
ceiling does not justify them (see "The ceiling, computed" above). They stay described here as
the design that *would* be correct if a measurement ever moves the share, and `design.md` keeps
their sections for that reason. What lands is instrumentation (Phase 3), narrative correction
(Phase 4), artifact integrity (Phase 7) and delivery hardening (Phase 8).

**Evidence (F-050)**

- `scripts/benchmark_latency.py` extended — not replaced — with an `observe_step` mode
  covering torch, `onnx_portable`, and (when present) `onnx_iobinding_*`, reusing its
  existing `--config`/`--checkpoint`/threshold/exit-code contract.
- `tests/performance/test_observe_step_budget.py` extended with the per-span breakdown,
  still budgeted by `MOUSEDROID_OBSERVE_STEP_BUDGET_MS`. No second source of truth for the
  10 ms / 33 ms numbers.
- Whole-tick evidence comes from the existing `mousedroid_tick_phase_ms{phase=...}`
  (`orchestrator.py:522`) and `mousedroid_tick_overruns_total`
  (`_registry_core.py:91`, written at `_telemetry_experience_mixin.py:114-126`). No new
  deadline counter.

**Delivery (F-051)**

- `scripts/deploy_remote.sh`: the `rsync -avz --delete` path (`:151`) gains a mandatory
  dirty-target refusal and a `rover/wip-<date>` preservation step, or is retired in favour
  of the git spine. It is the existing PC-side push command and rev A never named it.
- `docker-compose.jetson.yml`: a named volume for the TensorRT engine/timing cache, since
  nothing persists `HF_HOME` or a TRT cache today. The directory comes from schema config,
  not a literal.
- `scripts/docker_deploy.sh`: `health_check` gains a strict mode that fails on a dead
  telemetry endpoint, an unexpected ORT provider, or a model-digest mismatch. Today every
  leg is downgraded to a warning at `:241`.
- `deployments/jetson-image.json` extended in place per `config-compat.yml:8-10`, with the
  model digest and ORT provider recorded. No parallel manifest format.
- `docs/runbooks/jetson-onnx-benchmark.md`, `docs/runbooks/pc-to-jetson-promotion.md`, and
  rows in `docs/README.md`.

## What does NOT change

- The bind-mount-editable deployment model. `/opt/mousedroid:/opt/mousedroid`
  (`docker-compose.jetson.yml:80`) is a git checkout carrying an editable install and
  rover-local WIP commits. Rev A's `releases/<id>` + `current`/`previous` symlink scheme
  changes the identity of that path and breaks the compose mount, `sync_jetson_overlay.sh`,
  `jetson-nightly.yml:87` and `preflight_check.sh:29-31`. Rev C keeps the spine and uses
  the established rollback anchor — `docker tag mousedroid:jetson
  mousedroid:jetson-rollback-<date>` — which already rolls back offline without rebuilding.
- `imagine_step` stays PyTorch. `CompositeWorldModel` also delegates `get_safety_trace`
  to it (`composite.py:128-150`) — but round 2 found that method has **zero production
  callers**, and `factory/world_model.py:267-270`'s claim that "the safety monitor's CfC
  inspection" needs it is false (`safety/monitor.py` never references it). The MCTS
  rollout path is likewise reached only from `_try_cognitive_action`'s `except` branch.
  So the retained PyTorch engine serves one exception-path method and one dead one; both
  comments are corrected in Phase 3 rather than the engine being removed, because
  `imagine_step` on the degraded path is still real.
- No second direct-TensorRT runtime. ADR-008 already reserves `torch2trt` via
  `JetsonTensorRTCompiler` as a separate future option.
- No INT8. No Cosmos — already catalogued as F-049 `deferred`. No Isaac ROS. No stereo
  pipeline. No ROS 2 rewrite.
- No GitHub-hosted runner driving a LAN robot, and no new job on the existing
  `[self-hosted, jetson]` runner (`jetson-nightly.yml:53`).
- No `world_model:` block in `config/default.yaml`. Per the F-043 precedent
  ("Do not add isaac keys to config/default.yaml") opt-in subsystem knobs stay on schema
  defaults, and `loader.py:76-96` would otherwise deep-merge the block into all 17
  overlays.

## Charter

No CHARTER.md §3 carve-out is required, and this paragraph exists because
`charter-carveout/SKILL.md:52-60` requires the determination in writing even when the
answer is "none apply".

- **Q1, motion without a human gate — no.** Nothing here commands actuators. Both motion
  gates keep their defaults: `MOUSEDROID_SMOKE_ALLOW_MOTION` (`jetson_smoke_test.sh:29`,
  default `0`) and `ESP32Config.smoke_test_allow_motion` (`hardware.py:231`, default
  `False`). All on-rover work in Phase 7 runs the canonical no-motion invocation, which
  clears both (`jetson_full_validation.sh:423,429`). The delivery half does not gain the
  ability to restart the rover service unattended: the service switch stays an operator
  step, which is why rev A's `--approved-release` auto-confirm is removed in rev B.
- **Q2, LLM or training in the 30 Hz loop — no.** `engine: onnx_trt` is already the
  accepted in-loop design (ADR-008, `WorldModelConfig.engine`
  `Literal["torch","onnx_trt"]` at `world_model.py:39`). F-047's "never RSSM TensorRT" is
  scoped to loading a PPO graph through `vla/policy.py` into the 30 Hz loop — the isaac
  bundle's verdict table says so — not to the RSSM observe path ADR-008 accepts.
- **Q3, behaviour change by editing source rather than YAML/env — no.** Every new field is
  `Field(default=..., description=...)` preserving today's behaviour; FP16 and caches are
  opt-in via the `MOUSEDROID_WORLD_MODEL__*` environment variables.

FP16 does change recurrent-state numerics, which is a safety-adjacent property. It is
gated on multi-step parity plus recorded task replay (`specs/performance-evidence`) and
ships default-off. If the evidence in Phase 6 shows safety-envelope drift, FP16 is dropped
rather than carved out.

## Impact

Backwards compatibility is preserved by default: `engine` stays `"torch"`, every new
switch defaults off, and no shipped YAML gains a key.

The real blast radius is three-fold.

**`config-compat` silently swallows new fields.** `.github/workflows/config-compat.yml`
worktrees the SHA in `deployments/jetson-image.json` (`032942b5…`) and validates new YAML
against the old `Settings`. At that SHA `WorldModelConfig` is a plain `BaseModel`
(`schema.py:1597`), i.e. `extra="ignore"` — so a new `world_model.*` key in
`config/jetson_production.yaml` passes the gate and is then ignored by the pinned schema.
A new *top-level* block hard-fails instead. Rev C therefore adds no `world_model:` key to
any tracked overlay; activation is environment-based (see What Changes), and Phase 8
re-pins the record post-merge.

**That pin is one branch cleanup from repo-wide failure.** `deployments/jetson-image.json`
states it itself: zero tags exist, and the SHA's only reachability is ten stale feature
branches — "precisely the set `scripts/archive_stale_branches.sh` exists to delete". Any
SHA this change records inherits `mouse-droid-deploy-repin`'s pin-reachability spec
(remote-*tag* reachability, remedied by `scripts/repin_tags.sh`). Phase 8 runs the audit.

**The ONNX surface has no blocking CI.** `onnx-world-model-extras` is the only job
installing `[onnx_world_model]` and it carries `continue-on-error: true`
(`ci.yml:579-583`), tracked with `promote_after_days: 180` since 2026-05-16 — due
2026-11-12, and the real bar is a 7-consecutive-green-run count the tracker admits nobody
has re-derived. Rev C does not claim coverage it lacks: the config, provider-policy and
artifact-contract tests land in the blocking `test` job via `tests/regression`; only the
ORT-dependent integration tests stay advisory, and Phase 8 records the streak impact.

F-008 remains `todo` ("USB-C rover smoke passes on the physical Jetson"). Nothing here is
path-blocked by `freeze_gate.py` — `.claude/workforce.yaml` freezes only
`src/mousedroid/arm/**` — but Phase 7 competes with F-008 for the same bench, so this
change is explicitly sequenced *behind* it, in the house idiom "while F-008 is
bench-blocked".

## Spec Deltas

- `openspec/changes/mouse-droid-jetson-onnx-delivery/specs/onnx-runtime/spec.md`
- `openspec/changes/mouse-droid-jetson-onnx-delivery/specs/performance-evidence/spec.md`
- `openspec/changes/mouse-droid-jetson-onnx-delivery/specs/pc-to-jetson-delivery/spec.md`

## Tasks

See `openspec/changes/mouse-droid-jetson-onnx-delivery/tasks.md`.

## Validation

`bash scripts/validations/F-050.sh` and `bash scripts/validations/F-051.sh`
