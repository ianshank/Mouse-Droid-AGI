# Peer review — Jetson ONNX runtime + PC-to-rover delivery

Review of the external plan `jetson-onnx-runtime-and-pc-delivery` against the tree at
`18aba56` (default branch `claude/markdown-implementation-plan-aVJ2l`). The plan artifact
is not in the repository; claims are quoted from the submitted document. Every repo-side
fact below was read out of the tree.

**Verdict: REQUEST_CHANGES.** The plan's architecture decision (ORT + TensorRT EP, no
second direct-TensorRT runtime, observe-only scope) is right and matches ADR-008. Its
delivery half is built on a target layout the rover does not have, and eight of its
concrete claims are wrong against the tree.

## Verdict table

| Claim | Verdict |
|---|---|
| `openspec validate --strict` gates Task 0 / Task 9 | **REJECTED** — no OpenSpec CLI exists; `openspec/project.md:1-6`. Already refuted once in `mouse-droid-claude-workforce/peer-review.md:30` |
| `onnx_warmup_iterations` is a new config field | **REJECTED** — exists at `config/schema/world_model.py:77`, consumed `factory/world_model.py:265`, documented `ADR-008:47` |
| `config/default.yaml` shows a `world_model:` block | **REJECTED** — no `config/*.yaml` sets `world_model:`; schema defaults only |
| Benchmark drift gate covers `new_h`, `new_z`, `obs_embed`, `surprise` | **REJECTED** — `new_z` is a `randn_like` sample, intentionally excluded (`ADR-008` "Cross-engine equivalence"). A `new_z` gate fails by construction |
| FP32 parity tolerance `rtol=1e-4, atol=1e-5` | **REJECTED** — tightens the accepted `atol=1e-4` (ADR-008, `test_export_dual_stream_rssm_onnx.py`) with no evidence |
| New metrics land in `src/mousedroid/telemetry/metrics.py` | **REJECTED** — that path does not exist; `telemetry/metrics/` is a package of `_registry_*.py` modules (ADR-017) |
| `mousedroid_control_deadline_miss_total` is new | **REJECTED** — `_registry_core.py:91` `_tick_overruns`, written live at `_telemetry_experience_mixin.py:114-126` against `safety.loop_soft_budget_factor` |
| `mousedroid_onnx_provider_fallback_total` as the registry name | **REJECTED** — `primitives.py:422-429` appends `_total` at render; would ship `..._total_total` |
| Isaac sensor-parity change is "active … in-flight" | **REJECTED** — F-043–F-048 `status: done`, `implemented_in: 3b52789…`; `openspec/project.md` row says `implemented` |
| Task 15 gate "only if the catalog remains sequential" | **REJECTED** — ADR-013:63-74 makes F-009–F-014 and F-033 *intentional* holes; rule is next-free ≥ F-015 and unique |
| `MOUSEDROID_ESP32__ENABLED=false` is the repo-standard no-motion control | **REJECTED** — canonical form clears both motion vars: `jetson_full_validation.sh:423,429` |
| `/opt/mousedroid/releases/<id>` + `current`/`previous` symlinks | **REJECTED** — `/opt/mousedroid` is a git checkout bind-mounted at the same path (`docker-compose.jetson.yml:80`) holding an editable install and rover-local WIP commits |
| Cosmos deferral is new | **REJECTED** — already catalogued, F-049 `status: deferred` |
| Change id `jetson-onnx-runtime-and-pc-delivery` | **REJECTED** — 18/18 bundles are `mouse-droid-<slug>` |
| HF repo lacks `observe_step.onnx`; fail closed | **CONFIRMED** — repo holds only `.gitattributes` (1519 B) and `README.md` (877 B) |
| `common/onnx_session.py` resolves TRT→CUDA→CPU, lazy session, zero-input warmup | **CONFIRMED** — `DEFAULT_ORT_PROVIDERS:47-51`, `warmup_session:149-221`, `run_session_with_zeros:103-146` |
| `observe_step` packs on CPU, NumPy round-trip, 4 outputs | **CONFIRMED** with correction — 3 Tensors + 1 Python `float`; output names cached in `__init__:137`, not recomputed |
| Export is observe-only; `imagine_step` raises; composite keeps PyTorch | **CONFIRMED** with addition — PyTorch is also kept for `get_safety_trace` (`composite.py:128-150`) |
| No provider-option schema / strict policy / precision / cache / graph / IO-binding switch | **CONFIRMED** |
| No TensorRT engine-cache volume in `docker-compose.jetson.yml` | **CONFIRMED** |
| `scripts/deploy_jetson.sh` is not a PC-side promotion command | **CONFIRMED** — but `scripts/deploy_remote.sh` is, and the plan never names it |
| Observe-step optimization may not move whole-tick latency | **CONFIRMED as the central risk** — and unmeasured; see pin 1 |

## Load-bearing pins

1. **No on-rover latency evidence exists for any world-model stage.** `reports/` holds 3
   files, `smoke-reports/` 2; none carries an `observe_step`, `tick_phase`, or
   `world_model` measurement. The only on-device report (`smoke-reports/smoke_report.md`,
   2026-05-12) records `telemetry down`. The plan optimizes an unmeasured stage.
2. **The metric the plan builds on is dead in production.** `build_world_model(cfg)`
   (`factory/world_model.py:55`) takes only `cfg`; `_build_onnx_world_model` constructs
   `DualStreamRSSMOnnx(model_path=…, cfg=…, warmup_iterations=…)` at `:262-266` with no
   `metrics=`. `observe_step` records only `if self._metrics is not None`
   (`dual_stream_rssm_onnx.py:271-279`), so `mousedroid_world_model_observe_step_seconds`
   has no production writer and `WorldModelObserveStepLatencyHigh`
   (`config/prometheus/alerts.yml:388-421`) cannot fire. The live signal is
   `mousedroid_tick_phase_ms{phase="world_model"}` (`orchestrator.py:522`).
3. **The existing histogram is mis-scoped.** `dual_stream_rssm_onnx.py:255-258` brackets
   `session.run` only; the NumPy→Torch conversions at `:262-269` and
   `pack_observation` at `:236` are outside the timer. The metric is named and alerted as
   `observe_step` wall-clock but measures ORT execute. Removing the copy boundary would
   barely move it.
4. **`resolve_providers` cannot carry provider options.** `common/onnx_session.py:72-100`
   is `tuple[str, ...] -> tuple[str, ...]`; a `("TensorrtExecutionProvider", {...})`
   tuple fails `p in available`, the intersection empties, and it returns
   `("CPUExecutionProvider",)` silently. Every cache/precision/graph/workspace option the
   plan proposes is a provider option, so this signature and `warmup_session`'s
   `requested_providers` must both change type.
5. **`active_providers` is a prediction, not an observation.** `session.get_providers()`
   is called nowhere in the tree. `warmup_session:193` computes the tuple *before*
   constructing the session at `:207-210` and returns it at `:221`. Strict-provider
   policy and a fallback counter are unimplementable until that call is added.
6. **`common/onnx_session.py` is shared with the VLA hot path.** Consumers are
   `world_model/dual_stream_rssm_onnx.py:42` and `vla/policy.py:29`. Module neutrality
   and lazy ORT import are pinned by `tests/unit/common/test_onnx_session.py:391-415` and
   `tests/unit/world_model/test_onnx_vla_decoupling.py`. Two ORT stubs pin the exact
   `InferenceSession(model_path, providers)` arity — `test_onnx_session.py:98-103` and
   `tests/unit/vla/test_distilled_onnx.py:107` — so adding `sess_options=` breaks both.
7. **Three of the proposed fields already exist on `JetsonConfig`.** `hardware.py:580-588`
   carries `precision: Literal["fp32","fp16","int8"] = "fp16"`, `workspace_gb`, and
   `tensorrt_cache_dir`; `config/default.yaml` sets the first two. `WorldModelConfig`
   also already has `onnx_cache_dir` (artifact download cache,
   `factory/world_model.py:320-322`), which collides by name with the plan's
   `onnx_engine_cache_dir` (TRT plan cache).
8. **`engine: onnx_trt` is unreachable from the overlay ADR-008 documents.**
   `_build_onnx_world_model` raises when `cfg.model.cfc_hidden_dim <= 0`
   (`factory/world_model.py:246-253`); the schema default is `0`
   (`world_model.py:205`) and only `config/jetson_dual_stream.yaml:29` sets `64`.
   `config/jetson_production.yaml` does not. ADR-008's migration step 2 and
   `export_dual_stream_rssm_onnx.py:17-21` both name `jetson_production.yaml`.
9. **ADR-008 is the accepted decision record for this subsystem** and already fixes the
   config surface, the provider chain, the <10 ms Orin Nano target, the 33 ms portable
   gate, and the `new_z` exclusion. Thirteen new fields, a strict-provider policy, FP16,
   caches, and I/O binding amend it. Highest ADR is ADR-018, so a successor is ADR-019.
10. **An existing budget test already measures both engines.**
    `tests/performance/test_observe_step_budget.py:126,157` runs torch and ONNX
    `observe_step`, 50 iterations, budget via `MOUSEDROID_OBSERVE_STEP_BUDGET_MS`
    (33 ms default, 10 ms Jetson). `tests/performance/conftest.py` skips off-Jetson and
    the `performance` CI job is advisory, so nothing blocking measures this.
11. **`scripts/benchmark_latency.py` already exists** with the `--config`/`--checkpoint`
    + threshold + exit-code contract, covering `RSSM.imagine_step()` and
    `MCTSPlanner.plan()` — the two stages the plan leaves in PyTorch.
12. **`world_model/CLAUDE.md` invariant 3 forbids in-place tensor mutation.** I/O binding
    is pre-allocated buffer reuse by construction. The invariant must be scoped in
    writing or the feature dropped.
13. **Construction must stay cheap.** `test_dual_stream_rssm_onnx.py:111`
    (`test_construction_does_not_load_session`) and `:121` (protocol conformance) pin it;
    eager device-buffer allocation breaks the first. `observe_step` is synchronous and
    warms lazily on first call (`:223-224`), so a cold TRT engine build blocks the
    caller — already a hazard, worse with engine caches.
14. **`WorldModelProtocol.observe_step` returns `tuple[Tensor, Tensor, Tensor, float]`**
    (`world_model/protocol.py:17-28`, `@runtime_checkable`). An I/O-binding path must
    still cast `surprise` to a Python float.
15. **`/opt/mousedroid` is a git checkout, bind-mounted at the same path, editable-install
    based** (`docker-compose.jetson.yml:79-80`, `Dockerfile.jetson:141`). Rover-local WIP
    is preserved as commits on `rover/wip-<date>` branches, never by rsync or clean
    (`docs/superpowers/plans/2026-07-25-jetson-deploy-and-validation-campaign.md` B1).
    The established rollback anchor is a Docker tag —
    `docker tag mousedroid:jetson mousedroid:jetson-rollback-<date>` (B4) — which already
    satisfies offline rollback without rebuilding.
16. **`scripts/deploy_remote.sh:151` runs `rsync -avz --delete`.** The campaign plan names
    it as the script that "destroys rover work". The plan proposes a new controller
    without retiring or fencing this one.
17. **`PYTHONOPTIMIZE=1` is set in the image** (`Dockerfile.jetson:178`) and leaks into
    every `docker exec`. In-container validation must pass `-e PYTHONOPTIMIZE=0`; asserts
    are stripped otherwise. Consequently `S101` is blocking in `src/` and strict-provider
    enforcement must raise a named exception.
18. **`/etc/mousedroid/docker.env` can override the safe default.**
    `config/docker.env.example:66` carries a commented
    `MOUSEDROID_ESP32__ENABLED=true`; compose `env_file` beats the YAML overlay. Any
    no-motion claim must audit the live file, not just export a variable.
19. **Two distinct motion gates exist**: `MOUSEDROID_SMOKE_ALLOW_MOTION`
    (`jetson_smoke_test.sh:29`, default `0`) and `ESP32Config.smoke_test_allow_motion` /
    `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION` (`hardware.py:231-238`, default `False`).
    `jetson_full_validation.sh:423,429` clears both. `RUN-MOTION` is prose only — no code
    reads it (`docs/analysis/autonomy-baseline-peer-review-2026-09-17.md:82`).
20. **A release manifest carrying a SHA inherits the pin-reachability spec.**
    `deployments/jetson-image.json` is worktreed by the `config-schema-compat` gate;
    `mouse-droid-deploy-repin/specs/pin-reachability/spec.md:3-8` requires remote-*tag*
    reachability, remediable via `scripts/repin_tags.sh` (F-035). Re-pinning to a
    feature-branch SHA reproduces the `9c31968` repo-wide gate failure, which is why the
    re-pin is post-merge only.
21. **`onnx-world-model-extras` is the only CI path installing `[onnx_world_model]`, and
    it is advisory** (`.github/advisory_stages.yaml`, `since: 2026-05-16`,
    `promote_after_days: 180`, real bar a 7-consecutive-green-run count). The plan must
    not claim blocking CI coverage. The `performance` job is advisory too
    (`since: 2026-07-25`, 90 days — not yet overdue).
22. **Branch/PR state.** 94 remote branches, 2 open PRs, both dependabot: #226 (ruff
    0.16.2→0.16.7) and #218 (mcp bound). Every ONNX/Jetson branch that looks like
    in-flight overlap is a squash-merge leftover — #140 `dedupe-onnx-session`, #141
    `onnx-default-providers-common`, #92 `tier-b2-onnx-world-model`, #116
    `jetson-full-validation`, #142 `deploy-hardening-f013-f014`, #74
    `bump-jetson-deployment-sha`; #26 `jetson-deployment-readiness` closed unmerged.
    Nothing blocks this change, and the plan's baseline `2469588` is one commit stale.
23. **F-008 is still `todo`** ("USB-C rover smoke passes on the physical Jetson"). The
    F-024 freeze rule states hardware readiness preempts in-flight software streams, and
    the house convention labels such work "while F-008 is bench-blocked" (PR #222). Only
    `src/mousedroid/arm/**` is hard-blocked by `freeze_gate.py`, but the on-rover phase of
    this change competes with F-008 for the same bench.
24. **F-047 is deferred with the phrase "never RSSM TensorRT"**, scoped to loading a PPO
    graph into the 30 Hz loop (`features.yaml`, and the isaac bundle's verdict table).
    ADR-008 already accepts RSSM TensorRT for `observe_step`, so no new carve-out is
    needed — but the bundle must say so in writing rather than leave it implied.

## House-format corrections

| Plan | House format |
|---|---|
| `openspec/changes/jetson-onnx-runtime-and-pc-delivery/` | `mouse-droid-<slug>` (18/18) |
| 3 bundle files | 4 — `peer-review.md` is mandated (`openspec-change/SKILL.md:15-19`, 15/18 bundles) |
| `Task 0 … Task 15` | `**Phase N — label**` + flat `- [ ] N.M` checkboxes |
| No front-matter | Bullet block: `change_id`, `project`, `status`, `feature_id`, `epic`, `owner`, `created`, `basis_commit`, `rev` |
| No design structure | `## D-N.` sections, each naming the rejected alternative |
| F-number reserved at Task 15 | Reserved first (`openspec-change/SKILL.md:12-14`); `in_progress` while coding |
| Registry update unmentioned | `openspec/project.md` table: change-id, status, F-number, landed, authoritative artifacts |
| Charter question unmentioned | A written `## Charter` determination is required even when the answer is "none apply" (`charter-carveout/SKILL.md:52-60`) |
| One `tests/regression/test_onnx_artifact_contract.py` | Paired `test_f0NN_aqa.py` + `test_f0NN_backwards_compat.py`, spelled out in full (`regression-pair-scaffold/SKILL.md:152-161`) |
| `tests/unit/common/test_onnx_session.py` (new) | Already exists — extend |
| `tests/unit/world_model/test_dual_stream_rssm_onnx.py` (new) | Already exists — extend |
| `tests/integration/test_onnx_iobinding_cuda.py -m hardware` | `tests/hardware/` (24 files use the marker there; 0 in `tests/integration/`) |
| `tests/performance/test_world_model_benchmark_schema.py` | Schema pins belong in `tests/regression/*_aqa.py`; `tests/performance/` is for budgets, is Jetson-gated and advisory |

## What survives review unchanged

- ORT + TensorRT EP with CUDA/CPU fallback; no second direct-TensorRT runtime. Matches
  ADR-008's decision and its explicit note that `torch2trt` via `JetsonTensorRTCompiler`
  stays a future option.
- Observe-only scope; `imagine_step` stays PyTorch; MCTS profiled before any export.
- Content-addressed artifacts. ADR-008's own "Negative" section records that the factory
  cannot detect a stale `.onnx` against retrained weights — "wrong weights = wrong
  inference, silently". The SHA-256 + manifest requirement closes a documented gap.
- Fail closed when the HF filename is absent, and never substitute a `.pt` for an
  `.onnx`. Verified necessary: the repo has no `.onnx`, yet ADR-008 tells operators they
  may leave `onnx_path: null` and rely on the Hub.
- Default every new switch off; FP16 only in a dedicated opt-in overlay; INT8 deferred.
- Report component and whole-tick latency separately; no 30 Hz claim from a
  microbenchmark. This is the plan's strongest instinct and pin 1 makes it mandatory.
- Strict-provider policy, post-construction provider proof, fallback counter — correct
  requirements; pins 4 and 5 are about cost, not merit.
- Flat `onnx_*` field naming is consistent with the existing `WorldModelConfig`.
- `-m hardware` is the right marker (`pyproject.toml:384`, `--strict-markers` at `:392`).
- No secrets in PR jobs; no GitHub-hosted runner driving a LAN robot.
