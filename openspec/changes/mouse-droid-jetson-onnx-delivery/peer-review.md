# Peer review — Jetson ONNX runtime + PC-to-rover delivery

Review of the external plan `jetson-onnx-runtime-and-pc-delivery`. Round 1 was verified
against `18aba56` (the bundle's `basis_commit`); rounds 2 and 3 against `9aedfe2` and
`a46fbad`, which are this branch's own docs-only commits adding the bundle itself — no
`src/`, `config/`, `scripts/` or workflow file differs between them, so every code
citation below resolves identically at all three. Baseline for the tree under review:
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

---

# Round 2 — second pass

A deeper pass over the areas round 1 treated only at the surface: the exported graph's
shape and input contract, the artifact supply chain, and whether the tick actually behaves
the way the plan's gates assume. Verified against `9aedfe2`.

**Round 2 changes one conclusion.** Round 1 said the plan optimizes an unmeasured stage.
Round 2 finds the repository already knows the 30 Hz tick does not hold, has said so twice
in its own reviews, and already ships the test that would prove it. Rev C's whole-tick gate
must therefore be a *prerequisite*, not a deliverable, and the plan must stop proposing new
measurement tooling.

## Verdict table — round 2

| Claim | Verdict |
|---|---|
| Build `scripts/benchmark_world_model_jetson.py` for whole-tick evidence | **REJECTED** — `tests/performance/test_jetson_endurance.py` already runs the full orchestrator loop and gates "Loop time p95 < 33ms at 30Hz target" |
| Report "p90, p95, p99" | **REJECTED** — repo convention is p50/p95/p99 (`validation/latency_stats.py:29`, `_P50,_P95,_P99`); p90 exists nowhere |
| Whole-tick p95 ≤ 33.33 ms is achievable as a gate of this change | **REJECTED** — `LoopConfig.planning_hz` has zero consumers; MCTS blocks the tick synchronously. The repo's own review calls 30 Hz plan-act "fiction" |
| `onnx_profile_batch: 1` expresses the TensorRT shape profile | **REJECTED** — axis 0 is dynamic on all 6–10 inputs *and* all 4 outputs; TRT EP needs per-input min/opt/max shape strings |
| A SHA-256 digest is sufficient artifact integrity | **REJECTED** — the graph's input set is config-dependent; a matching digest can still be the wrong graph for the loading config |
| Fixed-batch export is "a separately proven optimization" | **PARTLY REJECTED** — it is a behaviour change: dynamic batch is deliberate, for training-time reuse of the same `.onnx` |
| The artifact can be produced and published | **UNPROVEN** — no checkpoint exists in the tree or at the path ADR-008 names, and no DualStream/CfC checkpoint is known to exist anywhere |
| Rev A's risk row "dynamic axes cause TensorRT build overhead" | **CONFIRMED** — `_dynamic_axes_for_inputs` marks every tensor dynamic |
| `performance` CI job promotion is overdue | **REJECTED** (correcting an earlier note) — `since: 2026-07-25`, 90 days, due 2026-10-23; `check_advisory_promotions.py` reports all stages within window |

## Load-bearing pins — round 2

25. **`LoopConfig.planning_hz = 10.0` has zero consumers.** Declared at
    `config/schema/misc.py:178` ("MCTS planning rate (Hz)") and set in
    `config/default.yaml:9`, but referenced nowhere in `src/`. There is no tick decimation,
    modulo or `_should_plan` gate in `orchestrator/`. So planning runs at `control_hz`
    (30 Hz, 33.33 ms), not 10 Hz. The repo has recorded this twice:
    `docs/analysis/autonomy-baseline-peer-review-2026-09-17.md:79` (S4) — "30 Hz plan-act
    is fiction… the multi-rate contract was declared and never wired" — and
    `docs/analysis/positioning-safety-peer-review-2026-09-19.md:148` (P5) — "**BOTH STILL
    OPEN** … `LoopConfig.planning_hz` still has zero consumers", with `:198` saying to
    "Close P5 … before publishing, not after". That second document landed in PR #232, one
    commit before this branch. The plan does not mention it.
26. **MCTS is the fallback action path and it is not cheap.** `_select_action`
    (`orchestrator/_action_mixin.py:30-68`) tries the cognitive core first, then the VLA
    branch (inert by default — `policy_selector='nav_agent'`), then falls through to
    `self._agents[0].act(...)` → `MouseDroidNavigationAgent.act`
    (`agents/navigation.py:84-88`) → `MCTSPlanner.plan()`. Per plan call:
    `n_simulations_base=50` × (one expansion `imagine_step` at `world_model/mcts.py:257`
    plus `rollout_depth=5` rollout calls at `:277`) ≈ **300 `imagine_step` invocations**,
    rising to ≈1200 at `n_simulations_max=200`. Against **one** `observe_step` per tick.
    The plan optimizes the one and leaves the rest in PyTorch on the same shared iGPU.
27. **The published model card puts MCTS at ~109 ms.** `ianshank/mousedroid-weights`'s
    README carries a `mcts_tuning` block: `p50_ms` 109–110, `p95_ms` 125, across every
    `ucb_c` value. Independently, `scripts/benchmark_latency.py` defaults to
    `--mcts-target-ms 50.0` and `--rssm-target-ms 15.0` — the repo's own targets already
    sum to roughly twice a 33.33 ms tick. Treat the 109 ms as indicative, not as a
    production tick measurement: the simulation count and host state behind it are not
    recorded.
28. **The fallback engages on an ordinary failure, and the plan would never measure it.**
    `factory/orchestrator.py:158-170` builds the cognitive core when
    `cfg.cognitive.enabled`; on exception, with `cognitive.fallback_to_mcts: true` — the
    schema default and set explicitly at `config/jetson_production.yaml:204` —
    `cognitive_core` stays `None` and every subsequent tick takes the MCTS path. A failed
    BDI weight download at boot is sufficient. Any 30 Hz claim is therefore conditional on
    a runtime branch, and evidence must cover both.
29. **MCTS cost scales up with surprise.** `agents/_planning.py:23-24` —
    `scale = min(surprise + 1.0, maximum/base)`, budget `clamp(int(base*scale), base,
    maximum)`. With base 50 / max 200 and `surprise.high_threshold: 2.0`, high surprise
    means 150 simulations and critical (5.0) means 200. The surprise input is
    `observe_step`'s own fourth return value. Worst-case planning latency therefore occurs
    in the most novel situations.
30. **`tests/performance/test_jetson_endurance.py` already is the whole-tick harness, and
    has never been run.** It runs the full orchestrator loop and validates GPU temperature,
    RSS stability within 10%, and **"Loop time p95 < 33ms at 30Hz target"**; it writes
    `duration_s`, `p95_ms`, `rss_start_mb`, `rss_end_mb`, `max_gpu_temp_c` to
    `${MOUSEDROID_ENDURANCE_REPORT_DIR:-reports/endurance}/endurance-<utc>.json`
    (`:50`, `:88-95`), is `pytestmark = [hardware, slow]` (`:53`), and takes
    `MOUSEDROID_ENDURANCE_DURATION_S` (default 60 s). `reports/endurance/` contains only
    `.gitkeep`. It is excluded from CI because the `performance` job runs
    `-m "not hardware"` — which
    `docs/analysis/positioning-safety-peer-review-2026-09-19.md:147` (P4) names as the
    reason that job is "a tripwire, not a benchmark", the remaining tests running at a
    deliberately loosened 2.0× budget.
31. **Loop-latency percentiles are survivorship-biased by design.**
    `_finish_tick_timing` (`orchestrator/_telemetry_experience_mixin.py`) latches the
    duration on every path but returns before recording when `ok is False`, so a tick that
    raises — including one cancelled by `asyncio.wait_for(self.tick(), tick_timeout_s)` —
    contributes neither a histogram sample nor a `tick_overruns` increment.
    `docs/analysis/positioning-safety-peer-review-2026-09-19.md:178` (D-9) records this as
    **addressed**: the bias is intentional per telemetry invariant 5 and stays; the
    denominator is now recorded and the caveat documented. So it is not an open defect —
    but any whole-tick p95/p99 this change publishes is a percentile *of successful ticks*
    and must be reported with its denominator and the timeout class alongside.
32. **The graph's input set is config-dependent, and nothing validates it.** `onnx_io.py`
    declares 6 always-present inputs and 4 gated on `cfg.<modality>_dim > 0`
    (`optional_input_names_for_cfg`). The export derives names from
    `build_example_inputs(cfg)` (`export_dual_stream_rssm_onnx.py:264`); the runtime
    derives feeds from `packed.<modality> is not None`
    (`dual_stream_rssm_onnx.py:238-253`); and `onnx_io.py`'s own
    `all_input_names_for_cfg` / `optional_input_names_for_cfg` / `required_input_names`
    accessors are referenced **only** by `tests/unit/world_model/test_onnx_io.py` — three
    independent derivations of one set, in the module whose docstring exists to prevent
    exactly that drift.
33. **A modality mismatch passes warmup and fails on the first real tick.**
    `run_session_with_zeros` builds its feeds from `session.get_inputs()`
    (`common/onnx_session.py:142`) — the graph, not the config — so warmup succeeds for any
    structurally valid graph. `session.get_inputs()` is used nowhere else. An artifact
    exported under a different modality set therefore loads, warms, reports healthy, and
    raises inside `observe_step` on the 30 Hz path. A SHA-256 check does not catch it.
34. **Dynamic batch is on every tensor, deliberately.** `_dynamic_axes_for_inputs`
    (`export_dual_stream_rssm_onnx.py:228-240`) sets axis 0 to the symbolic `batch`
    dimension for every input *and* every output, with the docstring citing
    `CFC_ONNX_SPIKE_REPORT.md`: "future training-time use of the same `.onnx` doesn't
    require a separate export per batch size". TensorRT EP needs
    `trt_profile_min_shapes` / `_opt_` / `_max_` as name-indexed strings covering every
    dynamic input, which a single `onnx_profile_batch` integer cannot express. And a
    fixed-batch export would remove the training reuse the dynamic axis exists for.
35. **The export produces no metadata, no digest, and never runs `onnx.checker`.**
    `scripts/export_dual_stream_rssm_onnx.py` has zero occurrences of `checker`,
    `onnx.load`, `metadata`, `sha256`, `hashlib` or `digest`. `_push_to_hf` uploads the
    `.onnx` alone. So the currently-documented publication path ships a structurally
    unvalidated, uncharacterised artifact.
36. **The artifact cannot be produced from the tree, and the checkpoint it needs is
    unverified.** `.gitignore` excludes `weights/`, `*.onnx` and `*.pt`, and no `weights/`
    directory exists. ADR-008's migration step 1 and `:112` name
    `--checkpoint weights/dual_stream_rssm/final.pt`; `scripts/benchmark_latency.py`
    defaults to `weights/rssm/final.pt`; `ianshank/mousedroid-weights` carries
    `rssm/final.pt` (2,116,713 B) plus 16 epoch checkpoints (~6.2–6.3 MB each). Its model
    card describes that file as "RSSM World Model — Recurrent State-Space Model" and tags
    the repo `rssm`, `bdi`, `constitutional-rl` — no CfC or dual-stream tag. But
    `DualStreamRSSM` requires `cfc_hidden_dim > 0` (`factory/world_model.py:246-253`).
    Whether a DualStream/CfC checkpoint exists at all is unestablished, and the path
    ADR-008 documents exists in neither Hugging Face repo.
37. **`onnx_cache_dir` is a `str`, not a `Path`, and its default is relative.**
    `config/schema/world_model.py:68` defaults it to `"weights/dual_stream_rssm"` — the
    same non-existent `dual_stream_rssm` name — with no path validation. It resolves
    against `WORKDIR /opt/mousedroid` (`Dockerfile.jetson:49`), which is how downloaded
    weights persist at all, since no volume mounts them and nothing sets `HF_HOME`.

## What round 2 changes in the plan

- **Do not build a whole-tick benchmark.** Run `test_jetson_endurance.py` on the rover with
  actuation disabled. Its JSON lands in `reports/endurance/`, which `.gitignore:339-340`
  ignores except `.gitkeep` and `evidence-commit/SKILL.md:24-38` lists among the six
  local-only families — so it closes out as a **declared local-only chain** (artifact on
  the rover plus a `CHANGELOG.md` reference), not as a git commit. Pin 54 states the same
  rule; an earlier draft of this line contradicted it. It
  already covers the tick p95 gate, RSS stability (which answers the both-engines memory
  question), and thermals.
- **The whole-tick gate is a prerequisite, not a deliverable.** Closing P5 — wiring
  `planning_hz` using the budget/timeout/fallback pattern `_try_vla_action` already
  demonstrates — comes before any 30 Hz claim. Until then this change reports component
  latency only, and says so.
- **Measure both action paths.** Evidence must cover the cognitive-core path and the MCTS
  fallback, because `fallback_to_mcts: true` makes the second reachable from a failed
  weight download.
- **Validate the graph against the config at warmup.** Compare `session.get_inputs()`
  against `all_input_names_for_cfg(cfg)`, fail closed with a named exception, and wire the
  export and runtime to that accessor so the three derivations collapse to one.
- **Record modality dims and shape profiles in the export metadata**, and derive TensorRT
  shape profiles per input rather than from a batch integer.
- **Establish the checkpoint before anything else.** Phase 6 has no artifact to integrity-
  check until a DualStream/CfC checkpoint is located or retrained, and ADR-008's checkpoint
  path needs the same narrative correction as its overlay name.
- **Use `validation/latency_stats.py::summarize`** for every percentile. Drop p90.

## Security pins — round 2

38. **The rover's network identity is already committed to a public repository, and gitleaks
    structurally cannot catch it.** `smoke-reports/smoke_report.md` and `smoke_report.json`
    are tracked (`git ls-files smoke-reports/`), are not gitignored
    (`git check-ignore` returns nothing), and landed in `02e585b` (PR #177). Between them
    they publish the rover's LAN IP (`:5`, `:6`, `:18`, `:265`), its mDNS name, the SSH
    username, the WiFi SSID, the dev-box identity, the JetPack version and the on-device
    HEAD. `.gitleaks.toml` extends only the default ruleset, and no default rule matches an
    IPv4 address, hostname, SSH alias or SSID — these files pass the blocking gate today,
    which is the proof. The plan proposes a deploy record and a benchmark report: the same
    artifact class. They must go to a gitignored `reports/<name>/` path (the precedent is
    `.gitignore:173-188`, which already ignores `reports/jetson_validate/`,
    `reports/jetson_smoke/*/`, `reports/jetson_full_validation/**`), and anything committed
    must carry only host-independent facts — `deployments/jetson-image.json` is the field-set
    model. Scrubbing `smoke-reports/` is pre-existing cleanup outside this diff, but it needs
    raising because reviewers will read it as precedent.
39. **Digest verification exists, fail-closed and metricised — on a different path.**
    `utils/weights_manager.py:251` (`verify_sha256`), `cloud/weight_update_poller.py:303`,
    the `sha256_manifest_filename` schema field (`config/schema/gcp_cloud.py:412-420`, "a
    download is refused if the local SHA does not match this manifest"), the
    `inc_cloud_weight_update_sha256_mismatch` counter
    (`telemetry/metrics/_registry_cloud.py:234`), and a deliberate
    do-not-mark-revision-seen retry policy. The world-model ONNX download uses none of it:
    `factory/world_model.py:290-361` calls `download_weights_from_huggingface` and then
    checks only `model_path.is_file()`, with no `revision=` pin. So *recording* a digest in
    a deploy record, which is all the plan asks for, is theatre — the digest must be
    **enforced at load**. Extend the OTA mechanism; do not build a parallel one.
40. **`torch.load(..., weights_only=False)` on a runtime path.**
    `world_model/checkpoint_migration.py:293` — the checkpoint-migration entry point a
    "promote a new model" flow calls — unpickles with no preceding digest check. It is the
    outlier: `growth/slot_store.py:121`, `learning/on_device/slot_store.py:166` and
    `learning/offline_rl.py:361,747` all use `weights_only=True`, and
    `efficiency/tensorrt.py:297-302` uses `False` but documents why (torch2trt) and scopes it
    to a 0700 cache — a mitigation stated in a comment and enforced nowhere. Separately,
    `scripts/export_dual_stream_rssm_onnx.py:353-359` attempts `weights_only=True` and
    **silently** falls back to the legacy unpickling loader on `TypeError`, with no log. The
    safe pattern is already in the tree at `growth/slot_store.py:112-121`: verify the digest,
    then load with `weights_only=True`. Note the threat models differ — the `.onnx` is data,
    not a pickle, so its risk is graph substitution affecting the safety trace; the RCE
    vector is the `.pt`. The plan should keep them distinct.
41. **The ONNX cache directory is created before it is validated, and validation is skipped
    on two of three branches.** `factory/world_model.py:320-321` does
    `Path(cfg.world_model.onnx_cache_dir).mkdir(parents=True, exist_ok=True)`, while the
    protected-root guard lives inside the download helper
    (`utils/weights_manager.py:120-132`, `.resolve()` then `_validate_download_directory`
    against `_PROTECTED_DOWNLOAD_ROOTS`). The `mkdir` runs first. And the helper is never
    reached at all when `model_path.is_file()` short-circuits at `:324-330`, or when an
    explicit `onnx_path` returns at `:305-307`. `onnx_cache_dir` is a bare `str` with no
    validator, whereas `config/schema/learning.py:15-58` (`_validate_relative_slot_dir`)
    already implements exactly the relative-only, no-`..` check these fields need. This
    matters more than it looks because `Dockerfile.jetson` declares no `USER` (the container
    runs as root) and `docker-compose.jetson.yml:21` sets `privileged: true` with
    `/opt/mousedroid` mounted read-write. Reachable only by someone who can already edit the
    rover's YAML or `docker.env`, so it is defence-in-depth rather than a remote chain — but
    the precedent is cheap and the plan is adding two more cache-path fields.
42. **`deploy_remote.sh` does not share the known-hosts store the rest of the tooling
    pins.** `jetson_discover.sh:56,193` and `jetson_bootstrap.sh:74,116` both set
    `-o UserKnownHostsFile="${HOME}/.mousedroid/known_hosts"`; `scripts/deploy_remote.sh`
    contains no `UserKnownHostsFile` at all, so it falls back to `~/.ssh/known_hosts`. The
    key accepted during discovery is not the key checked during deploy, and the
    trust-on-first-use window opens twice in two stores. `StrictHostKeyChecking` is never set
    to `no` anywhere — all seven sites use `accept-new`, which is defensible for a
    LAN-attached rover — so the defect is the split, not the policy. Compounding it,
    `jetson_discover.sh:19,187-199` SSH-probes three /24 subnets and selects the first host
    answering as `jetson` with `/etc/nv_tegra_release`; `deploy_remote.sh:115-124` persists
    that to `~/.mousedroid/jetson_host` and then runs `rsync -avz --delete` and
    `ssh … sudo --`. And `jetson_bootstrap.sh:156` grants the target user
    `NOPASSWD:ALL`. Since rev B extends `deploy_remote.sh`, the new path must adopt the
    shared store, require explicit operator confirmation of the host-key fingerprint rather
    than auto-promoting a scan result, and scope the remote sudo rule to the specific unit.
43. **Credential-in-argv precedent to avoid.** `scripts/jetson-runner-install.sh:124` passes
    `--token "${RUNNER_TOKEN}"` on a command line, visible in `ps` for the duration. It is
    the only place in the tree where a credential reaches argv; the token is otherwise
    correctly sourced from the environment at `:82`. The plan's promotion path must keep
    secrets in the child environment or on stdin.
44. **The Hugging Face push is structurally safe but documents a variable nothing reads.**
    No code in the repo reads `HF_TOKEN`, `HUGGINGFACE_TOKEN` or `HUGGING_FACE_HUB_TOKEN`;
    `HfApi()` is constructed with no `token=` (`export_dual_stream_rssm_onnx.py:481-487`),
    delegating to `huggingface_hub`'s own chain, and `_push_to_hf` logs only `repo_id`,
    `filename` and `local_path`. CI holds no HF credential — zero `secrets.*` references in
    any workflow — so an accidental `--push-to-hf` in CI cannot publish. Three defects the
    plan should fix while it is there: the `--push-to-hf` help text at `:410` names
    `HUGGINGFACE_TOKEN`, which nothing consumes, so a wrong-identity or 401 failure is
    silent; `--push-to-hf` is a bare `store_true` with no dry-run and no confirmation before
    publishing to a public repo; and the repo id is hardcoded at `:501,516`, duplicating the
    `onnx_repo_id` schema default. Adding an HF token to CI secrets would remove the property
    that makes the current posture safe.
45. **Naming constraint for any committed digest.** `deployments/jetson-image.json:4` already
    ships a 40-hex `"sha"` through the blocking full-history gitleaks scan, so a digest field
    is fine — but the default `generic-api-key` rule fires on keyword + assignment +
    entropy, and its keyword list includes `key`, `token`, `secret`, `auth`, `api`,
    `credential`. Name the field `sha256` or `digest`, never `model_token` or `bundle_key`.
    `.gitleaks.toml` itself is clean: four exact fake-key regexes, zero path allowlists, and
    a header stating the invariant ("allowlist by exact fake-key REGEX only, NEVER by path").
    The plan should add no allowlist entry.
46. **The existing self-hosted Jetson runner cannot be driven by a pull request.**
    `jetson-nightly.yml` triggers on `schedule` and `workflow_dispatch` only (`:24-35`), has
    `permissions: contents: read` (`:41-42`) with a written rationale about the long-lived
    physical footprint, and uses `concurrency` with `cancel-in-progress`. One residual: the
    `PILLARS` dispatch input reaches an intentionally unquoted shell expansion at `:110-115`.
    Reaching it requires write access, so it is privilege retention rather than escalation —
    but if a promotion job is ever added to this runner it must not widen the trigger set
    and must not add another unquoted dispatch input.

## Harness and coverage pins — round 2

47. **`WorldModelConfig` has zero executed test coverage anywhere in CI.**
    `tests/unit/factory/test_factory_world_model_engine.py` is the only test file in the
    repository that constructs `WorldModelConfig` (`grep -rln "WorldModelConfig("
    tests/` returns it alone). It opens with `importorskip("ncps")`, `importorskip("onnx")`
    and `importorskip("onnxruntime")` (`:22-24`), so it skip-alls in the blocking `test` job,
    which installs `.[dev,telemetry,mcp]` (`ci.yml:222`) and `[dev]` carries no ONNX
    packages. And the advisory `onnx-world-model-extras` job runs only
    `tests/unit/world_model` plus `tests/unit/training/test_export_dual_stream_rssm_onnx.py`
    (`ci.yml:600-601`) — not `tests/unit/factory/`. So the file executes in **no** CI path.
    Since no `config/*.yaml` declares a `world_model:` block, `config-validate` does not
    exercise it either. The plan adds eight fields to a class nothing currently pins, and
    `build_world_model`'s ONNX dispatch is likewise unpinned. Either add that file to the
    extras job's path list in the same change, or — better — add always-on integration and
    regression-pair coverage for the new fields.
48. **Suppression ratchets have zero headroom.** `python -m tools.ratchet_budgets` reports
    `noqa: 19` against a ceiling of 19, `type_ignore: 8` against 8, and
    `hardcoded_ok: 26` against 26 — all three at the ceiling, all three already past their
    early-warning threshold. Ceilings come from `.claude/workforce.yaml`
    `ratchet_budgets.items` and are enforced hard by
    `tests/regression/test_suppression_budget.py` in the blocking `test` job and by
    `python -m tools.ratchet_budgets --strict` in `local-gates` (`ci.yml:428`). So this
    change **cannot add a single `# noqa`, `# type: ignore` or `# hardcoded-ok`** under
    `src/mousedroid/**` without first removing an existing one. That matters specifically
    here: ORT's `io_binding()` and `InferenceSession` surfaces are `Any`-shaped, which is
    exactly where `mypy --strict` invites a `type: ignore`. Moving a suppression into
    `pyproject.toml` is not an escape either — `test_suppression_budget.py` budgets
    file-level `per-file-ignores` entries targeting `src/` against a frozen allowlist.
49. **Reserving F-050/F-051 early is safe, but the validation scripts must land with them.**
    `harness/spec.py:362-363` skips any feature whose `status != "done"` before both the
    `implemented_in` git-reachability check and the `runner(f)` call, and
    `features.schema.json:28-37` requires `validation_command` and `implemented_in` only when
    `status == "done"`. So `status: "in_progress"`, `implemented_in: null` with a
    `validation_command` set is a legal green state and the command is not executed. But
    `tests/regression/test_harness_spec_aqa.py:125-137`
    (`test_referenced_validation_scripts_exist`) has **no status filter** and runs in the
    blocking `test` job, so `scripts/validations/F-050.sh` and `F-051.sh` must exist and be
    non-empty in the same commit as the catalog entries. Note also that
    `scripts/validate.py:48-55` `--check <id>` bypasses both filters, and
    `feature-closeout/SKILL.md:76` tells operators to run it — so the stubs should fail
    honestly rather than `exit 0`.
50. **Two of the files this change touches most are subject to the branch-coverage floor.**
    `scripts/check_branch_coverage.py` exempts `config/schema/` and `telemetry/metrics/` via
    `_ALLOWED_DIR_PREFIXES` (`:59-64`) and `factory/world_model.py` via `_ALLOWED_FILES`
    (`:87`), but **not** `common/onnx_session.py` or `world_model/dual_stream_rssm_onnx.py`.
    The allowlist cannot be quietly extended: its exact contents are pinned by
    `tests/unit/scripts/test_check_branch_coverage_base_ref.py:429-484` and
    `tests/regression/test_f042_aqa.py:48,66`. The gate is local-only by design
    (`scripts/ci.sh:177-179`, with the omission itself pinned as deliberate at
    `test_ci_gate_wiring_aqa.py:543-549`), so it will not fail a PR — but it will fail
    `bash scripts/ci.sh` for any contributor.
51. **The blocking gate is satisfiable, and the repo already shows how.** Because
    `warmup_session` imports `onnxruntime` lazily inside the function
    (`common/onnx_session.py:190`) and `resolve_providers` is a pure function,
    `tests/unit/common/test_onnx_session.py` covers the whole helper with **no
    `importorskip`** — it injects a stub ORT into `sys.modules` via
    `monkeypatch.setitem` (`:106-115`, stub surface at `:48-103`).
    `tests/unit/vla/test_distilled_onnx.py:117-121` uses the identical fixture. So rev C must
    factor the new policy logic — execution-mode selection, the strict-versus-fallback
    decision, provider validation and its error text — as pure or stub-ORT-testable code in
    always-on test files, and confine real-`InferenceSession`, I/O-binding and numerical
    work to the advisory job and `tests/hardware/`. A strict-policy branch reachable only
    through a real session is uncoverable in the blocking and local gates.
52. **`common/` and `world_model/` are not exempt from the hardcoded-value gate.**
    `scripts/check_no_hardcoded_values.py:59-66` exempts `config/schema/`,
    `telemetry/metrics/`, `telemetry/server/`, `validation/runtime/`, `factory/` and
    `orchestrator/_` — not the two modules this change edits most. New literal provider
    names or thresholds there will trip it. Provider strings belong in schema config or in
    the existing `DEFAULT_ORT_PROVIDERS` constant (`common/onnx_session.py:47-51`).
53. **A `tier: hardware` validation command never runs in any GitHub-hosted path.**
    `harness.yml` is the only caller of `scripts/validate.py`: `validate-fast` runs
    `--tier fast` on pull requests, and `validate-slow` runs
    `--tier fast,slow --strict-git` on the nightly schedule only (`:89`) — the single
    `--strict-git` invocation in the repository. Neither installs ONNX extras. The hardware
    tier is deliberately left to the rover, and `jetson-nightly.yml` runs the Ten-Pillars
    script rather than `validate.py`. So if F-050 is ever flipped to `done` with
    `tier: fast`, its command runs on `ubuntu-latest` without onnxruntime, where an
    ORT-dependent test skips and `_run_always_on_pytest.sh` then fails it on
    `passed=0` ("skip-all is not done"). F-050's validation script must target always-on
    stub-ORT tests; the real-hardware claim belongs to F-051 as `tier: hardware`.
54. **`features.yaml` has no evidence-link field.** `features.schema.json:11` sets
    `additionalProperties: false` and the property list is exactly `id, epic, name,
    description, category, priority, status, tier, verification, validation_command,
    implemented_in, depends_on, notes`. `evidence-commit/SKILL.md:22` therefore routes
    evidence paths through the free-text `notes:` field. `reports/endurance` is one of six
    families gitignored by policy (`SKILL.md:24-38`, mirroring `.claude/workforce.yaml`
    `evidence.local_only_declared`), so the endurance JSON this change produces is closed out
    as a declared local-only chain — the artifact exists locally plus a `CHANGELOG.md`
    reference — not as a tracked artifact. That is also why the report must not be committed
    (pin 38) and why closing F-051 without either chain would be caught by
    `hw-evidence-auditor`.
55. **Tiers the plan is missing entirely.** There is no ONNX `tests/property/` test (zero
    grep hits) — provider and mode resolution is a natural Hypothesis target (order
    preservation, idempotence, strict implies raise-or-exact-match); no ONNX
    `tests/integration/` test, which is where `build_world_model`'s new fields would get
    their always-on exercise through a stubbed ORT seam; no `WorldModelConfig`
    backwards-compat regression test at all; and only a metrics-family pin in
    `tests/smoke/`. Also: do not register a new pytest marker —
    `pyproject.toml:382-387` registers exactly `slow`, `hardware`, `smoke`, `pillar` and
    `--strict-markers` turns an unregistered marker into a collection error.
56. **Both `tests/hardware/conftest.py` and `tests/performance/conftest.py` un-mock
    hardware unconditionally.** `tests/hardware/conftest.py:24-27` is an `autouse` fixture
    setting `MOUSEDROID_MOCK_HARDWARE=false` with **no** host guard; the Jetson check lives
    only in the opt-in `jetson_settings` session fixture (`:35-47`), and even there it
    demotes to mock via `model_copy` rather than skipping.
    `tests/performance/conftest.py:15-18` has the same shape, with the Jetson skip in the
    opt-in `runtime_settings` fixture. So a new file in either directory must gate itself at
    module level and must never rely on the conftest:
    `pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not is_jetson_host(), ...)]`
    plus `pytest.importorskip("onnxruntime")`. Without `pytest.mark.hardware` the file also
    runs nowhere on the rover, since `jetson_full_validation.sh:255` selects `-m hardware`.
    And per `hardware/CLAUDE.md` mock discipline, no `mock`/`patch` on the device under
    test — which is the reason the stub-ORT tests belong in `tests/unit/`.

## Runtime-reality pins — round 2 (and three corrections to the pins above)

These change the plan's premise more than anything in round 1.

57. **The production world model never loads trained weights.** `build_world_model` returns
    `DualStreamRSSM(cfg.model)` (`factory/world_model.py:108`) or `RSSM(cfg.model)` (`:113`)
    — freshly constructed. There is no `load_state_dict`, `torch.load` or `state_dict`
    anywhere in that module, nor in `orchestrator/`. The one OTA seam is dead:
    `factory/telemetry.py:315` `build_weight_update_loader` ends in an unconditional
    `return None` ("Tier C1 ships the seam; the operator pulls the concrete loader through
    configuration in a follow-up PR"). No config file carries a world-model checkpoint path.
    So `observe_step` on the rover is a forward pass through **random weights**. Making it
    faster changes nothing observable, and a numerical-parity gate against random weights is
    vacuous. The plan must state this precondition.
58. **Production does not even run the architecture the plan optimizes.**
    `config/jetson_production.yaml` has no `model:` block, so `cfg.model.cfc_hidden_dim` is
    the schema default `0`, `factory/world_model.py:91` (`if cfg.model.cfc_hidden_dim > 0`)
    is False, and the rover builds plain `RSSM` — not `DualStreamRSSM`. Only
    `config/jetson_dual_stream.yaml:29` sets `64`, and that file gates itself at `:8`: "Only
    enable (64+) after human review of training metrics." So the plan's A/B is `RSSM`
    (torch, ~513 k params) versus `DualStreamRSSM` through ORT/TensorRT (~574 k params) —
    **two different models**. Any measured speedup is confounded. This is sharper than pin 8:
    not only is `engine: onnx_trt` unreachable from the production overlay, the torch
    baseline is a different class.
59. **A cold TensorRT build will emergency-stop the rover.** `observe_step` warms lazily
    (`dual_stream_rssm_onnx.py:223-224`) with no startup call site;
    `_update_world_model` is **synchronous** (`_world_model_state_mixin.py:28`) and is called
    inside `tick()` (`orchestrator.py:521`); and `run()` wraps the tick in
    `await asyncio.wait_for(self.tick(), timeout=tick_timeout)`
    (`_lifecycle_mixin.py:366`) with `tick_timeout_s` defaulting to `1.0`
    (`config/schema/misc.py:181-184`), not overridden in the production overlay. On timeout,
    `_lifecycle_mixin.py:380-392` records a critical failure, calls
    `await self._esp32.emergency_stop()`, and plays the error voice. `asyncio.wait_for`
    cannot preempt a synchronous call, so a multi-second engine build runs to completion and
    then trips the timeout on the next scheduler pass. `loop_overrun_warmup_ticks: 30`
    (`config/schema/reward_safety.py:255-264`) does not help — it gates the safety monitor's
    `max_loop_time_ms` interlock, and the `tick_timeout_s` branch has no tick-index guard.
    Engine caches mitigate only warm starts; cold builds happen on first deploy and after
    every TensorRT, driver or ORT version change. Rev B's Phase 5.6 already moves warmup
    behind `asyncio.to_thread`; this pin makes it a precondition of landing any
    `engine: onnx_trt` code, not hardening.
60. **`onnxruntime` is never installed in the Jetson image.** `Dockerfile.jetson` runs
    `pip install --no-cache-dir -e "."` with **no extras**, and its own comment at `:88`
    says so: "the base `pip install -e "."` only resolves required deps." `onnxruntime-gpu`
    lives in the `[vla]` and `[onnx_world_model]` extras (`pyproject.toml:104-106`,
    `:110-124`). The only transitive source is `piper-tts` (`Dockerfile.jetson:146`), which
    pulls the generic **CPU** wheel — reporting neither `TensorrtExecutionProvider` nor
    `CUDAExecutionProvider`. And `resolve_providers` silently returns
    `("CPUExecutionProvider",)` on an empty intersection
    (`common/onnx_session.py:94-100`). So the plan could ship, run `engine: onnx_trt` on the
    CPU provider, be *slower* than the torch baseline, and emit only an INFO log.
    Whether the `dustynv/l4t-pytorch:r36.4.0` base preinstalls a GPU-enabled ORT is not
    knowable from the tree. The plan must add an explicit install stage and the fail-loud
    provider check, which is what `onnx_require_primary_provider` is for.
61. **`workspace_gb: 2.0` is 2 GiB of builder workspace for a ~574 k-parameter graph, and
    nothing currently reads it.** `config/jetson_production.yaml:66` sets it;
    `grep -rn "workspace_gb" src/` finds the schema field plus
    `efficiency/tensorrt.py:131,189,250` (the torch2trt compiler), whose builder
    `build_tensorrt_compiler` (`factory/hardware.py:422`) has **zero production callers**.
    `jetson.gpu_memory_fraction` is likewise inert — schema-only, with
    `config/jetson_production.yaml:62` setting `0.7` into a void. So rev B's D-5 decision to
    source workspace and precision from `cfg.jetson` picks the right *fields* but wires them
    from a path that currently reaches nothing, and inherits a value that is absurd for this
    graph. The weights themselves are a red herring at single-digit MB; the costs to budget
    are builder workspace, the TensorRT builder-resource library, and build-time peak —
    which the plan never separates from steady state.
62. **The rover is documented to be at its accelerator-allocation ceiling with one tenant.**
    `docs/runbooks/jetson-claude-pilot-deploy.md:118-122`: "Phi-3 fallback must run on CPU on
    this host: `MOUSEDROID_LLM__N_GPU_LAYERS=0`. The world model already occupies the shared
    7.4 GB iGPU, so a full-offload second model fails with `unable to allocate CUDA0
    buffer`." Corroborated at `docs/runbooks/jetson-full-bringup.md:31`,
    `deployments/jetson-image.json:6`, and as a STOP gate in
    `docs/superpowers/plans/2026-07-15-trunk-reconcile-jetson-docker-validation-v3.md:882`. Note the committed YAML
    (`config/jetson_production.yaml:119`, `n_gpu_layers: -1`) describes a configuration
    documented to crash, and only the uncommitted per-host `docker.env` override saves it.
    Container limit is `memory: 6G` with no `memswap_limit`, `shm_size` or `pids_limit`
    (`docker-compose.jetson.yml:145-153`) against 7.4 GB usable and ~4.6 GB measured
    available; unbounded swap means a 30 Hz loop thrashes rather than restarting cleanly,
    and every swapped page fault inside the synchronous `_update_world_model` counts against
    pin 59's 1.0 s timeout. **But** the attribution in that runbook is probably wrong: the
    world model is CPU-resident today (no `set_default_device`, no `.to(cuda)` on any live
    path, latents allocated on CPU at `orchestrator.py:468-477`, and
    `pack_observation(..., device=torch.device("cpu"))`). So nobody in the tree knows what
    the iGPU budget actually is — which cuts both ways for the plan.
63. **`get_safety_trace` has zero production callers — correcting this bundle's own
    proposal.** `grep -rn "get_safety_trace\|safety_trace"` outside `world_model/` and
    `tests/` returns only the comment at `factory/world_model.py:267-270` that claims "the
    safety monitor's CfC inspection" needs it. `safety/monitor.py` contains no reference to
    `cfc`, `safety_trace` or `world_model`; the only callers are
    `tests/unit/world_model/test_composite.py:195,207` and
    `test_dual_stream_rssm.py:336,348`. `composite.py:135`'s claim that delegation "keeps the
    safety monitor wired" is false — it was never wired. Relatedly, `build_planner`, which
    ADR-008:80 names, **does not exist**: the real builder is
    `factory/cognitive.py:34` via `build_agent`. So the second engine in the composite
    currently serves one exception-path method and one dead one. Both comments need
    correcting, and this bundle's own `proposal.md` overstated the case when it
    called the PyTorch graph "load-bearing for the safety monitor" — corrected in place.
64. **The adaptive MCTS budget is dead in production — correcting pin 29.**
    `_world_model_state_mixin.py:41` discards two of the four outputs
    (`self._h, self._z, _, _ = …`), and `SafetyContext.surprise` is only ever its default
    `0.0` (`safety/context.py:21`; nothing in `safety/` or `orchestrator/` assigns it from
    the world model). So `compute_mcts_budget(safety_ctx.surprise, …)`
    (`agents/navigation.py:82-86`) always receives `0.0` and the budget is permanently pinned
    at `n_simulations_base`. Pin 29's "worst-case latency in the most novel situations" is
    therefore a latent property of the code, not current behaviour — the surprise signal
    never reaches the planner. Two consequences for the plan: the ~300-call figure in pin 26
    is the *fixed* cost, not a floor; and an I/O-binding implementation should bind only
    `new_h` and `new_z` to persistent device buffers, since `obs_embed` and `surprise` are
    discarded every tick and paying a device-to-host copy for them is pure waste.
65. **Refining pin 26: the MCTS fallback is reached only from an exception branch.**
    `_try_cognitive_action` (`_action_mixin.py:210-277`) returns `None` **only** from its
    `except Exception` branch (`:264-277`), and `cognitive.enabled: true` in the production
    overlay (`:200`). So `self._agents[0].act(...)` at `:66` is an error path, not a
    steady-state one. That makes pin 28's point sharper rather than weaker: the expensive
    path is the *degraded* path, entered precisely when something has already gone wrong.

### What survives, stated plainly

The plan's structural premise holds. `observe_step` is unconditionally on the 30 Hz path —
`orchestrator.py:513-522` calls `_update_world_model` with no config gate, no cadence
divider and no enable toggle, and it runs *before* the `if safety_ctx.is_emergency:` branch
at `:531`, so it executes on emergency ticks too. And the instrumentation rev B's abandon
gate depends on already exists and is on by default: `TickPhaseLiteral`
(`config/schema/_primitives.py:72-81`) has exactly eight phases in tick order —
`sense`, `safety`, `world_model`, `plan`, `act`, `learn`, `telemetry`, `post` — written via
`_mark_phase` → `observe_tick_phase_ms` and gated by `metrics.track_tick_phases`, default
`True`.

There is no per-phase budget to compare against: `_primitives.py:84-89` states phase timings
"are diagnostics only — no phase timing has emergency-stop authority, because phases do not
sum exactly". The only real ceilings are whole-tick — `safety.max_loop_time_ms: 200.0` with a
3-tick debounce and 30-tick warmup, and `loop.tick_timeout_s: 1.0`. So "material share" is a
number rev C must choose and justify, not one it can look up.

### Ordering this imposes on rev C

Before any ONNX runtime work: state pins 57 and 58 as unmet preconditions (no trained
weights; production runs `RSSM`, not `DualStreamRSSM`), fix pin 59 (off-loop warmup) and
pin 60 (install `onnxruntime-gpu`, fail loud on provider downgrade). Pins 57 and 58 are not
this change's to fix, but a latency A/B across two architectures on random weights is not
evidence, and the proposal must say so rather than imply otherwise.

---

# Round 3 — AI/ML and DevOps lenses

A third pass through two lenses the first two rounds did not apply: the model itself, and
release engineering. Verified against `a46fbad`.

**Round 3 finds that this exact decision has already been made in this repository, against a
sibling optimization, with a written rubric — and the verdict was DEFER.**

## The house rubric already exists, and this plan scores zero against it

66. **`docs/analysis/alayaworld-distillation-spike.md:65-75` is a ratified three-part GO
    rubric for accelerating the world model**, all three required:

    > 1. **Primitive**: ≥3× p95 primitive speedup at the chosen k **on Jetson**.
    > 2. **Accuracy**: ≥0.90 action agreement at that k (trained-checkpoint teacher).
    > 3. **Consumer case**: a written plan for the MCTS rollout-leg integration whose
    >    projected end-to-end gain (bounded by the ~1.25-1.6× ceiling) justifies the added
    >    model + maintenance — or an alternative consumer … with its own budget case.

    Its verdict: **"Recommendation (provisional): DEFER."** Scoring this plan against it as
    the tree stands: (1) unmeasured — no on-device latency evidence exists anywhere;
    (2) unattainable — there is no trained checkpoint and production runs random weights
    (pin 57); (3) never performed. The plan proposes a larger change than the spike and
    supplies less evidence.

67. **The Amdahl ceiling is already computed, verified against the planner, and printed to
    the operator.** `scripts/spike_step_distillation.py:57-61`:

    > ```python
    > # Consumer-ceiling constants for the report (verified against MCTSPlanner at
    > # defaults: 9-candidate expand + 50 sims x depth-5 rollouts + re-expansions).
    > _PLAN_CALLS_ESTIMATE = "500-650"
    > _ROLLOUT_SHARE = "~40%"
    > _CONSUMER_CEILING = "~1.25-1.6x"
    > ```

    and `:340-342` prints: "MCTS plan() makes ~500-650 imagine_step calls; rollouts are ~40%
    of them, so end-to-end planner gain caps at ~1.25-1.6x **regardless of the primitive
    speedup above**." That bound applies to optimizing the *rollout* leg — 500-650 calls at
    ~40% share. This plan optimizes `observe_step`: **one** call per tick. Its share is
    strictly smaller, so its end-to-end ceiling is strictly worse, and it is derivable today
    by the same method without touching the rover. The plan must compute it in the proposal.

68. **Measured world-model step latency is sub-millisecond to ~1 ms.**
    `reports/spike_step_distillation.json` records `primitive_latency_ms` p50 of 0.274 ms
    (k=2), 0.532 ms (k=4) and 1.136 ms (k=8) on `container-cpu`. Against a 33.33 ms period
    that is roughly 1–3% of a tick. These are k-step primitives on a random-init `RSSM`, not
    `observe_step` on `DualStreamRSSM`, so treat them as scale-setting rather than as the
    number — but they are the only measured world-model latencies in the repository, and they
    put the optimization target an order of magnitude below the budget it is being justified
    against.

69. **Correcting pin 26.** I estimated ~300 `imagine_step` calls per `plan()` from
    `n_simulations_base × (1 + rollout_depth)`. The repo's own verified figure is
    **500-650**, because it also counts the 9-candidate initial expansion and re-expansions
    (`spike_step_distillation.py:57-58`). Use the repo's number.

70. **The house accuracy gate is task-level action agreement, not tensor tolerance.** The
    rubric asks for ≥0.90 action agreement; the spike measured 0.422 / 0.609 / 0.422 at
    k=2/4/8 and explicitly attributed the failure to the teacher: "the teacher is a
    random-init RSSM (near-flat reward surfaces make the argmax grid nearly a coin toss
    across similar candidates)" (`:86-89`), concluding "The Jetson run must use a trained
    checkpoint before this criterion is judged." Note the non-monotonicity in k — 0.42 →
    0.61 → 0.42 — which is what noise-dominated measurement looks like. Rev C's FP16 gate
    should adopt action agreement against the torch engine as its task-level criterion,
    alongside (not instead of) the tensor tolerances, because that is the metric this repo
    already uses to decide.

71. **`new_z` is empirically ~58× noisier than `new_h`, in the repo's own measurement.**
    `reports/drift_comparison.json` (F-023 drift comparison, 8 episodes, seq_len 48) reports
    baseline mean MSE `latent_h` 0.0304 versus `latent_z` 1.7669, and finals 0.0349 versus
    1.8447. That independently justifies ADR-008's exclusion of `new_z` from the parity
    check, and it means any FP16 tolerance placed on `z` would be meaningless — the signal is
    dominated by the sampling variance, not by precision.

72. **Three never-closed on-device gates form a pattern this plan would extend.**
    `reports/endurance/` holds only `.gitkeep` (pin 30); `reports/spike_step_distillation.json`
    carries `"environment": "container-cpu (Jetson measurement pending operator run)"`; and
    `docs/analysis/alayaworld-distillation-spike.md:54-63` has a "Results — Jetson Orin Nano
    (operator run — PENDING)" section whose table is a single row of em-dashes, with
    "**The Jetson criterion is UNMET until this section is filled**". The repository reliably
    produces careful CPU-side spikes and reliably does not close them on hardware. Rev C's
    Phase 4 is the fourth such gate, and it should say plainly that the pattern is the risk.

### What this changes in rev C

Replace the invented "material share" criterion in Phase 4 with the existing three-part
rubric, and compute the `observe_step` consumer ceiling — by the method
`spike_step_distillation.py` already implements — **in the proposal, before Phase 2**. It is
a desk calculation, needs no rover, and on the evidence above it is likely to return a number
that does not justify I/O binding, FP16 or CUDA Graph. That is a successful outcome for a
review, not a failed one: the instrumentation (Phase 2), narrative corrections (Phase 3),
artifact integrity (Phase 6), the `onnxruntime-gpu` install and provider proof (Phase 1b), and
the delivery and security hardening (Phase 7) all stand on their own merits and are worth
landing regardless of whether the latency work ever does.

## MLOps pins — round 3

73. **The production action path can run on random weights, observable only as one log
    line.** `_resolve_bdi_weights` (`factory/cognitive.py:38-85`) tries local weights, then a
    Hugging Face download, then falls through to
    `_log.warning("weights_not_found_using_random_initialization")` and returns `NeuralBDI()`
    — random init. The `weights_source` label it returns is used only in a structured log at
    `:133`; there is **no metric, no health-check gate and no startup refusal**. Combined with
    pin 57, the rover can be flying a random-weight world model *and* a random-weight
    cognitive core, and nothing observable distinguishes that from a trained deployment.
74. **The BDI download has no integrity verification either.** The same function calls
    `download_weights_from_huggingface(repo_id=..., filenames=[belief/desire/intention/affect
    .npz], ...)` with no `revision=` pin and no digest check (`grep sha256|verify_sha|revision`
    over `factory/cognitive.py` returns nothing). This is the path that actually executes on
    every boot, since `cognitive.enabled: true` and `auto_download: true` in
    `config/jetson_production.yaml:200,203`. Pin 39's point therefore applies more urgently
    here than to the ONNX artifact the plan is about: the fail-closed `verify_sha256` +
    `sha256.txt` machinery exists and this path does not use it.
75. **Generalising the plan's own best idea.** `onnx_require_primary_provider` — fail closed
    when the intended execution provider is not actually active — is the right pattern, and
    the repository has no equivalent for weights. Rev C should carry a matching requirement:
    a model that falls back to random initialisation SHALL increment a metric and SHALL be
    refusable by config, and any benchmark or promotion record SHALL state which weight
    source loaded. Otherwise the plan's evidence is measured against an unrecorded weight
    state, which is the same defect as an unrecorded execution provider.

## ML pins — round 3

76. **The plan's two parity requirements were mutually contradictory, and one is impossible.**
    `new_z` is not a leaf output: `_world_model_state_mixin.py:41` assigns `self._z` and feeds
    it back, and `dual_stream_rssm.py:283` builds
    `recurrent_input = torch.cat([z, prev_action], dim=-1)` for both the GRU (`:284`) and the
    CfC. So `new_h(t+1) = f(new_z(t))`. Excluding `new_z` from parity while gating multi-step
    `new_h` parity cannot hold — two independent RNG streams diverge by O(‖post_std‖) at t=2,
    orders of magnitude above any precision tolerance, so the gate would fail a **correct**
    FP32 implementation. Resolved by making `eps` an injected graph input (task 6.10), which
    makes `new_z` gateable and multi-step parity meaningful. Until then parity is single-step.
77. **ADR-008's justification for excluding `new_z` is false and needs correcting.** It says
    "consumers that depend on the specific sample (none, in the current architecture)". There
    are three: `observe_step` itself next tick, `MCTSPlanner.plan(h, z)` via
    `agents/navigation.py`, and the VLA policy's `VLAObservation(h=..., z=...)`. The
    *conclusion* (exclude it from single-step parity) is still right; the reasoning is not.
78. **My own "changing inputs produce changing outputs" scenario was vacuous.**
    `sample_gaussian` draws `randn_like` unconditionally on every call, so outputs differ even
    with byte-identical inputs and a completely broken buffer. Replaced with a fixed-input
    test asserting `new_h`/`obs_embed`/`surprise` identical while `new_z` differs. Note also
    that nothing in the repo asserts ONNX `new_z` varies across calls, so a graph whose sample
    was folded to a constant would pass the whole suite and silently stop being stochastic on
    the rover — the export passes `do_constant_folding: True`.
79. **Two `randn_like` draws occur per step, not one.** `observe_step_traceable` samples the
    prior and discards it. If ONNX dead-code elimination removes that unreachable draw while
    torch still consumes it, the streams desynchronise at *different* rates, so seeding alone
    could never align them. Removing the dead draw is also free latency on the hot path.
80. **FP16 has two unguarded overflow sites, and the sibling function shows the fix.**
    `latent_utils.py::kl_divergence` — the one `observe_step` calls — has no clamp and no
    float32 cast, while `balanced_free_bits_kl` ten lines away does both and says why. In fp16
    `post_logvar.exp()` overflows above ~11.09 and dividing by `prior_logvar.exp()` overflows
    for any O(1) numerator once `prior_logvar < ~-11.1`; `logvar_clamp` defaults to `10.0`,
    which does not save it. `sample_gaussian`'s `exp(logvar * 0.5)` overflows above ~22.2 and
    feeds `inf` into the recurrent state.
81. **The only production drift guard cannot fire.** `_validate_latent` warns when
    `norm(h) > model.latent_norm_threshold`, default `50.0`. A `nn.GRUCell` state is a convex
    combination of `tanh` outputs, so `norm(h) <= sqrt(256) = 16` at the production
    `hidden_dim`. Sub-NaN FP16 drift is unobservable, so "no safety-envelope regression" has
    no instrument behind it.
82. **An fp16 graph would fail on the first live tick, not at warmup.**
    `run_session_with_zeros` is dtype-aware and zero-fills `tensor(float16)` correctly, but
    `observation_packer.pack_observation` hardcodes `dtype=torch.float32` at every
    construction site. Warmup passes; `observe_step` raises `InvalidArgument` on the first
    sensor read. And `jetson.precision` cannot gate any of it: its only consumer is
    `efficiency/tensorrt.py` via `build_tensorrt_compiler`, which has no production caller.
83. **The training objective can collapse, and the repo already knows.** Both scripts that
    produced the published weights regress the decoder onto the encoder's **own
    grad-attached output**: `training/train_rssm.py:249,268`
    (`obs_embed = rssm.encoder(...)` then `mse_loss_fn(obs_recon, obs_embed)`), with no
    `.detach()` and no fixed target — and `train_dual_stream_rssm.py` does it three times.
    Encoder → 0 minimises both that term and the KL. The fix is documented on
    `rssm.py:280`: `train_sequence` reconstructs raw per-modality targets, "fixed targets, so
    the objective cannot collapse the way an `obs_embed` self-reconstruction would". It was
    never back-ported to `training/`. `DualStreamRSSM` — the class this change accelerates —
    has no `train_sequence` at all.
84. **The reward head has never received a gradient.** `reward_head` appears twice in the
    tree: constructed at `rssm.py:51`, used at `:231`. Zero occurrences in `training/`. No
    loss touches it, so Adam skips it and it stays at its `kaiming_uniform_` init. MCTS's only
    value estimator is `_rollout`'s accumulated `imagine_step` reward, so the planner's entire
    value signal is a random projection of an untrained dynamics model.
85. **`best_ucb_c: 1.41` in the published artifact is the unmodified input, not a result.**
    `training/warmstart_policy.py::tune_ucb` initialises `best_ucb = base_cfg.ucb_c` (1.41) and
    only overwrites it inside `if mean_reward > best_reward and p50_ms < target_ms`. No
    candidate met `p50_ms < 50` — the card's own p50 is ~109 ms — so the assignment never ran
    and the function reported its input back under a key asserting it is an output. The card's
    rewards make 1.41 the **worst** of the five sampled values (0.145 against 0.4133 at
    ucb_3.0). Nothing else in that artifact should be cited without re-deriving it.
86. **The gate set is inverted relative to consumer risk.** The plan gates FP16 on `surprise`
    — the most fp16-fragile output (pin 80) — which reaches nothing:
    `safety/monitor.py` never passes it to `SafetyContext`, so `compute_mcts_budget` always
    sees `0.0`. Meanwhile `new_h`/`new_z`, which reach MCTS, the VLA policy and the next tick,
    were excluded or ungateable. Pin 76's fix corrects this.

## DevOps pins — round 3

87. **The health gate the plan depends on is a hardcoded `"ok"`.**
    `_lifecycle_mixin.py:601-612` — `health_check()` returns
    `{"status": "ok", "platform": ..., "mock_hardware": ..., "agents": [...]}`
    unconditionally. There is one `def health_check` in `src/`. `main.py::_health_check`
    exits non-zero only when `status != "ok"`, which is unreachable, so `--health-check`
    detects DI/config wiring failures and nothing else. The REST surface is no better:
    `_rest_handlers.py::_handle_health` always returns 200 and `health/monitor.py::check_health`
    derives status from GPU temperature alone. `docker_deploy.sh` probes it with `curl -sf`,
    which passes on any 2xx. The plan's `--strict-health` mode must therefore be written
    against real signals — heartbeat freshness, `mousedroid_loop_time_ms`,
    `mousedroid_tick_overruns_total`, `mousedroid_safety_violations`, or container
    `Health: healthy` — not against these two.
88. **The systemd restart budget is silently inert.** `scripts/mousedroid-docker.service`
    puts `StartLimitIntervalSec=300` and `StartLimitBurst=5` at lines 68-69, inside
    `[Service]` (section starts line 23). `StartLimitIntervalSec` is only valid in `[Unit]`;
    `systemd-analyze verify` on a scratch copy reports "Unknown key name
    'StartLimitIntervalSec' in section 'Service', ignoring." The effective policy is the
    10 s default, and with `RestartSec=10` at most one or two restarts land per window, so
    the limiter never trips and the unit never enters `failed`. Combined with
    `restart: unless-stopped` in compose — which the unit's own comment notes may stop
    `--abort-on-container-exit` from firing — a bad promotion crash-loops with no failure
    state and nothing to alert on. `tests/regression/test_systemd_unit.py` greps for
    `Restart=on-failure` as text and never checks section placement.
89. **"Promote an immutable commit" is true of the git SHA and false of the artifact.** Both
    `FROM` lines in `Dockerfile.jetson` are mutable third-party tags (`dustynv/llama_cpp:r36.4.0`,
    `dustynv/l4t-pytorch:r36.4.0`), not digests; there is no lockfile or constraints file
    anywhere (`git ls-files` finds none) while `pyproject.toml` and the Dockerfile both use
    ranges; `docker_deploy.sh` builds with `--no-cache`, so every deploy re-resolves against
    live PyPI; roughly a dozen install layers are `|| true`, so an image missing `ncps`,
    `picamera2`, `anthropic` or the GCP SDK still tags green; the Jetson image is never built
    in CI (the `docker` job runs `docker build --check`, a linter, and really builds only
    `Dockerfile.dev`); and there is no registry, push, signing, SBOM or provenance. The
    rollback anchor therefore exists in exactly one local daemon, recoverable from nothing.
90. **The TensorRT cache would default inside the promotion boundary.**
    `JetsonConfig.tensorrt_cache_dir` defaults to `/opt/mousedroid/tensorrt_cache` — inside
    the bind-mounted git checkout the promotion operates on — and `efficiency/tensorrt.py`
    writes `engine_<fingerprint>.pth` via `torch.save` with no eviction, size cap or pruning.
    A `git clean -fdx` on a promotion that wants a deterministic tree destroys it and forces a
    full rebuild at the worst moment. It is also a root-writable, root-deserialised pickle
    cache in a bind mount shared with the host, with the 0700 its own comment assumes set
    nowhere. Rev C's named-volume decision (task 8.1) is right; this pin says why it is
    mandatory rather than tidy.
91. **`sync_jetson_overlay.sh` inverts config authority on the promotion path.** It is
    well-built — sha256 compare, atomic `mktemp`+`mv`, explicit error checks, a `RETURN` trap,
    and a strict `--verify` mode that never mutates — but it runs as a dash-prefixed
    (non-fatal) `ExecStartPre`, so the **checkout** wins over the deployed config on every
    start and a sync failure is ignored. A rollback that reverts the image but leaves
    `/opt/mousedroid` at the new commit silently re-applies the *new* config to the *old*
    image, reintroducing exactly the crash-loop the `config-compat` gate exists to prevent.
    The promotion path should call `--verify` (undashed) and fail on drift. Two other writers
    compete: `docker_deploy.sh` copies a broader `config/*.yaml` set from a script-relative
    root, and `jetson-nightly.yml:87` mutates rover config from whatever `/opt/mousedroid` is
    checked out at while validating code from the runner's own workspace — two commits in one
    validation run.
92. **Nothing bounds logs, and nothing alerts on disk or memory.** `docker-compose.jetson.yml`
    has no `logging:` block, so the default `json-file` driver grows without bound under a
    30 Hz structlog stream; `config/loki/loki.yml` sets no `retention_period`; there is no
    `daemon.json` or logrotate config; and `scripts/jetson_disk_cleanup.sh` is manual with no
    timer. `config/prometheus/alerts.yml` has ten groups and zero recording rules, with no
    alert on disk, memory, swap or CPU temperature — only `mousedroid_gpu_temp_celsius`.
    `scripts/preflight_check.sh`'s `MOUSEDROID_MIN_DISK_GB` guard is real but start-time only.
    The plan adds a disk-backed cache and a memory-hungry build; neither would be noticed.
93. **No backup, no DR, no restore runbook.** The only backup in the tree is
    `host_bootstrap.sh`'s timestamped `docker.env.bak.*` — on the same disk that would fail.
    Not backed up: the `mousedroid_experience` LMDB volume (the only irreplaceable artifact),
    downloaded weights, `/opt/voice_models`, the TRT cache, and the image itself (pin 89).
    Recovery from a dead NVMe is a reflash plus a rebuild against mutable tags — a different
    dependency set — and total loss of accumulated experience.
94. **`pip-audit` does not audit what ships.** The `security` job installs
    `.[dev,telemetry,mcp]` on `ubuntu-latest`/amd64. The rover additionally carries torch,
    CUDA and TensorRT from the base image (never audited), `llama_cpp`, `anthropic`, five
    `google-cloud-*`, `hailort`, `ncps`, `piper-tts`, `picamera2`, `Jetson.GPIO` and more.
    Coverage of the base image is zero, and that is where CUDA and TensorRT live.
95. **Correcting one agent claim, for the record.** A DevOps review asserted that an alert
    against `mousedroid_tick_overruns_total` "would silently match nothing" because the
    registry field is `f"{ns}_tick_overruns"` (`_registry_core.py:101`). That is wrong:
    `primitives.py::_render_counter` sets `metric_name = f"{name}_total"` at render, so the
    **exposed** name is `mousedroid_tick_overruns_total` and an alert against it is correct.
    Round 2's pin 48 and pin 6 had this right. What *is* true is the narrower point that the
    registry-side field name carries no suffix, which is why pin 6 requires the registry name
    be declared without one.

## Implementation record — the task 2.1 gate fired

Four review rounds argued the consumer case was unproven. Implementation computed it, and the
number settles it. `scripts/analyze_observe_step_ceiling.py` derives the bound from `MCTSConfig`
and `LoopConfig` with no literals and reproduces `scripts/spike_step_distillation.py:59-60`'s two
published constants exactly (`"500-650"` imagine_step calls, `"~40%"` rollout share) as its
correctness anchor.

96. **The ceiling is 1.0016x-1.0039x.** `plan()` issues 259-628 `imagine_step` calls at
    `n_simulations_base=50`; `observe_step` is one call per tick. At equal per-call cost that is
    0.159%-0.385% of the tick's RSSM work, so Amdahl caps end-to-end gain at 1.0016x-1.0039x
    *at any primitive speedup*. To clear 1.25x — the **low** end of the sibling rollout leg's own
    ceiling, and the bar the ratified rubric at
    `docs/analysis/alayaworld-distillation-spike.md:65-75` used to return DEFER — `observe_step`
    would have to occupy at least **20% of tick time**. That is 52x-126x the structural reference
    point. The rubric returns DEFER again, for the same reason and by a wider margin.
97. **The conclusion is conditional and the analyzer says so.** No measured `observe_step` share
    exists in this repository; `reports/` and `smoke-reports/` still carry zero `tick_phase` or
    `observe_step` values, re-checked after Phase 3 landed. The analyzer exits refusing to
    conclude unless `--observe-share` is supplied, and records seven `unknown_inputs`. What
    decides the gate is not the exact share but the *distance* to the required one.
98. **Two review findings turned out to be load-bearing bugs, found by the tests, not the review.**
    (a) Wiring `MetricsRegistry` onto an `nn.Module` broke `copy.deepcopy(self._base_rssm)` at
    `learning/on_device/rssm_refiner.py:241` with `TypeError: cannot pickle '_thread.lock'
    object` — caught by `tests/integration/test_on_device_sim_soak.py`. Fixed with
    `ObserveStepTimingMixin`, whose state hooks drop the sink on copy; that is also the correct
    semantics, since an off-loop refinement candidate must not write to the production histogram.
    (b) `tests/_script_loader.py` never registered the module in `sys.modules` before
    `exec_module`, so any script declaring a `@dataclass` with a bare-identifier annotation under
    `from __future__ import annotations` died inside `dataclasses._process_class`. Both were
    latent before this change and neither appears in rounds 1-4.
99. **What the gate does *not* close.** Phase 3's instrumentation lands anyway: the
    `mousedroid_world_model_observe_step_seconds` family had **no** production writer, so
    `WorldModelObserveStepLatencyHigh` (`config/prometheus/alerts.yml:388-421`) could never fire.
    That is an observability defect independent of any speedup, and fixing it is what would turn
    pin 97's derived number into a measured one. Phases 7 and 8 — artifact integrity and delivery
    hardening — were justified in the proposal independently of the optimization and land on that
    basis, not on this one.
100. **Task 6.6 was mis-filed and is now task 2.4.** Off-loop warmup reads as part of the
     optimization but is a correctness fix: `observe_step` warms lazily, `_update_world_model` is
     synchronous, and `asyncio.wait_for` cannot preempt a synchronous call, so a cold engine build
     on the first tick e-stops the rover rather than merely being slow. It landed with Phase 2,
     before the gate closed, and stays landed after it — the hazard exists for any future engine
     with a session to build, optimized or not.
