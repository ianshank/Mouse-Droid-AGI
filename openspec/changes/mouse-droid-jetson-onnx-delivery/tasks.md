# Tasks — Jetson ONNX runtime evidence + PC-to-rover delivery

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/ tests/ tools/ scripts/ && python -m ruff format --check src/ tests/ tools/
python -m mypy src/ --strict --ignore-missing-imports
bash scripts/validations/F-050.sh
```

Task ordering is binding: each task lands green before the next starts.

Two standing constraints apply to every task below.

**No new suppressions.** `python -m tools.ratchet_budgets` currently reports `noqa: 19`
against a ceiling of 19, `type_ignore: 8` against 8, and `hardcoded_ok: 26` against 26 — all
three at the ceiling. `tests/regression/test_suppression_budget.py` enforces them in the
blocking `test` job and `--strict` runs in `local-gates`. So no `# noqa`, `# type: ignore` or
`# hardcoded-ok` may be added under `src/mousedroid/**` without first removing an existing
one, and a file-level `per-file-ignores` entry is not an escape — that set is frozen too.
This bites specifically on ORT's `Any`-shaped `InferenceSession` / `io_binding()` under
`mypy --strict`.

**New branches must be coverable without onnxruntime.** `common/onnx_session.py` and
`world_model/dual_stream_rssm_onnx.py` are **not** in `check_branch_coverage.py`'s allowlist
(`_ALLOWED_DIR_PREFIXES` covers `config/schema/` and `telemetry/metrics/`; `_ALLOWED_FILES`
covers `factory/world_model.py`), and the allowlist's contents are themselves pinned by
`tests/unit/scripts/test_check_branch_coverage_base_ref.py:429-484` and
`tests/regression/test_f042_aqa.py`. Policy logic — mode selection, the strict-versus-fallback
decision, provider validation and its error text — must therefore be pure or stub-ORT-testable
and land in always-on test files, following the `sys.modules` stub fixture at
`tests/unit/common/test_onnx_session.py:106-115`. Only real-`InferenceSession`, I/O-binding
and numerical work may sit behind `importorskip`. Also note `common/` and `world_model/` are
**not** exempt from `check_no_hardcoded_values.py` (`:59-66`), so new provider literals belong
in schema config or in `DEFAULT_ORT_PROVIDERS`.

There is no `openspec` CLI in this repository (`openspec/project.md:1-6`), so the bundle is
validated by `make gates` plus the two validation scripts, never by `openspec validate`.
Narrow pytest selections use the `Makefile::regression` form —
`python -m pytest <paths> -m "not hardware" --import-mode=importlib --no-cov -q` — because
`pyproject.toml:392` already carries `-v --import-mode=importlib --strict-markers` and the
coverage plugin's `fail_under = 90` (`:411`) fails any narrow run without `--no-cov`.

**Phase 1 — Governance and catalog (no runtime change)**

- [ ] 1.1 Reserve `F-050` (runtime + performance evidence) and `F-051` (PC-to-rover
      delivery) in `features.yaml` with `status: "in_progress"`, `implemented_in: null`,
      `depends_on: ["F-008"]` for F-051 and `["F-002"]` for F-050. Required fields per
      `features.schema.json`: `id, name, category, priority, status, verification,
      depends_on`. IDs are next-free, not sequential — ADR-013:63-74 makes F-009–F-014 and
      F-033 intentional holes.
- [ ] 1.2 `scripts/validations/F-050.sh` and `F-051.sh` **must exist and be non-empty in
      this same commit**: `tests/regression/test_harness_spec_aqa.py:125-137`
      (`test_referenced_validation_scripts_exist`) has no status filter and runs in the
      blocking `test` job. They will not be *executed* while status is `in_progress` —
      `harness/spec.py:362-363` skips any feature whose `status != "done"` before both the
      git-reachability check and the `runner(f)` call — but `scripts/validate.py --check
      F-050` bypasses both filters, so the stubs must fail honestly rather than `exit 0`.
      F-050 must be `tier: fast` pointing at always-on stub-ORT tests; the real-hardware
      claim belongs to F-051 as `tier: hardware`, because `harness.yml` never installs ONNX
      extras and `_run_always_on_pytest.sh` fails on `passed=0` ("skip-all is not done").
      Shape follows `F-048.sh`:
      `set -euo pipefail`, `cd "$(dirname "$0")/../.."`, then
      `bash scripts/validations/_run_always_on_pytest.sh F-0NN <paths>`. Do not restate
      `--import-mode=importlib --no-cov -q` — the helper appends them and fails when
      `passed < 1`.
- [ ] 1.3 Register the bundle in `openspec/project.md`'s Changes table
      (change-id, status, F-number, landed, authoritative artifacts).
- [ ] 1.4 `docs/architecture/ADR-019-onnx-runtime-provider-policy.md` naming the ADR-008
      clauses it supersedes (config surface, provider proof, parity scope), plus a row in
      `docs/architecture/adr-log.md` —
      `test_doc_reconciliation_aqa.py::test_every_adr_on_disk_has_a_row_in_the_adr_log`
      fails without it.
- [ ] 1.5 Amend `src/mousedroid/world_model/CLAUDE.md` invariant 3 to scope
      "No In-Place Tensor Mutation" to the PyTorch autograd graph, per design D-12. This
      lands before any I/O-binding code.

**Phase 1b — Preconditions round 2 found (gate for everything after)**

- [ ] 1b.0 **Compute the consumer ceiling on paper, first.** This is a desk calculation and
      needs no rover. `scripts/spike_step_distillation.py:57-61` already implements the
      method and states the result for the rollout leg: "MCTS plan() makes ~500-650
      imagine_step calls; rollouts are ~40% of them, so end-to-end planner gain caps at
      ~1.25-1.6x **regardless of the primitive speedup**". Apply the same method to
      `observe_step`, which is **one** call per tick — a strictly smaller share, so a strictly
      worse ceiling. Write the number into `proposal.md`. If it does not justify I/O binding,
      FP16 and a cache, stop here and land only Phases 1b.3-1b.5, 2, 3, 6 and 7. Concluding
      "not worth it" on this calculation is a successful outcome, not a failed one.

- [ ] 1b.1 State in the proposal, not as a caveat: the production world model **loads no
      trained weights**. `build_world_model` returns `DualStreamRSSM(cfg.model)`
      (`factory/world_model.py:108`) or `RSSM(cfg.model)` (`:113`) freshly constructed; there
      is no `load_state_dict` in that module or in `orchestrator/`; and
      `factory/telemetry.py:315` `build_weight_update_loader` unconditionally returns `None`.
      A numerical-parity gate against random weights is vacuous. Not this change's to fix —
      but it must be said.
- [ ] 1b.2 State that production runs **plain `RSSM`, not `DualStreamRSSM`**:
      `config/jetson_production.yaml` has no `model:` block, so `cfc_hidden_dim` is `0` and
      `factory/world_model.py:91` falls through to `RSSM`. Only
      `config/jetson_dual_stream.yaml:29` sets `64`, self-gated at `:8` behind human review
      of training metrics. So a torch-vs-ONNX A/B compares two architectures. Any benchmark
      must run both arms on `config/jetson_dual_stream.yaml` or report the confound.
- [ ] 1b.3 **Off-loop warmup, before any `engine: onnx_trt` code lands.** `observe_step`
      warms lazily (`dual_stream_rssm_onnx.py:223-224`), `_update_world_model` is synchronous
      (`_world_model_state_mixin.py:28`), and `run()` wraps the tick in
      `asyncio.wait_for(..., tick_timeout_s)` defaulting to `1.0`
      (`_lifecycle_mixin.py:366`, `config/schema/misc.py:181-184`), whose timeout path calls
      `emergency_stop()` (`:380-392`). `wait_for` cannot preempt a synchronous call, so a
      cold TensorRT build e-stops the rover. Add a `start()`-time
      `await asyncio.to_thread(wm.warmup)` and a test that a slow warmup does not reach the
      tick. `loop_overrun_warmup_ticks: 30` does not cover this — it gates the safety
      monitor's `max_loop_time_ms`, not `tick_timeout_s`.
- [ ] 1b.4 **Install `onnxruntime-gpu` in `Dockerfile.jetson`.** It runs
      `pip install --no-cache-dir -e "."` with no extras and says so at `:88`; the only
      transitive ORT is the CPU wheel `piper-tts` pulls (`:146`). Since
      `resolve_providers` silently returns `("CPUExecutionProvider",)` on an empty
      intersection (`common/onnx_session.py:94-100`), `engine: onnx_trt` would run slower
      than the baseline with only an INFO log. Pair the install with
      `onnx_require_primary_provider` defaulting **true** in the FP16 overlay. Record whether
      `dustynv/l4t-pytorch:r36.4.0` already ships a GPU-enabled ORT — not knowable from the
      tree.
- [ ] 1b.5 Correct two false comments: `factory/world_model.py:267-270` ("the safety
      monitor's CfC inspection") and `composite.py:135` ("keeps the safety monitor wired").
      `get_safety_trace` has zero production callers, and `build_planner` — named at
      `ADR-008:80` — does not exist; the real builder is `factory/cognitive.py:34`. Fold into
      the Phase 3 narrative sweep.
- [ ] 1b.6 Size the memory budget before enabling caches. `docker-compose.jetson.yml:145-153`
      is `memory: 6G` with no `memswap_limit`/`shm_size`/`pids_limit`, against 7.4 GB usable
      and ~4.6 GB measured available. `config/jetson_production.yaml:66` sets
      `workspace_gb: 2.0` — 2 GiB of TensorRT builder workspace for a ~574 k-parameter graph
      — and nothing currently reads it on an ORT path (`build_tensorrt_compiler`,
      `factory/hardware.py:422`, has zero production callers; `gpu_memory_fraction` is
      schema-only). Separate build-time peak from steady state, and justify or shrink the
      workspace. Note `docs/runbooks/jetson-claude-pilot-deploy.md:118-122` records that a
      second full-offload GPU tenant already failed with `unable to allocate CUDA0 buffer`,
      though its attribution to the world model looks wrong — the world model is CPU-resident
      today (no `set_default_device`, no live `.to(cuda)`, latents on CPU at
      `orchestrator.py:468-477`).
- [ ] 1b.7 Bind only `new_h` and `new_z` to persistent device buffers.
      `_world_model_state_mixin.py:41` discards `obs_embed` and `surprise`
      (`self._h, self._z, _, _ = …`), and `SafetyContext.surprise` is never assigned from the
      world model (`safety/context.py:21` default `0.0` is its only value), so
      `compute_mcts_budget` is permanently pinned at `n_simulations_base`. Paying a
      device-to-host copy for two discarded outputs is waste.

**Phase 2 — Make the stage measurable**

- [ ] 2.1 Failing tests first: `build_world_model` accepts `metrics=` and threads it to
      `DualStreamRSSMOnnx`; the PyTorch engine emits the same observe-step histogram;
      construction still does not load a session
      (`test_dual_stream_rssm_onnx.py:111` must stay green).
- [ ] 2.2 `src/mousedroid/factory/world_model.py`: `build_world_model(cfg, *, metrics=None)`
      threaded to `_build_onnx_world_model` → `DualStreamRSSMOnnx(metrics=...)`. Update all
      call sites: `factory/orchestrator.py:121`, `factory/on_device_learning.py:102,274,304`,
      `validation/pillars.py:184`.
- [ ] 2.3 `src/mousedroid/factory/orchestrator.py`: move `build_metrics_registry(cfg)`
      (currently `:210`) above the `build_world_model` call (currently `:121`).
- [ ] 2.4 `src/mousedroid/telemetry/metrics/_registry_onnx_runtime.py` (new, per ADR-017's
      package split): `onnx_session_build_seconds{provider,cache_state,precision}` and
      `onnx_copy_seconds{direction,mode}`. Registry field names carry **no** `_total` /
      no double suffix — `primitives.py:422-429` appends `_total` at render. Bucket
      boundaries come from new `MetricsConfig` fields registered in the single
      `_validate_histogram_buckets` validator (`telemetry.py:485-496`). Labels need a
      module-level `frozenset` in `primitives.py` paired with a `Literal` alias in
      `config/schema/_primitives.py`. Seed `generate_metrics_sample()` (`registry.py`) or
      `test_prometheus_alerts_yml.py` and `test_grafana_dashboard_json.py` go red.
- [ ] 2.5 `src/mousedroid/world_model/dual_stream_rssm_onnx.py`: time `pack_observation`
      (`:236`) and the tensor conversions (`:262-269`) as `onnx_copy_seconds` spans. The
      existing `observe_world_model_observe_step_seconds` keeps its `session.run` scope
      (`:255-258`) so `alerts.yml:388-421` and `registry.py:221` stay valid.
- [ ] 2.6 `src/mousedroid/world_model/dual_stream_rssm.py`: emit the same observe-step
      histogram from the PyTorch engine, so the engines are comparable in production and
      not only inside the Jetson-gated advisory test.
- [ ] 2.7 Do **not** add a deadline-miss counter. `mousedroid_tick_overruns_total`
      (`_registry_core.py:91`) is already written live at
      `_telemetry_experience_mixin.py:114-126` against
      `safety.loop_soft_budget_factor / loop.control_hz`, and
      `mousedroid_tick_phase_ms{phase="world_model"}` (`orchestrator.py:522`) already gives
      per-phase attribution.

- [ ] 2.8 Close the pre-existing hole this change lands inside:
      `tests/unit/factory/test_factory_world_model_engine.py` is the **only** test file that
      constructs `WorldModelConfig`, and it executes in **no** CI job — it
      `importorskip`s `onnx`/`onnxruntime` (`:22-24`) so it skip-alls in the blocking `test`
      job, and the advisory `onnx-world-model-extras` job runs only `tests/unit/world_model`
      plus `tests/unit/training/test_export_dual_stream_rssm_onnx.py` (`ci.yml:600-601`).
      Either add `tests/unit/factory/` to that job's path list or add always-on integration
      coverage for `build_world_model`'s dispatch through a stubbed ORT seam. Without this,
      eight new config fields land completely unpinned.
- [ ] 2.9 Fill the missing tiers: a `tests/property/` test for provider and mode resolution
      (order preservation, idempotence, strict implies raise-or-exact-match — there is no
      ONNX property test today), a `tests/integration/` test exercising the new fields
      through the factory with a stubbed ORT, and a `tests/smoke/` assertion that the new
      config parses. Do **not** register a new pytest marker — `pyproject.toml:382-387`
      registers exactly `slow`, `hardware`, `smoke`, `pillar` and `--strict-markers` turns an
      unregistered marker into a collection error.

**Phase 3 — Correct the narrative and reach the ONNX path**

- [ ] 3.1 `narrative-correction-sweep` over the claim that `config/jetson_production.yaml`
      can enable `engine: onnx_trt`. It cannot: `factory/world_model.py:246-253` raises
      when `cfg.model.cfc_hidden_dim <= 0`, the schema default is `0`
      (`world_model.py:205`), and only `config/jetson_dual_stream.yaml:29` sets `64`.
      Fix `ADR-008` migration step 2 and `scripts/export_dual_stream_rssm_onnx.py:17-21`
      to name `config/jetson_dual_stream.yaml`. Prove the sweep against the original
      wording before ticking.
- [ ] 3.2 Correct every live surface claiming the Hugging Face repo carries the artifact.
      Verified against the Hub: `ianshank/mousedroid-dual-stream-rssm` holds only
      `.gitattributes` (1519 B) and `README.md` (877 B). Touch `ADR-008` (migration step 2
      invites `onnx_path: null`), `docs/architecture.md:1119`,
      `docs/planning/NEXT_STEPS.md:344`.
- [ ] 3.3 `tests/regression/test_f050_aqa.py` + `test_f050_backwards_compat.py` — the
      mandated pair, spelled out in full per `regression-pair-scaffold/SKILL.md:152-161`.
      AQA: every new field's `FieldInfo.default` and non-empty `description`, validator
      rejections, the artifact contract (a missing HF filename is a named failure, never a
      `.pt` substitution). Backwards-compat: all 17 `config/*.yaml` still load, absent-block
      defaults unchanged, a typo'd key raises under `extra="forbid"`
      (`_primitives.py:115-133`).

**Phase 4 — Rover baseline (gate for Phases 5-6)**

- [ ] 4.1 Sequenced behind F-008 — hardware readiness preempts in-flight software streams
      (F-024 rule). Run while F-008 is bench-blocked only with operator agreement on bench
      time.
- [ ] 4.2 No-motion invocation is the canonical one: clear **both** gates as
      `jetson_full_validation.sh:423,429` does —
      `MOUSEDROID_SMOKE_ALLOW_MOTION= MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=` — and
      audit the live `/etc/mousedroid/docker.env` for an uncommented
      `MOUSEDROID_ESP32__ENABLED=true` (`config/docker.env.example:66`), because compose
      `env_file` beats the YAML overlay. `RUN-MOTION` is prose only; no code reads it.
- [ ] 4.3 In-container commands pass `-e PYTHONOPTIMIZE=0` — `Dockerfile.jetson:178` sets
      `PYTHONOPTIMIZE=1` and it leaks into every `docker exec`, stripping asserts from a
      suite that has never run under `-O`.
- [ ] 4.4 Capture `mousedroid_tick_phase_ms` for all 8 phases plus
      `mousedroid_tick_overruns_total`, torch engine, actuation disabled, and commit the
      raw samples via `evidence-commit` with a `reports/jetson_onnx_benchmark/` entry added
      to `.gitignore`'s per-directory list (`.gitignore:173-196`).
- [ ] 4.5 **Decision gate — use the repository's existing rubric, not a new one.**
      `docs/analysis/alayaworld-distillation-spike.md:65-75` is the ratified three-part GO
      rubric for accelerating this world model, all three required: (1) ≥3x p95 primitive
      speedup **on Jetson**; (2) >=0.90 action agreement against the torch engine with a
      **trained-checkpoint** teacher; (3) a written consumer case whose projected end-to-end
      gain justifies the added surface. Its own verdict for a sibling optimization was
      **DEFER**, on criterion 3 alone — "This alone justifies DEFER over ADOPT regardless of
      accuracy" (`:93-95`). Criterion 2 is unattainable while pins 57-58 hold, so record it
      as blocked rather than passing it vacuously. If the rubric does not return GO, close
      F-050 on the instrumentation and artifact-integrity work and do not build I/O binding.
      This is a real exit.
- [ ] 4.6 Note the pattern this phase is joining: `reports/endurance/` holds only
      `.gitkeep`; `reports/spike_step_distillation.json` says "Jetson measurement pending
      operator run"; and `alayaworld-distillation-spike.md:54-63`'s "Results — Jetson Orin
      Nano (operator run — PENDING)" table is still a row of em-dashes with "**The Jetson
      criterion is UNMET until this section is filled**". Three CPU-side spikes, none closed
      on hardware. Do not open a fourth without an operator committed to closing it.

**Phase 5 — Provider options and proof**

- [ ] 5.1 Failing tests first, extending the two files that already exist rather than
      creating them: `tests/unit/common/test_onnx_session.py` (already pins every branch of
      the provider intersection) and `tests/unit/world_model/test_dual_stream_rssm_onnx.py`
      (already pins the TRT→CUDA→CPU chain and idempotent warmup). Assert the exact option
      dictionary handed to each provider, the post-construction
      `session.get_providers()` comparison, option redaction, and strict-mode rejection.
- [ ] 5.2 `src/mousedroid/config/schema/world_model.py`: `onnx_execution_mode`
      (`portable | cuda_iobinding`), `onnx_require_primary_provider`, `onnx_device_id`,
      `onnx_engine_cache_enabled`, `onnx_timing_cache_enabled`,
      `onnx_context_memory_sharing_enabled`, `onnx_profile_batch`. Precision, workspace and
      TRT cache directory are read from `cfg.jetson.precision` / `workspace_gb` /
      `tensorrt_cache_dir` (`hardware.py:580-588`) — not duplicated. `onnx_warmup_iterations`
      already exists (`:77`); do not re-add it. No new `world_model:` key in any tracked
      YAML (design D-7).
- [ ] 5.3 `src/mousedroid/common/onnx_session.py`: `resolve_providers` and `warmup_session`
      widened to a provider-spec sequence, matching on the name component; `sess_options`
      passed explicitly with a deliberate graph-optimization level; post-construction
      `session.get_providers()` returned alongside the requested tuple. Strict-mode failure
      raises a named exception — `S101` is blocking in `src/`. Module neutrality and lazy
      ORT import stay pinned (`test_onnx_session.py:391-415`).
- [ ] 5.4 Same commit: `src/mousedroid/vla/policy.py:283-306` and both ORT stubs —
      `tests/unit/common/test_onnx_session.py:98-103` and
      `tests/unit/vla/test_distilled_onnx.py:107` — which pin the exact
      `InferenceSession(model_path, providers)` arity and will raise `TypeError` on a third
      argument.
- [ ] 5.5 `onnx_provider_fallback` counter (registry field without the render suffix) and
      the `fallback` structured-log event. Cache directories keyed on the ONNX SHA-256, ORT
      version, TensorRT version, CUDA version, L4T identifier, precision and shape profile;
      a mismatch rebuilds rather than reusing.
- [ ] 5.6 Move ONNX warmup behind `asyncio.to_thread` at the factory boundary before
      enabling caches — `observe_step` is synchronous and warms lazily on first call
      (`:223-224`), so a cold TRT engine build blocks the 30 Hz caller for tens of seconds.
- [ ] 5.7 `tests/integration/test_onnx_provider_policy.py` (stubbed ORT, runs in the
      blocking `test` job) and `tests/hardware/test_onnx_iobinding_cuda.py` with
      module-level `pytestmark = pytest.mark.hardware` — `tests/hardware/`, not
      `tests/integration/`, where 24 files already use the marker and zero integration
      files do.

**Phase 6 — Artifact integrity**

- [ ] 6.1 Digest verification in `src/mousedroid/factory/world_model.py`, where the
      download happens — today it checks only `model_path.is_file()`. Precedent for the
      mismatch counter is `_registry_cloud.py:227`
      (`inc_cloud_weight_update_sha256_mismatch`). Not a deploy-time shell check.
- [ ] 6.2 Export metadata beside the artifact: checkpoint digest, config digest, git SHA,
      opset, input/output names, shapes, dtypes, tool versions. The IO contract is owned by
      `src/mousedroid/world_model/onnx_io.py` (`OBSERVE_STEP_OUTPUT_NAMES`) and pinned by
      `tests/unit/world_model/test_onnx_io.py` — go through that module.
- [ ] 6.3 `onnx.checker` plus ORT CPU inference on the PC, and multi-step seeded parity over
      `new_h`, `obs_embed`, `surprise` at `atol=1e-4`. `new_z` is **excluded** — ADR-008
      documents its `torch.randn_like` divergence; compare `post_mean`/`post_logvar`
      instead.
- [ ] 6.4 `scripts/export_dual_stream_rssm_onnx.py::_push_to_hf` (`:456`) extended to
      upload the metadata sidecar and refresh the model card, mirroring
      `training/upload_weights.py`'s `_COMPONENT_DOCS` pattern. Token from the environment,
      never logged, never uploaded from a test or a deploy script.
- [ ] 6.5 `scripts/benchmark_latency.py` extended with an `observe_step` mode across torch /
      `onnx_portable` / `onnx_iobinding_*`, reusing its existing
      `--config`/`--checkpoint`/threshold/exit-code contract. `tests/performance/
      test_observe_step_budget.py:126,157` extended with the per-span breakdown, still
      budgeted by `MOUSEDROID_OBSERVE_STEP_BUDGET_MS`. No new benchmark script and no second
      source of truth for the 10 ms / 33 ms numbers.

**Phase 7 — Delivery (F-051)**

- [ ] 7.0 Record the container resource budget alongside the cache volume:
      `docker-compose.jetson.yml:145-153` is `memory: 6G` with no `memswap_limit`,
      `shm_size` or `pids_limit`. Unbounded swap means a 30 Hz loop thrashes instead of
      restarting cleanly, and every swapped page fault inside the synchronous
      `_update_world_model` counts against `tick_timeout_s: 1.0`. Decide explicitly whether
      to bound swap before enabling engine builds on-device; note the campaign plan's
      ≥8 GiB disk guard and NVMe-swap requirement
      (`docs/superpowers/plans/2026-07-25-…:123-126`) and
      `scripts/jetson_disk_cleanup.sh`.
- [ ] 7.1 `docker-compose.jetson.yml`: named volume for the TensorRT engine/timing cache
      beside `mousedroid_experience` and `promtail_positions` (`:165-169`). Nothing persists
      a TRT cache or `HF_HOME` today. The directory comes from
      `cfg.jetson.tensorrt_cache_dir`, not a literal in the compose file. Note for the
      record: there is **no** weights mount — weights persist transitively via
      `WORKDIR /opt/mousedroid` (`Dockerfile.jetson:49`) plus the relative
      `onnx_cache_dir` default (`world_model.py:68`) landing inside the bind mount.
- [ ] 7.2 `scripts/docker_deploy.sh`: `--strict-health` failing on a dead telemetry
      endpoint, an unexpected ORT provider, or a digest mismatch. Today the telemetry leg is
      a `warn` (`:116-123`) and the whole function is softened at `:241`. Default behaviour
      unchanged (design D-10).
- [ ] 7.3 `scripts/deploy_remote.sh`: dirty-target refusal plus `rover/wip-<date>`
      preservation before the `rsync -avz --delete` at `:151`. Never `git clean`, never
      rsync-delete over uncommitted rover work.
- [ ] 7.4 `tests/unit/scripts/test_deploy_remote_guard.py` — there is no shellcheck gate in
      CI, so shell safety gets a Python test with a real fixture. Precedent:
      `tests/unit/scripts/test_repin_tags.py` (and the vacuous-assertion trap
      `mouse-droid-deploy-repin/tasks.md:19-32` records).
- [ ] 7.5 `tests/regression/test_f051_aqa.py` + `test_f051_backwards_compat.py`.
- [ ] 7.6 `deployments/jetson-image.json` extended in place with the model digest and proven
      provider — `config-compat.yml:8-10` documents the per-platform extension, and
      `test_config_no_dup_keys_and_deploy_record.py` pins
      `("sha","platform","image_tag")` plus a full 40-hex SHA. No parallel manifest.
- [ ] 7.7 Rollback drill with the network disabled, using the established anchor
      (`docker tag mousedroid:jetson mousedroid:jetson-rollback-<date>`) plus a checkout —
      no rebuild, no internet. Drill before the prior image is eligible for pruning.
- [ ] 7.8 `docs/runbooks/jetson-onnx-benchmark.md` and
      `docs/runbooks/pc-to-jetson-promotion.md`, each opening with a `# ` H1
      (`tests/regression/test_runbooks_structure.py`), plus rows in `docs/README.md` (the
      declared canonical runbook index). Flag the overlap with `docs/deployment.md`,
      `jetson-full-bringup.md`, `jetson-claude-pilot-deploy.md` and
      `docs/planning/JETSON_DEPLOY_RUNBOOK.md` rather than adding a fifth parallel path.
      If any skill doc backtick-references a new script, it lands in the same PR —
      `tools/validate_skill_commands.py` fails on a dead path.

**Phase 8 — Close out**

- [ ] 8.1 `bash scripts/ci.sh` green; `make gates`; `make test`.
- [ ] 8.2 Record the `onnx-world-model-extras` consecutive-green-run count in
      `.github/advisory_stages.yaml`'s `reason`, and either promote it or re-extend with the
      verified count. Its entry has been extended once already (2026-08-16) for want of that
      number. If a job is added, update the count in all three docs that pin it —
      `CLAUDE.md`, `docs/claude/surfaces/ci-gates.md`, `docs/claude/surfaces/README.md` —
      per `test_doc_reconciliation_aqa.py::test_ci_job_count_docs_match_the_real_workflow`.
- [ ] 8.3 `pin-reachability-audit`, then `bash scripts/repin_tags.sh` (and `--push`, an
      operator action) before any SHA this change records is relied on.
      `deployments/jetson-image.json` states the current pin has zero tags and only ten
      stale branch carriers.
- [ ] 8.4 Post-merge only: re-pin `deployments/jetson-image.json` to the squash-merge trunk
      commit. Re-pinning to a feature-branch SHA reproduces the `9c31968` failure the record
      documents, which killed the `config-compat` gate repo-wide.
- [ ] 8.5 Flip F-050/F-051 to `done` with a 40-char hex `implemented_in` — a branch name
      fails `test_harness_spec_aqa.py::test_done_features_pin_a_hex_sha_not_a_branch_name`.
      Update `CHANGELOG.md`, `openspec/project.md`, and this bundle's task state.

## Explicitly deferred (separate change, do not fold in)

- **CUDA Graph.** Removed from rev A's task list entirely. Requires stable buffer
  addresses, fixed batch-1 shapes, and a Phase 4 baseline showing the stage matters.
- **INT8.** Needs representative calibration data with checksums, recurrent multi-step
  drift, task replay and safety metrics. Never inferred from file size or microbenchmark
  speed.
- **Full `imagine_step` export.** Profile MCTS first —
  `scripts/benchmark_latency.py` already covers `RSSM.imagine_step` and
  `MCTSPlanner.plan`. Note `CompositeWorldModel` also delegates `get_safety_trace` to the
  PyTorch engine (`composite.py:128-150`), so the graph cannot simply be dropped.
- **Replacing the bind-mount-editable deployment model with immutable releases.** An
  ADR-level change touching both systemd units, `sync_jetson_overlay.sh`,
  `jetson-nightly.yml` and `deployments/jetson-image.json`'s rationale.
- **Isaac ROS / nvblox.** Needs a hardware spike first: the deployed sensor is a 2D
  FHL-LD19, and the camera abstraction exposes one CSI stream, not calibrated stereo.
- **Cosmos.** Already catalogued as F-049 `deferred`; offline augmentation only, never in
  the 30 Hz path.
- **Promoting `onnx-world-model-extras` to blocking.** Gated on its own
  7-consecutive-green-run count per `.github/advisory_stages.yaml`.
