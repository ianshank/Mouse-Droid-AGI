# Spec delta — Performance evidence for the world-model hot path

## ADDED Requirements

### Requirement: The observe-step metric SHALL have a production writer

`build_world_model` SHALL accept a `MetricsRegistry` and thread it to the ONNX runtime, and
`factory/orchestrator.py` SHALL build the registry before the world model.

Today `build_world_model(cfg)` (`factory/world_model.py:55`) takes only `cfg`;
`_build_onnx_world_model` constructs `DualStreamRSSMOnnx` at `:262-266` with no `metrics=`;
and `observe_step` records only `if self._metrics is not None`
(`dual_stream_rssm_onnx.py:271-279`). The consequence is that
`mousedroid_world_model_observe_step_seconds` has no production writer and
`WorldModelObserveStepLatencyHigh` (`config/prometheus/alerts.yml:388-421`) cannot fire on
the rover.

The PyTorch engine SHALL emit the same histogram, so the two engines are comparable in
production and not only inside a Jetson-gated advisory test.

#### Scenario: The alert becomes able to fire

- **GIVEN** a rover running `engine: onnx_trt` with telemetry enabled
- **WHEN** `observe_step` runs for 5 minutes
- **THEN** `mousedroid_world_model_observe_step_seconds_bucket` is populated
- **AND** `WorldModelObserveStepLatencyHigh` can evaluate against real samples

#### Scenario: Engines are comparable

- **GIVEN** the same overlay run once with `engine: torch` and once with `onnx_trt`
- **WHEN** the histogram is scraped in both runs
- **THEN** both populate the same family and can be compared without a benchmark script

### Requirement: Stage timing SHALL sum to wall-clock observe-step

The existing histogram SHALL keep its `session.run` scope
(`dual_stream_rssm_onnx.py:255-258`) so the alert and the golden sample
(`registry.py:221`) stay valid. New `onnx_copy_seconds{direction,mode}` spans SHALL cover
`pack_observation` (`:236`) and the tensor conversions (`:262-269`), so the family sums to
wall-clock `observe_step`.

Without this, removing the copy boundary would barely move the only metric that exists,
because that metric never measured the copies.

Metric names SHALL be declared as registry fields **without** the `_total` suffix that
`primitives.py:422-429` appends at render, or the exposition ships `..._total_total`.
Histogram buckets SHALL come from new `MetricsConfig` fields registered in the single
`_validate_histogram_buckets` validator (`telemetry.py:485-496`). Labels SHALL be bounded by
a module-level `frozenset` in `primitives.py` paired with a `Literal` alias in
`config/schema/_primitives.py`. New families SHALL seed `generate_metrics_sample()`, or
`test_prometheus_alerts_yml.py` and `test_grafana_dashboard_json.py` fail.

No metric SHALL be labelled with a commit SHA, file path, model digest, exception text or
mission ID. Those belong in structured logs and evidence artifacts.

#### Scenario: Copy time is attributable

- **GIVEN** portable mode on a CPU host
- **WHEN** `observe_step` runs
- **THEN** pack, execute and convert are separately visible and sum to wall-clock

### Requirement: No second deadline-miss counter SHALL be added

Whole-tick deadline evidence SHALL come from the existing
`mousedroid_tick_overruns_total` (`_registry_core.py:91`, written at
`_telemetry_experience_mixin.py:114-126` against
`safety.loop_soft_budget_factor / loop.control_hz`) and
`mousedroid_tick_phase_ms{phase=...}` (`orchestrator.py:522`, 8 phases including
`world_model`).

#### Scenario: Attribution without new metrics

- **GIVEN** a tick that exceeds its soft budget
- **WHEN** telemetry is inspected
- **THEN** `tick_overruns_total` increments and `tick_phase_ms` attributes the phase

### Requirement: A rover baseline SHALL gate the optimization work

Before any I/O-binding, FP16 or cache work lands, a no-motion rover run SHALL record
`mousedroid_tick_phase_ms` for all phases plus `mousedroid_tick_overruns_total` on the
`torch` engine, with raw samples committed.

No `reports/` or `smoke-reports/` artifact currently carries an `observe_step`,
`tick_phase` or `world_model` measurement, and the only on-device report records
`telemetry down`. If the baseline shows `phase="world_model"` is not a material share of the
tick, the optimization work SHALL be abandoned and the negative result recorded. This is an
exit, not a formality.

#### Scenario: The stage does not dominate

- **GIVEN** a baseline where `world_model` is a small share of the tick
- **WHEN** the gate is evaluated
- **THEN** I/O binding and FP16 are not built, and the finding is committed as evidence

### Requirement: Evidence runs SHALL be reproducible and honestly bounded

Benchmark records SHALL capture git commit and dirty state, model SHA-256, board model,
L4T/JetPack, CUDA, cuDNN, TensorRT, ORT, Torch and Python versions, `nvpmodel -q`, clock
state and whether clocks were changed, container image ID, requested and **observed**
providers with secret-safe options, cold and warm session-build times, per-stage latency,
distribution statistics from raw samples via `validation/latency_stats.py::summarize`
(count, min, mean, p50, p95, p99, max), stdev, deadline misses over 33.33 ms, `tegrastats` samples, throttling indicators, and drift
versus Torch for `new_h`, `obs_embed`, `surprise`.

Raw samples SHALL be stored, not only aggregates. At least three independent processes per
variant. Cold- and warm-cache starts measured separately. Clocks SHALL NOT be changed unless
explicitly requested and operator-approved. `tegrastats` SHALL be stopped by an EXIT trap.

Benchmarks SHALL run with actuation disabled using the canonical invocation that clears
**both** motion gates as `jetson_full_validation.sh:423,429` does
(`MOUSEDROID_SMOKE_ALLOW_MOTION=` and `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=`), and
SHALL audit the live `/etc/mousedroid/docker.env` for an uncommented
`MOUSEDROID_ESP32__ENABLED=true`, because compose `env_file` overrides the YAML overlay.
In-container commands SHALL pass `-e PYTHONOPTIMIZE=0`.

Latency budgets SHALL be expressed through `MOUSEDROID_OBSERVE_STEP_BUDGET_MS` and
`tests/performance/test_observe_step_budget.py`. Prose SHALL NOT restate the 10 ms / 33 ms
numbers as a second source of truth. Measurement work SHALL extend
`scripts/benchmark_latency.py`, which already carries the
`--config`/`--checkpoint`/threshold/exit-code contract for `RSSM.imagine_step` and
`MCTSPlanner.plan`.

#### Scenario: A provider fallback invalidates a labelled result

- **GIVEN** a run labelled TensorRT whose observed providers include a fallback
- **WHEN** the verdict is computed
- **THEN** the result is invalid for promotion and the fallback is recorded

#### Scenario: Interrupted benchmark

- **GIVEN** a benchmark interrupted mid-run
- **WHEN** the process exits
- **THEN** the EXIT trap stops `tegrastats` and leaves no orphan process

#### Scenario: Thermal throttling

- **GIVEN** throttling during a promoted run
- **WHEN** the report is written
- **THEN** the run is reported as invalid and the failed evidence is retained, not
  overwritten by a rerun

### Requirement: Component latency SHALL NOT be presented as end-to-end throughput

An `observe_step` p95 result SHALL be labelled a component measurement. No 30 Hz claim
SHALL be made unless whole-tick p95 is within the 33.33 ms period with a stated p99
guardrail, measured on the same run.

Every performance claim SHALL name the board, power mode, model digest, precision, observed
provider, cache state and measurement boundary.

#### Scenario: A fast component and a slow tick

- **GIVEN** `observe_step` p95 well under budget but whole-tick p95 over 33.33 ms
- **WHEN** results are reported
- **THEN** no 30 Hz claim is made and the dominant phase is named

### Requirement: ONNX CI coverage SHALL be described accurately

`onnx-world-model-extras` is the only CI job installing `[onnx_world_model]` and it carries
`continue-on-error: true` (`ci.yml:579-583`), tracked in `.github/advisory_stages.yaml`
since 2026-05-16 with `promote_after_days: 180`. The `performance` job is advisory too, and
`tests/performance/conftest.py` skips off-Jetson.

Config, provider-policy and artifact-contract assertions SHALL therefore land in
`tests/regression` so the blocking `test` job runs them. Schema pins SHALL NOT be placed in
`tests/performance/`, which is for budgets, is Jetson-gated, and cannot fail a PR.
Documentation SHALL NOT claim blocking coverage for ORT-dependent tests.

#### Scenario: A schema pin that can actually fail

- **GIVEN** a new config field with a wrong default
- **WHEN** CI runs on the PR
- **THEN** `tests/regression/test_f050_aqa.py` fails in the blocking `test` job

## ADDED Requirements — round 2

### Requirement: Whole-tick evidence SHALL come from the existing endurance test

`tests/performance/test_jetson_endurance.py` SHALL be the whole-tick harness. No new
whole-tick benchmark script SHALL be written.

It already runs the full orchestrator loop and validates GPU temperature, RSS stability
within 10%, and "Loop time p95 < 33ms at 30Hz target", writing `duration_s`, `p95_ms`,
`rss_start_mb`, `rss_end_mb` and `max_gpu_temp_c` to
`${MOUSEDROID_ENDURANCE_REPORT_DIR:-reports/endurance}/endurance-<utc>.json` (`:50`,
`:88-95`). It is `pytestmark = [hardware, slow]` (`:53`) with a configurable
`MOUSEDROID_ENDURANCE_DURATION_S` (default 60 s), and `MOUSEDROID_ENDURANCE_FORCE_REAL=1`
for the real-hardware opt-in. `reports/endurance/` contains only `.gitkeep`: it has never
been run and committed.

Its RSS-stability assertion is also the answer to the memory question this change raises —
`CompositeWorldModel` holds a PyTorch `DualStreamRSSM` alongside the ORT session, plus an
engine plan cache and persistent device buffers, on an 8 GB unified-memory board.

It is excluded from CI because the `performance` job runs `-m "not hardware"`, which
`docs/analysis/positioning-safety-peer-review-2026-09-19.md:147` names as the reason that
job is "a tripwire, not a benchmark", its remaining tests running at a deliberately loosened
2.0× budget. Evidence SHALL therefore be produced on the rover and committed, not expected
from CI.

#### Scenario: Whole-tick evidence produced

- **GIVEN** a rover with actuation disabled
- **WHEN** the endurance test runs for the full-validation duration
- **THEN** its JSON lands in `reports/endurance/` and is closed out as a **declared
  local-only** evidence chain — the artifact exists on the rover plus a `CHANGELOG.md`
  reference — because `.gitignore:339-340` ignores that directory except `.gitkeep`, and
  `evidence-commit/SKILL.md:24-38` lists `reports/endurance` among the six local-only
  families. It is NOT git-committed.

### Requirement: The whole-tick gate SHALL be treated as a prerequisite, not a deliverable

`LoopConfig.planning_hz = 10.0` (`config/schema/misc.py:178`, "MCTS planning rate (Hz)",
set in `config/default.yaml:9`) has **zero consumers** in `src/`. There is no tick
decimation, modulo or `_should_plan` gate in `orchestrator/`, so planning runs at
`control_hz` (30 Hz), not 10 Hz, and `_select_action` blocks the tick synchronously.

The repository has recorded this twice:
`docs/analysis/autonomy-baseline-peer-review-2026-09-17.md:79` (S4) — "30 Hz plan-act is
fiction… the multi-rate contract was declared and never wired" — and
`docs/analysis/positioning-safety-peer-review-2026-09-19.md:148` (P5) — "**BOTH STILL
OPEN** … `LoopConfig.planning_hz` still has zero consumers", with `:198` directing that P5
be closed "before publishing, not after". That second document landed one commit before
this bundle.

Until P5 closes, this change SHALL report component latency only and SHALL state that the
whole-tick gate is blocked on it. Wiring `planning_hz` — using the budget/timeout/fallback
pattern `_try_vla_action` already demonstrates — is named in that review as unblocked work
and belongs to its own change, not to this one.

#### Scenario: A 30 Hz claim is withheld

- **GIVEN** `planning_hz` still has zero consumers
- **WHEN** benchmark results are published
- **THEN** they are labelled component measurements and name P5 as the open prerequisite

### Requirement: Evidence SHALL cover both action paths

`_select_action` (`orchestrator/_action_mixin.py:30-68`) tries the cognitive core first,
then the VLA branch (inert by default — `policy_selector='nav_agent'`), then falls through
to `self._agents[0].act(...)` → `MouseDroidNavigationAgent.act`
(`agents/navigation.py:84-88`) → `MCTSPlanner.plan()`.

That fallback is reachable in production from an ordinary failure:
`factory/orchestrator.py:158-170` builds the cognitive core when `cfg.cognitive.enabled`,
and on exception with `cognitive.fallback_to_mcts: true` — the schema default, set
explicitly at `config/jetson_production.yaml:204` — `cognitive_core` stays `None` and every
subsequent tick takes the MCTS path. A failed BDI weight download at boot suffices.

The cost asymmetry is large. Per `plan()` call, `n_simulations_base=50` × (one expansion
`imagine_step` at `world_model/mcts.py:257` plus `rollout_depth=5` rollout calls at `:277`)
is roughly 300 `imagine_step` invocations, rising to about 1200 at `n_simulations_max=200`,
against **one** `observe_step` per tick. The published model card for
`ianshank/mousedroid-weights` records `mcts_tuning` at p50 109–110 ms and p95 125 ms, and
`scripts/benchmark_latency.py` defaults to `--mcts-target-ms 50.0` against
`--rssm-target-ms 15.0`. Treat those as indicative, not as production tick measurements:
the simulation count and host state behind them are not recorded.

The budget also scales with surprise: `agents/_planning.py:23-24` gives
`clamp(int(base × min(surprise+1, max/base)), base, max)`, so with
`surprise.high_threshold: 2.0` high surprise means 150 simulations and critical means 200.
The surprise input is `observe_step`'s own fourth return value, so worst-case planning
latency occurs in the most novel situations.

#### Scenario: Cognitive core unavailable

- **GIVEN** BDI weight loading fails and `fallback_to_mcts` is true
- **WHEN** the tick runs
- **THEN** evidence for the MCTS path exists separately and is not conflated with the
  cognitive-core path

### Requirement: Percentiles SHALL use the repo's existing summariser

All latency statistics SHALL come from `validation/latency_stats.py::summarize`, whose
`LatencySummary` is min / mean / p50 / p95 / p99 / max (`:29`, `_P50, _P95, _P99`).

p90 SHALL NOT be reported: it appears nowhere in the repository, and
`docs/analysis/positioning-safety-peer-review-2026-09-19.md:144` names p50/p95/p99 as the
house convention. A second percentile implementation SHALL NOT be introduced.

Any published whole-tick percentile SHALL carry its denominator and the timeout class
alongside it. `_finish_tick_timing`
(`orchestrator/_telemetry_experience_mixin.py`) latches the duration on every path but
returns before recording when `ok is False`, so a tick that raises — including one cancelled
by `asyncio.wait_for(self.tick(), tick_timeout_s)` — contributes neither a histogram sample
nor a `tick_overruns` increment. That bias is intentional per telemetry invariant 5 and is
recorded as addressed at `:178` (D-9), with the denominator now captured; so it is not an
open defect, but `histogram_quantile(0.99, …)` remains a p99 *of successful ticks* and must
be labelled as such.

#### Scenario: A published percentile is qualified

- **GIVEN** a whole-tick p99 derived from `mousedroid_loop_latency_ms`
- **WHEN** it is written into an evidence artifact
- **THEN** it is accompanied by the sample denominator and the count of failed or timed-out
  ticks

### Requirement: Evidence artifacts SHALL NOT commit host fingerprints

Benchmark and deploy reports SHALL be written to a gitignored `reports/<name>/` path, per the
precedent at `.gitignore:173-188`. Anything committed SHALL carry only host-independent
facts, as `deployments/jetson-image.json` does.

`smoke-reports/smoke_report.md` and `smoke_report.json` are tracked, not gitignored, and
publish the rover's LAN IP, mDNS name, SSH username, WiFi SSID, dev-box identity, JetPack
version and on-device HEAD in a public repository. `.gitleaks.toml` extends only the default
ruleset and no default rule matches an IP, hostname, SSH alias or SSID, so those files pass
the blocking gate — which is the proof that gitleaks is not the control for this class.

A committed digest field SHALL be named `sha256` or `digest`, never a name containing `key`,
`token`, `secret`, `auth`, `api` or `credential`, which are the default `generic-api-key`
rule's keywords.

#### Scenario: A benchmark report carries a hostname

- **GIVEN** a `tegrastats` report whose header includes the rover hostname
- **WHEN** it is produced
- **THEN** it lands on a gitignored path and is not committed


## ADDED Requirements — round 3

### Requirement: The consumer ceiling SHALL be computed before any optimization is built

The proposal SHALL carry an Amdahl bound for `observe_step`, derived by the method
`scripts/spike_step_distillation.py:57-61` already implements and prints: "MCTS plan() makes
~500-650 imagine_step calls; rollouts are ~40% of them, so end-to-end planner gain caps at
~1.25-1.6x regardless of the primitive speedup."

That bound is for the rollout leg. `observe_step` is one call per tick, so its share is
strictly smaller and its ceiling strictly worse. The calculation needs no hardware.

If the ceiling does not justify I/O binding, FP16 and an engine cache, the optimization
SHALL NOT be built, and the instrumentation, artifact-integrity, provider-proof and delivery
work SHALL land on their own merits.

#### Scenario: The ceiling does not justify the work

- **GIVEN** a computed end-to-end ceiling below what the added surface costs to maintain
- **WHEN** the proposal is reviewed
- **THEN** the latency work is deferred and the remaining phases proceed

### Requirement: Promotion SHALL use the repository's existing three-part rubric

`docs/analysis/alayaworld-distillation-spike.md:65-75` defines the GO rubric for accelerating
this world model, all three required: a ≥3× p95 primitive speedup **on Jetson**; ≥0.90 action
agreement at that operating point against a **trained-checkpoint** teacher; and a written
consumer case whose projected end-to-end gain justifies the added surface.

No new promotion rubric SHALL be invented. Criterion 2 SHALL be recorded as **blocked**, not
passed, while the production model loads no trained weights — the same spike recorded 0.422 /
0.609 / 0.422 action agreement and attributed it to a random-init teacher making "the argmax
grid nearly a coin toss". Non-monotonicity in that series is the signature of a
noise-dominated measurement.

FP16 promotion SHALL therefore be gated on action agreement against the FP32 torch engine, in
addition to the tensor tolerances in `specs/onnx-runtime`. Tensor tolerances alone do not
answer whether behaviour changed.

#### Scenario: Tensor parity passes but behaviour diverges

- **GIVEN** FP16 outputs within `rtol=1e-2, atol=1e-3` of the torch engine
- **WHEN** action agreement is measured over a recorded episode
- **THEN** agreement below the bar blocks promotion regardless of the tensor result

### Requirement: `new_z` SHALL NOT carry a precision tolerance

`reports/drift_comparison.json` reports baseline mean MSE of 0.0304 for `latent_h` against
1.7669 for `latent_z`, and finals of 0.0349 against 1.8447 — roughly 58× the variance. The
`z` signal is dominated by posterior sampling variance, so a precision tolerance on it
measures nothing. Compare `post_mean` and `post_logvar` instead, as
`specs/onnx-runtime` already requires.

#### Scenario: A tolerance is proposed for the sampled latent

- **GIVEN** a proposed FP16 drift gate over `new_z`
- **WHEN** it is reviewed against `reports/drift_comparison.json`
- **THEN** it is rejected as unmeasurable and replaced by a distribution comparison
