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
distribution statistics from raw samples (count, mean, median, p90, p95, p99, stdev, min,
max, deadline misses over 33.33 ms), `tegrastats` samples, throttling indicators, and drift
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
