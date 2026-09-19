# Spec delta — ONNX runtime provider policy and artifact integrity

Amends the runtime contract ADR-008 accepted. ADR-019 records which clauses are superseded.

## MODIFIED Requirements

### Requirement: The active execution provider SHALL be observed, not predicted

`warmup_session` SHALL call `session.get_providers()` after constructing the session and
SHALL return the observed tuple alongside the requested one. `DualStreamRSSMOnnx.
active_providers` SHALL surface the observed tuple.

Today `common/onnx_session.py:193` computes the provider tuple from
`ort.get_available_providers()` *before* building the session at `:207-210` and returns that
prediction at `:221`. A provider that is listed as available but fails to initialise for the
graph is reported as active.

`onnx_require_primary_provider=true` SHALL fail warmup with a named exception when the
configured primary provider is absent from the observed tuple. It SHALL NOT use `assert` —
`Dockerfile.jetson:178` sets `PYTHONOPTIMIZE=1`, and ruff `S101` is blocking in `src/`.

#### Scenario: TensorRT listed but not initialised

- **GIVEN** `TensorrtExecutionProvider` appears in `ort.get_available_providers()`
- **AND** ORT falls back internally when building the session for this graph
- **WHEN** warmup completes under a permissive policy
- **THEN** `active_providers` reports the provider ORT actually used
- **AND** the fallback counter increments and a `fallback` event is logged

#### Scenario: Strict policy on a CPU-only host

- **GIVEN** `onnx_require_primary_provider=true` and no TensorRT EP
- **WHEN** warmup runs
- **THEN** it raises a named exception before any motion can start

### Requirement: Provider options SHALL survive provider resolution

`resolve_providers` and `warmup_session` SHALL accept a sequence whose elements are either a
provider name or a `(name, options)` pair, matching availability on the name component only.

Today both are typed `tuple[str, ...]` (`common/onnx_session.py:72-100`). A
`("TensorrtExecutionProvider", {...})` element fails the `p in available` membership test,
empties the intersection, and silently returns `("CPUExecutionProvider",)` — losing
TensorRT with no log and no exception.

`sess_options` SHALL be passed explicitly with a deliberate graph-optimization level.
Cache options SHALL travel in the TensorRT provider tuple, never via process-global
environment variables.

`common/onnx_session.py` SHALL continue to import neither `onnxruntime`, `mousedroid.vla`,
`mousedroid.world_model`, nor `mousedroid.config` at module scope; new behaviour is
parameter-passed. `tests/unit/common/test_onnx_session.py:391-415` pins this.

#### Scenario: Options reach TensorRT only

- **GIVEN** engine-cache options and FP16 from `cfg.jetson.precision`
- **WHEN** the provider list is built
- **THEN** only the TensorRT entry carries them and the CPU fallback stays valid

#### Scenario: The VLA path moves with the helper

- **GIVEN** the widened signature
- **WHEN** `mousedroid.vla.policy.DistilledVLAOnnx` warms up
- **THEN** it resolves providers through the same helper without a type error

### Requirement: Precision, workspace and TRT cache directory SHALL have one home

The ORT session SHALL read precision from `cfg.jetson.precision`, builder workspace from
`cfg.jetson.workspace_gb`, and the engine/timing cache directory from
`cfg.jetson.tensorrt_cache_dir` (`hardware.py:580-588`). `WorldModelConfig` SHALL NOT
declare parallel fields for any of the three.

`WorldModelConfig.onnx_cache_dir` keeps its existing meaning — the *artifact download*
cache (`factory/world_model.py:320-322`) — and SHALL NOT be reused for TensorRT engine
plans.

`onnx_warmup_iterations` already exists (`world_model.py:77`) and SHALL NOT be re-declared.

#### Scenario: An overlay that already overrides workspace

- **GIVEN** `config/jetson_dual_stream.yaml` sets `jetson.workspace_gb: 2.0`
- **WHEN** the ONNX session is built from that overlay
- **THEN** the session uses 2.0 GB and no second workspace value exists

### Requirement: Engine caches SHALL be fingerprinted, and SHALL NOT block the tick

Cache directories SHALL include or validate the ONNX SHA-256, ORT version, TensorRT
version, CUDA version, L4T identifier, precision and shape profile. A mismatch SHALL
rebuild rather than reuse.

Because `observe_step` is synchronous and warms lazily on first call
(`dual_stream_rssm_onnx.py:223-224`), warmup SHALL be moved behind `asyncio.to_thread` at
the factory boundary before engine caches are enabled. A cold TensorRT build turns a
millisecond stall into tens of seconds on the 30 Hz caller.

#### Scenario: Model digest changes

- **GIVEN** a cached engine built from a different ONNX digest
- **WHEN** the session is created
- **THEN** the stale plan is not accepted as a warm cache and the build is recorded as cold

#### Scenario: Unwritable cache directory

- **GIVEN** an unwritable cache path
- **WHEN** config or preflight runs
- **THEN** it fails with a named error rather than silently disabling the cache

## ADDED Requirements

### Requirement: The runtime SHALL verify the artifact digest before session creation

Digest verification SHALL live in `factory/world_model.py`, where the download happens.
Today it checks only `model_path.is_file()` after `download_weights_from_huggingface`.
ADR-008's own "Negative" section records the consequence: "wrong weights = wrong inference,
silently". The mismatch-counter precedent is `_registry_cloud.py:227`.

A requested Hugging Face filename that does not exist SHALL be a named contract failure. A
`.pt` checkpoint SHALL NEVER be substituted for an `.onnx` filename. Verified against the
Hub, `ianshank/mousedroid-dual-stream-rssm` currently holds only `.gitattributes` and
`README.md`, so this path fails today and the documentation inviting it is corrected in the
same change.

Publication to the Hub SHALL remain a separate, reviewed command. It SHALL NOT run from a
test or a deploy script, and the token SHALL come from the environment and never be logged.

#### Scenario: Requested filename absent from the Hub

- **GIVEN** `onnx_filename: observe_step.onnx` and a repo without it
- **WHEN** the factory resolves the path
- **THEN** it raises a named contract failure naming the repo and filename

#### Scenario: Digest mismatch against the manifest

- **GIVEN** a local `.onnx` whose SHA-256 differs from the recorded digest
- **WHEN** the factory resolves the path
- **THEN** it refuses the artifact and increments the mismatch counter

### Requirement: `cuda_iobinding` SHALL avoid per-tick host round trips

When `onnx_execution_mode=cuda_iobinding` and a CUDA-capable provider is observed active,
recurrent `h` and `z` SHALL stay on device between ticks and outputs SHALL be returned as
Torch CUDA tensors without a NumPy conversion.

`observe_step` SHALL continue to return `tuple[Tensor, Tensor, Tensor, float]` —
`WorldModelProtocol` is `@runtime_checkable` (`world_model/protocol.py:17-28`) — so
`surprise` is still cast to a Python float at a named boundary.

Buffers SHALL be allocated after warmup, not in the constructor:
`tests/unit/world_model/test_dual_stream_rssm_onnx.py:111` pins that construction does not
load a session. Rebinding SHALL occur only when pointer, shape, dtype or device changes.
`portable` SHALL retain today's NumPy path for CPU hosts and for regression comparison.

This requirement depends on `src/mousedroid/world_model/CLAUDE.md` invariant 3 ("No
In-Place Tensor Mutation") being scoped in writing to the PyTorch autograd graph. That
amendment lands before any I/O-binding code.

CUDA Graph is out of scope for this change.

#### Scenario: Changing inputs do not replay stale output

- **GIVEN** ten consecutive calls with different sensor values
- **WHEN** each returns
- **THEN** the outputs differ, proving no buffer reuse replays a prior result

#### Scenario: CPU fallback cannot claim I/O binding

- **GIVEN** `cuda_iobinding` configured and only the CPU EP observed active
- **WHEN** warmup completes
- **THEN** under a permissive policy it selects portable mode and logs the downgrade
- **AND** under a strict policy it raises

### Requirement: Parity gates SHALL exclude the posterior sample

Torch↔ONNX parity SHALL compare `new_h`, `obs_embed` and `surprise` at `atol=1e-4`, and
SHALL compare the posterior distribution (`post_mean`, `post_logvar`) rather than `new_z`.

ADR-008's "Cross-engine equivalence guarantee" excludes `new_z` because its
`torch.randn_like` source diverges between the PyTorch and ONNX RNG paths. A gate over
`new_z` fails by construction on a correct implementation.

The FP32 tolerance SHALL NOT be tightened below the accepted `atol=1e-4` without a
measurement justifying it. FP16 parity SHALL start at `rtol=1e-2, atol=1e-3` over multiple
recurrent steps, and SHALL additionally require recorded task replay showing no
safety-envelope regression.

#### Scenario: A correct FP32 implementation

- **GIVEN** seeded deterministic observations
- **WHEN** parity runs over `new_h`, `obs_embed`, `surprise`
- **THEN** all three agree within `atol=1e-4` and `new_z` is not compared

#### Scenario: FP16 recurrent drift

- **GIVEN** FP16 enabled and a multi-step recurrent rollout
- **WHEN** drift is measured against the PyTorch reference
- **THEN** a single-call comparison is insufficient and the multi-step result governs

## ADDED Requirements — round 2

### Requirement: The graph's input set SHALL be validated against the loading config

At warmup, the runtime SHALL compare `session.get_inputs()` names against
`all_input_names_for_cfg(cfg)` and SHALL raise a named exception on any mismatch, naming the
missing and unexpected inputs.

The exported graph's input set is config-dependent: `onnx_io.py` declares six always-present
inputs plus `ultrasonic`, `audio`, `lidar` and `imu`, each gated on `cfg.<modality>_dim > 0`.
Today nothing checks them. `run_session_with_zeros` builds warmup feeds from
`session.get_inputs()` (`common/onnx_session.py:142`) — the graph, not the config — and
`session.get_inputs()` is used nowhere else, so an artifact exported under a different
modality set loads, warms, reports healthy, and then raises inside `observe_step` on the
30 Hz path. A SHA-256 check does not detect this: the digest can be perfectly valid for the
wrong graph.

The export and the runtime SHALL both derive their input-name set from `onnx_io.py`'s
accessors. There are currently three independent derivations of one set — the export via
`build_example_inputs(cfg)` (`export_dual_stream_rssm_onnx.py:264`), the runtime via
`packed.<modality> is not None` (`dual_stream_rssm_onnx.py:238-253`), and the accessors
themselves, which are referenced only by `tests/unit/world_model/test_onnx_io.py`. That is
exactly the drift `onnx_io.py`'s docstring exists to prevent.

The export metadata SHALL record the modality dimensions the artifact was built from.

#### Scenario: Artifact exported with a different modality set

- **GIVEN** an `.onnx` exported with `lidar_dim = 0` and a config with `lidar_dim > 0`
- **WHEN** warmup runs
- **THEN** it raises a named exception naming `lidar` as an expected-but-absent graph input
- **AND** the failure occurs at warmup, not on the first tick

#### Scenario: Digest valid, graph wrong

- **GIVEN** an artifact whose SHA-256 matches its manifest but whose modality set does not
  match the config
- **WHEN** the factory resolves and the runtime warms it
- **THEN** the digest check passes and the input-set check fails closed

### Requirement: TensorRT shape profiles SHALL be derived per input

`_dynamic_axes_for_inputs` (`export_dual_stream_rssm_onnx.py:228-240`) marks axis 0 as the
symbolic `batch` dimension for every input **and** every output. The TensorRT EP therefore
requires `trt_profile_min_shapes` / `_opt_` / `_max_` as name-indexed shape strings covering
every dynamic input.

A single scalar batch field cannot express that. The implementation SHALL build the profile
strings from `all_input_names_for_cfg(cfg)` plus the `ModelConfig` dimensions, and the
benchmark record SHALL report the profile actually passed.

A fixed-batch export SHALL NOT be treated as a pure optimization: the dynamic axis is
deliberate, so the same `.onnx` can be reused at training-time batch sizes, per the
`CFC_ONNX_SPIKE_REPORT.md` rationale quoted in that function's docstring. Removing it is a
behaviour change requiring its own decision record.

#### Scenario: Profile covers every dynamic input

- **GIVEN** a config enabling ultrasonic, lidar and imu but not audio
- **WHEN** the TensorRT provider options are built
- **THEN** the min/opt/max profile strings name all nine inputs present in the graph

### Requirement: The recorded digest SHALL be enforced at load, not merely recorded

Digest verification SHALL reuse `utils/weights_manager.py::verify_sha256` and the
`sha256.txt`-manifest-in-the-same-repo convention, SHALL increment a mismatch counter, and
SHALL fail closed. The Hugging Face fetch SHALL pin `revision=`.

This mechanism already exists, fail-closed and metricised, on the OTA weight path:
`cloud/weight_update_poller.py:303`, the `sha256_manifest_filename` field
(`config/schema/gcp_cloud.py:412-420`, "a download is refused if the local SHA does not match
this manifest"), and `inc_cloud_weight_update_sha256_mismatch`
(`telemetry/metrics/_registry_cloud.py:234`). The world-model ONNX path uses none of it:
`factory/world_model.py:290-361` checks only `model_path.is_file()` and passes no `revision`.
Recording a digest in a deploy record without enforcing it at load changes nothing.

Any `.pt` arriving over the network SHALL be loaded with `weights_only=True` after digest
verification, following `growth/slot_store.py:112-121`.
`world_model/checkpoint_migration.py:293` (`weights_only=False`, no preceding digest check)
and the silent legacy-loader fallback at `export_dual_stream_rssm_onnx.py:353-359` SHALL be
fixed or explicitly fenced with a stated trusted-source precondition.

#### Scenario: Digest mismatch on a downloaded artifact

- **GIVEN** a downloaded `.onnx` whose SHA-256 differs from the repo's `sha256.txt` entry
- **WHEN** the factory resolves the path
- **THEN** the load is refused, the mismatch counter increments, and the revision is not
  marked seen

### Requirement: Cache-path fields SHALL be validated before any directory is created

New engine- and timing-cache path fields SHALL carry a `field_validator` modelled on
`config/schema/learning.py:15-58` (`_validate_relative_slot_dir` — relative-only, no `..`,
POSIX and Windows semantics), and the directory SHALL be validated on **every** branch,
before `mkdir`.

Today `factory/world_model.py:320-321` calls `mkdir(parents=True, exist_ok=True)` on a bare
`str` config value, while the protected-root guard lives inside the download helper
(`utils/weights_manager.py:120-132`) which runs afterwards — and is skipped entirely when
`model_path.is_file()` short-circuits at `:324-330` or an explicit `onnx_path` returns at
`:305-307`. The container runs as root (`Dockerfile.jetson` declares no `USER`) with
`privileged: true` (`docker-compose.jetson.yml:21`) and `/opt/mousedroid` mounted
read-write, so an unvalidated config-supplied path is a root-authority write primitive with
host reach.

The 0700 mode that `efficiency/tensorrt.py:300` assumes in a comment SHALL be enforced for
any cache directory this change introduces. No requirement is made about rejecting unwritable
paths — no precedent for that exists in the tree.

#### Scenario: Traversal path rejected at config load

- **GIVEN** a cache dir of `../../etc/mousedroid`
- **WHEN** `Settings` loads
- **THEN** validation fails with an operator-actionable message and no directory is created

#### Scenario: Cache hit does not skip validation

- **GIVEN** a pre-placed artifact under an unvalidated path
- **WHEN** the factory resolves it
- **THEN** the path is validated before the file is accepted


### Requirement: A random-weight fallback SHALL be observable and refusable

Any model that falls back to random initialisation SHALL increment a metric and SHALL be
refusable by config, mirroring `onnx_require_primary_provider`'s fail-closed shape. Every
benchmark and promotion record SHALL state which weight source loaded.

`_resolve_bdi_weights` (`factory/cognitive.py:38-85`) currently falls through to
`NeuralBDI()` with only `_log.warning("weights_not_found_using_random_initialization")`; the
`weights_source` label it returns reaches nothing but a structured log at `:133`. And
`build_world_model` never loads weights at all. So the rover can run a random-weight world
model and a random-weight cognitive core with no metric, no health-check failure and no
startup refusal.

The same download path takes no `revision=` and performs no digest check, while the
fail-closed `verify_sha256` + `sha256.txt` machinery exists on the OTA path — and this is the
path that executes on every boot, since `config/jetson_production.yaml:200,203` set
`cognitive.enabled: true` and `auto_download: true`.

Evidence measured against an unrecorded weight state is the same defect as evidence measured
against an unrecorded execution provider.

#### Scenario: Weight download fails at boot

- **GIVEN** no local BDI weights and a failed Hugging Face fetch
- **WHEN** the cognitive core is built
- **THEN** a random-init metric increments and, under a strict policy, startup fails rather
  than proceeding on random weights

#### Scenario: A benchmark records its weight provenance

- **GIVEN** a promotion benchmark run
- **WHEN** the record is written
- **THEN** it names the weight source and digest for every model loaded, alongside the
  observed execution provider
