# Design — Jetson ONNX runtime evidence + PC-to-rover delivery

Each section states the alternative considered and why it was rejected. Where rev A of the
external plan chose differently, the rev-A choice is the rejected alternative.

## Which of these sections were built

The task 2.1 ceiling gate fired (`proposal.md` -> "The ceiling, computed"): the end-to-end
ceiling for accelerating `observe_step` is **1.0016x-1.0039x**, so I/O binding, FP16 and the
TensorRT engine cache are **not built**.

| section | built | note |
| --- | --- | --- |
| D-1 measure first | **partly** | `observe_step_timing.py`, both PyTorch engines, `build_world_model(cfg, *, metrics=)`. The three-span split D-1 also specifies is tasks 3.4-3.5 — **NOT BUILT (task 2.1 gate)**; those are `onnx_copy_seconds` spans timing the copy path the gate closed against. |
| D-2 thread metrics through the factory | **yes** | No global; the registry is built once, before the engine. |
| D-3 correct the docs to `jetson_dual_stream.yaml` | **yes** | Phase 4 narrative sweep. |
| D-4 widen `resolve_providers` | no | Provider options are Phase 6. `resolve_providers` is unchanged. |
| D-5 source precision/workspace/cache dir from `cfg.jetson` | no | Nothing reads them on an ORT path yet. |
| D-6 keep the bind-mount spine, Docker-tag rollback | **yes** | Phase 8. |
| D-7 no `world_model:` key in a tracked overlay | **yes** | Pinned by `test_f050_backwards_compat.py`. |
| D-8 extend `deployments/<platform>-image.json` | **yes** | Phase 8, in place. |
| D-9 extend the PC-side push path | **yes** | Phase 8 dirty-target refusal + WIP preservation. |
| D-10 strict health check as opt-in | **yes** | Phase 8; default behaviour unchanged. |
| D-11 exclude `new_z` from parity gates | n/a | No parity gate lands — it needs a trained checkpoint. |
| D-12 scope the in-place-mutation invariant | no | Nothing mutates in place, so the invariant stays as written. |

The unbuilt sections are kept rather than deleted: each remains the correct decision *if* a
measured `observe_step` share ever contradicts the derivation, and deleting them would lose the
reasoning along with the conclusion.

## D-1. Measure before optimizing, and fix the metrics seam first

**Decision.** Phase 1 wires `MetricsRegistry` into `build_world_model`, makes the PyTorch
engine emit the same histogram, and splits the observe-step timing into three spans
(pack / execute / convert). The first two landed; the three-span split did **not** — it is
tasks 3.4-3.5, whose `onnx_copy_seconds` family times the copy path the task-2.1 gate
closed against, so the ONNX engine still times `session.run` alone. Stated here rather
than only in the ledger above, because a reader who stops at this section would otherwise
come away believing the histogram family sums to wall-clock `observe_step`. Phase 2 captures a rover baseline of
`mousedroid_tick_phase_ms{phase=...}`. The I/O-binding, FP16 and cache work in Phases 4-6
is conditional on that baseline showing `phase="world_model"` as a material share of the
tick.

**Rejected: rev A's ordering** (config schema → provider options → I/O binding → benchmark
harness → evidence). It builds four optimizations and then measures. Three facts make that
unsafe here: the target metric has no production writer
(`factory/world_model.py:262-266` omits `metrics=`), the one timer that does exist brackets
`session.run` only (`dual_stream_rssm_onnx.py:255-258`) and so cannot see the copy
boundary I/O binding removes, and no rover latency evidence exists anywhere in `reports/`
or `smoke-reports/`. Optimizing under those conditions produces a number nobody can
attribute.

**Rejected: adding a fourth histogram for the full wall-clock.** The existing
`mousedroid_world_model_observe_step_seconds` is already named and alerted as
`observe_step` wall-clock (`alerts.yml:388-421`). Renaming it would break the alert and the
golden sample (`registry.py:221`); adding a parallel family would leave two metrics both
claiming to be `observe_step`. Keeping its ORT-execute scope and adding `onnx_copy_seconds`
spans that sum with it is the smaller change and keeps the alert honest once the seam is
live.

## D-2. Thread metrics through the factory rather than importing a global

**Decision.** `build_world_model(cfg, *, metrics: MetricsRegistry | None = None)`, and
`factory/orchestrator.py` moves `build_metrics_registry(cfg)` (currently `:210`) above the
`build_world_model(cfg)` call (currently `:121`).

**Rejected: a module-level registry singleton.** It would satisfy the seam in one line and
violate Factory-First DI (CLAUDE.md invariant 1) — concrete types stay imported inside
`factory/`, and application code stays typed against Protocols. It would also break
`tests/unit/world_model/test_dual_stream_rssm_onnx.py:111`
(`test_construction_does_not_load_session`), which pins cheap construction.

**Rejected: `getattr(metrics, "observe_...", None)` defensive lookup.** The B2 Story 2
interim did exactly that and `dual_stream_rssm_onnx.py:267-278` documents its removal. Do
not reintroduce it.

## D-3. Correct the docs to `jetson_dual_stream.yaml`, do not add `cfc_hidden_dim` to the production overlay

**Decision.** `engine: onnx_trt` requires `cfg.model.cfc_hidden_dim > 0`
(`factory/world_model.py:246-253`); the schema default is `0` (`world_model.py:205`) and
only `config/jetson_dual_stream.yaml:29` sets `64`. ADR-008's migration step 2 and
`scripts/export_dual_stream_rssm_onnx.py:17-21` both name `config/jetson_production.yaml`,
which cannot work. Fix the documentation to name `config/jetson_dual_stream.yaml`, and run
the on-rover ONNX evidence from that overlay composed with the FP16 overlay.

**Rejected: adding `cfc_hidden_dim: 64` to `config/jetson_production.yaml`.** It changes
the world-model architecture of the default production overlay as a side effect of a
performance change, and it walks into D-7's `config-compat` silent-ignore window. The
narrative fix is the honest one; `narrative-correction-sweep` covers the sweep so the false
claim is removed from every live prose surface, not just ADR-008.

## D-4. Widen `resolve_providers` to a provider-spec type; do not bypass it

**Decision.** Introduce an internal `ProviderSpec = tuple[str, Mapping[str, object]]` and
change `resolve_providers` / `warmup_session` to accept a sequence of either bare names or
specs, matching on the name component only. Add a post-construction
`session.get_providers()` comparison, and return both requested and active tuples.

**Rejected: passing `provider_options=` around `resolve_providers`.** Today
`resolve_providers` is `tuple[str, ...] -> tuple[str, ...]`
(`common/onnx_session.py:72-100`) and a `("TensorrtExecutionProvider", {...})` tuple fails
`p in available`, empties the intersection, and silently returns
`("CPUExecutionProvider",)`. Bypassing the function to avoid the signature change leaves
that trap armed for the next caller.

**Rejected: treating the pre-construction intersection as proof.** `warmup_session:193`
computes `active` before building the session at `:207-210` and returns it at `:221`;
`DualStreamRSSMOnnx.active_providers` (`:154-156`) surfaces it verbatim. If TensorRT is
listed but fails to initialise for this graph, ORT falls back internally and the repo still
reports TensorRT. `session.get_providers()` appears nowhere in the tree — adding it is the
prerequisite for both `onnx_require_primary_provider` and the fallback counter.

**Consequence accepted.** This is a cross-subsystem change: `vla/policy.py:29,283-306`
shares the helper, and two ORT stubs pin the exact `InferenceSession(model_path, providers)`
arity — `tests/unit/common/test_onnx_session.py:98-103` and
`tests/unit/vla/test_distilled_onnx.py:107`. Both move in the same commit. Module
neutrality and lazy ORT import stay pinned by `test_onnx_session.py:391-415`, so the new
behaviour is parameter-passed and `common/onnx_session.py` still imports nothing from
`mousedroid.config`.

## D-5. Source precision, workspace and TRT cache directory from `cfg.jetson`

**Decision.** `JetsonConfig` already carries
`precision: Literal["fp32","fp16","int8"] = "fp16"`, `workspace_gb: float = 1.0` and
`tensorrt_cache_dir` (`hardware.py:580-588`), and `config/default.yaml` sets the first two.
The ORT session reads those. `WorldModelConfig` gains only what has no home:
`onnx_execution_mode`, `onnx_require_primary_provider`, `onnx_device_id`,
`onnx_engine_cache_enabled`, `onnx_timing_cache_enabled`,
`onnx_context_memory_sharing_enabled`, `onnx_profile_batch`. `onnx_cuda_graph_enabled` is
**not** included: D-12 removes CUDA Graph from this change's scope, so adding its switch here
would put a deferred feature back into the schema.

**Rejected: rev A's `onnx_precision` / `onnx_max_workspace_bytes` /
`onnx_engine_cache_dir`.** They create two places to set TRT precision, two workspace
sizes and two engine-cache directories with nothing reconciling them — and
`config/jetson_dual_stream.yaml` already overrides `workspace_gb: 2.0`, so the two would
diverge immediately.

**Rejected: rev A's `onnx_warmup_iterations`.** It already exists
(`world_model.py:77`, consumed `factory/world_model.py:265`, documented `ADR-008:47`).

**Naming note.** `onnx_engine_cache_dir` is additionally unusable because
`WorldModelConfig.onnx_cache_dir` already means the *artifact download* cache
(`factory/world_model.py:320-322`). The TRT plan cache is `cfg.jetson.tensorrt_cache_dir`.

**Rejected: a nested `WorldModelOnnxRuntimeConfig` sub-model.** The repo does nest once a
group grows (`RoverConfig.sim/action/observation`, `TelemetryConfig.auth`), and eight new
fields on top of six is borderline. Flat `onnx_*` wins because `WorldModelConfig` is
already flat-`onnx_*` and a nested block would be a new top-level YAML shape — exactly the
case D-7 shows hard-fails `config-compat`.

## D-6. Keep the bind-mount-editable spine; roll back with the existing Docker tag anchor

**Decision.** `/opt/mousedroid` stays a git checkout bind-mounted at the same path
(`docker-compose.jetson.yml:79-80`, `Dockerfile.jetson:141`). Promotion is: preserve
rover WIP as a `rover/wip-<date>` commit, fetch and check out the immutable commit, pre-tag
`mousedroid:jetson-rollback-<date>`, rebuild, `up -d --force-recreate`, strict health
check, stabilization window. Rollback is `docker tag` restore plus a checkout — no network,
no rebuild.

**Rejected: rev A's `/opt/mousedroid/releases/<id>` + `current`/`previous` symlinks.** The
bind mount is an *editable install* whose finder points at an absolute path, so swapping the
mount root for a symlink farm breaks the install. It also changes the identity of
`/opt/mousedroid` for `sync_jetson_overlay.sh`'s `MOUSEDROID_INSTALL_DIR`,
`jetson-nightly.yml:87`, `preflight_check.sh:29-31`, and `docker_deploy.sh:165-169` (which
requires `${INSTALL_DIR}/pyproject.toml` to already exist). And it discards the rover-local
git workflow that is the repository's actual WIP-preservation mechanism. Replacing the
deployment model is an ADR-level change touching both systemd units and
`deployments/jetson-image.json`'s whole rationale — not two new scripts inside a
performance change.

**Rejected: rev A's `flock` deploy lock as new machinery.** Verified absent from the tree,
so it is greenfield rather than duplicated — but the serialization it buys is already
provided by the operator-gated single-writer promotion in D-9. A lock is added only if
Phase 7 shows concurrent apply is reachable.

## D-7. Add no `world_model:` key to any tracked overlay

**Decision.** New fields live on schema defaults, and the FP16/cache profile is selected
through `MOUSEDROID_WORLD_MODEL__*` environment variables in `/etc/mousedroid/docker.env` —
not through a tracked YAML overlay.

**Rejected: a `config/jetson_onnx_fp16.yaml` overlay** (rev A's proposal, and rev B's own
earlier wording). `check_config_compat.py` validates every changed `config/*.yaml` against
the schema at the pinned SHA, where `WorldModelConfig` is a plain `BaseModel`
(`032942b…:src/mousedroid/config/schema.py:1597`) with `extra="ignore"` — so such a file
passes the gate and has every key silently dropped by the pinned schema. An overlay and
"no `world_model:` key in tracked YAML" cannot both hold. Environment activation satisfies
both and matches the F-043 precedent.

**Rejected: rev A's "default all new switches OFF in `config/default.yaml`".** Two
problems. First, `loader.py:76-96` loads `default.yaml` as the base and deep-merges every
overlay on top, so a `world_model:` block there applies to all 17 overlays and
`jetson_dual_stream.yaml` would need an explicit override to get back what it has now.
Second and worse, `.github/workflows/config-compat.yml` worktrees the SHA in
`deployments/jetson-image.json` (`032942b5…`) and validates new YAML against the old
`Settings`; at that SHA `WorldModelConfig` is a plain `BaseModel` (`schema.py:1597`), i.e.
`extra="ignore"`. A new `world_model.*` key therefore **passes the gate and is silently
ignored by the pinned schema** — a worse outcome than a hard failure, from the very gate
that exists to catch "YAML merges, rover crash-loops". A new top-level block hard-fails
instead. Schema-only defaults avoid both.

## D-8. Extend `deployments/<platform>-image.json`; no parallel manifest

**Decision.** The deploy record gains the model digest and the proven ORT provider.
`config-compat.yml:8-10` already documents the matrix as extensible via additional
`deployments/<platform>-image.json` records, and
`tests/regression/test_config_no_dup_keys_and_deploy_record.py` pins
`("sha","platform","image_tag")` plus a full 40-hex SHA.

**Rejected: rev A's standalone `dist/release-manifest.json` with its own JSON Schema.** A
second manifest format for the same facts, with the pinned-SHA gate reading only the first.
Any SHA it carried would still inherit `mouse-droid-deploy-repin`'s pin-reachability spec —
remote-*tag* reachability, remedied by `scripts/repin_tags.sh` — so it would need that
plumbing twice.

**Digest verification belongs in the factory, not a shell check.**
`factory/world_model.py` checks only `model_path.is_file()` after
`download_weights_from_huggingface`. The precedent for a digest counter already exists at
`_registry_cloud.py:227` (`inc_cloud_weight_update_sha256_mismatch`).

## D-9. Extend the existing PC-side push path; keep the service switch operator-gated

**Decision.** `scripts/deploy_remote.sh` is the PC-side command
(`:1-18`, `--full|--code-only|--config-only`, host resolution via
`~/.mousedroid/jetson_host` or `jetson_discover.sh`). Its `rsync -avz --delete` at `:151`
gains a dirty-target refusal and a `rover/wip-<date>` preservation step before any
destructive sync. The service switch stays an explicit operator confirmation.

**Rejected: rev A's new `scripts/promote_jetson.py` as a sixteenth deploy script.** The
campaign plan already names `deploy_remote.sh` as the script that "destroys rover work";
adding a parallel controller leaves the dangerous one in place and forks the operator
surface across `deploy_remote.sh`, `deploy_jetson.sh`, `docker_deploy.sh`,
`sync_jetson_overlay.sh` and `jetson_full_validation.sh`.

**Rejected: rev A's `--approved-release <id>` skipping confirmation.** It converts the
promotion path into an unattended automated write to a robot. Keeping a human on the
service switch is what keeps the charter answer in Q1 "no".

**Lint note.** A Python helper under `scripts/` is `C901`-exempt
(`pyproject.toml:287`) but must still pass `ruff check scripts/` (blocking,
`ci.yml:112-113`); `ruff format --check` and `mypy --strict` do not cover `scripts/`. No
shellcheck gate exists in CI, so shell changes get a Python test instead — the precedent is
`tests/unit/scripts/test_repin_tags.py`.

## D-10. Strict health check as an opt-in mode, not a default flip

**Decision.** `scripts/docker_deploy.sh` gains `--strict-health`, which fails on a dead
telemetry endpoint, an ORT provider other than the configured primary, or a model-digest
mismatch. Promotion uses it; the general-purpose helper keeps today's permissive default.

**Rejected: making the existing health check fail-closed.** Every leg is currently
downgraded at `:241` (`health_check || warn "…non-fatal for deployment"`), and the
telemetry leg is explicitly a `warn` with "(This is normal if telemetry is disabled or
still starting)" at `:116-123`. Flipping that default would break the bring-up flows that
legitimately run before telemetry is up. The container `HEALTHCHECK`
(`scripts/mousedroid_healthcheck.sh`) only checks heartbeat freshness and cannot express
provider state at all, so the assertion has to be a new surface either way — and under
`PYTHONOPTIMIZE=1` (`Dockerfile.jetson:178`) any Python-side guard raises a named
exception, never `assert`.

## D-11. Exclude `new_z` from every parity gate

**Decision.** Parity compares `new_h`, `obs_embed`, `surprise` at `atol=1e-4`, and
compares the posterior *distribution* (`post_mean`, `post_logvar`) rather than the sample.

**Rejected: rev A's drift gate over `new_h`, `new_z`, `obs_embed`, `surprise`.** ADR-008's
"Cross-engine equivalence guarantee" states `new_z` is intentionally excluded because its
`torch.randn_like` source diverges between the PyTorch and ONNX RNG paths. A `new_z` gate
fails by construction on a correct implementation.

**Rejected: rev A's `rtol=1e-4, atol=1e-5` FP32 tolerance.** It tightens the accepted
`atol=1e-4` (ADR-008, `tests/unit/training/test_export_dual_stream_rssm_onnx.py`) with no
supporting measurement. Tighten only after the Phase 2 baseline justifies it.

## D-12. I/O binding needs the in-place-mutation invariant scoped in writing

**Decision.** Before any I/O-binding code lands, `src/mousedroid/world_model/CLAUDE.md`
invariant 3 ("No In-Place Tensor Mutation") is amended to scope itself to the PyTorch
autograd graph and to state that ORT device buffers owned by a runtime wrapper are exempt,
with the changing-input tests named as the compensating control.

**Rejected: landing I/O binding and treating the invariant as obviously inapplicable.**
I/O binding is pre-allocated buffer reuse with in-place writes by construction. Shipping it
silently weakens a documented subsystem invariant, which is the failure mode
`doc-reconciler` exists to catch.

**Rejected: rev A's eager buffer allocation in the constructor.**
`test_dual_stream_rssm_onnx.py:111` pins that construction does not load a session.
Buffers are allocated after warmup, which is itself lazy on first `observe_step`
(`:223-224`) — a synchronous call, so a cold TRT engine build blocks the caller. Phase 5
moves warmup behind `asyncio.to_thread` at the factory boundary before enabling caches,
since an engine-cache miss turns a millisecond stall into tens of seconds.

**CUDA Graph stays out of scope for rev B.** Rev A already defaulted it off; rev B removes
it from the task list entirely and records it as deferred. It cannot be justified before
Phase 2 shows the stage matters, and `run_with_iobinding` with stable buffers is the
prerequisite anyway.
