# Tasks: `mouse-droid-branch-hygiene-sweep` (F-052)

Task ordering is binding: each task lands green before the next starts. Deviations from
task wording are recorded inline — declared, not silent.

Tasks marked **[LANDED]** were executed while authoring this plan, because they are
confirmed security defects in code the branch under review introduced and leaving them
open while writing a document about them would be the exact failure this change exists to
correct. Everything else is unstarted.

---

## Phase 0 — Create ratchet headroom (blocks everything else)

All three suppression budgets are at ceiling. No later phase can add a suppression until
this one lands. It creates capacity by deletion, never by raising a ceiling.

- [ ] 0.1 Delete the three inert `hardcoded_ok` markers whose values are already in the
  gate's `ALLOWED_NUMERIC_VALUES = {0.0, 1.0, -1.0}`:
  `src/mousedroid/comms/_utils.py:23` (`= 1`), `:26` (`= 0`),
  `src/mousedroid/comms/command_set.py:71` (`= 1`). Verify with
  `python -m tools.ratchet_budgets` that the count drops 26 → 23 and no gate finding
  appears.
- [ ] 0.2 Replace the two `1000.0` duplications with `constants.MILLISECONDS_PER_SECOND`
  (`src/mousedroid/constants.py:67`) at `src/mousedroid/validation/latency_stats.py:30`
  and `src/mousedroid/comms/command_set.py:68`, deleting both markers. Count 23 → 21.
- [ ] 0.3 Ratchet `.claude/workforce.yaml` `hardcoded_ok` to `ceiling: 21`,
  `warn_threshold: 19`, with a comment recording *why* (inert markers + an existing
  constant), matching the precedent at `:118-128`. Do not bank the slack.
- [ ] 0.4 Sync the stale fallback at `tools/claude_hooks/config.py:353` (`ceiling=24,
  warn_threshold=22`) to the post-ratchet values, and add a test asserting every
  `RatchetBudgetItem` default equals the shipped `.claude/workforce.yaml` entry — the
  docstring at `:339-341` claims this and nothing enforced it.
- [ ] 0.5 `python -m tools.ratchet_budgets --strict` exits 0 and
  `tests/regression/test_suppression_budget.py` passes.

## Phase 1 — Confirmed security defects in this branch's own code

Each was verified by driving the real script, not by reading it.

- [x] 1.1 **[LANDED]** Widen `require_safe_remote_path`'s reject set: TAB, VT and FF were
  absent from the bracket class, and tab is an IFS character, so the value word-split at
  every argv boundary into a command running under `sudo` on the rover.
- [x] 1.2 **[LANDED]** Fix the backslash arm. It was spelt `*'\\'*`; a *quoted* backslash
  pair in a `case` pattern matches TWO literal backslashes, so a lone backslash passed.
  Now an unquoted `*\\*` arm with its own message.
- [x] 1.3 **[LANDED]** Reject glob metacharacters (`* ? [ ]`). They are not shell
  metacharacters, so no existing arm caught them, and the remote shell pathname-expands
  an unquoted value.
- [x] 1.4 **[LANDED]** `REMOTE_CONFIG_Q` was computed and never used while
  `${REMOTE_CONFIG}` crossed raw at `:339` and `:350`, contradicting the header's claim
  that every boundary is `%q`-quoted. Both boundaries now use the quoted form via the
  `bash -c` idiom already established at `:316`.
- [x] 1.5 **[LANDED]** Five further raw boundaries, found by the new widened test rather
  than by reading: three `remote_sudo bash "${REMOTE_SRC}/scripts/*.sh"` sites, the pip
  `-e "${REMOTE_SRC}[hardware,jetson]"` target (whose bracket expression survived only
  because `nullglob` is off by default), and the rsync destination.
- [x] 1.6 **[LANDED]** Tests: three new payload classes (word-splitting, single
  backslash, glob), `test_every_pre_quoted_form_is_used` — which is the test that would
  have caught 1.4, since the pre-existing one asserted the quoted form was *defined* and
  the dead variable satisfied that — and a raw-boundary scan widened from `${REMOTE_SRC}`
  alone to `${REMOTE_CONFIG}` and `rsync`. 47 pass.
- [ ] 1.7 Validate and quote `REMOTE_USER`. **Pre-existing** — env-overridable at the
  merge base (`18aba56:scripts/deploy_remote.sh:15`), so deliberately *not* folded into
  the landed fix above. Two exposures: it is interpolated raw into the root-executed
  remote string at `:316` (`chown -R ${REMOTE_USER}:${REMOTE_USER}`), and it leads the
  ssh destination `"${REMOTE_USER}@${HOST}"`, so a value beginning with `-` (e.g.
  `-oProxyCommand=…`) is parsed by ssh as an **option and executed locally** before any
  rover is contacted. Validate against `^[A-Za-z_][A-Za-z0-9_-]*$`, add `--` before the
  destination, and validate `HOST` the same way (`:211` only strips whitespace; `:220`
  trusts `jetson_discover.sh` stdout).
- [ ] 1.8 `scripts/rover_wip_guard.sh:125` — `trap "rm -rf -- '${staging}'" RETURN`
  interpolates at trap-set time (the `SC2064` disable at `:124` makes that deliberate), so
  a `TMPDIR` containing a single quote reshapes an `rm -rf` that runs on the rover. Change
  to a single-quoted trap body; the local is still in scope at RETURN.
- [ ] 1.9 `scripts/rover_wip_guard.sh:201` — `git checkout -b "${branch}"` passes
  `MOUSEDROID_ROVER_WIP_BRANCH` with no `--` separator and no validation. Add `--` and
  validate via `git check-ref-format --branch`.
- [ ] 1.10 `scripts/docker_deploy.sh:43-48` sources `/etc/mousedroid/docker.env` as shell
  code under `sudo`, and `:363-373` creates it with a default umask (0755/0644) — so any
  non-root write to the file is root code execution, and that file is the documented home
  of `MOUSEDROID_TELEMETRY_TOKEN` and `ANTHROPIC_API_KEY`. Replace `.` with a
  `KEY=VALUE`-only read loop and `chmod 600` at creation.
- [ ] 1.11 `scripts/docker_deploy.sh` — validate `CONTAINER_NAME` and `DEPLOY_RECORD`
  (both come from that sourced file) against `^[A-Za-z0-9_.-]+$` and insert `--` before
  the container name at `:234`, `:256`, `:264`, `:276`, `:394`, `:411`, `:417`.
- [ ] 1.12 `scripts/deploy_remote.sh:285-289` — the off-rover WIP archive can contain the
  rover's whole non-gitignored tree and is written with a default umask, never pruned,
  never encrypted. `install -d -m 700` the directory and `umask 077` around the
  redirection.
- [ ] 1.13 `.dockerignore` — no pattern for `*.pem`, `*.key`, `*.crt`, `*.p12`, `id_rsa*`,
  `.ssh/`, while `docker/Dockerfile.cloud:76-77` does `COPY scripts` + `COPY config` and
  `Dockerfile.dev:62` does `COPY config/`. An operator-local key is baked into a published
  layer. Add them to the secrets block at `:119-130`.
- [ ] 1.14 `.dockerignore` and `.gitignore` — the `.env.jetson` shape is covered by
  **neither** (`.env` matches the exact name; `*.env` needs a `.env` suffix), yet the
  shipped template is `config/.env.jetson.example`, so the natural
  `cp … config/.env.jetson` yields an untracked-but-not-ignored credential file. Add
  `.env.*` / `**/.env.*` with a `!**/.env.*.example` negation to both.
- [ ] 1.15 `.dockerignore:46` — `*.md` has no `**/` twin and `openspec/` has no rule, so
  every nested `src/mousedroid/**/CLAUDE.md` reaches the build context and, via
  `Dockerfile.jetson:57 COPY src/ ./src/`, the image. Add `**/*.md` with a `!README.md`
  negation, plus `openspec/` and `smoke-reports/`.
- [ ] 1.16 `tests/regression/test_secret_scan_gate.py:100-110` checks only top-level
  `allowlist.paths`, so a path waiver can re-enter through `[[rules]].allowlists[].paths`
  and stay green. Walk the parsed TOML at every depth and assert `extend.disabledRules`
  is absent.

## Phase 2 — Make the gates run where the work lands

- [ ] 2.1 Add a `bash -n` parse gate over every tracked `.sh`, blocking, as a regression
  test using `tests/_bash.py::requires_bash`. Measured green today: 58 files, 9,036 lines.
  Generalises the existing two-file convention
  (`tests/regression/test_jetson_full_validation_script.py:40-50`).
- [ ] 2.2 Add `shellcheck` to the `lint` job with `continue-on-error: true`, plus an entry
  in `.github/advisory_stages.yaml` carrying `since`, `promote_after_days` and a reason.
  Advisory because the finding count over 9,036 pre-existing lines is unknown and Phase 0
  gave this change no suppression allowance.
- [ ] 2.3 Add a `Changed-lines branch coverage` step to the `local-gates` job,
  `if: github.event_name == 'pull_request'`, mirroring the `Hardcoded-value gate` step at
  `ci.yml:434-441`. Update the stale "local-only by design (need heavy deps)" comment at
  `ci.yml:380-383` — that job installs those deps.
- [ ] 2.4 Add `_GATED_ORCHESTRATOR_FILES` to `tests/regression/test_f042_aqa.py`, mirroring
  the `_GATED_FACTORY_FILES` mechanism the same file already uses at `:14-20` / `:49-52`,
  and change the orchestrator assertion at `:68` from `on_disk == exempt` to
  `on_disk == exempt | _GATED_ORCHESTRATOR_FILES` with the same disjointness and
  `not _is_exempted_from_branch_gate` checks. Required first: `:68` is a **categorical**
  rule that every orchestrator `_*.py` is exempt, so there is currently no way to gate one.
  (Revised from the first draft, which said simply "remove it from `_ALLOWED_FILES`" —
  that would have broken this invariant rather than amended it. See `design.md` D-1.)
- [ ] 2.5 Remove `src/mousedroid/orchestrator/_lifecycle_mixin.py` from
  `check_branch_coverage.py::_ALLOWED_FILES` and add it to `_GATED_ORCHESTRATOR_FILES`.
  `_warm_world_model()` is algorithmic, not "pure DI wiring", and the file's own comment at
  `:66-68` says algorithmic modules stay gated.
- [ ] 2.6 Move `src/mousedroid/factory/orchestrator.py` from `_ALLOWED_FILES` into the
  existing `_GATED_FACTORY_FILES`. No new mechanism needed on this side.
- [ ] 2.7 Update the two remaining byte-for-byte pins:
  `tests/unit/scripts/test_check_branch_coverage_base_ref.py:455` and
  `tests/regression/test_f042_backwards_compat.py`.
- [ ] 2.8 Add the `[onnx_world_model]`, `[vla]` and `[hardware,jetson]` extras to the
  blocking `security` job's audit. `ci.yml:792` installs only `".[dev,telemetry,mcp]"` and
  `:802` runs `pip-audit --skip-editable` against *that* tree, so five packages this
  branch installs into the rover image are outside the blocking gate entirely.
- [ ] 2.9 Cap the unbounded majors in `pyproject.toml:110-124`
  (`onnxruntime-gpu>=1.18,<2`, `onnx>=1.15,<2`, `onnxscript>=0.2,<1`).

## Phase 3 — Pin the inventories so drift fails a test

- [ ] 3.1 Add `test_every_wired_hook_is_documented_in_the_runbook()` to
  `tests/regression/test_claude_workforce_aqa.py`, parsing hook modules out of
  `.claude/settings.json`. Assert wired ⟶ documented only; the reverse would forbid the
  runbook discussing an available-but-disabled hook.
- [ ] 3.2 Document `tools.claude_hooks.ratchet_budget_check` in
  `docs/runbooks/claude-workforce-hooks.md` — it is wired and running and appears in none
  of the runbook, `SKILLS.md`, `AGENTS.md` or `CLAUDE.md`. 3.1 must fail before 3.2 lands.
- [ ] 3.3 Add a `SessionStart` hook reporting the three ratchet counts against ceilings,
  which advisory windows are due, and working-tree cleanliness. Report-only, never
  blocking. Every one of those was manually re-derived during the F-050/F-051 work.
- [ ] 3.4 Add a `PreCompact` hook persisting change-bundle task state and the
  verified-findings list. The predecessor change lost the verified/asserted distinction
  across a compaction boundary — that is how its `design.md` came to mark D-1 built when
  half of it was not.
- [ ] 3.5 Both new hooks ship with tests under the `coverage.tools_line_min: 85` gate
  (`.claude/workforce.yaml:68`) and runbook entries, which 3.1 then requires.
- [ ] 3.6 Author skill `ceiling-gate`: run `analyze_observe_step_ceiling.py`, read the
  verdict against the three-part rubric at
  `docs/analysis/alayaworld-distillation-spike.md:65-75`, record GO/DEFER in the bundle.
- [ ] 3.7 Author skill `ort-provider-proof`: prove a provider claim by constructing the
  session and reading `session.get_providers()`, never by checking availability, and take
  the reading after every ORT-affecting install.
- [ ] 3.8 Author skill `remote-path-safety`: validate, then `printf '%q'` at *every*
  boundary — including argv-style ones, since ssh flattens argv and the remote shell
  re-parses. Phases 1.1–1.6 are its worked example.
- [ ] 3.9 Index all three in `SKILLS.md` (required by
  `test_claude_workforce_aqa.py:290`) and pass `python tools/validate_skill_commands.py`.

## Phase 4 — Close the test gaps that would actually have caught something

- [ ] 4.1 Extract `STRICT_PROBE_PY` (`scripts/docker_deploy.sh:139-230`) and drive all six
  `problems.append` branches: ORT not importable, artifact missing, session construction
  failure, provider downgrade, deploy record unreadable, record pins no `model_sha256`.
  `grep -rn STRICT_PROBE_PY tests/` returns nothing today — ~70 lines of the promotion
  gate, entirely unexecuted.
- [ ] 4.2 Add an **always-on** integration test that `build_world_model` returns a
  `CompositeWorldModel` that is `Warmable` and is warmed off the main thread, by
  monkeypatching the ONNX engine constructor with a torch-free Warmable fake. Today the
  only proof is `tests/unit/factory/test_factory_world_model_engine.py:132-134`, behind
  `pytest.importorskip("onnxruntime")` (`:22-24`), in the advisory
  `onnx-world-model-extras` job (`ci.yml:579-583`) — so the exact chain whose break was
  found empirically is still not proved by any blocking gate.
- [ ] 4.3 Add negative tests for `_validate_metadata_sidecar_filename`
  (`src/mousedroid/config/schema/world_model.py:174-190`): four `pytest.raises` cases
  (path separator, dot-path, bare `..`, collision with `onnx_filename`). Deleting the whole
  validator turns nothing red today.
- [ ] 4.4 Widen `tests/unit/test_bash_guard.py:178-201` from reading its own file to
  scanning `tests/**/test_*.py`, and hoist the two search literals to module constants so
  the scanner is not searching for its own body. As written it cannot fail unless someone
  edits that one module, and it proves nothing about the six other modules that inline the
  same `os.name` predicate (`tests/_bash.py:25-28`).
- [ ] 4.5 Replace the source-order pin at `tests/regression/test_f050_aqa.py:142-156`
  (`source.index(...) < source.index(...)`) with an object-graph assertion:
  `orch._world_model._metrics is orch._metrics_registry` on a real `build_orchestrator`.
  A reorder preserving both string positions while breaking the wiring passes today.
- [ ] 4.6 Add exact-value assertions for the five string defaults to
  `tests/regression/test_f050_backwards_compat.py` (`== "main"`, `== "sha256.txt"`,
  `== "observe_step.metadata.json"`), and give `CognitiveConfig`'s three new fields the
  backwards-compat half they lack entirely. Today they are pinned only as truthy
  (`test_f050_aqa.py:243-246`).
- [ ] 4.7 Add a unit tier for `scripts/docker_deploy.sh` mirroring the `deploy_remote.sh`
  shim harness: `--strict-health` exits non-zero on a dead endpoint, and the default run
  never invokes `docker exec`.
- [ ] 4.8 Cover `observe_step_timing.py:119` — the key-absent branch of `__getstate__`,
  reachable only by `del host._metrics`. One line.
- [ ] 4.9 Add property tests for `observe_step_latency`: any exception type propagates
  unchanged and records nothing; *n* blocks yield *n* finite non-negative samples.
- [ ] 4.10 Add a property test for the sidecar validator asserting its **post-condition**
  (`Path(name).name == name`, no leading dot, no collision) rather than its three
  enumerated rules — the post-condition form catches the rule the validator lacks.
- [ ] 4.11 Add a hermetic e2e: `docker compose -f docker-compose.jetson.yml config
  --quiet` behind a `shutil.which("docker")` skip. Today only the string shape of the
  `mousedroid_tensorrt_cache` interpolation is asserted, never that it resolves.
- [ ] 4.12 Add `tests/unit/scripts/test_deploy_remote_guard.py` to
  `scripts/validations/F-051.sh`, which currently lists only the two regression files —
  the feature's declared evidence chain excludes its only behavioural test.
- [ ] 4.13 Make `test_no_payload_ever_executed`'s canary `tmp_path`-scoped. The fixed
  global path makes it order-dependent and wrong under `pytest-xdist`.
- [ ] 4.14 Add a unit test for `resolve_config`'s fail-closed branch
  (`scripts/analyze_observe_step_ceiling.py:740-747`): no test passes `--config` a
  nonexistent path, and that guard is what stops the gate computing a valid-looking
  ceiling for the wrong MCTS budget.

## Phase 5 — Config correctness in the promotion gate

- [ ] 5.1 `scripts/docker_deploy.sh:150` — resolve the overlay through the repository's own
  resolver. It reads `MOUSEDROID_CONFIG` only; `_CONFIG_SINGLE_ENV_VARS` at
  `src/mousedroid/validation/runtime/_shared.py:25` honours `MOUSEDROID_JETSON_CONFIG` too,
  so a rover on the legacy key has the strict gate evaluate a different config than the
  rover runs — and **pass**.
- [ ] 5.2 Prove 5.1 with a negative test: set `MOUSEDROID_JETSON_CONFIG` to an overlay
  whose `world_model.engine` differs and assert the probe reports *that* engine. It fails
  today.
- [ ] 5.3 `scripts/docker_deploy.sh:54-55` — read `cfg.telemetry.port` and derive the path
  from `cfg.telemetry.api_prefix` from the `load_settings` call already made at `:151`.
  `MOUSEDROID_TELEMETRY_PORT` is not a pydantic-settings key: `root.py:189-191` sets
  `env_nested_delimiter="__"`, so the real key is `MOUSEDROID_TELEMETRY__PORT` and an
  operator moving the port the supported way leaves the probe on literal `8080`. Keep
  `MOUSEDROID_HEALTH_PORT`/`_PATH` as explicit overrides.
- [ ] 5.4 Prove 5.3 with a negative test asserting the probed URL carries
  `MOUSEDROID_TELEMETRY__PORT`.
- [ ] 5.5 Document the 13 env keys these scripts read that are absent from
  `config/docker.env.example` — the input to the `host_env_keys` preflight drift check
  (`src/mousedroid/validation/preflight.py:440`), so none is covered today. Follow the
  commented-entry precedent this branch set at `config/docker.env.example:121`.
- [ ] 5.6 Validate `MOUSEDROID_JETSON__TENSORRT_CACHE_DIR` before compose interpolates it
  into the container-side mount path (`docker-compose.jetson.yml:130`): a value of `/etc`
  shadows an image directory, and one containing `:` injects a third mount field.
- [ ] 5.7 `scripts/deploy_remote.sh:395,430` — derive the venv path from
  `REMOTE_SRC`/`MOUSEDROID_INSTALL_DIR` instead of the hardcoded `/opt/mousedroid/venv`
  (two copies). An install-dir override currently leaves the probe on the old tree and
  `pip_reinstall` silently falls through to a full `deploy_jetson.sh`.

## Phase 6 — Documentation, and the alert that does not exist

- [ ] 6.1 Add a `ModelArtifactDigestMismatch` rule to `config/prometheus/alerts.yml` for
  `mousedroid_model_artifact_sha256_mismatches_total`, plus a runbook paragraph.
  `_registry_replay_vla.py:195` states "Operator alert rules should page on any non-zero
  rate" and `grep` finds **zero** hits outside `src/` — no rule, no panel, no runbook.
  This is the F-050 defect class repeated inside F-050's own change. Not documentation:
  a digest mismatch means wrong weights, wrong inference, silently, and nothing pages.
- [ ] 6.2 Add a `### Added` block for F-050/F-051 under `CHANGELOG.md:9`. The file was not
  touched by this branch at all; 78 files are unrecorded. Include a forward reference to
  the PR #93 entry at `:4129-4151` whose "wired" claim F-050 contradicts — do not edit
  history.
- [ ] 6.3 Reword `src/mousedroid/world_model/CLAUDE.md:10-11`, which still states
  ONNX/TensorRT execution as present-tense fact. This is the same false claim the task-4.1
  narrative sweep corrected in ADR-008, the export script and the schema — this surface
  was missed.
- [ ] 6.4 `docs/architecture/c4-orchestrator.md:106-137` — add `_warm_world_model()` to the
  lifecycle sequence as the last step before the loop, noting it can abort startup; and add
  the `RSSM → metrics` relationship at `:30-34,47-56`.
- [ ] 6.5 `docs/architecture/ADR-008-world-model-onnx-engine.md` §Public surface — add the four new `WorldModelConfig` fields to
  the public-surface block, and an "Artifact integrity" subsection for the SHA-256 manifest
  gate the ADR never mentions.
- [ ] 6.6 `docs/architecture/c4-rssm-sim-pretraining.md:80` — `@torch.no_grad()` now sits on
  `_observe_step_impl` (`rssm.py:155`), not `observe_step`. The invariant holds; the pointer
  does not.
- [ ] 6.7 Add `docs/architecture/c4-world-model.md` and
  `docs/architecture/c4-pc-to-jetson-delivery.md`, and their rows in
  `c4-overview.md:116-128`. Two subsystem seams have no component diagram; no architecture
  doc mentions the delivery scripts at all.
- [ ] 6.8 `README.md:279-291` and `scripts/README.md:7` — add `deploy_remote.sh`,
  `rover_wip_guard.sh`, `analyze_observe_step_ceiling.py` and the two new runbooks.
- [ ] 6.9 Root `NEXT_STEPS.md` — add the four rover-gated open items (7.3, 7.4, 7.5, 8.7)
  under "Open engineering follow-ups", tagged F-008-sequenced, and index the three missing
  runbooks at `:196-204`.
- [ ] 6.10 Rename the two overpromising `features.yaml` entries: F-050 `:1422` says "ONNX
  provider proof" where provider observation is recorded NOT BUILT; F-051 `:1465` says
  "offline rollback drill" where the drill is recorded not run. A reader of
  `scripts/select_next.py` sees the name, not the caveat.
- [ ] 6.11 Move F-050 from `epic: "Jetson deployment"` to the existing `World model` epic.
- [ ] 6.12 `docs/architecture/adr-log.md:16` — annotate ADR-008 "amended 2026-09-19"; the
  branch reversed its multi-step cross-engine parity decision.
- [x] 6.13 **[LANDED]** `docs/analysis/positioning-safety-peer-review-2026-09-19.md` — D-7,
  P7 and the corrected-design map all still read as live. Done while merging the base:
  the drafted wording ("partly addressed by F-051 task 8.1, leaving the
  `_model_fingerprint` half open") was **wrong by the time it was written** — base PR #234
  landed `_runtime_identity()` and `cache_dir_is_private` and closed that half, while
  annotating D-25 and D-26 but not D-7. Both halves are now closed and the row says so.
- [x] 6.13a **[LANDED]** Six sites said the cache directory is "bind-mounted from the host"
  — false on the merged tree, since F-051 replaced that with the named volume
  `mousedroid_tensorrt_cache`. Two are operator-facing (`JetsonConfig.tensorrt_cache_dir`'s
  `description=`, `cache_dir_is_private`'s docstring). A semantic merge conflict with **zero
  file overlap**: `git merge` had nothing to flag, because the base's new prose and this
  branch's compose change are only inconsistent together.
- [ ] 6.13b Verify the interaction the merge created rather than assuming it: Docker creates
  a named volume root-owned `0755`, and `cache_dir_is_private` treats group/other-reachable
  as a MISS. Reading `_save_sync` shows it does `mkdir(mode=0700)` **plus** an explicit
  `os.chmod`, which hardens the mount point, so there is no permanent-miss loop — but the
  chmod is wrapped in `except OSError` for the not-our-directory case, so confirm on the
  rover that the container user can chmod the volume root. Not a claimed defect; a claimed
  unknown.
- [ ] 6.14 Append the four omitted artifacts to the `openspec/project.md:24` cell
  (`src/mousedroid/utils/artifact_integrity.py`, `src/mousedroid/world_model/onnx_export_metadata.py`,
  `src/mousedroid/world_model/composite.py`, `scripts/rover_wip_guard.sh`) and the two new runbooks.

## Phase 7 — Advisory ladder and supply chain

- [ ] 7.1 Promote `test-windows` to blocking: drop `continue-on-error`, remove its
  `.github/advisory_stages.yaml` entry. Window closes 2026-09-20 and the evidence is in
  hand — it caught four real defects in one change.
- [ ] 7.2 Teach `scripts/check_advisory_promotions.py` to count consecutive green runs per
  job, or change `onnx-world-model-extras`/`mlflow-extras` to a bar the checker can
  evaluate. Both currently defer to a 7-green-run rule nothing measures, and one was
  already re-extended *because* the streak could not be re-derived.
- [ ] 7.3 Add a `package-ecosystem: "docker"` entry to `.github/dependabot.yml` per
  Dockerfile directory, same `weekly` / limit-5 shape as the existing two.
- [ ] 7.4 Digest-pin the four base images (`dustynv/llama_cpp:r36.4.0`,
  `dustynv/l4t-pytorch:r36.4.0`, `python:3.11-slim`,
  `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`) and record the resolved digest in
  `deployments/jetson-image.json` beside `model_sha256`. The two rover bases are
  third-party community images on mutable tags.
- [ ] 7.5 `Dockerfile.jetson` — collapse the duplicated `r36.4.0` into an `ARG L4T_TAG`,
  the Python minor version at `:161-162` into an `ARG PY_VER`, and `LABEL version` into a
  build `ARG` from `pyproject.toml`. Compose already passes build args.
- [ ] 7.6 Remove the dependency version floors from `Dockerfile.jetson:59-178` that
  duplicate `pyproject.toml` extras — the file states that rule at `:79` and violates it at
  every other stage.

## Phase 8 — Validation and closeout

- [ ] 8.1 `make gates` passes.
- [ ] 8.2 `make test` passes (all four pytest steps).
- [ ] 8.3 `python -m tools.ratchet_budgets --strict` exits 0; the three budgets read
  19/19, 8/8, 21/21.
- [ ] 8.4 `python scripts/validate.py --tier fast` passes. Note: `implemented_in`
  warnings in a shallow clone are an environment artifact, not a finding — see
  `peer-review.md` §Disproved.
- [ ] 8.5 `bash scripts/ci.sh` passes.
- [ ] 8.6 Write `scripts/validations/F-052.sh` and register F-052 in `features.yaml` with
  `status: in_progress`, `implemented_in: null`, `depends_on: ["F-050", "F-051"]`.
- [ ] 8.7 Add the F-052 regression pair: `tests/regression/test_f052_aqa.py` +
  `test_f052_backwards_compat.py`.
- [ ] 8.8 Register the change in the `openspec/project.md` table; flip to `implemented`
  with the trunk SHA only after squash-merge, per the same rule F-050/F-051 follow.
