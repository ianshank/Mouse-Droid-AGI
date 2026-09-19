# Design: `mouse-droid-branch-hygiene-sweep` (F-052)

Each section is a decision with its rejected alternatives. Nothing here is built yet;
`tasks.md` is the execution order and `peer-review.md` is the adversarial record.

**Governing constraint on every section below.** All three suppression budgets sit at
ceiling (`noqa` 19/19, `type_ignore` 8/8, `hardcoded_ok` 26/26). A fix that needs a new
suppression cannot be written until headroom exists. **D-5 therefore runs first** — it is
the only section that creates capacity, and it creates it by deletion, not by raising a
ceiling.

---

## D-1 — Branch coverage runs in GitHub CI, and the exemption list stops lying

**Decision.** Add a `Changed-lines branch coverage` step to the existing `local-gates`
job, `if: github.event_name == 'pull_request'`, exactly mirroring how the
`Hardcoded-value gate` step was graduated there (`ci.yml:434-441`). Then remove
`src/mousedroid/orchestrator/_lifecycle_mixin.py` and
`src/mousedroid/factory/orchestrator.py` from `check_branch_coverage.py::_ALLOWED_FILES`.

**Why the stated rationale is spent.** `ci.yml:380-383` calls this gate "Still local-only
by design (need heavy deps)". `local-gates` installs `.[dev,telemetry,mcp]` and the `test`
job runs the whole suite — the deps are present in CI and have been since that comment was
written. The comment two lines above it states the job's own purpose: "These gates
previously ran ONLY in `bash scripts/ci.sh` — a GitHub-CI-only contributor bypassed all of
them." Branch coverage is the one that was left behind.

**Why the two removals, specifically.** The exemption is granted to "ADR-017 factory.py
split products that are pure DI wiring, plus the orchestrator mixin/_state split
products", and `check_branch_coverage.py:66-68` is explicit that "Algorithmic factory
modules stay off this set so the changed-line gate still applies to them."
`_warm_world_model()` has an `isinstance` guard, an `asyncio.to_thread` hop, timing, and a
failure-propagation path (`_lifecycle_mixin.py:97-160`). It is algorithmic. Leaving it
exempt does not merely miss coverage — it makes the *stated criterion* false, which is the
worse defect.

**Cost, measured not guessed.** `_lifecycle_mixin.py` currently reports 77% line coverage
with the new lines 126-145 fully covered; the 63 misses are pre-existing. Removing the
exemption gates *changed lines only*, so today's diff would pass. The cost is borne by
future edits to those files — which is the point.

**Rejected:** a new CI job (needless; `local-gates` is exactly this job's remit).
**Rejected:** keeping both exempt and adding a comment (a comment is not a gate; that is
the failure mode this whole change exists to stop).

**Blast radius — and a correction found by verifying this section's own claim.**
The first draft of this design said "remove the two files from `_ALLOWED_FILES`" and
listed three tests to update. Checking that against the tree showed it would fight a
*categorical* invariant rather than a list:
`tests/regression/test_f042_aqa.py:68` asserts `on_disk == exempt` for the orchestrator —
**every** `_*.py` mixin is exempt, by category, with no escape hatch.

The factory half of the same test file already has the escape hatch this needs.
`_GATED_FACTORY_FILES` (`:14-20`) enumerates three factory modules that stay *gated*, and
the assertion is `on_disk == exempt | _GATED_FACTORY_FILES` with the two sets asserted
disjoint (`:49-50`). So a factory module can be classified either way; an orchestrator
mixin cannot.

The revised decision is therefore to **mirror the factory mechanism onto the orchestrator
side** rather than to poke a hole in a byte-for-byte list: add `_GATED_ORCHESTRATOR_FILES`
with `_lifecycle_mixin.py` as its first member, change `:68` to
`on_disk == exempt | _GATED_ORCHESTRATOR_FILES`, and add the disjointness and
`not _is_exempted_from_branch_gate` assertions the factory half already carries.
`src/mousedroid/factory/orchestrator.py` needs no new mechanism — it moves into the existing
`_GATED_FACTORY_FILES`.

That is a better change than the original: the categorical rule "all orchestrator mixins
are pure split products" was true when F-042 wrote it and stopped being true the moment one
gained algorithmic logic. Replacing the category with a reviewed classification is what
F-042 already decided to do for the sibling directory.

Tests to update: `tests/regression/test_f042_aqa.py` (the new gated set and assertion),
`tests/regression/test_f042_backwards_compat.py` and
`tests/unit/scripts/test_check_branch_coverage_base_ref.py:455` (both pin
`_ALLOWED_FILES` / `_ALLOWED_DIR_PREFIXES` byte-for-byte).

## D-2 — Shell hygiene: `bash -n` blocking now, `shellcheck` advisory with a window

**Decision.** Two separate gates with two separate risk profiles.

1. **`bash -n` over every tracked `.sh`, blocking, immediately.** Measured: 58 tracked
   scripts, 9,036 lines, **all parse clean today**. A gate that is already green costs
   nothing to make blocking, and the repository already has the convention in two places
   (`tests/regression/test_host_bootstrap_script.py:5`,
   `test_jetson_full_validation_script.py:40-50`) — this generalises it from two files to
   the tree. Implemented as a regression test with `tests/_bash.py::requires_bash` so it
   skips correctly on `test-windows`, not as a workflow step.
2. **`shellcheck`, advisory, with a tracked promotion window.** Unknown finding count on
   9,036 pre-existing lines. Landing it blocking would either produce a red wall or force
   a suppression spree against three budgets that are already at ceiling. So it lands
   `continue-on-error: true` with an entry in `.github/advisory_stages.yaml` carrying a
   `since`, a `promote_after_days`, and a reason — which is precisely the ladder this
   repository built for exactly this situation (five jobs currently on it).

**Rejected:** `shellcheck` blocking on changed files only. Tempting, and it is how the
hardcoded-value gate works, but shell findings are frequently whole-file (quoting
patterns, `set -e` interaction) and a changed-lines view of them misleads. Advisory over
the whole tree tells the truth; the ladder then makes it blocking on evidence.

**Rejected:** skipping `shellcheck` because the scripts have unit tests.
`tests/unit/scripts/test_deploy_remote_guard.py` is genuinely behavioural (it shims
`ssh`/`rsync`/`scp` against a throwaway git repo) — and `docker_deploy.sh`, 506 lines,
has no unit-tier test at all. Tests prove the paths someone thought of. `shellcheck`
finds the quoting bug nobody thought of, which is the class this branch already shipped
once.

## D-3 — A wired hook that no document mentions must fail a test

**Decision.** Add `test_every_wired_hook_is_documented_in_the_runbook()` to
`tests/regression/test_claude_workforce_aqa.py`, parsing hook module names out of
`.claude/settings.json` and requiring each to appear in
`docs/runbooks/claude-workforce-hooks.md`. Then document
`tools.claude_hooks.ratchet_budget_check` in that runbook's table.

**Why this is the finding and not the symptom.** The undocumented hook is a symptom. The
mechanism is an asymmetry: that file already pins skills→`SKILLS.md` (`:290`) and
agents→`AGENTS.md` (`:315`), and separately pins that wired hooks reference *existing
modules* (`:386`) and that the expected modules are *present* (`:789`). Two of three asset
classes have a documentation-inventory pin. Hooks have an existence pin and no
documentation pin, so a hook can be wired, running, and invisible — which is exactly what
happened. Fixing the doc without fixing the asymmetry guarantees a repeat.

**Direction matters.** The test asserts wired ⟶ documented. It deliberately does not
assert documented ⟶ wired: the runbook legitimately discusses hooks that are available but
not enabled, and a bidirectional test would forbid that.

## D-4 — `docker_deploy.sh` resolves config the way the repository resolves config

**Decision.** Two HIGH defects, one root cause: the strict promotion probe re-implements
resolution instead of calling the repository's resolver.

- **C-1.** `docker_deploy.sh:150` reads `MOUSEDROID_CONFIG` only. The repository's
  resolver honours two keys — `_CONFIG_SINGLE_ENV_VARS = ("MOUSEDROID_CONFIG",
  "MOUSEDROID_JETSON_CONFIG")` (`src/mousedroid/validation/runtime/_shared.py:25`). A
  rover configured via the legacy key has the strict gate evaluate a *different config
  than the rover runs* — wrong `world_model.engine`, wrong artifact path — and **pass**.
  A promotion gate that passes for the wrong reason is worse than no gate, because it is
  cited as evidence. Fix: call the repository's own resolution helper.
- **C-2.** `docker_deploy.sh:54` falls back through `MOUSEDROID_TELEMETRY_PORT`, which is
  not a pydantic-settings key: `root.py:189-191` sets `env_prefix="MOUSEDROID_"` with
  `env_nested_delimiter="__"`, making the real key `MOUSEDROID_TELEMETRY__PORT`. An
  operator who moves the port the supported way leaves the health probe on literal `8080`.
  Fix: read `cfg.telemetry.port` and derive the path from `cfg.telemetry.api_prefix` from
  the `load_settings` call the probe already makes at `:151`. `HEALTH_PORT` /
  `HEALTH_PATH` remain honoured as explicit operator overrides — they just stop being the
  only way to be right.

**Each fix is proven by a negative test, not by reading the diff.** C-1: set
`MOUSEDROID_JETSON_CONFIG` to an overlay whose `world_model.engine` differs and assert the
probe reports *that* engine. C-2: set `MOUSEDROID_TELEMETRY__PORT` and assert the probed
URL carries it. Both would fail today. This is the lesson of the composite-warmup defect
applied before the fact: assert against what runs.

**Coupled to D-6/16.** The probe body (`STRICT_PROBE_PY`, `:139-230`) is ~70 lines of
Python with six `problems.append` branches and **zero** test references. Fixing C-1/C-2
without making that body executable in a test would mean shipping the fix the same way the
defect shipped.

## D-5 — Reclaim ratchet headroom by deletion; ratchet the ceiling to match

**Decision.** Remove five `hardcoded_ok` markers, taking the measured count 26 → 21, and
**ratchet the ceiling to 21 rather than banking the slack.**

- Three suppress nothing at all — their values are already in the gate's
  `ALLOWED_NUMERIC_VALUES = {0.0, 1.0, -1.0}`: `src/mousedroid/comms/_utils.py:23` (`= 1`),
  `:26` (`= 0`), `src/mousedroid/comms/command_set.py:71` (`= 1`). Deleting the marker is
  semantically inert; the gate never flagged those lines.
- Two duplicate an existing constant, which is *verbatim* the class the 28→26 ratchet
  already resolved (`.claude/workforce.yaml:123-128`):
  `src/mousedroid/validation/latency_stats.py:30` and
  `src/mousedroid/comms/command_set.py:68` both spell `1000.0` where
  `src/mousedroid/constants.py:67` `MILLISECONDS_PER_SECOND` exists. Import it.

**Ratchet, do not bank.** `.claude/workforce.yaml:100-131` documents this budget's own
discipline as down-only-except-with-a-recorded-reason, and records both a bump and a
ratchet-down as precedent. Banking 5 slots would hand future changes a silent suppression
allowance that no reviewer authorised. If a later fix genuinely needs a marker it can bump
the ceiling *with a reason*, which is the discipline working.

**Consequence, stated plainly.** After this section, the budgets are again at ceiling and
this change has *no* suppression allowance. Every later section must be written without
one. That is intentional: it is the constraint that produced the
`TYPE_CHECKING`-only `_StateHookBase` shape in `observe_step_timing.py:41-70`, which is
better code than the `type: ignore` it replaced.

**Also fixed here:** `tools/claude_hooks/config.py:353` carries a stale `hardcoded_ok`
fallback (`ceiling=24, warn_threshold=22`) against `workforce.yaml`'s `26`/`24`, while the
`noqa` and `type_ignore` fallbacks are in sync. The docstring at `:339-341` claims the
fallbacks reproduce the real budgets "exactly"; that is false, and a run with
`workforce.yaml` unreadable reports a phantom breach. Sync it to the post-ratchet values
and add a test that the fallbacks equal the shipped config, so the docstring's claim is
enforced rather than asserted.

## D-6 — Test-tier closures, ordered by what would actually have caught something

Not "add tests to every tier for every module" — that is volume, not coverage. Ordered by
defect-catching power, with each item's absence stated as a consequence:

1. **`STRICT_PROBE_PY` has no test at any tier.** ~70 lines, six distinct failure
   branches (ORT not importable, artifact missing, session construction failure, provider
   downgrade, deploy record unreadable, record pins no `model_sha256`). `grep -rn
   STRICT_PROBE_PY tests/` → nothing. This is the largest untested block the branch adds,
   and it is the promotion gate. Extract the heredoc body and drive all six branches.
2. **The factory→composite→Warmable chain is proved only in an advisory job.** The one
   test asserting `build_world_model(engine="onnx_trt")` returns a `CompositeWorldModel`
   is `tests/unit/factory/test_factory_world_model_engine.py:132-134`, gated behind
   `pytest.importorskip("onnxruntime")` at `:22-24` — and the only job installing that
   extra is `onnx-world-model-extras`, which is `continue-on-error: true`
   (`ci.yml:579-583`). Meanwhile `_warm_world_model()` *does* run at integration and e2e
   (`tests/integration/test_sense_plan_act.py:30`, `tests/e2e/test_full_pipeline.py:36`)
   but always takes the non-Warmable early exit, because those configs build a plain
   `RSSM`. **So the exact chain whose break was found empirically is, today, still not
   proved by any blocking gate.** Close it with an always-on integration test that
   monkeypatches the ONNX engine constructor with a torch-free Warmable fake, so
   `build_world_model` returns a real `CompositeWorldModel`, and asserts it is warmed off
   the main thread.
3. **`_validate_metadata_sidecar_filename` is deletable without turning anything red.**
   Three raise branches (`src/mousedroid/config/schema/world_model.py:174-190`) and only positive
   constructions in the suite. A 4-case `pytest.raises` unit test plus one property test
   asserting the *post-condition* (`Path(name).name == name`, no leading dot, no collision
   with `onnx_filename`) rather than the three enumerated rules — the post-condition form
   catches the rule the validator does not have.
4. **My own structural pin is self-referential.** `tests/unit/test_bash_guard.py:178-201`
   reads `Path(__file__)` — its own file, only its own file — and scans it for a pattern
   the same method body defines inline. It cannot fail unless someone edits that one
   module, and it proves nothing about the six other modules that inline the same
   `os.name` predicate (recorded in `tests/_bash.py:25-28`). Widen the scan to
   `tests/**/test_*.py` and hoist the search literals to module constants so the scanner
   is not searching for its own body.
5. **The registry-before-engine ordering is pinned as a source `str.index` comparison**
   (`tests/regression/test_f050_aqa.py:142-156`), not as an object graph. A reorder that
   preserved both string positions while breaking the wiring would pass. Add one
   integration assertion on the graph the factory actually returns.
6. **Five new string defaults are pinned only as truthy** (see C-4). Add exact-value
   assertions to `tests/regression/test_f050_backwards_compat.py`, and give
   `CognitiveConfig`'s three new fields the backwards-compat half they currently lack
   entirely.
7. **`docker_deploy.sh` has no behavioural tier** while its sibling got a full shim
   harness. Mirror the `deploy_remote.sh` pattern.
8. **One uncovered new line:** `observe_step_timing.py:119`, the key-absent branch of
   `__getstate__` — reachable only by `del host._metrics`, which no test does. One line
   closes it.
9. **Two property gaps:** `observe_step_latency` (any exception type propagates unchanged
   and records nothing; *n* blocks yield *n* finite non-negative samples) and the sidecar
   validator (item 3). Amdahl arithmetic got fuzzed; the timing core and a path-safety
   validator did not.
10. **No e2e touches the delivery path, not even a dry run.** `docker compose -f
    docker-compose.jetson.yml config --quiet` behind a `shutil.which("docker")` skip is
    cheap and hermetic, and would catch the `mousedroid_tensorrt_cache` interpolation
    failing to resolve — today only its string shape is asserted.

**Explicitly not doing:** filling every tier for every module. Items 1–4 are where the
evidence says defects live.

## D-7 — Extract skills from actions this branch proved, not from actions we imagine

**Decision.** Three skills, each earned by a procedure that was executed more than once
during the F-050/F-051 work and got re-derived each time. `grep` across all 24 existing
`SKILL.md` files for `ceiling|onnxruntime|deploy_remote|docker_deploy|warmup|Amdahl`
returns zero matches, so none of this is covered today.

1. **`ceiling-gate`** — run `analyze_observe_step_ceiling.py`, read the Amdahl verdict
   against the three-part rubric at `docs/analysis/alayaworld-distillation-spike.md:65-75`,
   and record a GO/DEFER decision in the change bundle. Executed once and it stood down
   two entire phases; the next optimisation proposal needs the same procedure and would
   otherwise re-derive it.
2. **`ort-provider-proof`** — verify an ONNX Runtime provider claim by *constructing the
   session* and reading `session.get_providers()`, never by checking availability, and
   place the reading after every ORT-affecting install. This is a corrected mistake, which
   is the best possible reason for a skill: the first version of the probe proved
   availability, and a later `piper-tts` install could silently downgrade the provider
   after the reading was taken.
3. **`remote-path-safety`** — the review checklist for making any path env-overridable in
   a script that shells out: validate, then quote with `printf '%q'` at *every* boundary.
   This branch introduced a command injection by making one path overridable, so the
   procedure exists as scar tissue and belongs written down.

**Rejected:** a skill for the warmup wiring. It is a one-time architectural change, not a
repeatable procedure. Skills are for procedures; `CLAUDE.md` and ADRs are for
architecture.

**Wiring is mandatory, not optional.** `tests/regression/test_claude_workforce_aqa.py:290`
already requires every skill directory to appear in `SKILLS.md`, and
`tools/validate_skill_commands.py` validates every backticked path. A skill that is not
indexed fails CI — correctly.

## D-8 — Hook events: add the two that are session-shaped, skip the rest

**Decision.** The repository uses `PreToolUse` and `PostToolUse` only. Add exactly two:

1. **`SessionStart`** — report the state a session needs and currently has to ask for:
   the three ratchet counts against their ceilings, which advisory windows are due, and
   whether the working tree is clean. Every one of those was manually re-derived during
   this branch's work. Report-only; never blocking.
2. **`PreCompact`** — persist the change-bundle task state and the verified-findings list
   before context is summarised. This change's own predecessor lost the distinction
   between "verified" and "asserted" across a compaction boundary, which is how
   `design.md` came to mark D-1 built when half of it was not.

**Rejected: `Stop`.** The hosting harness already enforces commit-and-push at stop in this
environment; a repo-level `Stop` hook would duplicate it and fight it.
**Rejected: `UserPromptSubmit`.** Nothing to validate on a prompt; it would be latency for
no signal.
**Rejected: `SubagentStop`.** Attractive — this change used four subagents — but
`tools/claude_hooks/` has no way to know a subagent's mandate, so the hook could only echo.

**Every new hook is a `tools/claude_hooks/` module with its own tests** under the
`coverage.tools_line_min: 85` gate (`.claude/workforce.yaml:68`), and a runbook entry,
which D-3's new test will then require.

## D-9 — Documentation: close the alert gap first, then the drift

**Decision.** One finding here is not documentation at all and is promoted out of this
section: `src/mousedroid/telemetry/metrics/_registry_replay_vla.py:195` states "Operator
alert rules should page on any non-zero rate" for
`mousedroid_model_artifact_sha256_mismatches_total`, and
`grep -rn model_artifact_sha256_mismatches config/prometheus/ docs/` returns **zero** hits
— no alert rule, no dashboard panel, no runbook line. A digest mismatch means "wrong
weights, wrong inference, silently" (ADR-008's own words) and nothing pages. **This is the
F-050 defect class repeated inside F-050's own change:** that change existed partly to give
`WorldModelObserveStepLatencyHigh` a writer, and shipped a new counter with no rule. Add
the rule and the runbook paragraph; `prometheus-check` then validates it.

The documentation drift proper, in one pass:

- **`CHANGELOG.md` was not touched by this branch at all** — the entire 78-file change is
  unrecorded. One `### Added` block under `[Unreleased]`.
- **`src/mousedroid/world_model/CLAUDE.md:10-11`** still states ONNX/TensorRT execution as
  present-tense fact. This is the *same false claim* the task-4.1 narrative sweep corrected
  in ADR-008, the export script and the schema — this surface was missed by the sweep.
  Reword to "available, gated on `model.cfc_hidden_dim > 0` plus a locally exported
  artifact; the shipped default is `engine: torch` with plain `RSSM`."
- **`docs/architecture/c4-orchestrator.md:106-137`** — the lifecycle sequence omits
  `_warm_world_model()`, which is the last step before the loop and can abort startup.
- **`docs/architecture/ADR-008-world-model-onnx-engine.md` §Public surface** — the public-surface block omits all four new
  `WorldModelConfig` fields, and the ADR never mentions the SHA-256 manifest gate that
  `src/mousedroid/factory/world_model.py` enforces before session build.
- **`docs/architecture/c4-rssm-sim-pretraining.md:80`** — points `@torch.no_grad()` at
  `observe_step`; it now sits on `_observe_step_impl` (`rssm.py:155`) behind the timing
  wrapper. The invariant holds; the pointer does not.
- **`docs/architecture/c4-overview.md:116-128`** — no row for the world-model/ONNX engine
  and none for the PC→rover delivery path, so two subsystem seams have no component
  diagram. Two new C4 files.
- **`README.md:279-291`, `scripts/README.md:7`, root `NEXT_STEPS.md`** — omit the new
  delivery scripts, the two new runbooks, and the four rover-gated open items (7.3–7.5,
  8.7).
- **`features.yaml:1422`/`:1465`** — both `name` fields overpromise against their own
  `verification` lists: F-050 says "ONNX provider proof" where the provider-observation
  work is recorded NOT BUILT; F-051 says "offline rollback drill" where the drill is
  recorded not run. A reader of `scripts/select_next.py` output sees the name, not the
  caveat. Rename to match what shipped.

**Rejected:** editing the historical `CHANGELOG.md` entry at `:4129-4151` whose "wired"
claim F-050 contradicts. History is a record. A forward reference in the new entry is
honest; a retroactive edit is not.

## D-10 — Advisory ladder and supply chain

**Decision, three parts.**

1. **`test-windows`: promote to blocking.** `since: 2026-08-20`, `promote_after_days: 30`,
   so the window closes tomorrow (`check_advisory_promotions.py` reports "within window"
   today only because its comparison is strict `>`). The evidence the window existed to
   collect is in hand: this branch drove four consecutive rounds of Windows-only failures
   to green — a `which("bash")` guard that fails open on the WSL shim, a host-separator
   assumption, guard tests asserting the positive branch against the real host, and a
   `Path()` built under a patched `os.name`. A gate that caught four real defects in one
   change has earned blocking status.
2. **Replace the calendar proxy for `onnx-world-model-extras` and `mlflow-extras`.** Both
   defer to a "7-consecutive-green-run" bar that no tooling counts, and the
   `onnx-world-model-extras` entry records being re-extended *because* the streak could
   not be cheaply re-derived. Teach `check_advisory_promotions.py` to count consecutive
   green runs per job from the Actions API, or change the bar to one the checker can
   evaluate. A window whose criterion nothing measures will be re-extended forever.
3. **Add the `docker` ecosystem to `.github/dependabot.yml`.** It covers
   `github-actions` and `pip` only, while `Dockerfile.jetson` and `Dockerfile.dev` pin
   base images by tag. This branch's whole provider probe exists *because* the base
   image's wheel content silently decides whether the rover runs accelerated or CPU-only —
   that is precisely the drift a `docker` entry watches. Same `weekly` /
   `open-pull-requests-limit: 5` shape as the existing two.

## D-11 — What this change deliberately leaves broken

Stated so no later reader mistakes silence for completion:

- **Tasks 7.3–7.5 and 8.7 stay open.** Real export, an HF token and the rover. Recorded in
  `NEXT_STEPS.md` by D-9, not closed.
- **The production world model still loads no trained weights.**
  `build_weight_update_loader` returns `None` unconditionally. Known debt; a trained
  `DualStream` checkpoint is the blocker, and it is not a hygiene item.
- **`build_cognitive_core`'s `fallback_to_mcts` `except Exception` still degrades a digest
  mismatch to MCTS.** A comment records it. Narrowing that except clause is a behavioural
  change to a safety path and belongs in its own change with its own review, not folded
  into a hygiene sweep.
- **Pre-existing large modules are not decomposed.** `src/mousedroid/sim/isaaclab/rover_env.py` (804),
  `src/mousedroid/learning/offline_rl.py` (755), `src/mousedroid/config/schema/hardware.py` (722) are untouched by this
  branch and by this change. No line-count gate is proposed for `src/` either: the
  repository's chosen instrument is `ruff C901` at complexity 15, every `src/` offender was
  decomposed to clear it with no baseline, and a line count would be a second, weaker
  instrument measuring something the first one already covers better.
