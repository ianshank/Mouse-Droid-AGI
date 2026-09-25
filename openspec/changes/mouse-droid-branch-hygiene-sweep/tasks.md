# Tasks: `mouse-droid-branch-hygiene-sweep` (F-052)

**Revision 3**, after an adversarial re-verification at `992da04` found six blocking defects in
revision 2 — including a diagnosis that was simply wrong, in a plan that had already merged. Ordering
within a slice is binding. Tasks marked **[LANDED]** were executed while authoring revision 1 and are
re-verified below; **one of them was false and is unmarked.**

## Revision 3 — what the re-verification overturned

### 1. BLOCKING: tasks 4.15/4.16 were wrong. It is slow, not hung — reproduced.

Revision 2 claimed a full-tree `pytest` sweep "blocks indefinitely in `ep_poll` … with system CPU at
0%, so it is hung, not slow." **Refuted by direct reproduction:**

```
python -m pytest tests/integration/test_e2e_5sec_run.py --timeout=40 --timeout-method=thread
-> mcts_planning budget=50 ... mcts_plan_complete n_simulations=50   (7-13 s per plan() call)
-> repeated, making progress the whole time
```

It is **CPU-bound in MCTS rollout** (`world_model/mcts.py::_rollout` -> `rssm.py::imagine_step` ->
`torch _VF.gru_cell`), not blocked on a selector. Three ways the original evidence was misread:

- `ep_poll` is what an asyncio selector shows when sampled *between* callbacks. It is the normal
  resting state of the loop while a tick body computes.
- My per-process CPU sampler had a bug: `ps -eo pid --no-headers | head -60` takes the first 60 PIDs
  in numeric order, and pytest was PID 17208 — **it was never sampled.**
- `tests/integration` is **step 1 of `make test`** (`Makefile::test-cov`) and CI's `test` job mirrors
  it, so this file runs in a blocking gate today. Revision 2's "no gate executes that selection" was
  wrong; the only tier a full sweep adds is `tests/performance`.

**Corrected task:** the integration tier is minutes-per-test on CPU-only hosts. Either mark these
`slow` or shrink the MCTS budget in the mock config. **Do not bisect a deadlock that does not exist.**
4.16 is deleted — it existed only to propagate 4.15's conclusion.

### 2. BLOCKING: task 6.13a was marked `[LANDED]` and is false; the same merge shipped a seventh site.

6.13a claimed six "bind-mounted from the host" sites corrected and "both halves are now closed."
**`CHANGELOG.md:19` still reads** *"`docker-compose.jetson.yml` bind-mounts the default location from
the host, so it outlives the image"* — in an operator-facing `### Security` entry, while
`docker-compose.jetson.yml:130` uses the named volume `mousedroid_tensorrt_cache`. Also open:
`src/mousedroid/efficiency/tensorrt.py:191` needs a contextual read to decide whether it is an eighth.

**This is the worst class of defect here**, because a `[LANDED]` marker stops a later pass from
checking. **6.13a is unmarked.** Its replacement is not a prose sweep but a regression test: no tracked
file may pair "bind-mount" with "tensorrt_cache".

### 3. BLOCKING: `proposal.md` A-4's evidence is false, and task 1.8 contradicts it.

A-4 said "the only two occurrences in the repo are `# shellcheck disable=SC2086` directives".
**Actual: 19 `shellcheck disable` directives across 10 files** — `scripts/prove_pin_fails.sh` alone has
9, and `scripts/rover_wip_guard.sh:124` carries the `SC2064` disable **task 1.8 cites by name.**

The true form is stronger: *shellcheck is invoked by no gate despite 19 inline, rule-specific
suppressions across 8 scripts that assume it runs.* It also means `peer-review.md`'s "nobody has run
it" is wrong, and §4's "any suppression declared inline and counted" starts from 19, not 0.

### 4. BLOCKING: task 2.2 is impossible as written.

`scripts/check_advisory_promotions.py::find_advisory_jobs` records only **job-level**
`continue-on-error: true`. A step-level one inside the blocking `lint` job is invisible, so an
`advisory_stages.yaml` entry immediately trips the checker's "stale metadata" branch — a contract
`tests/regression/test_ci_gate_wiring_aqa.py:305-311` pins verbatim. **shellcheck needs its own job**
with job-level `continue-on-error`, or no tracker entry.

### 5. BLOCKING: task 2.7 misses two surfaces.

Beyond the three pins it names: `tests/unit/scripts/test_check_branch_coverage_base_ref.py:487`
(`test_is_exempted_from_branch_gate_matches_prefix_precisely`) asserts **positively** at `:511` that
`_lifecycle_mixin.py` is exempt, and its docstring at `:495` explains why (ADR-017's measurement). And
`scripts/validations/F-042.sh` runs that whole file, so F-042's declared evidence chain goes red too.

**Also: 2.4-2.7 buy zero measured coverage now.** `tests/unit/orchestrator/test_world_model_warmup.py`
is 367 lines / 22 tests covering every branch of `_warm_world_model()`, and D-1 itself admits the gate
is diff-scoped and "today's diff would pass." The real benefit is future edits to a 683-line mixin —
and removing the exemption **reverses ADR-017 and F-042's recorded decision**, which needs an ADR
amendment the bundle tasks for a smaller change (6.12) and not for this one. State the honest benefit
or cut the slice.

### 6. BLOCKING: task 0.1 reverses the recorded F-030 decision.

The arithmetic is right (26 -> 23 -> 21, independently re-derived). The rationale is not.
`.claude/workforce.yaml:116-121` records the 24->28 bump as fixing *"inconsistent treatment of the same
kind of value"* — `ESP32_CMD_TYPE_*` constants were unmarked while identical vendor-protocol constants
carried markers. Task 0.1 strips `ESP32_CMD_TYPE_VELOCITY` (=1) and `_STOP` (=0) while
`_BATTERY` (=2) **must** keep its marker, because 2 is not in `ALLOWED_NUMERIC_VALUES`. That recreates
the exact inconsistency the bump removed, inside a phase arguing "the discipline working."

**Corrected:** take only the two `1000.0` markers (26 -> 24, ratchet to 24/22). Or delete all four
protocol markers and record F-030's note as superseded. Do not silently invert it.

### Non-blocking corrections, all landing in the same edit

| # | wrong | right |
|---|---|---|
| 7 | "`local-gates` installs those deps, so the comment is stale" | identical install to `test` (`ci.yml:399` vs `:222`). The real barrier is `timeout-minutes: 20` against the whole unit+property+integration selection |
| 8 | "all 23 CI jobs are green" | **17** jobs; 23 is check-runs after matrix expansion |
| 9 | "70 tasks across 9 phases" | **92** |
| 10 | task 7.2 would make the streak count accurate | enforcement is zero three times over: the step lives in the advisory `vulture-audit` job, has no `actions: read` permission, and runs without `--strict`. Fix those first or 7.2 is motion without effect |
| 11 | "`prometheus-check` then validates it" | `promtool check rules` validates YAML shape and PromQL only, and is skipped entirely if its install fails. Pair 6.1 with a test asserting every metric in `alerts.yml` appears in `registry.render_prometheus()` |
| 12 | "`test-windows` window closes 2026-09-20" | the 30-day window **elapsed 2026-09-19**; the checker's strict `>` defers the WARN to 2026-09-20 |
| 13 | every open `deploy_remote.sh` line number | stale by ~14 lines after 1.1-1.6 landed (`:316`->`:330`, `:211`->`:225`, `:395,430`->`:410,447`). **Cite by symbol**, per the repo's own convention |
| 14 | "`openspec/project.md:24`" in task 6.14 | `:24` is now this bundle's own row. Cite by change-id |
| 15 | four citations off by a few lines | `config.py:339-341`->`:335-337`; `workforce.yaml:68`->`:65`; "13 env keys"->**14**; `_lifecycle_mixin.py:97-160`->`:97-145` |
| 16 | `test_deploy_remote_guard.py` "607 lines"; "9,036 lines" of bash | **703**; **9,053** |

### What survived, and it is most of the bundle

Every row of `proposal.md` §6's "do not churn" table reproduces at `992da04`, independently re-run:
`mypy --strict` clean over 424 files, `ruff NPY` clean, exactly 3 `C901` offenders in `scripts/`, 58
`.sh` files all `bash -n` clean, 22 validation commands resolving, all 7 config fields carrying
`Field(default=…, description=…)`.

Every §2 finding except A-4 holds exactly: A-1 (no workflow invokes `check_branch_coverage.py`), A-6,
A-7, B-1 through B-4, C-1 through C-4, D-3. Tasks 1.1-1.6 **are** genuinely landed — only their line
numbers are stale. Phase 0's five markers and the `config.py` fallback mismatch are all confirmed.
D-1's `_GATED_ORCHESTRATOR_FILES` mechanism is sound; it is incompletely tasked, not wrong.

---

## Revision 2 — the shape problem, and the slices that fix it

Revision 1's own peer-review named the defect and did nothing about it: *"This plan has 70 tasks
across 9 phases. That is a programme, not a change."* It then grew to 92. A 92-task bundle does not
ship; it gets cherry-picked by whoever reads it next, in whatever order, with no record of what the
ordering was protecting.

So revision 2 keeps every task and regroups them into **seven slices that can each ship as one PR**,
with the dependency between them stated. Nothing is deleted. The task numbers are unchanged so
existing references still resolve.

### The slices, in dependency order

| slice | what | tasks | ships alone? |
|---|---|---|---|
| **A — Headroom** | reclaim 5 inert/duplicated `hardcoded_ok` markers (26 -> 21), ratchet the ceiling, sync the stale hook fallback | Phase 0 | yes, and **everything else depends on it** |
| **B — Gates that do not run** | `bash -n` blocking (already green, zero cost), branch coverage into `local-gates`, the `_GATED_ORCHESTRATOR_FILES` mechanism, `shellcheck` advisory, `pip-audit` over the missing extras | Phase 2 | yes |
| **C — Security still open** | `REMOTE_USER`/`HOST` validation, the `rover_wip_guard.sh` trap quoting, `docker.env` sourcing under sudo, `.dockerignore` secrets patterns, `.env.*` in both ignore files, the gitleaks depth walk | 1.7-1.16 | yes |
| **D — Config correctness** | the two HIGH `docker_deploy.sh` defects: the resolver that ignores `MOUSEDROID_JETSON_CONFIG`, and the telemetry port that is not a settings key | Phase 5 | yes |
| **E — The missing alert** | `ModelArtifactDigestMismatch` for a counter whose own docstring says operators should page on it | 6.1 | yes |
| **F — Inventory pins and hooks** | hooks->runbook pin, `SessionStart`, `PreCompact`, the three skills | Phase 3 | yes |
| **G — Test gaps** | `STRICT_PROBE_PY`'s six unexecuted branches, the always-on Warmable proof, the sidecar validator negatives, the self-scanning pin, the full-tree hang | Phase 4 | yes, but largest |
| **H — Docs and ladder** | `CHANGELOG`, C4, `test-windows` promotion, dependabot `docker`, digest pinning | Phases 6-7 | yes |

**A is the only hard dependency.** All three suppression budgets sit at ceiling, so any slice that
needs a new marker is blocked until A lands. B through H are mutually independent.

### If this must be cut to a third, keep A, C, D, E

That is: create the headroom, close the security items the last change left open, fix the promotion
gate that currently **passes for the wrong reason**, and add the alert rule for a digest mismatch that
nothing pages on.

What breaks if the rest never happens, stated plainly:

- **Without B**, the changed-lines branch-coverage gate keeps running in no workflow, and
  `_lifecycle_mixin.py` stays exempt as "pure DI wiring" while holding algorithmic code. New logic
  lands ungated. `shellcheck` keeps not running over ~9,000 lines of Bash.
- **Without F**, the next wired hook goes undocumented the same way `ratchet_budget_check` did, because
  the asymmetry that allowed it is still there.
- **Without G**, the promotion gate's 70-line probe body keeps zero test coverage, and the
  factory->composite->Warmable chain stays proved only inside an advisory job. That chain already broke
  once, silently.
- **Without H**, `test-windows` stays advisory past its window, and no base image is watched by
  dependabot.

None of those is an emergency. All four are the same shape: a check that exists and does not run.

---

## Phase 0 — Create ratchet headroom (blocks everything else)

All three suppression budgets are at ceiling. No later phase can add a suppression until
this one lands. It creates capacity by deletion, never by raising a ceiling.

- [~] 0.1 **DECLINED** (see correction 6). Delete the three inert `hardcoded_ok` markers whose values are already in the
  gate's `ALLOWED_NUMERIC_VALUES = {0.0, 1.0, -1.0}`:
  `src/mousedroid/comms/_utils.py:23` (`= 1`), `:26` (`= 0`),
  `src/mousedroid/comms/command_set.py:71` (`= 1`). Verify with
  `python -m tools.ratchet_budgets` that the count drops 26 → 23 and no gate finding
  appears.
- [x] 0.2 Replace the two `1000.0` duplications with `constants.MILLISECONDS_PER_SECOND`
  (`src/mousedroid/constants.py:67`) at `src/mousedroid/validation/latency_stats.py:30`
  and `src/mousedroid/comms/command_set.py:68`, deleting both markers. Count 23 → 21.
- [x] 0.3 Ratchet `.claude/workforce.yaml` `hardcoded_ok` to `ceiling: 21`,
  `warn_threshold: 19`, with a comment recording *why* (inert markers + an existing
  constant), matching the precedent at `:118-128`. Do not bank the slack.
- [x] 0.4 Sync the stale fallback at `tools/claude_hooks/config.py:353` (`ceiling=24,
  warn_threshold=22`) to the post-ratchet values, and add a test asserting every
  `RatchetBudgetItem` default equals the shipped `.claude/workforce.yaml` entry — the
  docstring at `:339-341` claims this and nothing enforced it.
- [x] 0.5 `python -m tools.ratchet_budgets --strict` exits 0 and
  `tests/regression/test_suppression_budget.py` passes.

### Slice A landed — what changed against the task wording

Correction 6 above is binding, so Phase 0 shipped the corrected shape, not the written one.
Ratchet is now `hardcoded_ok` **24/22** with a measured count of 24, at ceiling, strict green.

- **0.1 declined, not deferred.** Stripping `ESP32_CMD_TYPE_VELOCITY` (=1) and `_STOP` (=0)
  while `_BATTERY` (=2) keeps its marker is the inconsistency the recorded F-030 bump exists
  to remove. `MIN_HEARTBEAT_WINDOW_MS` (=1) is the same class. The alternative the correction
  allows — delete all four protocol markers and record F-030 as superseded — is a decision
  about vendor-protocol marker policy, not headroom arithmetic, so it is left to the user
  rather than taken silently inside a hygiene slice.
- **0.3 landed at 24/22, not the written 21/19.** 21/19 was the pre-correction arithmetic,
  which assumed 0.1's three deletions. With 0.1 declined the count is 24, so 21 would have
  been an immediate self-inflicted breach. The YAML comment records the reason and states
  explicitly why the ESP32 markers were left alone, so the next reader does not re-derive
  the reversed decision from the ceiling alone.
- **0.4 needed no source edit, and that is the finding.** The stale fallback already read
  `ceiling=24, warn_threshold=22` — which the 26 -> 24 ratchet made correct *by coincidence*.
  So the drift closed itself, and nothing would have caught the next one. The test the task
  asks for was added (`test_hook_fallback_budgets_match_workforce_yaml`) and **proven to
  fail**: reverting the fallback to 26/24 reds it with a message naming both sides; the
  measured count and both gates return green once aligned.
- **Headroom delivered is 0, by design.** Count 24 against ceiling 24 means later phases
  still cannot add a suppression. Phase 0's premise — "creates capacity by deletion" — is
  only satisfied by 0.1, which is declined. Any later phase needing a marker must either
  reclaim elsewhere or come back to the ESP32 policy question. This is stated rather than
  papered over by banking slack, which this budget's own discipline forbids.

Verification: `python -m tools.ratchet_budgets --strict` exits 0; a probe 25th marker makes
it exit 1 and reds `test_hardcoded_value_marker_budget.py` with `assert 25 <= 24`, proving the
new ceiling binds. `make gates` green (295 passed, hook coverage 98.27%). 639 unit tests across
`validation`, `comms` and `claude_hooks` plus 80 F-025/heartbeat tests pass — both edits
substitute an identical `1000.0`, so behaviour is unchanged by construction.

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
- [x] 1.10 `scripts/docker_deploy.sh:43-48` sources `/etc/mousedroid/docker.env` as shell
  code under `sudo`, and `:363-373` creates it with a default umask (0755/0644) — so any
  non-root write to the file is root code execution, and that file is the documented home
  of `MOUSEDROID_TELEMETRY_TOKEN` and `ANTHROPIC_API_KEY`. Replace `.` with a
  `KEY=VALUE`-only read loop and `chmod 600` at creation.
- [x] 1.11 `scripts/docker_deploy.sh` — validate `CONTAINER_NAME` and `DEPLOY_RECORD`
  (both come from that sourced file) against `^[A-Za-z0-9_.-]+$` and insert `--` before
  the container name at `:234`, `:256`, `:264`, `:276`, `:394`, `:411`, `:417`.
### Slice C landed — two corrections to the task wording

Both defects were confirmed on the current tree before anything changed, and both fixes are
pinned by `tests/unit/scripts/test_docker_deploy_env_loading.py` (22 tests), written to the
`test_deploy_remote_guard.py` contract: real fixtures driven through `bash`, specific messages
asserted, and every pin proven to fail before it was trusted.

**1.10 — the dot-source.** Replaced with `_load_env_file_as_data`, which follows systemd's
`EnvironmentFile` rules. The justification is stronger than "sourcing is risky": *every* other
consumer of that file already parses it — `mousedroid-docker.service:47`,
`mousedroid-trend.service:34`, and `docker-compose.jetson.yml:64` — so this script was the lone
one treating it as code, and the only one running as root. It was a correctness bug too: a value
containing `$` or a backtick was expanded here and taken literally by systemd, so the script and
the units it installs disagreed about the same file. Proven both ways — the old `set -a` +
dot-source creates a sentinel file from `PAYLOAD=$(touch ...)`; the parser does not, and yields
the literal text systemd would. A `;` comment, which systemd accepts, is a bash *syntax error*,
so the old dot-source aborted the whole file on one and silently dropped every key below it.

**1.10 — CORRECTION: the directory must NOT be tightened.** The task says `:363-373` creates the
file "with a default umask (0755/0644)" and asks for `chmod 600` at creation. The file fix landed
and is proven: without it the operator's token lands `0o644`, and the test reds with that exact
mode. But tightening the *directory* to 0700, which the 0755 half implies, would break the rover:
`scripts/mousedroid.service:23` drops to `User=jetson` and `:26` points `MOUSEDROID_CONFIG` at a
YAML inside that same directory. So the secret is confined to one file and that file is tightened;
the directory is deliberately left as-is, with the reason recorded at the `mkdir` and pinned by
`test_the_config_directory_is_deliberately_left_group_readable`, which fails if
`mousedroid.service` ever stops dropping privileges (at which point tightening becomes safe).
The 0600 mode is not a new policy: `scripts/host_bootstrap.sh:99,103` already applies exactly it
to exactly this file, for exactly this reason. Two scripts created the same credential file and
only one protected it.

**1.10 — CORRECTION: the severity claim is overstated, and the fix still stands.** The task says
"any non-root write to the file is root code execution". A non-root write is not actually reachable
on a default host: `/etc` is root-owned `0755` and the `mkdir -p` created `/etc/mousedroid` the
same way, so an unprivileged user could not write `docker.env` even while it was `0644`. The real,
demonstrated problems are the two that do not depend on that: the script and the systemd units
**disagreed about the meaning of the same file** (expansion here, literal there), and the
credentials were **world-readable at `0644`** — confirmed by test, not argued. Both are fixed. The
escalation framing is left corrected rather than repeated, because a fix that has to be oversold
is a fix nobody re-examines. Not addressed, and recorded as a follow-up rather than silently
skipped: the file can still set `PATH`, `IFS` or `LD_PRELOAD` for a root process that calls
`docker` by bare name. That is unchanged from before this slice, matches what systemd itself would
pass through, and needs the same "who can write this file" analysis to prioritise.

**Review round 1 — four defects in the first cut, all verified before fixing.** Copilot flagged
these on PR #245; each was reproduced against the tree first, and each fix was then proven by
reverting it and watching the new pin go red.

- **HIGH, and the worst of the four: the parser's warning echoed the offending line.** The line
  that fails to parse is exactly a mistyped `ANTHROPIC_API_KEY` or `MOUSEDROID_TELEMETRY_TOKEN`, so
  a fix whose purpose was to stop that file being dangerous had introduced a path that writes the
  credential to stderr and the journal — against this repo's own rule that secrets are
  presence-checked, never echoed. Now reports `<file>:<lineno>` and nothing else, with the line
  count including comments and blanks so the location is actionable.
- **HIGH: the `chmod` covered only the creation path.** Every rover seeded by the old script still
  had `0644` credentials, and nothing else revisits them — so the remediation would have left the
  entire existing fleet exposed while reading as applied. It now runs on every deploy, and
  re-running the deploy is what repairs the fleet.
- **MEDIUM: `KEY="v"   ` exported its quote characters.** The quote test ran before trailing blanks
  were stripped, so an anchored pattern could not match a line ending in a space. That is the exact
  script-vs-systemd disagreement this function exists to remove, for a line shape systemd accepts.
  Trim now precedes the quote strip; blanks inside the quotes still survive.
- **MEDIUM: the budget pin ignored `scope_glob`.** It decides which files are counted, so a YAML
  scope change would have passed the pin while the fallback measured a different set. Now compared.

One fixture bug of my own surfaced while verifying the second item: a fixture seeded from the real
`config/docker.env.example` sets `MOUSEDROID_CONFIG_DIR`, and the env file legitimately overrides
the process environment (as the old dot-source also did), so the script redirected to
`/etc/mousedroid` and never reached the step under test. The first "after: 644" reading was that,
not the fix failing. Recorded because the fixture's comment now explains it, and because a
verification run that fails for the wrong reason is indistinguishable from a broken fix.

**1.11 — CORRECTION: the specified regex does not close the hole.** The task asks for
`^[A-Za-z0-9_.-]+$`. `-` is a member of that class, so `-uroot` *matches* it and the guard would
have admitted the very value it exists to reject. Landed with Docker's own container-name rule,
`^[A-Za-z0-9][A-Za-z0-9_.-]*$`, which anchors the first character and makes a leading dash
unrepresentable. `test_the_weak_pattern_would_have_admitted_that_name` pins that reasoning so the
pattern is not "simplified" back. Because a leading dash is now impossible, the `--`
end-of-options insertion at seven call sites is **not needed and was not done** — which also
avoids betting the deployment path on `--` placement in `docker exec`, unverifiable without a
Docker daemon. `DEPLOY_RECORD` is a path, not an identifier, so the same regex would have been
wrong for it; it is validated as absolute instead, which catches the real failure (silent
resolution against the operator's cwd). With both guards neutered the three refusal tests go red
and `-uroot` is shown reaching a live `docker` call.

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
- [ ] 4.15 **The integration tier is minutes-per-test on a CPU-only host.** Revision 2 called this a
  deadlock; it is not — see Revision 3 §1. Reproduced: `tests/integration/test_e2e_5sec_run.py` logs
  `mcts_plan_complete n_simulations=50` every 7-13 seconds, making progress throughout, CPU-bound in
  `world_model/mcts.py::_rollout` -> `rssm.py::imagine_step` -> torch `_VF.gru_cell`. It runs in a
  **blocking** gate today (step 1 of `make test`), and passes in CI because CI runners are faster.
  Fix: mark the affected tests `slow`, or shrink the MCTS budget in the mock config so a 5-second
  simulated run does not need 50 simulations per tick. Do **not** bisect for a hang.
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
- [ ] 6.13a **UNMARKED — was falsely [LANDED].** A seventh site survives: `CHANGELOG.md:19` Six sites said the cache directory is "bind-mounted from the host"
  — false on the merged tree, since F-051 replaced that with the named volume
  `mousedroid_tensorrt_cache`. Two are operator-facing (`JetsonConfig.tensorrt_cache_dir`'s
  `description=`, `cache_dir_is_private`'s docstring). A semantic merge conflict with **zero file overlap**. Revision 2 claimed all six were closed; `CHANGELOG.md:19` — an operator-facing `### Security` entry — still says `docker-compose.jetson.yml` bind-mounts the default location from the host, while `docker-compose.jetson.yml:130` uses the named volume. Replace the prose sweep with a regression test: no tracked file may pair "bind-mount" with "tensorrt_cache". Also read `src/mousedroid/efficiency/tensorrt.py:191` in context — it may be an eighth.
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
