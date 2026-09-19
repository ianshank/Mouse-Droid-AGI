# Peer review: `mouse-droid-branch-hygiene-sweep` (F-052)

Adversarial record for the plan itself, not for the code it proposes. Written before any
of Phases 0 and 2–8 exist, so it reviews the *findings* and the *reasoning*.

Method: five specialist passes over the branch delta (`18aba56...b0759da`, 78 files),
each with a self-contained brief — documentation reconciliation, test-tier analysis,
configuration and hardcoded-value audit, security and supply chain, plus a direct pass on
gate wiring and the workforce inventory. Every load-bearing claim was then re-verified
against the tree by the author before it entered `proposal.md`. Claims that failed that
re-verification are in §Disproved and are **not** in the plan.

---

## What the review changed about the plan

### The plan was going to propose god-file decomposition. The measurement refused it.

`scripts/analyze_observe_step_ceiling.py` is 1,111 lines — the largest file the branch
adds — and the obvious hygiene move is to split it. Measuring first:

```
ruff check scripts/ --isolated --select C901 --no-cache \
    --config 'lint.mccabe.max-complexity = 15'
→ Found 3 errors      (check_branch_coverage.py::main, two ::_probe)
```

Those three are precisely the offenders `pyproject.toml:282-287` already names as the
baseline for the `scripts/**` `C901` ignore. The 1,111-line file contributes **none** of
them: it holds 29 top-level definitions and no function over the ceiling. The first
`ruff` invocation I ran was wrong — `--select C901` alone still honours
`per-file-ignores`, so it printed "All checks passed!" and would have let me claim the
scripts tree was clean. `--isolated` was needed to bypass the ignore, and then the
default ceiling of 10 gave a different, also-wrong number (7). Only the third form
measures what the repository actually enforces.

Two corrections came out of that: the plan now says explicitly *do not decompose it*
(`proposal.md` §5), and the real `scripts/` gap is the blanket `["D", "T20", "S603",
"S104", "E501", "C901"]` exemption plus the absence of any shell linting — not
complexity.

### The sign-off condition caught a defect in this bundle, before it shipped

`peer-review.md`'s own acceptance rule is that any line marked "verified" must reproduce
when a reader runs the cited command. Applying it to D-1's blast-radius paragraph — which
said "remove the two files from `_ALLOWED_FILES`, update three tests" — found that the
change as written would *break* an invariant rather than amend one.

`tests/regression/test_f042_aqa.py:68` asserts `on_disk == exempt` for the orchestrator:
every `_*.py` mixin is exempt **by category**, with no escape hatch. Removing one file from
the list does not narrow a reviewed enumeration; it violates a categorical rule.

The factory half of that same file already solved this. `_GATED_FACTORY_FILES` (`:14-20`)
enumerates three factory modules that stay gated, asserted disjoint from the exempt set at
`:49-50`. The revised D-1 mirrors that mechanism onto the orchestrator side instead of
poking a hole in a list, which is both smaller and more faithful to what F-042 decided.

Worth stating plainly: this was the *only* defect the sign-off rule caught in its first
application, and it was in the section the author was most confident about.

### Three of the request's named items turned out to be already satisfied

Reported as verified-clean rather than silently dropped, because "we fixed lint" would be
a false claim:

- `mypy --strict` — `Success: no issues found in 424 source files`.
- `ruff --select NPY` including `NPY002` — clean across `src/` and `training/`.
- Dependabot — `.github/dependabot.yml` exists. The gap is a missing `docker` ecosystem,
  not a missing file.

### The security pass found defects in the author's own code, so the plan stopped being a plan

Phase 1.1–1.6 were executed, not scheduled. Four confirmed holes in
`require_safe_remote_path` — a validator this branch introduced, whose header claims two
defences:

```
tab              PASS       ← IFS character; word-splits under sudo on the rover
single backslash PASS       ← the arm was *'\\'*, and a QUOTED pair matches TWO
glob star / ? [] PASS       ← not shell metacharacters, so no arm caught them
REMOTE_CONFIG_Q  unused     ← computed at :110, referenced nowhere
```

Each verified by driving the real script, not by reading the pattern. The fourth is the
one worth keeping: the pre-existing test asserted `REMOTE_CONFIG_Q="$(printf ` was
*present* in the source, which a dead variable satisfies perfectly. The replacement
asserts every `*_Q` form is **used**, which is the property the header actually claims.

Then the widened raw-boundary scan — extended from `${REMOTE_SRC}` alone to
`${REMOTE_CONFIG}` and `rsync` — immediately failed on **five more** boundaries the author
had not spotted. That is the plan's own thesis executing on the plan's own author: a test
found in nine seconds what four rounds of reading did not.

`REMOTE_USER` and `HOST` have the same class of exposure and were **not** folded in:
`git show 18aba56:scripts/deploy_remote.sh:15` shows `REMOTE_USER` was already
env-overridable at the merge base, so it is pre-existing debt. It is task 1.7 with the
full exploit path recorded, not a silent PR widening.

---

## Disproved — investigated, and not in the plan

Recorded so a later pass does not re-raise them, and so the plan's accuracy is falsifiable.

| Claim | Why it is wrong |
|---|---|
| "9 features have unresolvable `implemented_in` SHAs" — `scripts/validate.py --tier fast` warns for F-039…F-048 | **Environment artifact.** `.git/shallow` exists; the clone holds 71 commits. `harness.yml:52-55` and `:74-77` both set `fetch-depth: 0` with the comment "so `git rev-parse` can resolve implemented_in refs". CI resolves them; this container cannot. Not a finding. |
| "`analyze_observe_step_ceiling.py` is a god file" | 29 top-level definitions, zero functions over the repo's complexity ceiling. Length without complexity is not a god file. |
| "`scripts/` has new complexity offenders" | Three offenders, all pre-existing and all already named in `pyproject.toml`. The branch added none. |
| "The new shell scripts don't parse" | All 58 tracked `.sh` files pass `bash -n`, including all six the branch adds. This is why task 2.1 can land blocking at zero cost. |
| "A `validation_command` points at a missing script" | All 22 referenced `scripts/validations/F-*.sh` exist. F-047 and F-049 are `deferred` with no command, which is correct. |
| "The directory allowlists in `check_no_hardcoded_values.py` are hiding literals" | Verified empirically: **zero** literals on this branch's added lines in `factory/`, `orchestrator/_` or `config/schema/` would be flagged even with the allowlist removed. The allowlist earned nothing here. The real blind spot is different — the gate parses only *numeric* literals, so every string literal in shell and Docker is invisible to it in any directory. |
| "New config fields break backwards compatibility" | All 7 carry `Field(default=..., description=...)`, asserted programmatically at `tests/regression/test_f050_aqa.py:219-233`. Both strict switches default `False`; `revision="main"` reproduces `hf_hub_download`'s own default. Existing YAML loads byte-identically. The real gap is narrower and is task 4.6: the *values* are pinned only as truthy. |
| "`assert` was added under `src/`" | Zero. The only `assert` on added `src/` lines is prose inside `src/mousedroid/world_model/CLAUDE.md` — text that itself explains why `PYTHONOPTIMIZE=1` makes an assert-based guard no guard. |
| "`.gitleaks.toml` needs a new allowlist entry for this branch" | It needs none and got none. The file is unmodified; its four regexes are narrow literals with `regexTarget = "match"`, no `paths`, no `.*`-class pattern. |

---

## Open weaknesses in this plan

Stated because a review that finds nothing wrong with its own output is not a review.

1. **`shellcheck`'s finding count is unknown.** `shellcheck` is not installed in this
   container, so task 2.2 sizes a gate nobody has run against 9,036 lines. The advisory
   landing is chosen *because* of that ignorance, not despite it — but if the count is in
   the hundreds, Phase 2 grows and the promotion window in
   `.github/advisory_stages.yaml` will need a realistic `promote_after_days`. The first
   action in 2.2 should be to run it and record the number in this file.
2. **Phase 0 leaves the change with no suppression allowance.** Deliberate (`design.md`
   D-5), and it produced better code once already — the `TYPE_CHECKING`-only
   `_StateHookBase` in `observe_step_timing.py:41-70` exists because a
   suppression was unavailable. But if a later phase genuinely needs one, the budget must
   be bumped *with a recorded reason*, not quietly. If that happens twice, the ratchet-to-21
   decision was wrong and should be revisited rather than repeatedly overridden.
3. **Task 4.2's fake could drift from the real ONNX engine.** An always-on integration
   test that monkeypatches the engine constructor proves the *factory wiring*, not the
   engine. If the real `DualStreamRSSMOnnx` stops being `Warmable`, the fake still passes.
   Mitigation: assert `isinstance(DualStreamRSSMOnnx, WarmableProtocol)` at class level in
   the same test, which needs no `onnxruntime` import at runtime. Without that, 4.2
   reintroduces the assert-what-you-wrote failure in a new place.
4. **This plan has 70 tasks across 9 phases.** That is a programme, not a change. Phases
   0, 1 and 2 are the ones with teeth; 6 and 7 are largely mechanical. If it must be cut,
   cut 6 and 7 — but cut them *explicitly*, because task 6.1 is not documentation: it is a
   safety-relevant metric with a documented paging expectation and no alert rule.
5. **The four specialist passes were not independently cross-checked against each other.**
   They agreed where they overlapped (all four independently flagged the absence of
   `shellcheck`; two independently flagged the `docker` dependabot gap), which is weak
   corroboration, not proof. Every claim that entered the plan was re-verified by the
   author against the tree; claims that were not re-verifiable were dropped rather than
   softened.

---

## Revision 2 — the weakness this bundle named and then ignored

Revision 1's weakness list said, at item 4: *"This plan has 70 tasks across 9 phases. That is a
programme, not a change."* It then grew to 92 and shipped anyway. Naming a defect is not fixing one,
and a bundle that does that is doing the thing it criticises elsewhere.

Revision 2 changes nothing about the tasks and everything about the shape: the same 92 are regrouped
into **eight slices that each ship as one PR**, with the single hard dependency stated (slice A, the
ratchet headroom, because all three budgets sit at ceiling and every other slice that needs a
suppression is blocked until it lands). `tasks.md` carries the table and the "if cut to a third"
answer — keep A, C, D, E: create the headroom, close the security items left open, fix the promotion
gate that passes for the wrong reason, add the alert for a digest mismatch nothing pages on.

**Every premise was re-verified in the merged tree** rather than carried forward from `b0759da`: the
five reclaimable markers, the three at-ceiling budgets, `mypy --strict` over 424 files after PR #234
added code, the three `C901` offenders, `ruff NPY`, and the `test-windows` window arithmetic. No row
of `proposal.md` §6 has gone false. That check mattered: a "do not churn" table that silently rots is
worse than no table, because the next pass trusts it.

### Where this bundle now overlaps F-053

The two plans touch the same surfaces and should not be implemented blind to each other:

- **Task 4.5** here defers validating the ~50 existing mermaid fences under `docs/`. F-053 establishes
  that **no renderer exists anywhere in the toolchain** — nothing in `pyproject.toml`,
  `.github/workflows/ci.yml`, `scripts/ci.sh` or the `Makefile`, and no Python parser installed. So
  4.5 is not a small follow-up; it is a Node dependency in a Python-only CI. F-053's answer —
  generate the diagrams so correctness comes from the generator's tests rather than a renderer — is
  the cheaper route for this backlog too.
- **Phase 3** here adds a hooks→runbook inventory pin. F-053's Phase 2 adds an `AGENTS.md`→importer
  pin. They are the same shape of gate (an asset exists ⟹ something references it) and should share
  one helper rather than growing two.
- **Task 6.1** here adds the missing `ModelArtifactDigestMismatch` alert. Unaffected by F-053, and the
  highest-value single item across both plans.

## Sign-off condition

This plan is accepted when a reader can take any line marked "verified" in
`proposal.md` §2 or §6, run the command or open the file:line cited, and get the stated
result. Any that does not is a defect in this bundle, not in the code.
