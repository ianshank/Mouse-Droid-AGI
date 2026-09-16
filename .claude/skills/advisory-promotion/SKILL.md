---
name: advisory-promotion
description: Promote an advisory CI job (continue-on-error) to blocking, or extend its window with a recorded reason — touching every coupled surface, not just ci.yml. Use when scripts/check_advisory_promotions.py reports a job overdue, when a promotion window is about to close, or when a green-run bar has been met.
status: active
---

# Advisory Promotion

Flip one `continue-on-error: true` job in `.github/workflows/ci.yml` to
blocking — or extend its window with a dated reason — and land every coupled
edit in the same change.

Use this when `scripts/check_advisory_promotions.py` reports **promotion
overdue**, when a window is days from closing, or when a job's stated bar (a
green-run count rather than a calendar window) has been met.

## Why this exists

The recipe has been executed three times and is not a one-file edit:

| Date | Job | Outcome |
|---|---|---|
| 2026-08-07 | `gitleaks` | promoted; protocol recorded in `docs/runbooks/secret-scanning.md` |
| 2026-09-16 | `security` (pip-audit) | promoted; **3 surfaces missed** (below) |
| due 2026-09-19 | `test-windows` | 30-day window closes |
| bar met (8/8 green) | `onnx-world-model-extras`, `mlflow-extras` | `docs/planning/TECH_DEBT_REMEDIATION_PLAN.md` §7 |

The `security` promotion updated five places and left three stale, which is
the whole argument for a checklist:

- `docs/claude/surfaces/ci-gates.md` still says *"6 jobs carry
  `continue-on-error: true`"* (real count: 5), still lists `security` as
  *(advisory)* in its stage table, and still carries its retired
  `advisory_stages.yaml` bullet.
- `scripts/ci.sh` still prints `security - pip-audit --skip-editable (advisory
  in CI)` in the not-reproduced-locally banner.
- `.github/workflows/ci.yml` still has an inline comment calling `security`
  *"the advisory `security` job: that job is continue-on-error and would
  swallow every failure"*.

Nothing catches any of these. `tests/regression/test_doc_reconciliation_aqa.py`
deliberately does **not** match a bare `N jobs` (its own comment explains why:
it would flag the true claim *"5 jobs run *(advisory)*"*), and
`scripts/check_advisory_promotions.py` reads workflows and metadata only — it
never reads prose. Step 6 below is the sweep that closes the gap.

## Inputs

- `$ARGUMENTS` — the CI job name exactly as it appears in `ci.yml`, e.g.
  `test-windows`. Required.

## Steps

### 1. Establish the ground truth before deciding

```bash
python scripts/check_advisory_promotions.py --today "$(date -u +%F)"
python -c "import yaml,pathlib; d=yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print(sorted(n for n,j in d['jobs'].items() if isinstance(j,dict) and j.get('continue-on-error') is True))"
```

The first prints overdue/untracked/stale-metadata warnings; it is **WARN-only
and exits 0** without `--strict`, so read the output, never the exit code. The
second is the authoritative advisory roster — the count every prose surface in
step 6 must agree with.

### 2. Read the job's own recorded bar — it is often not the calendar

The `reason:` field in `.github/advisory_stages.yaml` is the contract, and for
several jobs it overrides `promote_after_days`:

- `onnx-world-model-extras` and `mlflow-extras` state a **7-consecutive-green-run**
  gate, not a date. Their 180-day windows exist only so the tracker has a
  clock. Promoting them requires a run count, not a calendar check.
- `vulture-audit` is findings-only **by design**; its window ends in a decision
  between "keep advisory forever + record an ADR" and "strict mode on a curated
  allowlist" — not in a promotion.
- `performance` carries shared-runner variance; it is green only with CI's
  `MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET`, which `scripts/ci.sh` does not set.

If the bar is not met, **stop here and go to step 7** (extend, don't promote).

### 3. Prove the job is actually green under its own conditions

Run the job's own steps locally, with the job's own extras — not `make gates`.
A promotion justified by "CI looks green" is the failure mode that made
`security` sit behind a `|| echo ::warning::` swallow for its entire life. Read
the job's `steps:` in `.github/workflows/ci.yml` and reproduce them verbatim,
including the `pip install -e ".[...]"` line, because `mypy --strict` and
pytest both report *different* findings under a different extras set rather
than fewer.

### 4. Edit the two machine-readable surfaces

`.github/workflows/ci.yml`:

- delete the job's `continue-on-error: true` line;
- add a dated promotion comment above the job (follow `gitleaks`' shape);
- update the `# Stage N:` banner comment to read `BLOCKING`, as `gitleaks`'
  banner does — the `security` promotion left its banner unmarked;
- rename any step whose name embeds its advisory-ness, e.g.
  `Security audit (advisory via continue-on-error)` →
  `Security audit (blocking since <date>)`.

`.github/advisory_stages.yaml`: **remove** the job's entry and leave a dated
comment in its place. Do not mark it done in situ — the tracker covers only
jobs still carrying the flag, so a leftover entry makes
`check_advisory_promotions.py` emit `stale metadata`. Both prior promotions did
it this way and both recorded why.

### 5. Flip the regression pin, do not delete it

`tests/regression/test_ci_gate_wiring_aqa.py` holds a per-job class
(`TestSecurityJob`, `TestPerformanceJob`, …) whose assertion is the *inverse*
of what you just did. Invert the assertion and rewrite the failure message to
rest on the job's **purpose**, not on `continue-on-error`:

```python
assert job.get("continue-on-error") is not True, (
    "<job> is blocking since <date> — <the evidence>. Re-adding "
    "continue-on-error also needs an advisory_stages.yaml entry, or "
    "check_advisory_promotions.py WARNs 'untracked advisory stage'."
)
```

Keep every other assertion in that class. The half worth keeping is usually the
flag-pair / no-shell-`||`-swallow check, which is what makes the job's output
mean anything and is independent of advisory-ness.
`TestAdvisoryTracking::test_all_advisory_jobs_tracked` needs no edit — it
derives its roster from the workflow.

### 6. Sweep the prose surfaces — this is the step that gets skipped

```bash
git grep -n -- "$ARGUMENTS" -- CLAUDE.md CHANGELOG.md scripts/ci.sh docs .github | grep -i "advisory\|continue-on-error"
git grep -n -iE "[0-9]+ jobs (run|carry)" -- CLAUDE.md docs
```

Every hit is a surface to update. The known set, all of which the `security`
promotion had to touch and not all of which it did:

| Surface | What to change |
|---|---|
| `CLAUDE.md` | the `Stage N` bullet's `*(advisory)*` tag, **and** the `N jobs run *(advisory)*` count |
| `docs/claude/surfaces/ci-gates.md` | the numbered stage entry, the `N jobs carry continue-on-error` count, **and** the per-job bullet under *Advisory Stages & Promotion Ladder* |
| `scripts/ci.sh` | the `CI jobs NOT reproduced locally` banner, if it labels the job advisory |
| `.github/workflows/ci.yml` | inline comments elsewhere in the file that call the job advisory |
| `CHANGELOG.md` | a `### Changed —` entry under `[Unreleased]` naming the window, the bar, and the evidence |

If the job has an operator runbook (`docs/runbooks/secret-scanning.md` is the
worked example), add a `## Promotion protocol` paragraph there too: the dated
flip, the evidence, and what re-demoting would require.

### 7. Extending instead of promoting

Legitimate, and both prior windows were extended at least once. Keep the entry
and edit it in place:

- leave `since` **unchanged** — it is when the stage landed advisory, not when
  it was last reviewed; moving it silently resets the clock;
- raise `promote_after_days`;
- rewrite `reason:` to state what was checked, what the real bar is, and the
  concrete next action ("count consecutive green runs for this job"), as
  `onnx-world-model-extras`' own extension note does.

No `ci.yml`, test, or prose edit is needed for an extension — only
`.github/advisory_stages.yaml`.

### 8. Verify

```bash
python scripts/check_advisory_promotions.py
python -m pytest tests/regression/test_ci_gate_wiring_aqa.py \
    tests/regression/test_doc_reconciliation_aqa.py \
    tests/unit/scripts/test_check_advisory_promotions.py -q --no-cov -o addopts=""
docker run --rm -v "$PWD:/repo" -w /repo rhysd/actionlint:1.7.12 -color
```

`check_advisory_promotions.py` must report neither `stale metadata` for the
promoted job nor a changed count of other warnings.
`tests/unit/scripts/test_check_advisory_promotions.py` contains
`test_repo_advisory_stages_are_all_tracked_and_in_window`, which evaluates the
**real** repo files — it is the pin that catches a half-finished edit to
`.github/advisory_stages.yaml`. `actionlint` has no local equivalent outside
that pinned container (`scripts/ci.sh` says so in its own banner).

## Definition of done

- the job has no `continue-on-error` and its `.github/advisory_stages.yaml`
  entry is removed, replaced by a dated comment
- its `# Stage N` banner and any advisory-labelled step name say blocking
- its class in `tests/regression/test_ci_gate_wiring_aqa.py` asserts the new
  state and justifies itself by purpose, not by the flag
- `git grep` for the job name returns no surface still calling it advisory
- the advisory count in `CLAUDE.md` **and** in `docs/claude/surfaces/ci-gates.md`
  equals the roster printed in step 1
- `CHANGELOG.md` `[Unreleased]` names the window, the bar, and the evidence
- step 8 is green

**Verify:** `python tools/validate_skill_commands.py` must report zero issues —
it is the gate behind `tests/regression/test_skill_commands_aqa.py`, and it
fails on broken front-matter, an empty description, an unknown `status`, a
hardcoded IPv4 literal, or a backtick-wrapped repo path that does not resolve.
A new directory under `.claude/skills/` also needs its row in the *Workforce
skills* table in `SKILLS.md`, or
`test_claude_workforce_aqa.py::test_every_skill_directory_is_mentioned_in_the_index`
fails — the `.gitignore` negation `!.claude/skills/` already makes the
directory shippable, so no ignore-file edit is required.
