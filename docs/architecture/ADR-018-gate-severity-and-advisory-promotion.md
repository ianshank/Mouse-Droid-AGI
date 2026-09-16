# ADR-018 — Gate Severity Split + Advisory-Promotion Bar

* **Status:** Accepted
* **Date:** 2026-09-16
* **Owners:** Ian Cruickshank
* **Scope:** `tools/ratchet_budgets.py` (`BudgetFinding`, `classify_budget_item`,
  `classify_all_budgets`, `--strict`), `.github/workflows/ci.yml`
  (`local-gates` ratchet step; `security` job), `scripts/ci.sh`,
  `.github/advisory_stages.yaml` (`security` entry removed), `SECURITY.md`,
  `docs/CHARTER.md`, `docs/claude/surfaces/ci-gates.md`, `NEXT_STEPS.md`,
  `README.md`, `tests/regression/test_doc_reconciliation_aqa.py`.

## Context

Two gates were nominally in place and neither was actually gating.

**`ratchet_budgets --strict` was structurally unusable.** The flag existed but was never
wired into any CI job, because it failed on *any* finding — including the
"approaching budget" warning. That is not a tuning problem, it is a contradiction with
the ratchet discipline itself: a ratchet lowers each ceiling to the *current* count, so a
healthy budget sits exactly at its ceiling and therefore permanently above its
`warn_threshold`. Wiring the flag in as written would have made `local-gates` perma-red on
day one, for budgets that were in perfect health. The flag could not be used, so the
budgets were report-only in practice and nothing stopped a ceiling from being exceeded.

**The `security` job had spent its whole life soft-failed.** `pip-audit` ran behind
`|| echo "::warning::"`, which made it invisible to `scripts/check_advisory_promotions.py`
— an advisory stage with no promotion clock is a permanent soft gate, not a temporary one.
Converting the swallow into `continue-on-error: true` gave it a clock (60 days from
2026-07-25, due 2026-09-23). This ADR is the decision made against that clock.

## Decision

**(a) We will make budget findings severity-aware, and gate `--strict` on the breach only.**

A new frozen `BudgetFinding` dataclass carries a `breached: bool` that splits a ceiling
breach (a contract violation) from an approaching-budget warning (advisory by design).
`--strict` fails only on a breach; an approaching warning still prints and still exits 0.
`check_budget_item` / `check_all_budgets` keep their `list[str]` signatures for their
existing consumers — the `PostToolUse` hook and the regression tests — and delegate to the
new classifier, so the message text has exactly one source. `--strict` is now wired into
`ci.yml`'s `local-gates` job and into `scripts/ci.sh`.

The severity split is not a refinement of the flag; it is the precondition for the flag
existing usefully at all. Without it there is no wiring, and the budgets stay decorative.

**(b) We will promote `security` from advisory to blocking, 53 days into its 60-day
window (7 days before the 2026-09-23 deadline),
on a findings-triage bar rather than a green-run count.**

`pip-audit --skip-editable` reports zero vulnerabilities across the resolved
`[dev,telemetry,mcp]` set — 123 packages, including every network-facing one (aiohttp,
cryptography, pyjwt, starlette, uvicorn, h11). The question a dependency audit answers is
"are there open findings", and the answer is no; counting further green days would not
change it. The `advisory_stages.yaml` entry is **removed** rather than marked done,
following the gitleaks precedent recorded in that file: the tracker's contract is "every
job carrying `continue-on-error: true` has an entry here", so a promoted job has no entry.
This leaves five advisory jobs.

## Consequences

* A ceiling breach now fails `local-gates`, which is blocking. The ratchet contract is
  machine-enforced for the first time; raising a ceiling is now a deliberate, reviewed
  edit to `.claude/workforce.yaml` rather than something a commit can do silently.
* An approaching-budget warning remains advisory and always will. Anyone later tempted to
  "tighten" `--strict` to cover warnings should read the Context above first — that change
  reintroduces the perma-red.
* **A future upstream CVE in any of ~123 packages now blocks PRs.** This is the real cost
  of (b) and it is accepted deliberately: an advisory dependency audit is an audit nobody
  reads. The remedy when it fires is to bump or pin the dependency, not to re-add
  `continue-on-error`. If a finding is genuinely unfixable and non-exploitable here, the
  correct move is an explicit, dated ignore with a recorded reason — visible in review —
  rather than reverting the whole job to soft.
* Promoting on a triage bar rather than a day count sets a precedent: the advisory window
  is a *deadline*, not a *minimum*. A job whose promotion question is already answered may
  be promoted early. Jobs whose bar genuinely is a green-run count
  (`onnx-world-model-extras`, `mlflow-extras`) keep that bar, recorded per-entry.
* Advisory-count prose lives in five files and only `advisory_stages.yaml` is
  machine-checked, which is how `docs/claude/surfaces/ci-gates.md` came to claim six
  advisory jobs while the root `CLAUDE.md` said five. A regression pin in
  `tests/regression/test_doc_reconciliation_aqa.py` now derives the count from
  `ci.yml` and asserts every doc stating one agrees.

## Alternatives considered

**A separate `--fail-on-breach` flag, leaving `--strict` as-is.** Rejected. Two overlapping
severity flags is a worse API than one flag with correct semantics: every caller then has
to know which of the two it wants, and the documented meaning of `--strict` stays wrong.
The compatibility argument for it was also empty — no existing test depended on the old
behaviour, because every `--strict` exit-1 test already used a real breach (25 occurrences
against a ceiling of 19). Fixing `--strict` in place broke nothing.

**Letting the `security` window run to 2026-09-23 and promoting then.** Rejected. The bar
is findings, not calendar days, and the findings were already triaged to zero. Waiting
would have left the job soft for three more days for no additional evidence.

**Marking the `advisory_stages.yaml` entry `promoted: true` instead of removing it.**
Rejected for consistency with the gitleaks promotion: the tracker exists to catch jobs
*still* carrying `continue-on-error`, and a residual entry for a blocking job would make
`check_advisory_promotions.py`'s "untracked advisory stage" check ambiguous. The promotion
history lives in the CHANGELOG, in the removed entry's replacement comment, and here.
