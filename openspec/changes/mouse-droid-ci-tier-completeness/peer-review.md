# Peer review — CI tier completeness

## Verdict table

| Claim | Verdict |
|---|---|
| The three tiers run in zero CI paths | **CONFIRMED** — 0 references across `scripts/ci.sh`, `.github/workflows/ci.yml`, `Makefile` |
| `tests/security/` is the only coverage of the pre-egress filter | **CONFIRMED** — two files, both in that tier |
| The tiers still pass | **CONFIRMED** — 17 passed in 2.44 s on a `[dev,telemetry,mcp]` install |
| The `security` job would swallow failures | **CONFIRMED** — `continue-on-error: true`, pinned advisory by the same AQA module |
| The new pin can fail | **CONFIRMED** — wiring removed → 3 of 4 assertions red; restored clean |

## Corrected-design map

| Original intent | Corrected |
|---|---|
| "Put the security tests in the security job" | Rejected — that job is advisory and would swallow failures. All three tiers go in the blocking `test` job. |
| "Wire the tiers" | Insufficient alone — a wiring fix without a pin re-drifts (PR #178's recorded lesson). The pin is part of the change. |
| "Fix the private-attribute assertions while here" | Rejected — couples a half-day wiring bundle to a test rewrite. Deferred to F-031's arc and recorded in D-4. |

## What survives review unchanged

- Placement outside the `MOUSEDROID_CI_SLIM` gate, mirroring the smoke stage.
- Reuse of the existing AQA module and its `_job_run_text` / `_CI_SH` helpers
  rather than a new test module.
- No production code, no config fields, no new dependencies.

## Load-bearing pins any implementation must satisfy

1. All three tiers appear in the blocking `test` job **and** in `scripts/ci.sh`.
2. `tests/security` appears in **neither** the advisory `security` job nor any
   `continue-on-error` job.
3. The `ci.sh` stage precedes the `MOUSEDROID_CI_SLIM` gate.
4. `TestOrphanTierWiring` asserts the advisory precondition of the `security`
   job before asserting the negative, so a future promotion surfaces as a
   rethink rather than a silent pass.

## Appendix — accepted deviation

D-4 records that this change makes two tests blocking which assert on the
private `orch._motor` attribute — the same anti-pattern criticised elsewhere in
the plan. Accepted knowingly: leaving the security tier unenforced is worse.
