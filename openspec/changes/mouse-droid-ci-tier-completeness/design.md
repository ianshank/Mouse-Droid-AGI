# Design — CI tier completeness

## D-1. Which job the tiers belong in

They go in the **blocking `test` job**, not the job named `security`.

`.github/workflows/ci.yml` already has a job called `security`, and the obvious
reading — "security tests belong in the security job" — is a trap. That job is
`continue-on-error: true` and is pinned as *honestly advisory* by
`tests/regression/test_ci_gate_wiring_aqa.py`. Putting `pytest tests/security`
there would swallow every failure and recreate the orphan-tier problem wearing a
name that looks correct. The `security` job runs `pip-audit` over dependencies;
it is a supply-chain check, not a test tier.

`TestOrphanTierWiring.test_tiers_are_not_in_the_advisory_security_job` pins this
negatively, and asserts the advisory precondition first so that a future
promotion of that job surfaces as a rethink rather than a silent pass.

## D-2. Placement relative to the slim gate

`scripts/ci.sh` gates its memory-heaviest stages behind `MOUSEDROID_CI_SLIM`,
set when the Jetson Phase-1 `ci.sh` run retries after an OOM kill. The three
tiers total ~2.5 s and import only core modules, so there is no memory-pressure
case for dropping them. They go **before** the slim gate, mirroring the smoke
stage and its recorded rationale.

## D-3. Why the pin is part of the change

A wiring fix without a pin re-drifts. That is the explicit lesson of PR #178,
written into the AQA module docstring. The pin lives in the existing
`tests/regression/test_ci_gate_wiring_aqa.py` rather than a new module, because
that file already owns "does the CI wiring say what we think it says" and
already carries the `_job_run_text` and `_CI_SH` helpers.

The pin was verified to fail: removing the wiring from both files turns three of
its four assertions red. A pin that cannot fail is not a pin.

## D-4. What this change deliberately does not fix

`tests/user_journey/test_operator_mission_journey.py` and
`tests/functional/test_mission_safety_interlocks.py` both assert on the private
attribute `orch._motor`, and both exercise `build_autonomous_orchestrator` —
a component F-031 records as off the production path. Making them blocking
promotes that pattern.

This is accepted knowingly. The alternative — leaving the security tier
unenforced — is materially worse, and coupling a wiring bundle to a test rewrite
would grow it without bound. The two files are named in F-031's disposition ADR
as the component's remaining exercisers, and moving them to observable
assertions is left to that arc.
