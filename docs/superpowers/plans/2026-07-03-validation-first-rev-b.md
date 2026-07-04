# Mouse-Droid-AGI — Validation-First Implementation Plan (rev. B, post peer review)

> **For agentic workers:** This is the peer-reviewed revision of `mdagi-validation-first_v1`.
> It was produced by a four-agent review (three repo fact-checkers + one design-logic critic)
> against the working tree at commit `b125543` and the live GitHub API, so — unlike rev. A —
> every `[Certain]` tag below has been verified with file:line evidence. Corrections from
> rev. A are marked **[FIXED]**. CONFIRM-FIRST gates are hard stops — do not proceed past one
> without an explicit user decision.
>
> Date: 2026-07-03 · Author: Claude (peer-review session) · Status: **Approved** ·
> Authored on branch: `claude/review-agents-plan-4qfxev` ·
> ID: `mdagi-validation-first_v2` · Basis commit: `b125543` (verified HEAD at review time)

---

## Peer-review summary (what changed and why)

The rev. A framing survives review: the critical path is not software, and the freeze
discipline, findings-only audit posture, and CONFIRM-FIRST register are sound. Five findings
drove the revision:

1. **CRITICAL — the rev. A DAG inverted its own thesis.** WS-6 (hardware gate) depended on
   WS-3→4→5 completing, so a repaired ESP32 would have waited on env-render machinery and
   Grafana panels. Rev. B adds a preemption rule: hardware readiness trumps all in-flight
   software streams.
2. **F-009…F-014 are already burned.** `features.yaml` holds F-001..F-008, but
   `SMOKE_REPORT.md:199-204` + CHANGELOG + NEXT_STEPS use an independent operational-findings
   F-sequence already reaching F-014 (commit `3015283` is literally the "F-013/F-014
   closeout"). Every new ID rev. A proposed collided. Rev. B renumbers to F-015+.
3. **WS-4.3's preflight timer was an operational hazard.** `jetson_full_validation.sh`
   Phase 2 stops the container specifically for exclusive device access; a timer-driven
   real-mode preflight would fight the running orchestrator for camera/UART. Rev. B adds a
   container-aware contract.
4. **Stale and refuted claims:** Issue #109 is closed and its work landed
   (`fire_on_startup`, pr109 integration + hardware tests); the `[vla]` CI leg is already
   promoted/blocking (the advisory ones are `jetson-nightly` and `onnx-world-model-extras`);
   `config/prometheus/alerts.yml` + `promtool check rules` already exist in CI;
   `scripts/mousedroid-docker.service` already runs `sync_jetson_overlay.sh` as ExecStartPre.
5. **Wrong precedents in the doctrine:** the coverage gate is 85% **line** (branch coverage
   is measured nowhere despite `check_branch_coverage.py`'s name); `test_smoke_harness.py`
   is source-text assertion, not pytest-subprocess; the SUMMARY generator is an untested
   inline bash function (`write_summary()` in `jetson_full_validation.sh`).

The full finding-by-finding audit trail (three fact-check reports + design critique) is
preserved in the review session; each `[Certain]` below carries its evidence there.

---

## 0. Framing (revised)

The framing survives review: the delivered software stack is landed green under mock
hardware; **F-008 is the only `todo` feature** [Certain]. Two corrections:

- **[FIXED]** F-008 is "USB-C rover smoke passes on the physical Jetson"
  (`validation_command: bash scripts/jetson_smoke_test.sh`, `implemented_in: null`) — the
  ESP32 is one blocker *inside* it, not its definition. The gate should assume a **defect
  inventory** (the smoke run's Stage 12 carries a "non-blocking until camera /dev/video0
  fixed" annotation — there is likely more than one physical defect).
- **[FIXED — CRITICAL]** New rule replacing the rev. A DAG dependency: **hardware readiness
  preempts all in-flight software streams.** WS-6 depends ONLY on: ESP32 responsive + WS-0
  rotation done. WS-3/4/5 are best-effort-before-gate — they never block bring-up.

## Feature-ID policy (new, fixes the collision)

`features.yaml` holds F-001..F-008; but F-009..F-014 are **already burned** as
operational-findings IDs in SMOKE_REPORT.md/CHANGELOG/NEXT_STEPS (F-013/F-014 were "closed
out" in commit `3015283`). **New harness features start at F-015.** WS-1 gains a
reconciliation task: one paragraph in HARNESS_SPEC.md declaring the two namespaces (harness
catalog vs smoke findings), so `implemented_in` provenance and grep-ability stay clean.
(Alternative if preferred: the schema id pattern permits a prefix, e.g. `OPS-F-001`.)

## DAG (revised)

```
WS-0 (security) ─┬─ WS-1 (truth) ─┬─ WS-2 (PR hygiene) — all parallel
                 │                │
                 └──► WS-3 (ops) ─┴─► WS-4.3 (timer only)
WS-4.1/4.2/4.4 (journal/trend/tests) — parallel with everything  [FIXED: falsely serialized]
WS-5 (observability) — parallel, pre-gate ONLY as endurance-run instrumentation
WS-6 GATE — starts the moment hardware + WS-0 are ready, preempting all of the above
WS-7 — post-gate only.  WS-8 — continuous, findings-only.
```

## WS-0 — Security (P0; rotation is human, scan gate lands regardless)

- [ ] 0.1 **Key-consumer inventory before rotating** [FIXED]: rover
      `/etc/mousedroid/docker.env`, GitHub Actions secrets, any dev-machine `.env` —
      rotation with an unknown consumer breaks that consumer. [CONFIRM-FIRST #5]
- [ ] 0.2 Rotate in Anthropic console; replace on rover; restart service.
- [ ] 0.3 Verify: `tools/llm_latency_probe.py --iterations 3` (flag verified present) **and
      record a rotation artifact** — date + old-key-revoked confirmation in
      NEXT_STEPS/CHANGELOG. [FIXED: rev. A had no evidence step; today nothing in-tree
      confirms rotation ever happened.]
- [ ] 0.4 Secret-scan CI stage (gitleaks or detect-secrets — none exists today [Certain]):
      **new feature `F-015`** (epic `Security`, tier fast). **[FIXED]** Drop the
      planted-fake-key wrapper unit test (self-sabotaging — the scanner flags its own
      fixture; gitleaks is tested upstream). Ship: thin CI invocation, **one-time
      full-history scan** (a leak already happened once), and a pre-commit hook entry.
      Advisory for one green week, then blocking (matches the repo's own promotion pattern).

## WS-1 — Truth reconciliation (scope corrected)

- [ ] 1.1 Split NEXT_STEPS.md (37,265 bytes, 72 ✅ marks [Certain]) — ✅/LANDED →
      CHANGELOG.md; forward items stay. **[FIXED] Also reconcile the SECOND next-steps
      file** `docs/planning/NEXT_STEPS.md` — "Phase 5" means physics-sim (deferred) in root
      but LLM-gateway (done) there; one file must own the phase vocabulary.
- [ ] 1.2 T3-contradiction resolution [CONFIRM-FIRST #1] — recommend pause-at-T2 with
      explicit unfreeze condition. Contradiction verified at NEXT_STEPS.md:71 vs :106.
- [ ] 1.3 Foundry plan doc banner/relocation — unchanged (verified: added in b125543,
      self-declares "NOT this repo").
- [ ] 1.4 Convert remaining live priorities to features (**F-016+**). **[FIXED]**
      Precondition: frozen/deferred work must NOT enter the catalog as `todo`, or
      `select_next.py` will recommend it. Either (a) only convert actionable pre/post-gate
      items and leave frozen work in prose with its unfreeze condition, or (b) first extend
      the schema with a `deferred` status — that is harness capability work; budget it
      explicitly if chosen. Recommend (a).
- [ ] 1.5 Skill re-scope. **[FIXED]** The inventory is bigger than rev. A claimed:
      `.claude/skills/` (3), `.github/skills/jetson-hardware-debug`, `docs/openclaw_skills/`
      (4, doc-paired to builtins by test). Scope: mark the three `.claude` skills frozen.
      There is **no `status:` frontmatter today** and F-004's validator doesn't check
      invocation — so this = add `status:` frontmatter + extend
      `tools/validate_skill_commands.py` to parse it (small, but real code with tests, not
      just docs).
- [ ] 1.6 **[FIXED — advisory only]** NEXT_STEPS size guard lands as a WARN in the existing
      advisory pattern, not a hard-fail (a red PR for adding a paragraph of prose is process
      theater). **Drop the ✅-age date parser entirely** — new markdown convention + freeform
      date parsing = false-positive generator; "✅ moves to CHANGELOG" is enforced by review.
- [ ] 1.7 **[NEW]** Delete the two stale `claude/*` scratch branches (local + remote) — rot
      the rev. A plan itself missed. [CONFIRM-FIRST #2, bundled with WS-2.1]

## WS-2 — PR hygiene (mechanics corrected)

- [ ] 2.1 Close #26/#30/#48 [CONFIRM-FIRST #2] — PR list verified live.
- [ ] 2.2 Ruff 0.8→0.15.20: **[FIXED]** three pin locations move atomically —
      `pyproject.toml:155`, `ci.yml:66`, **and `release.yml`'s unpinned `ruff>=0.4`**
      (already-drifting third location). The pyproject comment itself mandates the
      pyproject/ci sync. Sequence **last** in the pre-gate window (or post-gate): a
      7-minor-version format churn right before hardware bring-up is diff noise with zero
      validation value, and it conflicts with WS-1's doc PRs.
- [ ] 2.3 Batch actions bumps (#147/#146/#122) — unchanged (actionlint gate verified).
- [ ] 2.4 mlflow-skinny (#145) gated on T2 logger tests in a matrix leg — unchanged
      (logger + tests verified to exist).

## WS-3 — Ops-as-code (mechanism right-sized)

- [ ] 3.1 **[FIXED — replaces the YAML→env render layer]** The render machinery was
      disproportionate for ~2 variables and had a hole: the file's most important content
      (`ANTHROPIC_API_KEY`) cannot live in committed YAML, so "render" was really
      merge-with-secret-injection, breaking its own determinism tests. Instead:
      (a) extend `config/docker.env.example` to enumerate `MOUSEDROID_LLM__ENABLED` /
      `MOUSEDROID_LLM__N_GPU_LAYERS` (verified missing today);
      (b) add a preflight WARN check that the deployed `/etc/mousedroid/docker.env` key-set
      ⊇ template key-set (**names only, never values**);
      (c) `scripts/host_bootstrap.sh` stays but as a thin installer (copy template if
      absent, install units) with **pre-overwrite backup + `--rollback`** [FIXED: no
      rollback story existed] and the verified `jetson-runner-install.sh --dry-run`
      convention. Feature **F-017** (epic `Deployment`).
- [ ] 3.2 Root-ownership fix [CONFIRM-FIRST #3] — unchanged; fallback chown +
      `ownership_drift_detected` WARN probe stands.
- [ ] 3.3 **[FIXED — scope shrunk]** `scripts/mousedroid-docker.service` ALREADY runs
      `sync_jetson_overlay.sh` as ExecStartPre — the YAML-overlay side is solved; rev. A
      conflated it with the env-file side (and swapped NEXT_STEPS items 3↔4). Remaining
      work: only ensure the *env-file* survives reflash, which 3.1 covers. No new
      ExecStartPre.

## WS-4 — Validation instrumentation (unserialized; timer contract added)

- [ ] 4.1 Thread `--journal-path` through `jetson_full_validation.sh:327` Phase-2 preflight
      — verified real gap; CLI flags already exist (`cli/preflight.py:62,75`). **Runs in
      parallel with WS-0..3** [FIXED: only 4.3 depends on WS-3].
- [ ] 4.2 Surface `--trend` in the Phase-4 SUMMARY. **[FIXED]** The generator is
      `write_summary()`, an *inline bash function* in `jetson_full_validation.sh:455-475`,
      currently untested — `test_smoke_harness.py` covers a different script via
      source-text assertions, not pytest-subprocess. Either test the extended
      `write_summary` with a real subprocess-fixture test (new pattern — the doctrine's
      cited precedent doesn't exist) or extract summary-writing to a small Python helper
      under the existing 85% gate. Recommend the latter.
- [ ] 4.3 Trend timer — **[FIXED, was an operational hazard]** contract required: while the
      orchestrator container is running, timer-mode preflight performs **non-exclusive
      checks only** (existence/config/disk — never opens `/dev/video*` or UARTs; the
      Phase-2 docker-stop contract proves device probes require exclusivity); full device
      probes only when the container is stopped. Journal gets a size cap/rotation (SD card;
      `jetson_disk_cleanup.sh` vacuums journald to 50M as precedent). One test: timer mode
      never touches exclusive devices with the container up. Feature **F-018** (epic
      `Validation efficiency` — an *existing* epic, F-006).
- [ ] 4.4 Fixture tests for journal-threading — unchanged.
- [ ] 4.5 openclaw validation overlay — unchanged (verified real: NEXT_STEPS item 6,
      Test C skip).

## WS-5 — Observability (rationale stated; scope shrunk)

**Pre-gate justification, stated explicitly [FIXED]:** these panels/alerts exist to
*observe the WS-6 endurance run*. If the gate opens first, WS-5 moves post-gate without
discussion.

- [ ] 5.1 Add the four LLM families to `docs/grafana_dashboard.json` — verified genuinely
      missing. **[FIXED]** Use registry base names (`{ns}_llm_tokens`,
      `{ns}_llm_gateway_served`, `{ns}_llm_latency_budget_exceeded`, histogram
      `{ns}_llm_gateway_latency_ms`); `_total` is only the Prometheus counter render suffix
      — panel exprs must match what the panel→sample test
      (`tests/unit/test_grafana_dashboard_json.py:103-141`) resolves.
- [ ] 5.2 **[FIXED — shrunk]** `config/prometheus/alerts.yml` exists and
      `promtool check rules` already runs in CI (`ci.sh:92`, `ci.yml:256`). New work = LLM
      alert *rules* only.
- [ ] 5.3 Reverse advisory check (emitted-but-not-dashboarded WARN) → **post-gate** (pure
      hygiene tooling; fails the plan's own scoping rule pre-gate).
- [ ] 5.4 Feature **F-019** (epic `Observability`).

## WS-6 — HARDWARE GATE (expanded from one line into the real work stream)

- [ ] 6.0 **Repair work stream:** diagnosis checklist (power chain probe via
      `src/mousedroid/diagnostics/power_chain.py`, UART loopback, `scripts/flash_esp32.sh`
      reflash); **time-box: 2 bench sessions**; then repair-vs-replace decision
      [CONFIRM-FIRST #4 — replacement driver board identified and priced *now*, not after
      the time-box]; **firmware provenance** — record which .bin was flashed (path + hash)
      alongside F-008's `implemented_in` (the harness's provenance discipline currently
      stops at Python).
- [ ] 6.1 **+45-day gate review:** if hardware still isn't up, the freeze itself is
      re-evaluated — an unbounded stall is not a plan.
- [ ] 6.2 Probe-first bring-up per `docs/runbooks/jetson-full-bringup.md` — unchanged
      (verified). Expect a **defect inventory** (camera Stage-12 annotation), not a single
      blocker; exit requires criticals triaged to zero, not just ESP32.
- [ ] 6.3 Full validation with trend journaling (whatever of WS-4 has landed — the gate
      never waits for it).
- [ ] 6.4 `scripts/validate.py --tier hardware` → F-008 `done` with run SHA — verified
      convention.
- [ ] 6.5 Endurance run (`MOUSEDROID_ENDURANCE_FORCE_REAL=1`); **[FIXED]** snapshot
      explicitly seeds the trend-journal baseline (rev. A implied but never wired the loop
      that WS-7.3's "re-evaluate with real trend data" depends on).

## WS-7 — Post-gate

- [ ] 7.1 30-day clock **anchored to the deployed SHA's continuous uptime, restart on
      redeploy** [FIXED: calendar-anchored clock is theater when WS-6 explicitly
      anticipates defect-fix redeploys]. Record in features.yaml notes.
- [ ] 7.2 **[FIXED — issue #109 is done]** Issue #109 closed 2026-06-14; `fire_on_startup`
      + pr109 integration/hardware tests are in-tree. The *remaining* #109 tail is exactly
      post-gate shaped: run the hardware-tier greeting test on the live rover and decide
      the `fire_on_startup` default flip (its own risk note requires live-rover
      verification). Second candidate: smoke-finding **F-010 (VLM progress is a constant
      mock — still open)**. Start after the **first 7 clean soak days**, not at gate-exit.
- [ ] 7.3 Unfreeze re-evaluation at clock+30d — unchanged.

## WS-8 — Redundancy/gap audit (claims corrected; continuous, findings-only, deletion prohibited)

- [ ] 8.1 vulture/ruff advisory stage — unchanged (verified absent today; F401/F811 already
      active via ruff's `F` family, so scope the new signal to what ruff misses).
- [ ] 8.2 Import-graph freeze check — **[FIXED] specify granularity:** module-top-level
      assertions (the arm platform is imported by `factory.py` only lazily/config-gated
      inside functions; a grimp-style check following function-local imports would fail
      today). Also note HC-SR04 "parked" is doc-status only — `config/default.yaml:63-70`
      ships a populated `ultrasonic:` block that `jetson_production.yaml` inherits; if
      parked is meant literally, that's a config change (surface as a finding, don't apply).
- [ ] 8.3 Promotion-lag check — **[FIXED]** the two current examples are `jetson-nightly`
      and `onnx-world-model-extras` (NOT `[vla]`, which is already promoted/blocking as
      Tier C3.1).
- [ ] 8.4 Feature **F-020** (epic `Hygiene`, tier slow) — audit pipeline planted-defect
      test unchanged.

## Doctrine corrections

- **Coverage:** the gate is **85% line** (`--cov-fail-under=85`); branch coverage is
  measured nowhere — `scripts/check_branch_coverage.py` never passes `--cov-branch` despite
  its name. State "≥85% line"; optionally file a WS-8 finding to either enable branch
  measurement (will need a ratchet — enabling it drops numbers) or rename the script. Don't
  silently claim "line+branch".
- **"100% on new pure modules"** is aspiration, not precedent — no per-module threshold
  exists (spec.py's 100% is unverified). Keep as a goal, drop the "precedent" citation.
- **Bash testing:** the "pytest-subprocess pattern (`test_smoke_harness.py` precedent)"
  doesn't exist — that file does source-text assertions. Either establish the subprocess
  pattern in WS-4.2 or amend the doctrine to "source-assertion tests + extracted-Python
  helpers for logic."
- Logging doctrine unchanged (survived review intact).

## CONFIRM-FIRST register (revised)

| # | Decision | Location |
|---|----------|----------|
| 1 | T3/arm: pause-at-T2 vs un-defer | WS-1.2 |
| 2 | Close stale PRs #26/#30/#48 (+ delete 2 stale claude/* branches) | WS-2.1 / WS-1.7 |
| 3 | Container user-mapping vs chown-probe fallback | WS-3.2 |
| 4 | ESP32 repair-vs-replace after 2-session time-box | WS-6.0 |
| 5 | Key rotation timing vs consumer inventory completeness | WS-0.1 |

(rev. A's #4 — doc-hygiene thresholds — deleted; advisory WARNs don't need a decision
ceremony.)

## Verification

- **Of the review:** every `[Certain]` verdict carries file:line evidence from the
  fact-check agents or direct GitHub API results.
- **Of this plan on execution:** each WS lands with its feature entry (F-015…F-020) whose
  `validation_command` must exit 0 under `scripts/validate.py` — the repo's own Golden Rule
  is the acceptance mechanism. WS-6 exit = F-008 `done` + defect inventory at zero criticals
  + endurance snapshot seeding the trend baseline.
- **Immediate sanity checks after adopting rev. B:** `python scripts/select_next.py` still
  returns F-008; `pytest tests/regression/test_harness_spec_aqa.py` green after any
  features.yaml edits; `bash scripts/ci.sh` green before any PR.
