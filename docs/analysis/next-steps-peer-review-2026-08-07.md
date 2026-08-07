# Next-steps peer review (line-level verification, 2026-08-07)

Independent verification of two prior planning inputs against the actual source tree
and CI history:

- the **vendor-documentation audit** ([`rover-jetson-integration-gaps.md`](rover-jetson-integration-gaps.md),
  PR #182) — every R/J finding re-checked at file:line granularity;
- the **prioritized list in `NEXT_STEPS.md`** as it stood at commit `5556a97` — checked
  for claims that contradict the git record or the repo's own governance tooling.

Findings only — the companion reconciliation change rewrites `NEXT_STEPS.md` and adds
catalog entries; this document records *why*, with evidence, so severities are
inherited correctly rather than re-derived. Severity is impact on the rover working,
matching the audit's convention.

---

## 1. Verification table — vendor audit claims

| Claim | Verdict | Key evidence |
| --- | --- | --- |
| R1 protocol mismatch | **Confirmed — worse than stated** | see §1.1 |
| R2 baud 1 M vs 115 200 | Confirmed | `config/jetson_production.yaml:32`; schema default `src/mousedroid/config/schema.py:408` |
| R3 encoder assertion unsatisfiable | **Overstated — downgraded High → Medium** | see §1.2 |
| R4 IMU + magnetometer unwired | Confirmed (strongest form) | zero word-boundary hits for `imu\|gyro\|accelerometer\|magnetometer\|qmi8658\|ak09918` across `sensing/`, `hardware/`, `comms/` |
| R5 battery protection disabled | Confirmed | `config/jetson_production.yaml:13-14` zeroes `battery_critical_v` / `battery_warn_v` |
| R6 no chassis heartbeat failsafe | Confirmed — with precision | see §1.3 |
| J1 `power_mode` dead config | Confirmed | only reads are `schema.py:613`, YAML overlays, and `tests/unit/test_config_schema.py` value asserts; `nvpmodel -m 0` hardcoded at `scripts/jetson_system_setup.sh:112` |
| J2 `dla_enabled` | **Corrected — not inert** | consumed at `src/mousedroid/efficiency/tensorrt.py:133,390,410`; residual risk is only that `true` assumes an unverified DLA. Low |

### 1.1 R1 is aggravated by a missing firmware artifact

- `src/mousedroid/comms/base_driver.py:139` — `get_battery_voltage()` sends
  `{"T": ESP32_CMD_TYPE_BATTERY}` (`{"T":2}` = stock `CMD_SET_MOTOR_PID`, a **write**)
  and parses `data.get("v", 0.0)`, so a non-answering firmware yields **0.0 V
  silently** — indistinguishable from a flat pack reading.
- `src/mousedroid/diagnostics/power_chain.py:51,60,62` — probe order is battery →
  `send_velocity` → `emergency_stop`, i.e. the PID-write lands immediately before
  motion is commanded.
- `scripts/flash_esp32.sh:3` documents `firmware/waverover_mousedroid.bin` as its
  usage example; **`firmware/` exists nowhere in the tree** and no firmware source is
  committed. Any board attached post-repair runs stock firmware and ignores the
  private protocol. The retarget (catalogued as **F-025**) is therefore mandatory, not
  optional, for the F-008 path.

### 1.2 R3 downgrade — the encoder assertion is double-guarded

`tests/hardware/test_motor_smoke.py:76-79`: the motion-quality assertion runs only
under `if smoke_test_allow_motion and not mock_hardware`, and
`ESP32Config.smoke_test_allow_motion` defaults `False`. On every default path the
check never executes — it is **not** a "permanent red" blocking F-008 today. It *is* a
latent trap that fires in exactly the F-008 closing scenario (operator opts into
motion on an encoder-less WAVE ROVER chassis), so the scope decision — re-scope to
"command ACKed + e-stop within budget", or source odometry from the IMU (R4) — is an
acceptance criterion of F-025, not an independent blocker.

### 1.3 R6 precision — the existing watchdog cannot stop wheels

Every "heartbeat" in the repo is the *software liveness* surface:
`src/mousedroid/health/watchdog.py` (systemd/file notifiers), pinged per successful
tick at `src/mousedroid/orchestrator/orchestrator.py:2020-2022`. It proves the loop is
alive to the *host*; nothing arms the chassis-side failsafe
(`CMD_HEART_BEAT_SET` = 136), so a wedged Jetson or dropped USB link leaves the last
velocity command executing indefinitely.

---

## 2. `NEXT_STEPS.md` claims contradicted by the record

### 2.1 Coverage-gate history stated backwards (item 0)

Item 0 claimed *"93.01% line coverage (gate promoted to 93%)"*. `git show 5556a97`
(PR #182) shows the inverse:

```
pyproject.toml:  -fail_under = 93          →  +fail_under = 90
ci.yml:          -MINIMUM_COVERAGE: "85"   →  +MINIMUM_COVERAGE: "90"
```

Read together: before #182 the repo carried **two disagreeing gates — CI enforcing 85
while pyproject advertised 93**. #182 unified every gate literal at 90 (a genuine
fix), and `tests/regression/test_coverage_gate_single_source.py` now pins the
equality. The 93.01% figure survives as a *measurement* from the 2026-07 validation
campaign; it was never an enforced gate. The reconciliation rewrites item 0
accordingly.

### 2.2 Overdue advisory promotion invisible in the plan

`python3 scripts/check_advisory_promotions.py` (run 2026-08-07):

```
WARN: promotion overdue: job 'gitleaks' (ci.yml) has been advisory 34 days
      (window 30d since 2026-07-03)
```

CI evidence: the `gitleaks` job is green on the most recent runs **including** the
2026-08-06 failing run `31060761758` (only `test (3.11)` failed there, on a
dependabot `mcp` bump; every advisory job passed). The gate-wiring AQA
(`tests/regression/test_ci_gate_wiring_aqa.py`) pins `performance` and `security` as
advisory but computes the tracked-set check dynamically — promoting gitleaks is
mechanically safe. Sharpest framing: the roadmap's P0 item is an unrotated leaked
`ANTHROPIC_API_KEY`, while the gate built to prevent recurrence is still non-blocking,
past its own deadline. The companion change promotes it.

### 2.3 Minor drift

- Two items were numbered 9 (F-023 follow-ups and the demo clip).
- `CLAUDE.md` frames `actionlint` as "CI Stage 0" guarding the chain; in `ci.yml` it
  has no dependents (`needs: null` everywhere else points at `lint`) — it runs in
  parallel and does not gate. 15 jobs total, 5 advisory.

---

## 3. New class-level finding: declared budgets without consumers

Two independent instances of "config that looks like enforcement and enforces
nothing":

1. **J1** — `JetsonConfig.power_mode` (`schema.py:613`), read by nothing.
2. **`DocsConfig`** — `tools/claude_hooks/config.py:141-147` declares
   `core_max_lines: 250` and `surfaces_dir: "docs/claude/surfaces"`; no hook consumes
   either field, `docs/claude/` does not exist, and `CLAUDE.md` stands at 941 lines
   against the declared 250. (The F-024 bundle's docs-consolidation phase is the
   intended consumer — until it lands, the field is a silent no-op.)

Two instances is a pattern. Catalogued as **F-026**: wire-or-delete each knob, plus an
AQA test asserting every declared budget/mode field has at least one consumer outside
schema/config parsing.

---

## 4. Positive findings (recorded so they aren't re-audited)

- **Zero** `TODO`/`FIXME`/`XXX`/`HACK` markers across `src/mousedroid/`.
- Suppressions exactly at their ratchet ceilings: 19 `noqa` / 8 `type: ignore`
  (budgets in `tests/regression/test_suppression_budget.py:43-44`). At-ceiling means
  the next torch-adjacent change must fix or argue — the ratchet working as designed.
- 618 test files, 0 `xfail`, 15 skips, 111 `importorskip` gates.
- The advisory-stage tracker itself (`.github/advisory_stages.yaml` +
  `check_advisory_promotions.py`) surfaced §2.2 — the governance loop functioned;
  only the human follow-through was missing.
