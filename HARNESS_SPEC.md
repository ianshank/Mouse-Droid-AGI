# HARNESS_SPEC.md — MouseDroidAGI

**Spec-Driven Development & Validation harness — Harness Engineering v2.1**

> Adopted from `PROJECT_HARNESS_SPEC_TEMPLATE.md` (v2.1). See `docs/architecture/ADR-012-spec-driven-harness.md`
> for the adoption decision and the deliberate deviations from the upstream template
> (ADRs live in `docs/architecture/`, not `docs/decisions/`; `F-001` is validated by a
> standalone script rather than a self-referential `--check`).

The single rule that matters: **`features.yaml` is the source of truth only because
`scripts/validate.py` enforces it.** A feature is `done` only when its
`validation_command` exits 0 under the runner — there is no hand-set `passes` flag to
game.

---

## 1. Project Metadata

```yaml
project:
  name: "MouseDroidAGI"
  slug: "mousedroid"
  version: "0.3.0"
  status: "active"          # planning | active | validated | archived
  owner: "Ian Cruickshank"
  harness_version: "2.1"
  repo: "https://github.com/ianshank/mouse-droid-agi"
  # No last_updated — git is the timestamp authority.
```

**Tags:** ai-agent, spec-driven, harness, robotics, jetson, agentic-world-model

---

## 2. Executive Intent (The Seed)

MouseDroidAGI is a Star Wars MSE-6 autonomous navigation system and hierarchical
robot-arm training platform on a Jetson Orin Nano, implementing the 10 Pillars of the
Ideal Neural Network as a cohesive agentic system. "Shipped and valuable" means the
rover senses, plans, and acts on a deterministic 30 Hz reactive loop, with a
deliberative cloud/local LLM brain translating natural language into goals *outside* the
hot loop, all under a constitutional safety monitor — and that every claim of "done" is
backed by an executable check, not a human assertion.

This harness exists because an autonomous, agent-developed codebase drifts the moment
completeness is inferred from code presence. The catalog (`features.yaml`) plus the
runner (`scripts/validate.py`) make completion mechanically checkable, DAG-ordered, and
CI-gated, so agents and humans share one ground truth.

**Why this matters:** the rover ships behaviour that controls physical hardware. A false
"green" is not a cosmetic bug — it is a robot acting on an unvalidated assumption. The
harness converts "I think this works" into "the command exits 0."

---

## 3. Success Definition & Done Criteria

**Project-level Definition of Done:**

- All `critical` and `high` priority features have `status: done`.
- `python scripts/validate.py --tier fast,slow` exits 0 (hardware tier proven on the rover).
- This spec + ADRs (`docs/architecture/`) + runbooks (`docs/runbooks/`) are current and agent-legible.
- Observability covers all critical paths (structlog events + Prometheus families).
- The existing CLAUDE.md architecture invariants (Protocol DI, factory wiring, no hardcoded values, structlog, asyncio, `mypy --strict`, 85% coverage) hold.

**Non-goals / Out of Scope:**

- This harness does not replace `scripts/ci.sh` or the 6-stage `ci.yml`; it is an
  additional spec-alignment gate layered on top.
- Multi-agent parallel development (opt-in only — see §9 Concurrency).

---

## 4. Harness Architecture & Principles

- **The harness enforces; the document only describes.** Any rule not backed by a script or CI gate is advisory and will drift.
- **Explicit specification beats inference.** Agents must not infer completeness from code presence.
- **Ground truth lives in structured, *validated* artifacts** — `features.yaml` checked against `features.schema.json`.
- **Cognitive load management:** scoped tools, capped outputs, structured lookups (`select_next.py`) over raw grep.
- **Reversible handoffs.** Every session ends clean and committed.
- **Feedback loops match the domain:** unit + integration + a domain-appropriate end-to-end check before `done`. Map cost to `tier`.
- **Durable rationale outlives the chronological log.** Permanent decisions go to ADRs in `docs/architecture/` (§11).

---

## 5. Feature Catalog (`features.yaml`)

`features.yaml` is the cognitive anchor. **There is no `passes` field** — completion is
whatever `validate.py` reports when it runs `validation_command`. CI checks both the
schema half (a `done` feature must declare a command + provenance) and the runtime half
(the command actually passing).

### Field reference

| Field | Owner | Notes |
|-------|-------|-------|
| `id` | human/agent | Unique, sortable. `F-001` or `EPIC-F-001`. |
| `epic` | human | Grouping label. |
| `name` / `description` | human/agent | What the feature is. |
| `category` | human | `functional` \| `non-functional` \| `infrastructure` \| `validation` |
| `priority` | human | `critical` \| `high` \| `medium` \| `low` |
| `status` | agent | `todo` \| `in_progress` \| `done` \| `blocked` \| `deferred` |
| `tier` | human | `fast` \| `slow` \| `hardware`. Controls *when* the command runs. Defaults to `fast`. |
| `verification` | human/agent | **Required, ≥1 entry.** Human-readable checks — the intent behind the command. |
| `validation_command` | human/agent | Shell command (prefer a script path) that exits 0 iff the feature works. **Required** before `done`. |
| `implemented_in` | agent | Commit SHA or branch. Set when `done`. `validate.py` verifies it resolves. |
| `depends_on` | human/agent | Feature IDs that must be `done` first. The DAG. |
| `notes` | agent | Free text. |

**Prefer script files over inline shell.** Complex checks live in
`scripts/validations/F-XXX.sh` (a real, testable editing surface). Reserve inline
commands for trivial one-liners (`python -m pytest path -q`, `test -f ...`).

There is **no `blocks` field** — it is the inverse of `depends_on`; reverse edges are computed when needed.

### Editing rules

- Agents may freely change `status`, `implemented_in`, `notes`, and append to `verification`.
- **Structural changes** (add/remove/rename/re-scope/reorder) are allowed, but each must be logged as a `progress.md` entry with a one-line rationale. Permanent technical decisions also get an ADR (§11).

### F-number namespaces

Two **independent** F-number sequences coexist in this repo and must never be
conflated:

1. **The harness catalog** (`features.yaml`, this spec) — the only IDs in the
   dependency DAG, validated by the schema, executed by `validate.py`.
2. **Operational smoke findings** (`SMOKE_REPORT.md`, echoed in CHANGELOG /
   older planning docs) — report-local triage IDs from the live-Jetson smoke
   campaigns, which independently reached `F-014`.

Because the findings sequence already burned `F-009`–`F-014` (e.g. commit
`3015283` "F-013/F-014 closeout" refers to *findings*, not catalog features),
**new catalog entries continue from `F-015`** — the catalog deliberately skips
9–14 rather than shadowing them. When cross-referencing, say "smoke finding
F-0xx" vs "feature F-0xx".

---

## 6. `features.schema.json`

JSON Schema draft 2020-12 at the repo root. Validates structure; `verification` is
required and non-empty; a `done` feature must carry a real `validation_command` and
`implemented_in`. `verification: minItems 1` is a forcing function — name a check before
you start. The command actually passing is enforced at runtime by `validate.py`.

---

## 7. `scripts/select_next.py`

Picks the next feature **respecting the dependency DAG** — it will not select a feature
whose `depends_on` are unmet. In-progress work is resumed first; otherwise the
highest-priority ready `todo` wins. Run it; do not eyeball `features.yaml`.

---

## 8. `scripts/validate.py` + CI

Validates structure against the schema, checks DAG integrity (dangling edges + cycles),
verifies `implemented_in` resolves to a real git commit, and runs the
`validation_command` of every `done` feature **in the selected tier(s)**.

`scripts/validate.py` and `scripts/select_next.py` are **thin CLI shims** — the
enforcement logic lives in the importable, unit-tested package module
`src/mousedroid/harness/spec.py` (mirroring how `cli/preflight.py` wraps
`validation/preflight.py`). This keeps the harness guarantees under the project's
85% coverage gate (`tests/unit/harness/test_spec.py`) rather than untested in a script.

```bash
python scripts/validate.py --tier fast               # inner loop / every push
python scripts/validate.py --tier fast,slow          # nightly
python scripts/validate.py --tier hardware           # on the rover only
python scripts/validate.py --check F-005             # single feature, any tier
```

**CI gate:** `.github/workflows/harness.yml`. `fetch-depth: 0` so `git rev-parse` can
resolve `implemented_in`. The push job installs `.[dev]` (validation commands delegate
to `pytest`) and runs `--tier fast`; the nightly job adds `--tier fast,slow`.

**Deviations from the upstream template (recorded in ADR-012):**

- The push job runs **without `--strict-git`** (warn-only) — feature-branch refs are
  brittle pre-merge. The **nightly** job uses `--strict-git`. **Post-merge maintenance:**
  replace any branch-name `implemented_in` (currently `F-001`, `F-003`) with the
  squash/merge SHA so the nightly strict-git job stays green on `main`.
- The harness CI installs the full dev toolchain rather than only `pyyaml jsonschema`,
  because the seeded validation commands run real pytest suites.
- The `hardware` tier is intentionally absent from hosted CI — run it on the self-hosted
  Jetson runner: `python scripts/validate.py --tier hardware`.

---

## 9. Agent Session Protocol

### Startup sequence (every session)

1. `pwd` — confirm working directory.
2. Read `HARNESS_SPEC.md` (this file).
3. Read the top of `progress.md` + `git log --oneline -10`. Read relevant ADRs in `docs/architecture/` when touching an area with prior decisions.
4. Run `python scripts/select_next.py` to pick the feature. **Do not eyeball `features.yaml`.**
5. Run `scripts/init.sh` to reach a known-good baseline.
6. Run `python scripts/validate.py --tier fast`. If it fails, **fix the baseline before new work.**
7. Begin work on the selected feature.

**Copy-paste agent prefix:**

```
You operate inside an enforced harness. Read HARNESS_SPEC.md, the latest progress.md,
and any relevant ADRs in docs/architecture/. Run scripts/select_next.py to pick the
feature — do not choose manually. Run scripts/validate.py --tier fast; if it fails,
fix that first. Work ONE feature at a time. You cannot mark a feature done unless its
validation_command passes under validate.py. Record permanent technical decisions as
an ADR. End in a clean, committed state.
```

### Session rules

- **One feature per session by default.** Parallel work is opt-in (Concurrency, below).
- After every meaningful change, run the relevant validation level (§10).
- Write complex checks to `scripts/validations/F-XXX.sh`, not inline YAML.
- **Before ending a session:** update `status`/`implemented_in`/`notes`; run
  `python scripts/validate.py --tier fast` (must pass); append a `progress.md` entry
  (newest on top); add/append an ADR if a permanent decision was made;
  `git add -A && git commit -m "feat(F-XXX): <desc>"`.

### Concurrency (opt-in)

Each agent works in its own git worktree on its own branch with that branch's
`features.yaml`; status changes merge back via PR. No shared-file live editing.

### Context guardrails

Prefer `select_next.py` / `features.yaml` lookups over raw grep. Summarize older turns explicitly when history grows large.

---

## 10. Validation Levels

| Level | Purpose | When | Example tools |
|-------|---------|------|---------------|
| Lint / Syntax | Mechanical errors | Every edit | ruff, mypy |
| Unit | Isolated logic | Implementation | pytest |
| Integration | Component interaction | Cross-feature | factory wiring tests |
| E2E / Domain | Real-world flow | **Before `done`** | orchestrator e2e, hardware-in-the-loop |
| Spec alignment | Catalog ↔ reality | Continuous (CI) | `scripts/validate.py` |

Map cost to `tier`: cheap lint/unit → `fast`; E2E and long integration → `slow`; physical target → `hardware`.

**Golden Rule.** A feature is `done` only when: its `validation_command` exits 0 under
`validate.py`; `implemented_in` references a resolvable commit; and `progress.md`
references the evidence. The command passing *is* the truth.

---

## 11. Progress Log + Architecture Decision Records

### `progress.md` — chronological session log

Newest entry on top. Set the date with `date +%F`; never copy a literal date.
**Rotation:** keep ~10 sessions; move older entries to `progress-archive/YYYY-QN.md`.

### `docs/architecture/` — ADRs (durable rationale)

Permanent technical decisions are append-only ADRs that persist independently of the
session log: `docs/architecture/ADR-NNNN-short-title.md` (this project already uses
ADR-004…ADR-011; the harness adds **ADR-012**). ADRs are never rotated; they are
superseded in place.

---

## 12. Known Failure Modes & Mitigations

| Failure mode | Mitigation |
|--------------|-----------|
| Agent declares victory on partial work | `validate.py` runs the real command; CI fails on mismatch. |
| Completeness inferred from code presence | A feature without a passing `validation_command` is not done. |
| State drift (status vs reality) | Single completion signal, checked in CI. |
| Picking a blocked feature | `select_next.py` filters by satisfied `depends_on`. |
| Broken baseline compounds | Mandatory `validate.py --tier fast` in startup; fix-before-new-work. |
| DAG rot (dangling/cyclic edges) | `validate.py` + `test_harness_spec_aqa.py` check edges and cycles. |
| CI/startup bottleneck as suite grows | `tier` gating: `fast` on every push, `slow`/`hardware` deferred. |
| Mangled inline shell in YAML | Checks live in `scripts/validations/*.sh`. |
| Fake/typo'd commit refs in `implemented_in` | `git rev-parse` provenance check (warn, or error under `--strict-git`). |
| Lost rationale after log rotation | ADRs persist decisions in `docs/architecture/`. |
| Parallel agents corrupting state | Per-branch `features.yaml`, PR merges. |

**Self-improvement:** after a stuck session, ask which guardrail would have caught it, and add it.

---

## 13. Architectural Invariants

Non-negotiable rules, mechanically checked where possible (see CLAUDE.md for the full set):

- Protocol-based DI; concrete types only in `src/mousedroid/factory.py` — enforced by review + factory tests.
- No raw `print()` in `src/mousedroid` production paths — structlog; ruff scope.
- No hardcoded values — `scripts/check_no_hardcoded_values.py` (AST gate) + config validation (`F-002`).
- `mypy --strict` passes; 85% coverage gate — `scripts/ci.sh`.
- Config backwards compatibility — `config-compat` gate + `F-002`.

---

## 14. Tooling, Observability & Environment

Runtime: Python ≥3.10, `pip`. Toolchain: ruff 0.8.0 (pinned), mypy 2.1.0 (pinned),
pytest. Observability: structlog structured events + Prometheus families. ADRs in
`docs/architecture/`. **`scripts/init.sh`** is idempotent and fast: `pip install -e
".[dev]"`, then a `validate.py --tier fast` health check ending in `baseline ready`.

---

## 15. References

- **SWE-agent (Princeton NLP), arXiv:2405.15793** — the agent-computer interface (ACI).
- **OpenAI — Harness Engineering** — origin of the term.
- **Anthropic — Claude Code / agent patterns** — initializer agent, structured ground truth, progress files, clean-state handoffs.
- **awesome-agent-harness** — `github.com/AutoJunjie/awesome-agent-harness`.

---

## 16. Changelog

| Version | Change |
|---------|--------|
| 2.1 (adopted) | Adopted the v2.1 template into MouseDroidAGI: schema + tier-gated `validate.py` + DAG-aware `select_next.py` + standalone `harness.yml`. Seeded 8 features mapping to real checks. ADRs reuse `docs/architecture/`. `F-001` validated by `scripts/validations/F-001.sh` (avoids the template's recursive `--check`). See ADR-012. |
| 2.1 + F-namespaces | Declared the two independent F-number sequences (catalog vs SMOKE_REPORT findings); catalog continues from F-015, skipping the finding-burned 9–14. Landed with the rev. B software work streams (F-015..F-020, PR #151). |
