# Tasks — mouse-droid-claude-workforce (rev. B)

Order is binding. Every task lands green (`ruff check`, targeted pytest,
`tools/validate_skill_commands.py`, and — where `features.yaml` is touched —
`scripts/validate.py --tier fast`) before the next starts. Status ticks are updated as
work lands. Deviations from this wording are recorded inline — declared, not silent.

**Declared interpretation (layout):** tasks are grouped by bold phase labels with `N.M`
numbering, matching the flat-checkbox house style of the memory-distill bundle; phase
labels are navigation, not structure.

> **Progress.** Phases 1–2 have landed (config + AQA foundation, all three
> hooks, CI wiring, `features.yaml` F-024 reserved as `in_progress`). Phase 0.1
> is folded into the peer review's inventory rather than a separate surfaces
> doc, since `docs/claude/surfaces/` does not exist until Phase 6 — declared
> here rather than silently skipped. Phases 3–6 remain open.

**Phase 0 — ground truth (no behavior changes)**

- [x] 0.1 Baseline inventory — captured in `peer-review.md` ("Verdict table" +
      "Load-bearing pins inventory"). **Deviation:** the original task wrote this
      to `docs/claude/surfaces/00-baseline-inventory.md`, but that directory is
      created in Phase 6; duplicating the inventory would have created two
      sources of truth for the same facts. Covers the `.claude/` census (three
      skills + frontmatter proof), the fourth skill at
      `.github/skills/jetson-hardware-debug/SKILL.md`, the absence of
      agents/hooks/`.mcp.json` at the basis commit, and the load-bearing test
      pins. Becomes `doc-reconciler`'s baseline.
- [x] 0.2 Reserve **F-024** in `features.yaml`, epic `Hygiene`,
      `implemented_in: null` — schema-valid, harness fast tier green.
      **Deviation:** recorded as `status: "in_progress"` rather than `"todo"`,
      because Phases 1–2 landed in the same change; `"todo"` would have
      understated the state. (F-009–F-014 are burned per ADR-013; never reuse.)
- [x] 0.3 Flip this change's row in `openspec/project.md` from `proposed` to `in progress`.

**Phase 1 — config + AQA foundation (everything else depends on it)**

- [x] 1.1 `tools/claude_hooks/__init__.py` + `tools/claude_hooks/config.py`
      (`WorkforceConfig`, Pydantic v2, `extra="forbid"`, range-validated) +
      `.claude/workforce.yaml` (D-2 keys). Tests: valid load, unknown-key rejection,
      range violations. No literal thresholds anywhere else.
      **Addition:** three reusable primitives the design implied but did not name —
      `paths.py` (repo-root resolution + separator-respecting glob matching),
      `logging_setup.py` (stderr-only structured logging, structlog-or-fallback),
      `hookio.py` (hook stdin/stdout protocol), plus `portability.py` for the AQA rule.
- [x] 1.2 `tests/regression/test_claude_workforce_aqa.py` + `scripts/ci.sh` stages
      (workforce-config parse, hook `mypy --strict`, dedicated hook coverage) +
      `TestCiSh` presence-pin additions.
- [x] 1.3 `.github/workflows/ci.yml` lint step gains `tools/` (closes the ci.sh↔ci.yml
      ruff-scope divergence).

**Phase 2 — hooks (mechanical governance)**

- [x] 2.1 `tools/claude_hooks/secret_scan.py` + tests (deny / allow / timeout /
      absent-binary warn+allow / `strict: true` deny). **Deviation:** the scanner is
      exercised through a synthesised stub executable rather than a marker-gated real
      `gitleaks` run — deterministic on any host, and it covers the exit-code contract
      the real binary implements.
- [x] 2.2 `tools/claude_hooks/freeze_gate.py` + tests (F-008 `todo`→deny,
      `in_progress`→deny, `done`→allow; missing-key/malformed catalog→fail-closed deny;
      override-env allowed + logged; glob matching from config only).
- [x] 2.3 `tools/claude_hooks/post_edit_check.py` + tests (report-only; never blocks).
- [x] 2.4 Wire all three into `.claude/settings.json` **additively** (`hooks` block only;
      per-hook `timeout`). **Deviation:** commands are
      `cd "$CLAUDE_PROJECT_DIR" && python3 -m tools.claude_hooks.<module>`, not a direct
      path to the module file — running the file by path leaves the repo root off
      `sys.path` and every hook fails with `ModuleNotFoundError`. Verified by hand and
      pinned by the AQA test. Operator guide: `docs/runbooks/claude-workforce-hooks.md`.

**Phase 3 — subagents (roster reviews its own successors from 3.1 on)**

- [ ] 3.1 `.claude/agents/peer-reviewer.md` first (exemplar, D-4) — reviews every
      subsequent task's diff.
- [ ] 3.2 `security-scanner`, `config-guardian`.
- [ ] 3.3 `openspec-author`, `test-engineer`.
- [ ] 3.4 `doc-reconciler`, `hw-evidence-auditor`. First auditor run attached to the PR —
      expected findings: the 2026-07-12 evidence gap (stays deferred) and the README
      coverage claim (fixed at 6.5).
- [ ] 3.5 AQA green over all seven: platform-supported frontmatter keys, bare tool names
      (reject `(` / `*` tokens), each file ≤ 60 lines.

**Phase 4 — skills (existing three byte-untouched)**

- [ ] 4.1 Verify-only proof: `git diff` shows zero changes to
      `.claude/skills/{sim-test,robot-arm-trainer,train-policy}/SKILL.md`;
      `test_next_steps_reconciled.py` + skill validator green.
- [ ] 4.2 New skill `openspec-change` (active).
- [ ] 4.3 New skill `coverage-gate` (active) — dedicated tools-coverage invocation +
      advisory branch report + PR delta table.
- [ ] 4.4 New skill `evidence-commit` (active) — tracked-artifact path AND declared
      local-only chain path per D-4/evidence config.
- [ ] 4.5 New skill `worktree-flow` (active).
- [ ] 4.6 New skill `jetson-smoke` — documents both the RUN-MOTION consent phrase and the
      `MOUSEDROID_SMOKE_ALLOW_MOTION` mechanical gate; Tier-3 section `frozen` with the
      typed-consent unfreeze condition.
- [ ] 4.7 `.github/skills/jetson-hardware-debug` disposition: recommended
      placeholder-swap of both IPv4 literals (real values live in
      `docs/runbooks/claude-code-on-jetson.md`) then widen the AQA sweep to that
      directory; fallback = declared exclusion with reason in the AQA docstring.
      All new skills are validator-clean at authoring time (backtick paths exist).

**Phase 5 — MCP + worktrees**

- [ ] 5.1 `.mcp.json`: `mousedroid` server per `docs/MCP_OPERATOR_GUIDE.md` (+
      `MOUSEDROID_MOCK_HARDWARE` expansion default `true`) + GitHub MCP (token via env
      expansion; no literals).
- [ ] 5.2 Tick the `.mcp.json` checkbox at `docs/MCP_NEXT_STEPS.md:51`.
- [ ] 5.3 `docs/runbooks/worktrees.md`; dogfood — Phase 6 is executed inside a worktree
      created by `worktree-flow`.
- [ ] 5.4 Evaluate-first notes for Grafana/HF MCP committed under
      `docs/claude/surfaces/` (decision recorded; servers NOT added).
- [ ] 5.5 Smoke: one read-only GitHub-MCP call from a session; transcript in the PR.
- [ ] 5.6 AQA: `.mcp.json` parses, is secretless (gitleaks-clean), and the `mousedroid`
      entry matches the operator guide.

**Phase 6 — docs consolidation (LAST, so it documents reality)**

- [ ] 6.1 Root CLAUDE.md trim to `docs.core_max_lines`: evergreen sections + corrected
      CI-pipeline section + orchestration directive (I-6) + freeze rule + surface map.
- [ ] 6.2 Nested per-directory CLAUDE.md files for subsystem-scoped surface contracts.
- [ ] 6.3 `docs/claude/surfaces/` + index for cross-cutting surfaces; `doc-reconciler`
      verifies zero broken references.
- [ ] 6.4 AGENTS.md dedupe on its declared worker-contract axis.
- [x] 6.5 README.md truth-fix: "85% branch coverage" → "85% line coverage" (branch
      is measured + reported for `tools/claude_hooks/` only, and stays advisory until
      its own promotion change). **Deviation:** landed with Phases 1–2 rather than in
      the docs phase — the claim was false the moment branch measurement arrived, and
      the dev-governance spec's truthful-claims requirement forbids leaving it.
- [ ] 6.6 Extended banned-token AQA sweep over nested CLAUDE.md + surfaces (repo slug
      allowlisted); `test_portfolio_reframe_aqa.py` stays green.
- [ ] 6.7 Dedicated tools-coverage stage live in `scripts/ci.sh`
      (`--cov=tools/claude_hooks --cov-branch`, line gate mirrored from
      `coverage.tools_line_min`; branch advisory) + `TestCiSh` pin additions.
- [ ] 6.8 Close out: F-024 → `done` (`validation_command` set, `implemented_in` set);
      registry row → `implemented`; CHANGELOG entry.

**Explicitly deferred (separate changes, do not fold in)**

- History purge (`git filter-repo`) + repo rename.
- Commit/backfill the 2026-07-12 on-device validation evidence.
- On-device measurement of the 30 Hz loop-rate target.
- Branch-coverage threshold promotion advisory → blocking.
- gitleaks CI job promotion advisory → blocking (7-green-run tracker in
  `.github/advisory_stages.yaml`).
- Grafana / Hugging Face MCP adoption.
- Any F-02x capability work (gated on F-008 by the freeze-gate hook this change ships).
- WS-F7b foundry adoption (separate plan; coexistence declared in `proposal.md`).
