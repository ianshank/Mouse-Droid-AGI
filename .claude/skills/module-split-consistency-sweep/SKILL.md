---
name: module-split-consistency-sweep
description: After splitting a flat "god" file into a package, or a god class into mixins, run the 7-check sweep that catches what the split itself doesn't fail loudly on — MRO-shadowed dead code, wrong mock.patch targets, wholesale-copied dead imports, dangling re-exports, stale doc references, and gate dilution. Use before merging any module-split PR.
status: active
---

# Module Split Consistency Sweep

A flat-file-to-package or class-to-mixins split is "extraction only, zero
behavior change" in intent, but the mechanics of extraction create failure
modes that look nothing like the bug you'd get from writing new logic. Ruff,
mypy, and the test suite catch most of them — but not all, and the ones they
miss are silent by construction: they pass CI and still ship a defect.

## Why this exists

The `factory.py` → `factory/` (19 submodules + facade, 20 files total) and
`src/mousedroid/orchestrator/orchestrator.py` → 7-mixin decomposition
(ADR-017) hit six confirmed regressions across two audit rounds — this
session's own follow-up sweep, then a separate independent adversarial peer
review of that sweep's own commit — none of them a logic mistake in the
moved code itself:

1. **MRO-shadowed dead code.** An earlier truncation pass cut
   `orchestrator.py` after the wrong line, leaving 7 lifecycle methods
   duplicated in both `orchestrator.py` and `_lifecycle_mixin.py`. Python's
   MRO silently resolved every call to the `orchestrator.py` copy — ruff,
   mypy, and the existing test suite all passed. Only a coverage read (23%
   on `_lifecycle_mixin.py`, exactly the shadowed line ranges) surfaced it,
   and only a new test that compares the concrete class against its mixins —
   `tests/regression/test_orchestrator_mixin_surface.py::test_concrete_class_defines_only_init_and_tick`
   — turned it into a hard failure. A test that only compares mixins against
   *each other* (the first version of that file) does not catch this class
   of bug.
2. **Wrong `mock.patch` target.** Two tests patched
   `mousedroid.factory.build_cognitive_core` expecting to intercept an
   internal call, but the consuming module's own eager
   `from mousedroid.factory.cognitive import build_cognitive_core` binds a
   separate reference — "patch where it's used, not where it's defined."
   The tests still passed; they just stopped testing anything.
3. **Wholesale-copied dead imports.** All 19 `factory/` submodules inherited
   the original flat file's full `TYPE_CHECKING` import block. Per-file, most
   of it was unused. 1,263 ruff findings, mostly F401.
4. **Stale doc references.** 22 docs (`CLAUDE.md`, `AGENTS.md`,
   `HARNESS_SPEC.md`, `SKILLS.md`, `docs/CHARTER.md`, multiple ADRs and C4
   diagrams, two skill files) still named the pre-split file and, in two
   cases, a literal line number a regression test asserted against directly.
5. **Coverage/hardcoded-value gate dilution.** A large file's blended
   average had hidden an already-under-tested function; splitting it exposed
   the same pre-existing gap as a much bigger percentage swing in the
   smaller resulting file — 9 files newly failed the branch-coverage gate
   with no new untested logic. Confirmed by reading the actual uncovered
   lines before deciding it wasn't a new regression.
6. **Dangling `__all__` entries after a dead-import cleanup.** Fixing
   incident 3's wholesale-copied `TYPE_CHECKING` block (removing it via
   `ruff check --fix`) left two `__all__` string literals —
   `"TYPE_CHECKING"`, `"TypeVar"` — with nothing bound behind them, since
   both names were previously reachable only through that block.
   `from mousedroid.factory import *` raised `AttributeError` at import
   time. Neither `ruff --select F822` nor `mypy --strict` catches a
   dangling `__all__` entry on a plain module (confirmed empirically — both
   passed clean with the bug present), and the facade-completeness test
   this decomposition added checked only one direction: "is every real
   symbol re-exported," never the inverse "does every `__all__` entry
   actually resolve." A second, independent adversarial review of the
   first round's own fix commit is what found it — the first round's own
   `mypy --strict`-clean / full-suite-green claim was true and still missed
   it, because neither tool checks this relationship.

Each of these is invisible to "did the tests pass" and needs its own check.

## The procedure

Run all seven after the mechanical split, before opening the PR:

1. **Subsystem-boundary carve-out check.**
   `python scripts/check_subsystem_boundaries.py` — if the new package is a
   DI composition root (like `factory/`) rather than a real subsystem, it
   needs excluding from `_discover_subsystems()`'s auto-discovery, the same
   way `factory` is excluded today. Run the checker *before* deciding this,
   not on assumption — confirm it actually reproduces a violation first,
   then add the exclusion and confirm it passes.

2. **MRO-shadowing / dead-code duplication check.** For a class-to-mixins
   split, add or extend a characterization test that asserts, on the
   concrete class itself, both: (a) no two mixins (or the concrete class and
   a mixin) define the same method name, and (b) the concrete class defines
   *only* the methods it's supposed to still own. Comparing mixins against
   each other is not enough — the regression above was concrete-class vs.
   mixin, not mixin vs. mixin. See
   `tests/regression/test_orchestrator_mixin_surface.py` for the shape.

3. **Patch-target verification.** Grep the whole test tree for
   `mock.patch(` / `monkeypatch.setattr(` referencing the pre-split module's
   dotted path. For each hit, confirm the target still resolves to the
   module that actually holds the live reference post-split — a consumer
   module's own eager import may now shadow the facade-level name.

4. **Dead-import sweep.** Run `ruff check --fix <new-package>/` immediately
   after the mechanical move, before any manual edits — a wholesale-copied
   `TYPE_CHECKING` block or top-level import list is almost always mostly
   unused per resulting file. Follow with `mypy --strict` to confirm nothing
   genuinely needed was auto-removed. Then check the other direction too:
   if the removed block was the only thing binding a name, and that name
   also appears as a string literal in `__all__` (or another re-export
   list), removing the import strands the `__all__` entry — neither
   `ruff --select F822` nor `mypy --strict` flags a dangling entry on a
   plain module. `python -c "import <pkg> as m; print([n for n in m.__all__
   if not hasattr(m, n)])"` catches it in one line.

5. **Doc-staleness sweep.** This is `.claude/skills/narrative-correction-sweep/SKILL.md`,
   scoped to the old file's path and any symbol:line references — run it
   rather than re-deriving it. Check the constitutional doc first
   (`docs/CHARTER.md` outranks `CLAUDE.md` per this repo's own framing), then
   `CLAUDE.md`, `AGENTS.md`, `HARNESS_SPEC.md`, `SKILLS.md`, subsystem
   `CLAUDE.md` files, and any ADR that cites a symbol from the split file.
   Grep for the literal old filename as a string, not just as a path — a
   regression test can hardcode it as a CLI arg or a dict key, not just a
   backtick reference.

6. **Coverage / hardcoded-value gate exemption — only after reading the
   actual uncovered lines.** A same-PR module split can dilute a blended
   average into a per-file gate failure with no new untested logic. Before
   exempting anything, read the flagged lines and confirm they were already
   uncovered pre-split (`git show <pre-split-ref>:<old-file>` plus the old
   coverage report). If confirmed, extend the existing exemption pattern
   rather than lowering the gate threshold or leaving it failing:
   `ALLOWED_DIR_PREFIXES/ALLOWED_FILES` in
   `scripts/check_no_hardcoded_values.py`, and
   `_ALLOWED_DIR_PREFIXES`/`_is_exempted_from_branch_gate` in
   `scripts/check_branch_coverage.py`. Both keep the real percentage
   visible in output — exemption suppresses the failure, never the number.

7. **Facade / class-surface characterization tests — both directions.** For
   a flat-file → package split, add a `dir()`-equality snapshot test
   asserting the package's public surface matches the pre-split module's
   exactly (see `tests/unit/factory/test_facade_completeness.py` — walk the
   AST forward, don't hardcode a snapshot list, or the test stops catching
   *future* drift the moment it's written). That alone only proves "every
   real symbol is re-exported" — add the inverse assertion too ("every
   `__all__` entry actually resolves via `hasattr`"), or a dangling entry
   like incident 6 above passes silently forever. For a class → mixins
   split, the test from step 2 doubles as this, plus its own inverse: every
   `_state.py`-style shared-stub method must have at least one real
   implementor among the mixins/concrete class, not just a type-only
   declaration (see
   `tests/regression/test_orchestrator_mixin_surface.py::test_every_state_stub_is_overridden_somewhere_reachable`).

## Guardrails

- **A truncation-based extraction (`head -n`, a line-range copy) must cut at
  a semantic boundary, not a physical one.** The dead-code duplication
  regression came from cutting after a method's position in the *old* file
  rather than after only the methods meant to stay. If you truncate instead
  of surgically removing, diff the result against the target file list
  before trusting it.
- **Don't trust a bulk text-transform script on prose without re-reading
  every output.** A scripted docstring rewrap this session inserted a blank
  line mid-sentence in 8 files — caught by ruff's `D205`, not by the script
  itself. A bulk edit across many similar files still needs each result
  individually verified, not just linted.
- **An exemption is a gate you're allowed to widen for one documented
  reason — never a way to make a number look better.** Step 6 exists to
  keep the real percentage visible even when the failure is suppressed. If
  you can't point at the specific pre-split evidence that a flagged line was
  already uncovered, don't exempt it — fix it or leave the gate red.
- **Steps 1-7 are cheap to skip and expensive to skip silently.** All six
  regressions above passed the existing CI ladder before being found — one
  of them (incident 6) passed it *twice*, on two separate commits, because
  the fix for one blind spot didn't happen to close the other. Budget time
  for this sweep as part of the split, not as optional polish after, and
  don't treat a clean `mypy --strict` / full-suite run as proof a fix is
  complete when the whole point of this sweep is the class of bug those
  tools don't catch.

## When to run it

- After any flat-file → package split (a `config/schema/`-shaped change).
- After any class → mixin-composition split (an `orchestrator.py`-shaped
  change).
- Before opening the PR — not after review flags one of these six
  regression classes by hand.
