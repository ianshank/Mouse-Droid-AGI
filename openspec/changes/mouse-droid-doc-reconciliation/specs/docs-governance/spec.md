# Spec delta — Docs governance

## ADDED Requirements

### Requirement: Live documentation surfaces SHALL state only claims that hold against the current tree

A CI job count, a coverage-gate percentage, a class/function/file name, or a wiring claim
("X is/is not instantiated by `factory.py`") stated in a live governance doc SHALL match
what the referenced source artifact actually contains. A doc that once matched and later
drifted SHALL be caught by an automated check tied to the same source artifact, not
re-verified only by manual re-read.

#### Scenario: every live doc stating the total CI job count states the real one

- **GIVEN** `.github/workflows/ci.yml`'s `jobs:` mapping
- **WHEN** `CLAUDE.md`, `docs/claude/surfaces/ci-gates.md`, and
  `docs/claude/surfaces/README.md` are inspected for a total-job-count phrasing
  (`"(N Jobs)"` or `"N-job"`)
- **THEN** every `N` found equals the real count of top-level keys under `jobs:`

#### Scenario: no live doc claims an unqualified 85% floor for the src/mousedroid gate

- **GIVEN** `pyproject.toml`'s `[tool.coverage.report] fail_under`
- **WHEN** `HARNESS_SPEC.md`, `tests/agent.md`, and `docs/architecture/c4-spec-harness.md`
  are inspected line by line for "85%"
- **THEN** every remaining "85%" line also names `tools/claude_hooks` on that same line
- **AND** the separate `tools/claude_hooks` floor itself stays pinned at 85% so a future
  edit cannot silently tighten it to match the `src/mousedroid` gate

#### Scenario: a subsystem CLAUDE.md names only symbols that exist

- **GIVEN** `src/mousedroid/orchestrator/CLAUDE.md`
- **WHEN** every backtick-wrapped class or file name in it is checked against `src/`
- **THEN** each one resolves to a real symbol
- **AND** any component with zero production callers is explicitly labelled as such rather
  than presented as the production path

#### Scenario: a module-wiring claim is checked everywhere it is repeated, not just where it was first fixed

- **GIVEN** a module gains a `factory.py` builder, invalidating a "not yet wired" claim about
  it that was copied into more than one live doc
- **WHEN** the claim is corrected in one location
- **THEN** the same module name is re-checked against every other live doc that repeats the
  claim (a whole-file, whitespace-normalized sweep, not a single remembered location)
- **AND** every stale repetition is corrected in the same pass, not left for a later,
  separate discovery

### Requirement: The skills and subagent catalog SHALL enumerate every real invocable capability

`SKILLS.md` SHALL mention every real `.claude/skills/<name>/` directory at least once, and
its Subagent skills table SHALL name every real `.claude/agents/*.md` file and no others.
Both SHALL be enforced by discovery (globbing the real directories/files) rather than a
hardcoded expected-list, so a capability added without a matching doc entry fails the check
instead of shipping silently undocumented.

#### Scenario: a new skill directory without an index mention fails the check

- **GIVEN** a new `.claude/skills/<name>/SKILL.md` directory
- **WHEN** `<name>` appears nowhere in `SKILLS.md`
- **THEN** `test_every_skill_directory_is_mentioned_in_the_index` fails, naming `<name>`

#### Scenario: a stale or fictional agent roster fails the check

- **GIVEN** the Subagent skills table in `SKILLS.md`
- **WHEN** a real `.claude/agents/*.md` stem is absent from that table's text
- **THEN** `test_every_agent_is_listed_in_the_subagent_skills_table` fails, naming the
  missing agent
