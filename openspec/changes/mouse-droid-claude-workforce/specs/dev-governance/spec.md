# Spec Delta: dev-governance

## ADDED Requirements

### Requirement: Evidence-Backed Claims
Every hardware or performance claim in README.md or `features.yaml` SHALL trace to either
(a) a tracked artifact under a directory listed in `evidence.tracked_roots`, or (b) a
declared local-only evidence chain — a report family the repository deliberately gitignores
("Local-only artefacts — never checked in") together with a CHANGELOG or plan-doc
reference. A claim with neither SHALL be an audit finding naming the expected artifact
path.

Implementation note (declared interpretation): option (b) exists because the repository's
`.gitignore` policy intentionally keeps several report families out of git; an auditor
that flagged every such artifact would indict policy rather than drift. BENCHMARKS.md is
not referenced — the file does not exist.

#### Scenario: Tracked artifact passes
- **GIVEN** a claim referencing a run whose artifact is tracked under `reports/`
- **WHEN** the `hw-evidence-auditor` sweeps
- **THEN** the claim passes with the artifact cited

#### Scenario: Declared local-only chain passes
- **GIVEN** a claim whose artifact family is gitignored by policy and whose run is
  recorded in CHANGELOG or a dated plan doc
- **WHEN** the auditor sweeps
- **THEN** the claim passes with the declared chain cited

#### Scenario: Naked claim is flagged
- **GIVEN** a referenced run with neither a tracked artifact nor a declared chain (the
  2026-07-12 on-device run is the canonical precedent)
- **WHEN** the auditor sweeps
- **THEN** the claim is listed as unevidenced with the expected artifact path

### Requirement: Single Configuration Source for Workforce Tooling
All thresholds, gate keys, path globs, and budgets consumed by workforce hooks, the AQA
module, and skills SHALL be read from `.claude/workforce.yaml`, validated by
`tools/claude_hooks/config.py::WorkforceConfig` with `extra="forbid"`; numeric literals in
hook/AQA code SHALL be limited to schema defaults.

Implementation note (declared interpretation): the repository's magic-number enforcement
mechanisms are `scripts/check_no_hardcoded_values.py` (changed lines under
`src/mousedroid`) plus review — the ruff `PL` family is not enabled, and this delta does
not claim it.

#### Scenario: Unknown configuration key
- **GIVEN** `workforce.yaml` containing a key absent from `WorkforceConfig`
- **WHEN** configuration loads
- **THEN** validation fails naming the unknown key

#### Scenario: Threshold change is config-only
- **GIVEN** an operator changing a workforce threshold
- **WHEN** they edit `.claude/workforce.yaml`
- **THEN** no Python source change is required

### Requirement: Tested Governance Tooling
Every hook module and the workforce AQA module SHALL be covered by a dedicated coverage
invocation over `tools/claude_hooks` meeting `coverage.tools_line_min`, with branch
coverage measured and reported.

Implementation note (declared interpretation): the repository-wide gate measures
`src/mousedroid` only and cannot see `tools/` — this is a NEW invocation, additive to that
gate. The branch number starts advisory (reported, non-blocking) because no baseline
exists; promotion to blocking is a separate deferred change.

#### Scenario: Hook code below the line threshold
- **GIVEN** hook code whose line coverage falls below `coverage.tools_line_min`
- **WHEN** the dedicated ci.sh coverage stage runs
- **THEN** the stage fails

#### Scenario: Branch coverage is reported, not blocking
- **GIVEN** the same stage
- **WHEN** it completes
- **THEN** a branch-coverage figure is present in the report and does not gate the run

### Requirement: Truthful Coverage Claims
Repository documentation SHALL NOT claim a coverage metric that is not measured.
README.md's current "85% branch coverage" claim SHALL be corrected to "85% line coverage"
until branch measurement is enabled and promoted by its own change.

#### Scenario: Claim matches measurement
- **GIVEN** the coverage configuration measuring line coverage only
- **WHEN** the docs-consolidation phase completes
- **THEN** no doc claims branch-coverage enforcement, and the README badge text matches
  the measured metric

### Requirement: Additive Compatibility for Existing Assets
Updates to pre-existing skills, `.claude/settings.json`, and CLAUDE.md SHALL be additive:
the three frozen skills stay byte-identical; existing permission entries remain; hooks are
added as a new block (platform semantics merge hooks across scopes); `.claude/commands/`
stays deleted; `tools/validate_skill_commands.py` is not removed before WS-F7b executes
its planned retirement.

#### Scenario: Legacy commands directory stays deleted
- **GIVEN** the completed change
- **WHEN** `test_skill_commands_aqa.py::test_legacy_commands_dir_stays_deleted` runs
- **THEN** it passes

#### Scenario: Prior permission entries survive
- **GIVEN** the pre-change `.claude/settings.json` allowlist
- **WHEN** the hooks block is added
- **THEN** every pre-existing permission entry is still present and unchanged
