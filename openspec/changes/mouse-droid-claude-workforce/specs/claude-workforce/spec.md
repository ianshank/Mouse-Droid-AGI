# Spec Delta: claude-workforce

## ADDED Requirements

### Requirement: Skill Lifecycle Contract
Every file under `.claude/skills/**/SKILL.md` SHALL declare a non-empty `description` and,
when it declares `status`, that value SHALL be one of `active`, `frozen`, `deferred` (the
existing validator vocabulary — no new statuses). A `frozen` skill SHALL carry an
`unfreeze:` note stating its unfreeze condition.

Implementation note (declared interpretation): `unfreeze:` stays free-form text, matching
the shipped contract (`tests/regression/test_next_steps_reconciled.py` asserts presence by
substring; the validator does not parse it). Machine-parsing `unfreeze:` would be a new
contract and is out of scope.

#### Scenario: Out-of-vocabulary status is rejected
- **GIVEN** a skill file whose front-matter `status` is outside the allowed set (for
  example the original draft's proposed `legacy`)
- **WHEN** `tools/validate_skill_commands.py` sweeps the repo
- **THEN** it reports `invalid-status` naming the file and exits non-zero, and
  `tests/regression/test_skill_commands_aqa.py::test_all_skills_are_valid` fails

#### Scenario: The frozen trio stays byte-stable through this change
- **GIVEN** the three existing skills (`sim-test`, `robot-arm-trainer`, `train-policy`)
  already frozen with identical `unfreeze:` conditions
- **WHEN** this change's tasks complete
- **THEN** `git diff` shows zero changes to those three files and
  `test_next_steps_reconciled.py::test_frozen_skills_carry_status_frontmatter` passes

#### Scenario: Absent status stays valid
- **GIVEN** a SKILL.md with no `status:` key (external layouts such as `.github/skills/`)
- **WHEN** the validator sweeps it
- **THEN** no lifecycle issue is reported (backwards compatible)

### Requirement: Workforce Asset Portability
No file under `.claude/` SHALL contain an absolute filesystem path or an IPv4 literal;
repo-relative access SHALL use `$CLAUDE_PROJECT_DIR` or configuration from
`.claude/workforce.yaml`.

#### Scenario: Hardcoded IPv4 in a workforce asset
- **GIVEN** an agent or skill file containing an IPv4 literal
- **WHEN** `tests/regression/test_claude_workforce_aqa.py` runs (reusing
  `tools.validate_skill_commands.find_hardcoded_hosts`)
- **THEN** the test fails naming the offending file and matched literal

#### Scenario: Dangling repo path in a workforce asset
- **GIVEN** a workforce asset whose backtick-quoted repo path does not exist on disk
- **WHEN** the AQA test runs (reusing `referenced_repo_paths`)
- **THEN** the test fails naming the file and the missing path

### Requirement: Agent Tool Declarations
Every file under `.claude/agents/*.md` SHALL declare frontmatter limited to
platform-supported keys, and its `tools:` field SHALL list bare tool names only.

#### Scenario: Permission-pattern syntax in an agent tools field
- **GIVEN** an agent file whose `tools:` value contains `(` or `*` (for example a
  `Bash(...)`-style pattern, which the platform does not support in agent frontmatter)
- **WHEN** the AQA test runs
- **THEN** it fails naming the file and the offending token

### Requirement: Capability Freeze Gate
A PreToolUse hook SHALL deny Write/Edit operations targeting paths matching
`freeze.frozen_paths` while the feature identified by `freeze.feature_key` has any status
other than `done` in `freeze.features_file`, unless the environment variable named by
`freeze.override_env` is set; overrides SHALL be logged. An unreadable or malformed
feature catalog SHALL deny (fail-closed).

#### Scenario: Capability edit while the gate is closed
- **GIVEN** F-008 has `status: "todo"` and `freeze.frozen_paths` includes
  `src/mousedroid/arm/**`
- **WHEN** a session attempts an Edit under `src/mousedroid/arm/`
- **THEN** the hook denies, quoting the rev-B preemption rule ("hardware readiness
  preempts all in-flight software streams")

#### Scenario: Gate self-disables on completion
- **GIVEN** F-008 status becomes `done`
- **WHEN** the same edit is attempted
- **THEN** it proceeds with no hook change and no code deployment

#### Scenario: Override is permitted but logged
- **GIVEN** the override environment variable is set
- **WHEN** a frozen-path edit is attempted
- **THEN** the edit proceeds and the hook emits a logged override record

#### Scenario: Broken catalog fails closed
- **GIVEN** `freeze.features_file` is missing, unreadable, or malformed, or
  `freeze.feature_key` is absent from it
- **WHEN** a frozen-path edit is attempted
- **THEN** the hook denies and reports the catalog problem

### Requirement: Edit-Time Secret Scanning
A PreToolUse hook SHALL scan pending Write/Edit content with the repository's configured
scanner and allowlist (`secret_scan.command` + `secret_scan.config`, i.e. gitleaks +
`.gitleaks.toml`) before the write occurs, denying on findings. The allowlist SHALL be
honored by regex only, never by path. When the scanner binary is absent, behavior SHALL
follow `secret_scan.strict`: warn-and-allow when false (default, mirroring the advisory CI
posture), deny when true.

#### Scenario: Pending content contains a secret
- **GIVEN** a Write whose content matches a gitleaks rule not covered by the regex
  allowlist
- **WHEN** the hook runs
- **THEN** the write is denied and the finding's rule id is reported

#### Scenario: Scanner absent, default posture
- **GIVEN** the scanner binary is not on PATH and `secret_scan.strict` is false
- **WHEN** a Write is attempted
- **THEN** the write proceeds and a warning names the missing scanner

#### Scenario: Scanner absent, strict posture
- **GIVEN** the scanner binary is not on PATH and `secret_scan.strict` is true
- **WHEN** a Write is attempted
- **THEN** the write is denied naming the missing scanner

### Requirement: MCP Configuration
The repository SHALL carry a checked-in, secretless `.mcp.json` that references
credentials only via environment-variable expansion, and it SHALL include the repository's
own `mousedroid` MCP server configured per `docs/MCP_OPERATOR_GUIDE.md`.

#### Scenario: No literal credentials
- **GIVEN** the checked-in `.mcp.json`
- **WHEN** the secret scanner and the AQA test inspect it
- **THEN** no credential literals are found; tokens appear only as `${VAR}` /
  `${VAR:-default}` references

#### Scenario: mousedroid server matches the operator guide
- **GIVEN** the `mousedroid` entry in `.mcp.json`
- **WHEN** the AQA test compares its `command`/`args`/`env` against the stanza in
  `docs/MCP_OPERATOR_GUIDE.md`
- **THEN** they match (with the declared addition of the motion-safe
  `MOUSEDROID_MOCK_HARDWARE` expansion default)
