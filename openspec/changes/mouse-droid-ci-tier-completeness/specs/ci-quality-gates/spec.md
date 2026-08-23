# Spec delta — CI quality gates

## ADDED Requirements

### Requirement: Every test tier SHALL execute in at least one enforced CI path

A test tier that exists in `tests/` but is invoked by no CI path can rot
invisibly. The repository SHALL run every tier from either `scripts/ci.sh` or a
blocking job in `.github/workflows/ci.yml`, and SHALL pin that wiring so its
removal fails a test.

#### Scenario: the three previously-orphaned tiers run in the blocking test job

- **GIVEN** the tiers `tests/functional`, `tests/user_journey` and `tests/security`
- **WHEN** the `test` job of `.github/workflows/ci.yml` is inspected
- **THEN** every one of the three appears in that job's `run` text
- **AND** the job carries no `continue-on-error: true`

#### Scenario: the security tier never lands in the advisory security job

- **GIVEN** the `security` job in `.github/workflows/ci.yml` is `continue-on-error: true`
- **WHEN** that job's `run` text is inspected
- **THEN** it does not contain `tests/security`
- **AND** a change promoting that job to blocking fails the precondition
  assertion rather than silently passing

#### Scenario: removing the wiring fails the pin

- **GIVEN** the wiring present in `scripts/ci.sh` and `.github/workflows/ci.yml`
- **WHEN** either invocation is removed
- **THEN** `TestOrphanTierWiring` reports at least one failing assertion
