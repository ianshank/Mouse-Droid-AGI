# CI/docs honesty (F-038)

## Requirement: parked tiers are labelled parked

CI and the two test modules MUST NOT read as production operator-path coverage.

### Scenario: CI step

- **GIVEN** `.github/workflows/ci.yml`
- **THEN** the step that runs `tests/functional` is named with `parked-autonomous`

### Scenario: imports

- **GIVEN** `tests/functional/` and `tests/user_journey/`
- **THEN** they import `build_autonomous_orchestrator` and not `build_orchestrator`

## Requirement: Current Next Steps stays forward-looking

### Scenario: no LANDED

- **GIVEN** the Current Next Steps section of root `NEXT_STEPS.md`
- **THEN** it contains no `LANDED` token

### Scenario: done ids

- **GIVEN** a line in that section that names a catalog id whose status is `done`
- **THEN** the same line contains `leftover`
