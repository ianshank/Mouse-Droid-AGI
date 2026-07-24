## Why
<!-- The motivation for this change. The diff shows the "what". -->

## Changes
<!-- Bullet the notable changes. -->

## Testing
<!-- Commands run and their results. -->

## Checklist
- [ ] `ruff check` + `ruff format --check` clean
- [ ] `mypy --strict` clean
- [ ] `pytest` passes; branch coverage ≥85%
- [ ] New config fields have defaults + a regression/AQA test; existing YAML loads unchanged
- [ ] Docs updated (README / `docs/`) and a `CHANGELOG.md` entry added
- [ ] No invariant weakened / scope expanded without maintainer sign-off (docs/CHARTER.md §6)
