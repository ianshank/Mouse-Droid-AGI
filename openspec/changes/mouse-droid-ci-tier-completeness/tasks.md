# Tasks — CI tier completeness

Quality gate for every task below, run before it is ticked:

```
python -m ruff check src/ tests/ tools/ && python -m ruff format --check src/ tests/ tools/
python -m mypy src/ --strict --ignore-missing-imports
python -m pytest tests/regression/test_ci_gate_wiring_aqa.py -q
```

Task ordering is binding: each task lands green before the next starts.
Deviations from task wording are recorded inline — declared, not silent.

**Phase 1 — Wire the tiers**

- [x] 1.1 Add a functional/user-journey/security stage to `scripts/ci.sh`, before the `MOUSEDROID_CI_SLIM` gate.
- [x] 1.2 Add the matching step to the blocking `test` job in `.github/workflows/ci.yml`.
- [x] 1.3 Confirm `bash -n scripts/ci.sh` and a YAML parse of the workflow both pass.

**Phase 2 — Pin the wiring**

- [x] 2.1 Add `TestOrphanTierWiring` to `tests/regression/test_ci_gate_wiring_aqa.py`.
- [x] 2.2 Pin negatively that `tests/security` never appears in the advisory `security` job.
- [x] 2.3 Prove the pin fails: remove the wiring from both files, confirm red, restore.
- [x] 2.4 Add `TestOrphanTierMarkerParity`: the pytest marker expression is identical
      across every site that runs these tiers, and still excludes `hardware` at each —
      parity alone would be satisfiable by deleting the filter everywhere. Sites are
      **discovered** (every `ci.yml` job, `ci.sh`, the `Makefile`, every
      `scripts/validations/*.sh`), because a listed roster of three missed a fourth,
      `make behaviour`, which agreed only by luck.
- [x] 2.5 Comment-strip `ci.yml` run blocks as well as the shell scripts. `yaml` parses
      `run:` as a literal scalar, so `#` lines survive `safe_load` and the ci.yml-side
      pins were satisfiable by a commented-out invocation — the same defect fixed on the
      `ci.sh` side in 2.3, left standing on the other.

**Phase 3 — Catalog**

- [x] 3.1 Add the F-028 entry to `features.yaml` and validate against `features.schema.json`.
- [x] 3.2 Write `scripts/validations/F-028.sh` and confirm it exits 0.
- [x] 3.3 Register the change in `openspec/project.md`.

**Phase 4 — Prove the pins mechanically**

- [x] 4.1 Run `scripts/prove_pin_fails.sh` over this bundle's pins rather than
      asserting the manual revert was done:
      `bash scripts/prove_pin_fails.sh --from <base-ref>`
      `--paths "scripts/ci.sh .github/workflows/ci.yml"`
      `--tests "tests/regression/test_ci_gate_wiring_aqa.py"`.
- [x] 4.2 Note any pin the tool cannot cover and why. Two here:
      `scripts/validations/F-028.sh` does not exist at the base ref (nothing to
      revert to), and a catalog-wide invariant gate — one asserting a rule over
      every entry rather than protecting one change — is green against a clean
      baseline by construction, so the tool reports it as toothless. Prove those
      by injecting a violation instead.

## Explicitly deferred (separate changes, do not fold in)

- Rewriting `tests/user_journey/test_operator_mission_journey.py` and
  `tests/functional/test_mission_safety_interlocks.py` off private-attribute
  assertions (`orch._motor`) — belongs to the F-031 disposition arc, see D-4.
- Adding the three tiers to `Makefile` targets — the Makefile is a thin
  discoverability wrapper; `scripts/ci.sh` stays the authoritative local superset.
- Promoting the advisory `security` job to blocking — tracked separately in
  `.github/advisory_stages.yaml`.
