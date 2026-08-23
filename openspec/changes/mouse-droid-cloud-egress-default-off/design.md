# Design — Cloud egress defaults OFF

## D-1. Why the AQA asserts the partial-block scenario

The obvious assertion — "nothing egresses by default" — is **vacuously true**,
because `Settings.gcp` is `Optional` and defaults `None`. A test asserting it
would pass before and after the change and prove nothing.

The assertion that carries weight is `GCPConfig(project_id="x")` yielding
`logging.enabled is False` and `monitoring.enabled is False`: the operator who
adds a `gcp:` block for one reason and gets two egress channels for free. That
is the failure this change exists to prevent, so that is what the test pins.

## D-2. Why the pair asserts the opt-in path too

`test_explicit_opt_in_still_works` exists so the default cannot be mistaken for
a hard-coded `False`. Without it, someone could satisfy the AQA by making the
field non-configurable, which would break the digital-twin overlay while the
suite stayed green. The same reasoning as the "validator isn't just always-raise"
case in `tests/regression/test_pr106_aqa.py`.

## D-3. Assertions read FieldInfo, not round-tripped values

Per `.claude/skills/test-tier-mirror/SKILL.md`, AQA checks schema properties off
`model_fields[...]` rather than through `model_validate`. A refactor swapping
`Field(...)` for a property override would slip past a value-only check.

## D-4. The backwards-compat test pins the overlay's explicit opt-in

`config/gcp_digital_twin.yaml` sets `logging.enabled: true`,
`monitoring.enabled: true` and `firestore.enabled: false` explicitly, so the
default flip is inert for it. The pair asserts this **twice** — once through the
parsed model and once against the raw YAML — because the second is what catches
a future edit that deletes the explicit lines and leans on the default. Without
that, the rover would quietly cease exporting and nothing would go red.

## D-5. Relationship to F-032

This is the compensating control for the operator decision to wire
`CloudLoggingSink` / `CloudMetricsExporter` / `CloudFirestoreSync` **without** a
ratified CHARTER §3 carve-out. F-029 must land first; F-032's `notes:` field
records the deviation.
