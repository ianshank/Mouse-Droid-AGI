# Spec delta — Cloud egress

## ADDED Requirements

### Requirement: Every off-device egress channel SHALL be opt-in

A configuration block that causes data to leave the device SHALL default to
disabled. Enabling egress SHALL require an explicit operator action naming that
specific channel. Adding a parent configuration block for an unrelated purpose
SHALL NOT enable any egress channel as a side effect.

#### Scenario: a partial GCP block opens no egress channel

- **GIVEN** an operator adds a `gcp:` block declaring only `project_id`
- **WHEN** the configuration is validated
- **THEN** `logging.enabled`, `monitoring.enabled` and `firestore.enabled` are all `False`

#### Scenario: explicit opt-in still enables the channel

- **GIVEN** a `gcp:` block setting `logging.enabled: true` and `monitoring.enabled: true`
- **WHEN** the configuration is validated
- **THEN** both resolve to `True`
- **AND** the default is demonstrably a default rather than a hard-coded value

#### Scenario: the shipped digital-twin overlay is unaffected

- **GIVEN** `config/gcp_digital_twin.yaml`, the only shipped overlay declaring a `gcp:` block
- **WHEN** the default is flipped from enabled to disabled
- **THEN** the overlay resolves to the same effective state as before
- **AND** it does so because it sets each flag explicitly, asserted against the raw YAML
