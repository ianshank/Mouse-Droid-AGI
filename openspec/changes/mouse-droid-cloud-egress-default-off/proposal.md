# Proposal — Cloud egress defaults OFF

- change_id: mouse-droid-cloud-egress-default-off
- project: mouse-droid
- status: implemented
- feature_id: F-029
- epic: Quality Gates
- owner: ianshank
- created: 2026-08-22
- basis_commit: df928d9
- rev: A

## Why

`GCPLoggingConfig.enabled` and `GCPMonitoringConfig.enabled` defaulted `True`
while `GCPFirestoreConfig.enabled` defaulted `False`. That asymmetry is the bug.

`Settings.gcp` is `Optional` and defaults `None`, so nothing egressed on a stock
config — which is exactly what made this easy to miss. The real hazard is narrower:
an operator adding a **minimal `gcp:` block for one reason** (Firestore, storage)
silently enabled **two** off-device egress channels they never named.

`docs/CHARTER.md` §6 requires new capabilities be additive and opt-in, never a
silent behaviour change. This also has to land before F-032 wires the cloud
components — wiring egress while a default-ON path exists opens a window where
any operator adding a `gcp:` block starts egressing.

## What Changes

Both flags default `False` in `src/mousedroid/config/schema/gcp_cloud.py`, with
descriptions carrying the rationale. Plus the mandatory regression pair.

## Impact

Behaviour changes for **no shipped config**: `config/gcp_digital_twin.yaml` is
the only overlay in the tree declaring a `gcp:` block, and it sets all three
flags explicitly rather than relying on defaults. Verified, and pinned by the
backwards-compat test so a future edit cannot quietly start leaning on the
default again.

## Spec Deltas

`openspec/changes/mouse-droid-cloud-egress-default-off/specs/cloud-egress/spec.md`

## Tasks

See `openspec/changes/mouse-droid-cloud-egress-default-off/tasks.md`.

## Validation

`bash scripts/validations/F-029.sh`
