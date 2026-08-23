# Peer review — Cloud egress defaults OFF

## Verdict table

| Claim | Verdict |
|---|---|
| Logging and monitoring defaulted `True`, firestore `False` | **CONFIRMED** — the asymmetry is the bug |
| Nothing egresses on a stock config | **CONFIRMED** — `Settings.gcp` is `Optional`, defaults `None` |
| A partial `gcp:` block opened two unnamed channels | **CONFIRMED** — reproduced before the fix |
| No shipped config changes behaviour | **CONFIRMED** — `config/gcp_digital_twin.yaml` sets all three explicitly |
| The pair can fail | **CONFIRMED** — reverting both defaults turns 3 assertions red; restored clean |

## Corrected-design map

| Original intent | Corrected |
|---|---|
| Assert "no egress by default" | Rejected as **vacuous** — true before and after, since `Settings.gcp` defaults `None`. Replaced with the partial-block scenario. |
| Assert the defaults only | Insufficient — a non-configurable `False` would satisfy it while breaking the twin overlay. Added the explicit opt-in case. |
| Assert the overlay through the parsed model | Insufficient alone — would still pass if a future edit deleted the explicit YAML lines and leaned on the default. Added the raw-YAML assertion. |

## What survives review unchanged

- Flipping both flags rather than adding a master kill-switch: the sub-configs
  already have the right shape, and a second gate would be more config surface
  for the same guarantee.
- Leaving `GCPFirestoreConfig.enabled` untouched — it already defaulted `False`.
- Reading `FieldInfo` off `model_fields` rather than round-tripping values.

## Load-bearing pins any implementation must satisfy

1. All three GCP egress sub-configs default `enabled=False`.
2. `GCPConfig(project_id=...)` alone opens no channel.
3. Explicit `enabled: true` still works.
4. `config/gcp_digital_twin.yaml` resolves to logging=True, monitoring=True,
   firestore=False, asserted through both the model and the raw YAML.

## Appendix — open follow-up

The other `gcp_cloud.py` sub-configs (pubsub, storage, training, simulation)
were not audited for the same default-ON asymmetry. Recorded in `tasks.md`
under "Explicitly deferred" rather than silently widened into this change.
