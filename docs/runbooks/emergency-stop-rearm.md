# Runbook — clearing a latched emergency stop

> Applies when `safety.emergency_latch.enabled` is `true`. It ships **false**,
> so on a stock deployment there is nothing to clear and this runbook is inert.

## What a latch is, and why it exists

`MouseDroidSafetyMonitor.evaluate` recomputes `is_emergency` from `False` on
every tick. Without a latch, the instant the triggering condition clears the
next tick resumes driving — no human involved. The shipped units compound it:
`scripts/mousedroid.service` and `scripts/mousedroid-docker.service` both set
`Restart=on-failure`, and `docker-compose.jetson.yml` sets
`restart: unless-stopped`, so a restart clears it too.

ISO 3691-4 requires that an emergency stop is reset **only by deliberate human
action**, and that the reset does not by itself command motion. The latch
holds the stop across ticks *and* across restarts; this procedure is the reset.

Recorded as defect D-5 in
`docs/analysis/positioning-safety-peer-review-2026-09-19.md`.

## Recognising it

The rover starts normally but will not move, and the boot log carries:

```
estop_latch_active_at_boot  reason=... causes=[...] tripped_at_iso=...
    rearm_cmd="python -m mousedroid.cli.rearm --operator <you> --confirm-area-clear"
```

A latched rover **starts on purpose**. Refusing to start would meet
`Restart=on-failure` on a unit with no `StartLimitBurst` and crash-loop; a
crash-looping process never reaches `_halt_actuators`, so nothing would be
holding the motors stopped, and telemetry would be dead too.

## Procedure

**1. Look at the rover before you touch the keyboard.** Read the `reason` and
`causes` from the boot log first — they name which interlock fired
(`forward_clearance_violation`, `battery_critical`, `sensor_stale`,
`insufficient_valid_sensors`, `loop_overrun`, `lidar_emergency`,
`human_proximity`). Confirm the hazard is gone and the path is clear.

**2. Check the state without changing it.**

Bare metal:

```bash
/opt/mousedroid/venv/bin/python -m mousedroid.cli.rearm \
    --config /etc/mousedroid/jetson_production.yaml --status
```

Container — must run *inside* the service, because the latch lives in the
`mousedroid_experience` named volume, which the host cannot reliably read:

```bash
docker compose -f docker-compose.jetson.yml exec mousedroid \
    python -m mousedroid.cli.rearm --status
```

Exit `1` means latched, `0` means clear.

**3. Clear it.** Both flags are required; neither has a default or an env
fallback, because an unattributed re-arm is not a re-arm.

```bash
... -m mousedroid.cli.rearm --operator "<your name>" --confirm-area-clear
```

**4. Restart the service.** Clearing the record does not command motion — the
rover arms on its next start, which keeps the reset and the resumption of
motion two separate acts.

```bash
sudo systemctl restart mousedroid-docker    # or mousedroid
```

## What will not clear it, deliberately

| Surface | Why not |
|---|---|
| Telemetry REST API | LAN-reachable behind a bearer token — that is a *remote* re-arm, and it shares an ingress with the natural-language mission endpoint. |
| MCP tool | Those are the tools a language model calls. It would let the model that caused the stop undo it. |
| `scripts/preflight_check.sh` | Runs as systemd `ExecStartPre` with no human attached, so anything it clears, `Restart=on-failure` clears automatically — the defect again. |
| Waiting | There is no TTL and there must never be one: an age-based auto-clear *is* an automatic restart after an emergency stop. |

## If the record is damaged

A corrupt, truncated or unreadable record reads as **latched**, with reason
`corrupt_latch_file`, `latch_store_unreadable` or
`unknown_latch_schema_version`. That is intentional: a damaged record is
evidence that something wrote a latch and the write or the media failed.
Treat it as a real stop — inspect the rover, then clear it the normal way.
