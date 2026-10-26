# Runbook — clearing a latched emergency stop

> Applies when `safety.emergency_latch.enabled` is `true`. It ships **true** on
> `config/jetson_production.yaml` (F-059 CF-2) and every stack built on it,
> except the lidar-only stack, which overrides it to **false**; configs that do
> not start from production leave it at the schema default, false. Where it is
> false there is nothing to clear and this runbook is inert.

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
    rearm_cmd="python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --status"
    clear_requires="after steps 1-2 of docs/runbooks/emergency-stop-rearm.md, clear by running the same command with --operator <you> --confirm-area-clear in place of --status"
```

The `rearm_cmd` is the **status** form: pasting it only looks. It is built from the overlay
files this process was actually started with — one `--config` per overlay, in the order they
were applied, as absolute paths — not from the environment, so it inspects the latch this
rover is holding and changes nothing (exit `1` = latched). A process whose settings were
built directly rather than loaded from a file has no overlay to name and logs the command
with no `--config`; add the overlay yourself.

The log never carries the clearing command. `--operator` and `--confirm-area-clear` are your
own statement that you have looked at the rover and its path is clear, so you type them
yourself after steps 1–2 below; `clear_requires` only says so. `<you>` in this runbook is a
placeholder for your own name, never a value to paste.

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

**2. Check the state without changing it.** Always pass `--config`. It names the overlay
that says where the latch lives (`experience.path`); a CLI that is not told cannot know, and
a latch it does not find reads as clear.

Bare metal:

```bash
/opt/mousedroid/venv/bin/python -m mousedroid.cli.rearm \
    --config /etc/mousedroid/jetson_production.yaml --status
```

Container — must run *inside* the service, because the latch lives in the
`mousedroid_experience` named volume, which the host cannot reliably read:

```bash
docker compose -f docker-compose.jetson.yml exec mousedroid \
    python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml --status
```

Exit `1` means latched, `0` means clear — for the overlay you named.

If `--config` is left out the CLI does not guess. It falls back, as `validation.runtime` does
(the smoke and probe scripts resolve this way; `cli.preflight` and `cli.validate_pillars` take
`--config` only), to the first of `MOUSEDROID_CONFIGS`, `MOUSEDROID_JETSON_CONFIGS`
(comma-separated lists), then `MOUSEDROID_CONFIG`, `MOUSEDROID_JETSON_CONFIG` that is set, and logs which one as
`rearm_config_resolved` (`source=env_csv` or `env_single`). The shipped container sets
`MOUSEDROID_CONFIG`, so a bare `exec` there still finds the production latch. That is a safety
net, **not the documented form**: it only helps where the variable is set (your bare-metal
shell does not inherit the service unit's `Environment=`), and a list variable outranks the
single one, so an exported leftover can point the CLI at a stack the rover never loaded.
With neither flag nor variable it loads `default.yaml` alone, where the latch is off, writes
`no config overlay was given` to stderr, and still exits `0` — read that `0` as "I did not
look", not "clear".

**3. Clear it.** `--operator` and `--confirm-area-clear` are both required; neither has a
default or an env fallback, because an unattributed re-arm is not a re-arm. Keep the
`--config` from step 2: it is what points the CLI at this rover's latch.

Bare metal:

```bash
/opt/mousedroid/venv/bin/python -m mousedroid.cli.rearm \
    --config /etc/mousedroid/jetson_production.yaml \
    --operator "<your name>" --confirm-area-clear
```

Container:

```bash
docker compose -f docker-compose.jetson.yml exec mousedroid \
    python -m mousedroid.cli.rearm --config /etc/mousedroid/jetson_production.yaml \
    --operator "<your name>" --confirm-area-clear
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
