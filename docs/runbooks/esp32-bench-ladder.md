# ESP32 bench ladder — F-008 retest

> Companion to [`jetson-full-bringup.md`](jetson-full-bringup.md) (its step 3 is the production-posture
> probe) and [`jetson-rover-smoke.md`](jetson-rover-smoke.md) (per-stage smoke triage).

## Purpose

Decide, from evidence, whether the Wave Rover's ESP32 driver board answers on the rover's USB-C bridge, and what the repair is if it does not.
Each step either proves the board alive or narrows the fault, and the ladder ends at one of four leaves (L1–L4) with a recorded decision.
The captured evidence closes F-008 (in PR-H); this is Phase 3 of the F-059 rover gate-closeout bundle, `openspec/changes/mouse-droid-rover-gate-closeout/`.

## Safety frame

Applies to every step:

- Chassis **lifted on blocks, wheels clear**.
- Operator at the power switch.
- The container keeps `esp32.enabled: false` (mock) until the board answers.
- Motion only in 3.10c, with **both** `MOUSEDROID_SMOKE_ALLOW_MOTION=1` and `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` on the command line.
- Commands run from the PC via `ssh $ROVER_USER@$ROVER_HOST` unless marked PC.
- Keep a bench log: `| tee -a ~/bench-$(date +%F).txt`.
- The main session drives the checklist and dispatches `hw-evidence-auditor` at 3.10f; the `mousedroid` MCP's `read_encoders`/`system_info` are allowed for triage, its `esp32_diagnostics` is a stub until task 4.3.

## Variables

| Variable | Meaning | Where it comes from |
|---|---|---|
| `ROVER_HOST` | The Jetson's USB device-mode address, reached over the USB-C cable in the Jetson's device port | Your environment: see your `~/.mousedroid/jetson_host` |
| `ROVER_USER` | The login on the Jetson | Your environment: `MOUSEDROID_REMOTE_USER` (the scripts also read `SSH_USER`) |
| `SHA` | The trunk commit deployed in 3.3 | Recorded in 3.0 |
| `BENCH_CFG` | Production overlay plus bench overlay, comma-separated; forwarded as `MOUSEDROID_JETSON_CONFIGS` on every direct `docker exec` | Rover shell, set in 3.4 |
| `ESP_PORT` | The canonical CP2102N bridge's device node, resolved from `/dev/serial/by-id` | Rover shell, set in 3.5 |

```bash
# PC shell: fill these in from your environment; this runbook hard-codes neither the address nor the login
ROVER_USER=<jetson-login>
ROVER_HOST=<device-mode-address>
SHA=<full-sha-recorded-in-3.0>

# Rover shell: set in 3.4 and 3.5, and again after any reconnect
BENCH_CFG=/opt/mousedroid/config/jetson_production.yaml,/opt/mousedroid/config/jetson_bench_esp32.yaml
ESP_PORT=$(readlink -f /dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_cc0ff02aab6cef119a38a4adc169b110-if00-port0)
```

- PC blocks are POSIX shell (Git Bash or WSL on a Windows PC), run from the repo root: `scripts/rover_wip_guard.sh`, `firmware` and `reports/` below are relative to it.
- `BENCH_CFG` and `ESP_PORT` live in the rover's shell: set them in the same `ssh $ROVER_USER@$ROVER_HOST` session that runs the `docker exec` commands.
- Tags in the steps (CF-n, R-n, PR-x, `task n.n`) index into the bundle named under Purpose: decisions in `proposal.md`, review findings in `peer-review.md`, this ladder as D-8 in `design.md`, tasks in `tasks.md`.

## Steps

Tick a box only when the expected output was seen.

- [ ] **3.0 Pre-flight (PC, before the Jetson arrives, 60 min).** Key, host and login for the repo scripts (CF-3: they default the login to `jetson`), and the trunk SHA to deploy.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/mousedroid_ed25519 -N ""
mkdir -p ~/.mousedroid && printf '%s\n' "$ROVER_HOST" > ~/.mousedroid/jetson_host
export MOUSEDROID_REMOTE_USER=$ROVER_USER SSH_USER=$ROVER_USER
git fetch origin && git rev-parse origin/claude/markdown-implementation-plan-aVJ2l
```

Also add this alias to `~/.ssh/config` (substitute the `ROVER_HOST` and `ROVER_USER` values):

```text
Host rover
    HostName <ROVER_HOST>
    User <ROVER_USER>
    IdentityFile ~/.ssh/mousedroid_ed25519
```

Keep the SHA printed above as `SHA`: it must contain PR-B/C/D (bench overlay, probe, loopback gate, stock flip), plus PR-G (hash-pinned flasher and firmware record) if 3.9 may be needed. Stage firmware per task 2.5 step 5 (never `combine/target.bin`).

- **Contingency** if any is unmerged: replace `BENCH_CFG` below with `-e MOUSEDROID_ESP32__ENABLED=true -e MOUSEDROID_ESP32__COMMAND_SET=waveshare_stock` on every direct `docker exec`, and skip 3.10d.
- Install Windows "Flash Download Tool 3.9.5" as the fallback flasher (COM port; offsets `0x1000/0x8000/0xE000/0x10000`; 115200).

- [ ] **3.1 First contact (10 min).** USB-C to the Jetson device port, then (PC; on Windows `ping.exe` counts with `-n`):

```bash
ping -c 2 $ROVER_HOST
ssh $ROVER_USER@$ROVER_HOST 'echo ok; head -1 /etc/nv_tegra_release; docker ps --format "{{.Names}} {{.Status}}"; git -C /opt/mousedroid rev-parse --short HEAD; git -C /opt/mousedroid status --porcelain | wc -l'
ssh-copy-id -i ~/.ssh/mousedroid_ed25519.pub $ROVER_USER@$ROVER_HOST
```

Expect `ok`, `R36…`, `mousedroid Up … (healthy)`, the current SHA, the dirty-line count. No container → on the rover:

```bash
docker compose -f /opt/mousedroid/docker-compose.jetson.yml --env-file /dev/null up -d
```

and wait for `healthy` (`loop.start_grace_s` = 60 s).

- [ ] **3.2 Checkpoint A — healthy container, no chassis (10 min).** Rover USB-C to the driver board **unplugged**.

```bash
docker inspect -f '{{.State.Health.Status}}' mousedroid
docker exec -w /opt/mousedroid mousedroid python3 -m mousedroid.main --health-check
```

→ `healthy`; exit 0. **This runs against the mock**: it proves the brain and the rollback anchor, not the chassis. Run the no-chassis smoke with the bench overlay opted out (`power`, and `camera` too: F-008's `validation_command` is `all`, so an open F-007 (`/dev/video0` missing, 2026-05-12) blocks the closeout regardless of the ESP32 — decide at 3.10e whether to narrow the catalog command):

```bash
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY= mousedroid bash scripts/jetson_smoke_test.sh power
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_SMOKE_BENCH_ESP32_OVERLAY= mousedroid bash scripts/jetson_smoke_test.sh camera
```

`power` → PASS (mock, 12.0 V — recorded as such).

- [ ] **3.3 Deploy trunk via git-fetch (20–30 min; CF-4 fallback is the primary path today).** From the PC:

```bash
ssh $ROVER_USER@$ROVER_HOST "find /opt/mousedroid -not -user $ROVER_USER | wc -l"          # ~6400 expected (ownership drift)
ssh $ROVER_USER@$ROVER_HOST "sudo chown -R $ROVER_USER:$ROVER_USER /opt/mousedroid && git config --global --add safe.directory /opt/mousedroid"
ssh $ROVER_USER@$ROVER_HOST 'git -C /opt/mousedroid status --porcelain | head'    # non-empty => preserve first:
mkdir -p ~/.mousedroid/rover-wip && ssh $ROVER_USER@$ROVER_HOST 'bash -s -- preserve /opt/mousedroid --archive -' \
    < scripts/rover_wip_guard.sh > ~/.mousedroid/rover-wip/rover-wip-$(date -u +%Y%m%dT%H%M%SZ).tar.gz
ssh $ROVER_USER@$ROVER_HOST 'git -C /opt/mousedroid rev-parse HEAD > ~/rollback-commit.txt; docker tag mousedroid:jetson mousedroid:jetson-rollback-$(date -u +%Y%m%d)'
ssh $ROVER_USER@$ROVER_HOST "git -C /opt/mousedroid fetch origin && git -C /opt/mousedroid checkout --detach ${SHA}"   # "HEAD is now at …"
ssh $ROVER_USER@$ROVER_HOST 'docker exec -w /opt/mousedroid mousedroid python3 -c "import mousedroid.main, mousedroid.factory, mousedroid.validation.runtime; print(\"import ok\")"'
```

On `ModuleNotFoundError: No module named 'X'`, install the package into the running container, re-run the import, and log every package (writable layer — gone on `--force-recreate`; today use `restart` only):

```bash
docker exec mousedroid python3 -m pip install --no-cache-dir "<spec from pyproject.toml>"
```

More than three packages or any native build → start a rebuild in `tmux` (45–90 min; the running container is unaffected) and continue with 3.5–3.8, which need no trunk code:

```bash
docker compose -f /opt/mousedroid/docker-compose.jetson.yml --env-file /dev/null build
```

**Code first, overlay second**: only now sync the overlay → `OK overlay_sync_replaced` (the 032942b schema ignores `command_set`/`emergency_latch`, and un-zeroed thresholds e-stop on a 0.0 V reading — review focus 3):

```bash
sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh
```

Restart with `sudo systemctl restart mousedroid-docker` if `systemctl is-enabled mousedroid-docker` prints `enabled`, else with compose:

```bash
systemctl is-enabled mousedroid-docker
sudo systemctl restart mousedroid-docker                                                               # when it printed "enabled"
docker compose -f /opt/mousedroid/docker-compose.jetson.yml --env-file /dev/null restart mousedroid    # otherwise
```

**Checkpoint A′:** `healthy` within 60 s; `--health-check` exit 0; the log shows `command_set=waveshare_stock` and the mock driver:

```bash
docker inspect -f '{{.State.Health.Status}}' mousedroid
docker exec -w /opt/mousedroid mousedroid python3 -m mousedroid.main --health-check
docker logs mousedroid --since 2m | grep esp32_driver_built
```

- [ ] **3.4 Bench overlay (5 min).** On the rover:

```bash
BENCH_CFG=/opt/mousedroid/config/jetson_production.yaml,/opt/mousedroid/config/jetson_bench_esp32.yaml
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false mousedroid python3 -c "from pathlib import Path; from mousedroid.config.loader import load_settings; c=load_settings(Path('config/jetson_production.yaml'), Path('config/jetson_bench_esp32.yaml')); print(c.esp32.enabled, c.esp32.command_set, c.esp32.serial_baud, c.esp32.smoke_test_allow_motion)"
```

→ `True waveshare_stock 115200 False`. Nothing is copied to `/etc/mousedroid`; the overlay is forwarded per command.

- [ ] **3.5 Chassis power-up, passive signs of life (15 min).** Lift; plug the rover USB-C into the Jetson; power on. Watch: OLED banner within 3 s; WiFi SSID `UGV_*`/`WAVE_ROVER_*`; then on the rover:

```bash
ls -l /dev/serial/by-id/ | grep CP2102N
dmesg | grep -i cp210x | tail -3
```

Expect `…cc0ff02aab6cef119a38a4adc169b110-if00-port0 -> ../../ttyUSBn`. Resolve the port and listen 3 s at 115200 with an EN pulse (the container runs the mock, so the port is free; `privileged: true` exposes `/dev/ttyUSB*`):

```bash
ESP_PORT=$(readlink -f /dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_cc0ff02aab6cef119a38a4adc169b110-if00-port0)
docker exec -i mousedroid python3 - "$ESP_PORT" <<'PY'
import serial, sys, time
s = serial.Serial(sys.argv[1], 115200, timeout=0.5)
s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False       # EN pulse via the CP2102N auto-reset wiring
t0 = time.time(); n = 0
while time.time() - t0 < 3.0:
    d = s.read(256); n += len(d); sys.stdout.write(d.decode("utf-8", "replace"))
print(f"\nBYTES={n}")
PY
```

Alive: `rst:0x1 (POWERON_RESET)` banner and/or `{"T":1001,…}` frames. Silent: `BYTES=0`.

- [ ] **3.6 Active USB-C probe (5 min).** Same heredoc with `s.write(b'{"T":130}\n')` and three `s.readline()` attempts (2 s timeout) → expect a line with `"T":1001` carrying `L`,`R`,`v`. Trunk equivalent (on the rover):

```bash
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_JETSON_CONFIGS="$BENCH_CFG" mousedroid python3 -m mousedroid.cli.preflight --checks esp32 --json
```

→ `"status": "ok"`, `FEEDBACK_BASE_INFO after 1 poll(s)`. Answer → **skip 3.7–3.9, go to 3.10**.

- [ ] **3.7 ttyTHS1 jumper probe (15 min, only if 3.6 is silent).** Power everything off. Driver board 40-pin `U0TX` (pin 10) → Jetson pin 10 (UART1_RXD); `U0RX` (pin 8) → Jetson pin 8 (UART1_TXD); GND ↔ GND (pin 6); 3.3 V both sides. Power on, then on the rover:

```bash
systemctl is-active nvgetty && sudo systemctl stop nvgetty
```

Rerun 3.5 and 3.6 with `/dev/ttyTHS1` (no RTS pulse). Answer → the fault is the CP2102N/USB-C path, not the ESP32.

- [ ] **3.8 esptool sync (5 min).** On the rover:

```bash
docker exec mousedroid python3 -m pip install --no-cache-dir esptool        # only if esptool is absent
docker exec mousedroid esptool.py --chip esp32 --port "$ESP_PORT" --baud 115200 --before default_reset --after no_reset chip_id
```

→ `Chip is ESP32-D0WD…`, `Detected flash size: 4MB`. On `Failed to connect`: hold BOOT, tap EN/RESET, retry once. PC fallback:

```bash
py -m esptool --port COMx --baud 115200 chip_id
```

- [ ] **3.9 Reflash (only if silent on BOTH ports in 3.6+3.7 AND 3.8 syncs — CF-5).** This needs PR-G in the deployed SHA: the hash-pinned `scripts/flash_esp32.sh` (dry run by default, `--bin-dir`, `--write`) and the firmware record. A checkout without PR-G still has the original script, which takes `<port> <firmware.bin>` and writes at once; if the deployed checkout is that one, stop and use the Flash Download Tool from 3.0 instead.

Stage the images under the bind-mounted checkout, not the home directory: the container never mounts `/home/<user>`, so `docker exec` cannot see anything staged there. From the PC, in the repo root:

```bash
scp -r firmware $ROVER_USER@$ROVER_HOST:/opt/mousedroid/firmware
```

Then on the rover, the dry run (review the argv):

```bash
docker exec -w /opt/mousedroid mousedroid bash scripts/flash_esp32.sh --port "$ESP_PORT" --bin-dir /opt/mousedroid/firmware
```

Add `--write`:

```bash
docker exec -w /opt/mousedroid mousedroid bash scripts/flash_esp32.sh --port "$ESP_PORT" --bin-dir /opt/mousedroid/firmware --write
```

→ `Hash of data verified` ×4 and a clean `verify_flash`. Power-cycle; repeat 3.5 and 3.6. Then once: heredoc with `s.write(b'{"T":900,"main":2,"module":2}\n')`; power-cycle; confirm the OLED reports WAVE ROVER and 130→1001 answers.

- [ ] **3.10 Smoke ladder.** Rover lifted. Each stage below is this `docker exec`, with the stage command in place of `…`:

```bash
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_JETSON_CONFIGS="$BENCH_CFG" mousedroid …
```

- [ ] **3.10a** `bash scripts/jetson_smoke_test.sh serial`:

```bash
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_JETSON_CONFIGS="$BENCH_CFG" mousedroid bash scripts/jetson_smoke_test.sh serial
```

→ `PASS: serial send/receive` with `Got response: {"T":1001…}`. `PASS: serial open/write (no response…)` counts as **silent**, not green (task 4.1 makes it a FAIL).

- [ ] **3.10b** `bash scripts/jetson_smoke_test.sh power` (no motion vars):

```bash
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_JETSON_CONFIGS="$BENCH_CFG" mousedroid bash scripts/jetson_smoke_test.sh power
```

→ `PASS: power chain smoke`; log `power_chain_probe_complete … cmd=0.000m/s estop=<ms>ms (budget=50ms)`; a frame without `v` FAILs (R-6).

- [ ] **3.10c** **Motion** — add `-e MOUSEDROID_SMOKE_ALLOW_MOTION=1 -e MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` to the prefix, then `bash scripts/jetson_smoke_test.sh motor`:

```bash
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_JETSON_CONFIGS="$BENCH_CFG" -e MOUSEDROID_SMOKE_ALLOW_MOTION=1 -e MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true mousedroid bash scripts/jetson_smoke_test.sh motor
```

Wheels spin at 0.1 m/s then 0.15 m/s briefly → `PASS: motor loopback smoke`. Abort = power switch.

- [ ] **3.10d** Wrapper, on the host (from the checkout):

```bash
MOUSEDROID_JETSON_CONFIGS="$BENCH_CFG" bash scripts/jetson_full_smoke_run.sh
```

→ `SUMMARY.md` with system/usbc/gpio/serial/power/lidar/app_health/llm_probe/new_features PASS. Until task 4.2 lands the wrapper cannot arm the Pydantic gate (its `docker exec` forwards no `MOUSEDROID_ESP32__*` var), so its motor stage is zero-velocity by construction.

- [ ] **3.10e** The F-008 check, with both motion gates on the command line:

```bash
docker exec -w /opt/mousedroid -e MOUSEDROID_MOCK_HARDWARE=false -e MOUSEDROID_JETSON_CONFIGS="$BENCH_CFG" -e MOUSEDROID_SMOKE_ALLOW_MOTION=1 -e MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true mousedroid python3 scripts/validate.py --check F-008 2>&1 | tee /opt/mousedroid/reports/jetson_smoke/f008-check-$(date -u +%Y%m%dT%H%M%SZ).txt
```

→ last line `F-008: OK`. `spec.VALIDATION_TIMEOUT_S` is 600 s; on `validation_command timed out`, record the direct `bash scripts/jetson_smoke_test.sh` exit 0 as evidence and narrow the catalog command in PR-H.

- [ ] **3.10f** Evidence. From the PC, in the repo root:

```bash
mkdir -p reports/esp32_bench/2026-10-20
scp -r $ROVER_USER@$ROVER_HOST:/opt/mousedroid/reports/jetson_smoke/<stamp> reports/esp32_bench/2026-10-20/smoke-<stamp>/
```

Plus probe transcripts and `f008-check-*.txt` (`.txt`/`.md` only — `*.log` is gitignored at `.gitignore:112`); write `decision.md` in that same dated directory (leaf, commands, frames, firmware hash or "stock as shipped, unflashed"). Dispatch `hw-evidence-auditor`:

```text
Trace every hardware claim in `decision.md` to a file under `reports/esp32_bench/2026-10-20/`.
```

- [ ] **3.10g** Closeout (PR-H) per `.claude/skills/feature-closeout/SKILL.md`: F-008 `status: "done"`, `implemented_in: <full SHA the rover ran>`, `notes` += evidence path + firmware provenance; on the PC:

```bash
python scripts/validate.py --tier fast
python scripts/validate.py --tier fast,slow --strict-git
```

Then CHANGELOG/progress/NEXT_STEPS item 2. Side effect: the `.claude/workforce.yaml` freeze on `src/mousedroid/arm/**` self-disables.

## Decision tree

| Leaf | Evidence | Action | Record |
|---|---|---|---|
| L1 answers on USB-C (3.6) | `{"T":1001,…}` | no flash; 3.10a–g | "stock firmware as shipped; F-008 closed at `<sha>`" |
| L2 silent on USB-C, answers on ttyTHS1 (3.7) | 3.7 reply | try the secondary CP2102N (`7cbb01b7…`) by pinning `esp32.serial_port` in the bench overlay; answers → continue 3.10; else production over `/dev/ttyTHS1` needs a compose device + `serial_port` + `usbc_discovery` `required: false` change — a PR, not a bench hack | "ESP32 alive; CP2102N/USB-C path faulty; repair = bridge rework or PCB swap" |
| L3 silent on both, esptool syncs (3.8) | `Chip is ESP32…` | 3.9; re-run 3.5–3.6; still silent after a verified flash → UART-path fault → replace PCB | "ROM alive, app absent/corrupt; reflashed `<app sha256>`" or "replace" |
| L4 no sync | `Failed to connect` after the BOOT retry | multimeter: 3.3 V at the ESP32 module — rail absent → regulator/PSU; rail present → module/PCB dead | "replace General Driver board" |

## Time box

Session 1 (the first bench day) = 3.1–3.9 + the decision; session 2 (the next day) = one repair attempt + re-run of 3.5–3.10. No 1001 frame by the end of session 2 → order the replacement board; F-008 stays `todo` with `notes` → `decision.md`; NEXT_STEPS item 2 updated; Phase 4/5 proceed regardless.
