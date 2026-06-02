# Runbook — Jetson Claude-Pilot Deploy (PR #107)

Deploy the Anthropic Claude mission-translation gateway + Phi-3 off-network
fallback to the rover. Design: `docs/superpowers/specs/2026-06-02-jetson-claude-pilot-deploy-design.md`.

## Prerequisites
- Deploy branch pushed to `origin` (`feat/jetson-claude-pilot-deploy`).
- An `ANTHROPIC_API_KEY` (`sk-ant-...`) to provision (cloud tier; fallback works without it).
- SSH: `ssh ian@mousedroid.local` (WiFi). Container `mousedroid` healthy.

## Deploy sequence (ordering matters — see design §5)

The ordering is forced by two facts: a `docker exec ... pip install` lands in the
container's **writable layer** (wiped by `--force-recreate`), and the API key lives
only in the `env_file` (read at container **creation**). So: recreate first, install
the SDK second, `restart` (not recreate) third.

1. **Backup + validate** the live config:
   ```bash
   TS=$(date +%Y%m%d_%H%M%S)
   sudo cp -a /etc/mousedroid/jetson_production.yaml /etc/mousedroid/jetson_production.yaml.bak.$TS
   ```
   Then validate the backup parses (after source sync, step 2):
   `docker exec mousedroid python3 /opt/mousedroid/scripts/validate_configs.py --config-dir /etc/mousedroid --include-default`
2. **Sync source** (record current branch first for rollback):
   ```bash
   git -C /opt/mousedroid rev-parse --abbrev-ref HEAD     # record for rollback
   git -C /opt/mousedroid status --porcelain              # must be empty
   git -C /opt/mousedroid fetch origin
   git -C /opt/mousedroid checkout feat/jetson-claude-pilot-deploy
   ls /opt/mousedroid/src/mousedroid/llm_gateway/anthropic_gateway.py   # verify present
   ```
3. **Write config**: `sudo cp /opt/mousedroid/config/jetson_production.yaml /etc/mousedroid/jetson_production.yaml`; validate parse.
4. **Provision key** (editor, NOT `echo >>` — avoids shell history):
   ```bash
   sudo nano /etc/mousedroid/docker.env      # add: ANTHROPIC_API_KEY=sk-ant-...
   sudo chmod 600 /etc/mousedroid/docker.env
   ```
5. **Recreate** (loads env + config + source):
   `docker compose -f docker-compose.jetson.yml up -d --force-recreate mousedroid`
   → Stage-1 validation (fallback; SDK still absent).
6. **Hot-install** the SDK: `docker exec mousedroid python3 -m pip install "anthropic>=0.40"`
7. **Restart** (preserves the writable-layer SDK; recreate would wipe it):
   `docker compose -f docker-compose.jetson.yml restart mousedroid`
   → Stage-2 validation (cloud).

## Validation
- Probe (no motors needed):
  `docker exec mousedroid python3 /opt/mousedroid/scripts/translate_mission.py --mission "patrol left then stop"`
  - Stage-1 (no key/SDK): prints `tier=local-fallback (degraded primary)` + a GoalVector.
  - Stage-2 (key set, SDK installed): prints `tier=primary` + a GoalVector.
- Structured-log grep recipes (`docker logs mousedroid` / Loki):
  - `anthropic_gateway_degraded` — primary unreachable (expected off-network / no key).
  - `anthropic_gateway_recovered` — primary self-healed after cooldown re-probe.
  - `anthropic_gateway_slow` — cloud call exceeded latency_target_ms (tune if noisy).
- Confirm the 30 Hz loop / telemetry is unaffected throughout (Grafana dashboards).

## Rollback
1. Restore `/etc/mousedroid/jetson_production.yaml` from the validated `.bak.<ts>`.
2. `git -C /opt/mousedroid checkout <recorded-prior-branch>`.
3. `docker compose -f docker-compose.jetson.yml up -d --force-recreate mousedroid`.

## Durability note
The hot-installed `anthropic` survives `restart`/reboot but NOT a future
`--force-recreate`/rebuild. The Dockerfile `Stage 4b` bake (this PR) ensures
future image builds include it, so a later rebuild won't silently lose the cloud
tier.
