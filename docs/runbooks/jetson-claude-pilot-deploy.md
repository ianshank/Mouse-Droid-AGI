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
   # compose runs as `ian`; chown so the owner can read, then 600 (see findings #4).
   sudo chown ian:ian /etc/mousedroid/docker.env && sudo chmod 600 /etc/mousedroid/docker.env
   ```
5. **Recreate** (loads env + config + source):
   `docker compose -f docker-compose.jetson.yml up -d --force-recreate mousedroid`
   → Stage-1 validation (fallback; SDK still absent).
6. **Hot-install** the SDK (pin to the rover-validated version for a reproducible
   recovery): `docker exec mousedroid python3 -m pip install "anthropic==0.105.2"`
   `0.105.2` is the version validated live on the rover; the `Dockerfile.jetson` /
   `pyproject.toml` keep the `anthropic>=0.40` lower bound for the image build.
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
tier. Manual recovery commands in this runbook pin `anthropic==0.105.2` (the
version validated live on the rover) for reproducibility; the `Dockerfile.jetson`
/ `pyproject.toml` deliberately keep the `anthropic>=0.40` lower-bound range for
the image build.

## Live-deploy findings (first deploy, 2026-06-02) — host-specific reality

These were discovered deploying to the actual rover and are required for any
re-deploy until the image is rebuilt with this PR's `Dockerfile.jetson`:

1. **`MOUSEDROID_LLM__ENABLED=false` in `docker.env` gates everything.** A prior
   operator override disabled the gateway entirely; env beats YAML in
   pydantic-settings. Set `MOUSEDROID_LLM__ENABLED=true` or the gateway never runs
   regardless of `llm.enabled: true` in the overlay.

2. **Phi-3 fallback must run on CPU on this host: `MOUSEDROID_LLM__N_GPU_LAYERS=0`.**
   The world model already occupies the shared 7.4 GB iGPU, so a full-offload
   (`n_gpu_layers: -1`, the repo default) second model fails with
   `unable to allocate CUDA0 buffer`. CPU offload loads the GGUF via mmap (sits in
   reclaimable page cache, ~0 hard RAM, no swap pressure). The repo YAML keeps `-1`
   for hosts with GPU headroom — this is a **per-host `docker.env` override**, not a
   committed change. Trade-off: CPU inference is slower (acceptable for an
   off-network fallback) and may cause transient 30 Hz `loop_overrun` *only while a
   fallback translation is actively running*; idle steady-state is unaffected
   (verified 0 overruns post-startup).

3. **Recreate ordering (the SDK-wipe trap).** `docker compose up -d --force-recreate`
   wipes the writable-layer `anthropic` install (image lacks it until rebuilt). So
   the moment you change any `docker.env` value and recreate, the cloud primary
   degrades until you **reinstall + `restart`**:
   ```bash
   docker exec mousedroid python3 -m pip install --no-cache-dir "anthropic==0.105.2"
   docker compose -f docker-compose.jetson.yml restart mousedroid   # NOT recreate
   ```
   `restart` re-runs the process (re-imports the SDK, reloads config) without wiping
   the writable layer. Confirm both tiers via the `fallback_gateway_started` event:
   `primary_ready: true` (Claude) **and** `secondary_ready: true` (Phi-3).

4. **`docker.env` must be readable by the compose-running user.** `chmod 600` alone
   breaks `env_file` loading (compose runs as `ian`, file was root:root). Use
   `chown ian:ian` + `chmod 600` so the owner can read it (more secure than the
   prior world-readable `755`).

5. **Validating with `scripts/translate_mission.py` inside the live container** loads
   a *second* gateway → a second Phi-3 copy → GPU/RAM contention. Always pass
   `--config /etc/mousedroid/jetson_production.yaml` **or** rely on the container's
   `MOUSEDROID_CONFIG` env var — the probe now resolves it via
   `resolve_runtime_config_paths` (same as the orchestrator), so inside the container
   it picks up the production overlay automatically. The probe reports the
   actually-serving tier (`tier=primary` / `tier=secondary (local fallback)` /
   `none — both tiers degraded`). For the live production gateway state, the
   orchestrator's own `anthropic_gateway_*` / `fallback_gateway_started` log events
   remain authoritative.
