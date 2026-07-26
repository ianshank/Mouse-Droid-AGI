# Runbook — Jetson Claude-Pilot Deploy (PR #107 / #111)

Deploy the Anthropic Claude mission-translation gateway + Phi-3 off-network
fallback to the rover. Design: `docs/superpowers/specs/2026-06-02-jetson-claude-pilot-deploy-design.md`.

## Current deployed image (PR #111)

> **`deployments/jetson-image.json` is authoritative for the deployed SHA — this
> narrative is not.** The record was re-pinned after #111 (the original
> `9c31968` was a squash-source commit that became unreachable when its branch
> was deleted, which killed the `config-compat` gate repo-wide). Always read the
> current `sha` out of that file; never re-pin it to a feature-branch commit.

As of PR #111 the rover image `mousedroid:jetson` is rebuilt from the SHA
recorded in `deployments/jetson-image.json` and **bakes both** the PR #107
LLMConfig schema **and** the `anthropic` SDK (Dockerfile `Stage 4b`). Consequently
the cloud tier survives `docker compose up -d --force-recreate` with **no manual
hot-install** — the recommended deploy path below is a plain recreate. The
hot-install dance is retained only as a **fallback** (see step 5a) for an image
that predates #111, or where the *non-fatal* `Stage 4b` layer failed at build
time (e.g. PyPI unreachable).

## Prerequisites
- Deploy branch pushed to `origin` (`feat/jetson-claude-pilot-deploy`).
- An `ANTHROPIC_API_KEY` (`sk-ant-...`) to provision (cloud tier; the Phi-3 fallback works without it).
- SSH: `ssh ian@mousedroid.local` (WiFi). Container `mousedroid` healthy.

## Deploy sequence

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
5. **Recreate** on the baked image (loads env + config + source):
   `docker compose -f docker-compose.jetson.yml up -d --force-recreate mousedroid`
   With the #111 image, `anthropic` is baked and the key is in `docker.env`, so
   **both tiers come up directly** — no hot-install needed. Confirm via the
   `fallback_gateway_started` event: `primary_ready: true` (Claude) **and**
   `secondary_ready: true` (Phi-3). Proceed to Validation.

   **5a. Fallback — ONLY if `anthropic` is not baked** (pre-#111 image, or the
   non-fatal `Stage 4b` failed at build): if the log instead shows
   `anthropic_gateway_degraded_no_sdk`, the running image lacks the SDK. Recover
   it in the writable layer, then `restart` (NOT recreate — recreate would wipe
   the writable-layer install):
   ```bash
   docker exec mousedroid python3 -m pip install --no-cache-dir "anthropic==0.105.2"
   docker compose -f docker-compose.jetson.yml restart mousedroid
   ```
   `0.105.2` is the version validated on the rover; `Dockerfile.jetson` /
   `pyproject.toml` keep the `anthropic>=0.40` lower bound for the image build.

## Validation
- Probe (no motors needed):
  `docker exec mousedroid python3 /opt/mousedroid/scripts/translate_mission.py --mission "patrol left then stop"`
  - Online + key provisioned: prints `tier=primary` + a GoalVector (cloud Claude).
  - Off-network OR key absent: prints `tier=secondary (local fallback)` + a GoalVector (Phi-3).
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
The #111 image **bakes** `anthropic` (Dockerfile `Stage 4b`), so the cloud tier
survives `--force-recreate`/reboot — the prior writable-layer hot-install (which a
recreate would wipe) is no longer required for the deployed image. Keep
`deployments/jetson-image.json` pointed at the built-from SHA whenever the image
is rebuilt, so the CI config-schema-compat gate validates config YAML against the
schema that is actually deployed. The step-5a fallback commands pin
`anthropic==0.105.2` (the rover-validated version); the image build itself uses
the `anthropic>=0.40` lower-bound range in `Dockerfile.jetson` / `pyproject.toml`.

## Host-surface durability (F-017 — supersedes the manual procedure)

The per-host `docker.env` overrides documented below survive a reflash via
`sudo bash scripts/host_bootstrap.sh` (seeds `docker.env` from
`config/docker.env.example` only-if-absent; `--force` backs up first;
`--rollback` restores; `--dry-run` plans). Drift is self-diagnosing: enable
`host_env.enabled` in the Jetson overlay and the `host_env_keys` preflight
check WARNs when the deployed env file is missing template keys (names only
— values never reach a log).

## Live-deploy findings (first deploy, 2026-06-02) — host-specific reality

Discovered during the first deploy. Items 1, 2, 4, 5 are ongoing host realities;
item 3 (SDK-wipe) is **resolved for the deployed image** by the #111 rebuild and
kept here only as a fallback note.

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

3. **(Resolved by the #111 rebuild) Recreate SDK-wipe trap.** *Before* #111 the
   deployed image lacked `anthropic`, so `docker compose up -d --force-recreate`
   wiped the writable-layer install and the cloud primary degraded until a
   `reinstall + restart`. The #111 image **bakes** the SDK
   (see the SHA in `deployments/jetson-image.json`), so recreate is now safe and the
   cloud tier persists. If you ever run an image WITHOUT the baked SDK, recover via
   the step-5a fallback (`pip install "anthropic==0.105.2"` then `restart`, NOT
   recreate). Confirm both tiers via `fallback_gateway_started`:
   `primary_ready: true` (Claude) **and** `secondary_ready: true` (Phi-3).

4. **`docker.env` must be readable by the compose-running user.** `chmod 600` alone
   breaks `env_file` loading (compose runs as `ian`, file was root:root). Use
   `chown ian:ian` + `chmod 600` so the owner can read it (more secure than the
   prior world-readable `755`).

5. **Validating with `scripts/translate_mission.py` inside the live container** loads
   a *second* gateway → a second Phi-3 copy → GPU/RAM contention. Always pass
   `--config /etc/mousedroid/jetson_production.yaml` **or** rely on the container's
   `MOUSEDROID_CONFIG` env var — the probe resolves it via
   `resolve_runtime_config_paths` (same as the orchestrator), so inside the container
   it picks up the production overlay automatically. The probe reports the
   actually-serving tier (`tier=primary` / `tier=secondary (local fallback)` /
   `none — both tiers degraded`). For the live production gateway state, the
   orchestrator's own `anthropic_gateway_*` / `fallback_gateway_started` log events
   remain authoritative.
