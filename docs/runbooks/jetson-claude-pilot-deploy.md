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

## Request profile — `llm.anthropic` (F-059 WS-A)

Retargeting `llm.model_name` is not an id swap. Claude Haiku 5.5 and Sonnet 5.5
answer HTTP 400 to any non-default `temperature`/`top_p`/`top_k`, and Haiku 5.5
thinks by default (thinking tokens count toward `max_tokens`). A bare swap would
400, flip the gateway to `_degraded`, and `FallbackLLMGateway` would quietly serve
Phi-3 while the overlay still said Haiku. The optional `llm.anthropic` block says
what the new model needs:

```yaml
llm:
  model_name: "claude-haiku-5-5"
  anthropic:
    omit_sampling_params: true   # default: send no temperature
    thinking: disabled           # or adaptive; omit to take the model default
    effort: low                  # low | medium | high; omit for the model default
```

- **Block absent (the default; both shipped overlays today):** the request is
  byte-identical to before — `temperature` is sent, nothing else is added.
- **Block present:** `temperature` is omitted unless `omit_sampling_params: false`;
  `thinking` / `output_config.effort` are sent only when set. `thinking: disabled`
  is accepted by Haiku 5.5 at effort `high` or below (the schema stops at `high`)
  and is a 400 on Sonnet 5.5, which needs `between_tools` instead.
- **SDK version:** the gateway asks the *installed* `anthropic` what
  `messages.create` takes. `anthropic>=1.0` removed the `temperature` keyword
  (passing it is a `TypeError` before any request is made, i.e. a silent fall
  back to Phi-3), so the legacy temperature is forwarded through `extra_body`; an
  SDK without `output_config` is sent no `effort` (one
  `anthropic_effort_unsupported_by_sdk` warning). `Dockerfile.jetson` installs
  `anthropic>=0.40`, which resolves to 1.x on a rebuild; the rover-validated
  hot-install above is `0.105.2`.
- **Env:** `MOUSEDROID_LLM__ANTHROPIC__THINKING=disabled` sets one key;
  `MOUSEDROID_LLM__ANTHROPIC=null` removes the block (the "before" half of a
  paired probe run, with `MOUSEDROID_LLM__MODEL_NAME=claude-haiku-4-5`).

**The overlay retarget lands with the probe evidence**, not before. The probe
builds the same cloud + Phi-3 composite as the rover, so it loads a second Phi-3
copy (see finding 5 below: pass the production config, mind iGPU/RAM contention).
The block can be exercised without touching any overlay, using env alone (which
also keeps the `config-compat` gate out of the way until the retarget PR):
`MOUSEDROID_LLM__MODEL_NAME=claude-haiku-5-5 MOUSEDROID_LLM__ANTHROPIC__THINKING=disabled`.
On the Jetson:

```bash
docker exec mousedroid python3 /opt/mousedroid/tools/llm_latency_probe.py \
    --config /etc/mousedroid/jetson_production.yaml --iterations 20 \
    --report-json /opt/mousedroid/reports/llm_latency/<date>/haiku-5-5.json
```

The report records `model_requested`, `model_resolved` (via `client.models.retrieve`),
`served` and p50/p95/p99. `served` is `primary` only when the cloud tier answered
**every** iteration; a run a fallback answered reads `fallback`, `passed: false`, and
exits `4` — the latency of Phi-3 is not evidence about Haiku. Retarget the overlays
only after `served: primary` with p95 under `latency_target_ms`, then record the run
with `tools/credit_ledger.py` (caps live in `.claude/workforce.yaml` under `credit:`;
`python -m tools.credit_ledger status`, run from the repo root, prints spend against
them). The overlays carry an `anthropic:` key only from the retarget PR on; the
deployed image's schema rejects unknown keys, so that PR also bumps
`deployments/jetson-image.json` to a trunk SHA that has `AnthropicRequestConfig`.

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
