# Design: Deploy PR #107 (Claude LLM Gateway + Cloud→Local Failover) to the Jetson

- **Date:** 2026-06-02
- **Author:** Ian Cruickshank (with Claude Code)
- **Status:** Peer-reviewed + revised (2026-06-02) — pending user sign-off → writing-plans
- **Deploy branch:** `feat/jetson-claude-pilot-deploy`
- **PR base:** `claude/markdown-implementation-plan-aVJ2l`
- **Subject of deploy:** PR #107 — "Anthropic Claude LLM gateway + cloud/local failover for rover missions" (merge commit `8c1ba09`)
- **Target device:** `ian@mousedroid.local` (WiFi `192.168.4.34`; USB-C `192.168.55.1` currently down)

---

## 1. Goal & Scope

Bring PR #107's **deliberative mission-translation path** live on the rover:

- **Primary tier:** Claude (Anthropic Messages API) translates a natural-language
  mission → normalised `GoalVector` (vx, vy, omega ∈ [-1, 1]).
- **Fallback tier:** local Phi-3-mini GGUF (via `llama_cpp`) serves the same
  translation when `api.anthropic.com` is unreachable (off-network autonomy).
- The **30 Hz reactive hot path** (RSSM world model → MCTS → ESP32 velocity
  command) is **untouched** — no LLM in the E-stop / motor path.

Two deliverables:

- **A — Repo changes** on a branch with full CI (Dockerfile dependency layer,
  config merge, env example, tests, runbook).
- **B — Live deploy** to the running Jetson container (source sync, hot-install,
  secret provisioning, config deploy, restart, staged validation).

### Non-goals

- No change to the reactive controller, safety monitor, or motor-command path.
- No Docker image rebuild during this effort (deferred; baked into `Dockerfile.jetson`
  for the *next* scheduled rebuild — see §4.1). The 14 GB free eMMC headroom makes a
  full rebuild risky and unnecessary for a pure-Python wheel.
- No physical ESP32 repair (tracked separately). The deliberative path is validated
  by inspecting the produced `GoalVector`, which does not require working motors.

---

## 2. Current State (verified via live recon 2026-06-02)

| Component | Live state on Jetson | Implication |
|---|---|---|
| Container `mousedroid:jetson` | Up 10 h (healthy); full Grafana/Prometheus/Loki/Promtail/node-exporter stack up | Restart, don't rebuild |
| `/opt/mousedroid` source | branch `feat/jetson-rover-usbc-smoke` @ `f96b9c2` (PR #106 era) | **No PR #107 code** — must sync |
| Container interpreter | `/usr/local/bin/python3` (3.10.12) | Hot-install via `python3 -m pip` |
| `anthropic` SDK | **Absent** (`ImportError`); not installed by `Dockerfile.jetson` | Bake layer + hot-install |
| `llama_cpp` | Present (`import llama_cpp` OK) | Fallback runtime ready |
| Local GGUF | `Phi-3-mini-4k-instruct-q4.gguf` (2.3 GB) staged & GPU-tuned | **Reuse** (decision §3) |
| `ANTHROPIC_API_KEY` | **Not** in `/etc/mousedroid/docker.env` (only `MOUSEDROID_LLM__ENABLED`) | Operator-provided (decision §3) |
| `/etc/mousedroid/` | Root-owned **file copies** (not a git checkout); operator keeps timestamped `.bak`s | `sudo` write + back up before edit |
| `config/jetson_production.yaml` (repo == live) | `llm:` block is **local-only** but F-006-tuned (`n_gpu_layers: -1`) | **Merge**, never overwrite |
| Free disk / RAM | 14 GB free eMMC (75% used); 5.7 GB RAM available | Hot-install fine; rebuild risky |

---

## 3. Key Decisions

1. **Fallback model = reuse staged Phi-3-mini.** The pilot example
   (`config/jetson_claude_pilot.yaml`) names `llama-3-8b`, but a GPU-tuned
   Phi-3-mini-4k GGUF is already present and wired. Phi-3-mini handles the simple
   JSON `GoalVector` translation; reusing it avoids a redundant ~4.7 GB download
   on a 14 GB-free disk and honours "no hardcoded values / reuse what exists".
   `model_path` already points at it — the fallback `llama_cpp` tier inherits it
   for free (see §4.2 schema note).
2. **Dependency delivery = bake + hot-install.** Add a non-fatal `anthropic` layer
   to `Dockerfile.jetson` (reproducibility for the next rebuild) **and**
   `docker exec … pip install` into the running container now (immediate bring-up,
   no rebuild).
3. **Secret = operator-provisioned.** Claude requires `ANTHROPIC_API_KEY` (or
   `MOUSEDROID_LLM__API_KEY`) in `/etc/mousedroid/docker.env`. Claude Code prepares
   the exact line + instructions; the operator pastes the `sk-ant-…` value. The
   secret never lands in logs, commits, or chat. Cloud-tier validation runs only
   after the operator confirms the key is set.
4. **Branching:** `feat/jetson-claude-pilot-deploy` off the integration branch tip
   (`8c1ba09`, which contains #107). PR targets `claude/markdown-implementation-plan-aVJ2l`.
5. **Deploy ordering (post-review):** recreate → hot-install → `restart` (not a
   second recreate), because the SDK lands in the writable layer and the key lives in
   the `env_file`. See §5 rationale.
6. **Networking:** `docker-compose.jetson.yml` uses `network_mode: host` (verified),
   so egress to `api.anthropic.com` needs **no** Docker network config — it works iff
   the Jetson host has internet. WAN-loss simulation must therefore be a host-level
   egress block, not a container network toggle.

---

## 4. Change-set A — Repo

### 4.1 `Dockerfile.jetson` — anthropic dependency layer

Add a non-fatal install mirroring the established pattern (Stage 4, the LLM-gateway
area near the `llama_cpp` copy + `huggingface-hub` install):

```dockerfile
# Stage 4b: Anthropic Claude SDK (cloud LLM tier) — non-fatal, graceful fallback.
# PR #107 gateway degrades to the local llama_cpp tier when this is absent, so a
# failed install must not break the build (matches the hardware/GCP layer policy).
RUN pip install --no-cache-dir "anthropic>=0.40" \
    || echo "WARNING: anthropic install failed (cloud Claude tier disabled; local fallback only)"
```

- Pure-Python wheel; no native build, no OOM risk (unlike `llama_cpp`).
- Preserves the graceful-degradation invariant: image still builds if PyPI is
  unreachable.

### 4.2 `config/jetson_production.yaml` — merge anthropic + fallback into the `llm:` block

**Minimal additive diff.** The principle (revised after peer review): change
**only** the fields the cloud tier requires, and **retain every existing
local-tuned value verbatim** — they are inherited by the `llama_cpp` fallback tier
(which reuses the same `LLMConfig` with only `backend` overridden — verified in
`factory._build_single_llm_gateway` / `build_llm_gateway`). All new fields have
schema defaults, so existing YAML loads byte-identical (backwards-compat).

The existing (live + repo) `llm:` block is:

```yaml
llm:
  enabled: true
  model_path: "/opt/mousedroid/models/Phi-3-mini-4k-instruct-q4.gguf"
  context_length: 2048
  n_threads: 6
  n_gpu_layers: -1        # F-006: offload all layers to iGPU
  n_batch: 32
  max_tokens: 128
  temperature: 0.1
  latency_target_ms: 500.0
```

**Diff to apply — six lines changed/added, the rest untouched:**

```yaml
llm:
  enabled: true

  # --- CHANGED: select Claude (Anthropic) as the primary tier ---
  backend: "anthropic"                 # was implicit "llama_cpp" (default)
  model_name: "claude-haiku-4-5"       # ADDED — no model id is hardcoded in the gateway
  request_timeout_s: 20.0              # ADDED — cloud round-trips are seconds, not ms
  latency_target_ms: 5000.0            # CHANGED 500.0 -> cloud value (avoids anthropic_gateway_slow spam)
  # api_key intentionally absent — supplied via ANTHROPIC_API_KEY /
  # MOUSEDROID_LLM__API_KEY in /etc/mousedroid/docker.env (SecretStr).

  # --- ADDED: off-network fallback tier (local Phi-3-mini via llama_cpp) ---
  fallback_backend: "llama_cpp"        # ADDED — wraps primary+secondary in FallbackLLMGateway
  fallback_retry_cooldown_s: 30.0      # ADDED — re-probe degraded cloud primary on this cadence

  # --- RETAINED verbatim (drive the llama_cpp fallback tier — do NOT change) ---
  model_path: "/opt/mousedroid/models/Phi-3-mini-4k-instruct-q4.gguf"
  context_length: 2048
  n_threads: 6
  n_gpu_layers: -1
  n_batch: 32            # KEEP: already the deployed value (NOT 512); tuned for Phi-3 on the iGPU
  max_tokens: 128        # KEEP existing; sufficient for a GoalVector JSON. Raise only if a
                         # verbose model truncates (shared by both tiers).
  temperature: 0.1
```

> **Schema note (verified against `schema.py` @ `8c1ba09`):** The cooldown field
> is **`fallback_retry_cooldown_s`** (default `30.0`, `gt=0`). There is **no**
> `primary_recovery_interval_s` field — the PR #107 round-2 narrative used an
> earlier name that was consolidated before merge.
>
> **Peer-review correction (false positive caught):** an independent reviewer
> flagged `n_batch: 32` as a regression "from the schema default 512". It is **not** —
> `32` is the value **already deployed** in both the live `/etc/mousedroid/jetson_production.yaml`
> and the repo `config/jetson_production.yaml`; we retain it unchanged. Likewise
> `max_tokens` stays at the existing `128` (the earlier draft's `256` and a stray
> `max_command_len: 512` were dropped — they were needless changes to the shared
> fallback tier, not additions the cloud tier requires).
>
> `fallback_model_name` is **not** set: the `anthropic → llama_cpp` pairing needs no
> override (llama_cpp loads `model_path`, not `model_name`). `latency_target_ms`,
> `max_tokens`, `temperature`, `request_timeout_s` are shared by both tiers (one
> `LLMConfig`); the cloud `latency_target_ms` just means the local tier never logs
> "slow" (Phi-3 on GPU is well under 5 s) and ignores `request_timeout_s` (in-process).

### 4.3 `config/docker.env.example` — document the secret

Append commented placeholders (no value):

```sh
# --- Claude (Anthropic) cloud LLM tier (PR #107) ---
# Provide ONE of these to enable the cloud mission-translation tier. Never commit
# a real key. The Anthropic SDK reads ANTHROPIC_API_KEY natively; the schema-mapped
# override is SecretStr-wrapped and kept out of YAML.
# ANTHROPIC_API_KEY=sk-ant-...
# MOUSEDROID_LLM__API_KEY=sk-ant-...
```

### 4.4 New operator probe — `scripts/translate_mission.py` (NEW WORK, not reuse)

> **Peer-review correction:** the earlier draft (§7) implied an NL→`GoalVector`
> dry-run CLI already existed to "reuse". It does **not** — verified: `scripts/`
> has only `greet_intro.py` (voice), `src/mousedroid/cli/` has only
> `preflight.py`/`validate_pillars.py`, `pyproject [project.scripts]` exposes only
> `mousedroid`, and `llm_gateway/mission_parser.py` is a rule-based parser with no
> entry point. So this probe is **new work**, scoped into Change-set A.

A small, reusable diagnostic so the deliberative path can be validated live
**without the (physically dead) ESP32**:

- `python scripts/translate_mission.py --mission "patrol left then stop" [--config <path>]`
- Loads `Settings` via the existing loader, calls `build_llm_gateway(cfg)`, awaits
  `start()`, translates the mission, prints the resulting `GoalVector` + the
  gateway's `is_degraded`/tier-served state, then `stop()`. No motor command issued.
- Follows the `greet_intro.py` CLI shape (argparse, structlog, exit codes).
- Note: called outside the orchestrator, `build_llm_gateway` builds its own default
  injection filter (each gateway constructs one when `injection_filter=None`) — fine
  for a probe; documented so the behaviour isn't mistaken for the shared-filter path.

### 4.5 Tests (project tier discipline) — **two distinct tests, distinct fixtures**

- **Config-pinning (regression)** (`tests/regression/test_jetson_claude_pilot_config.py`):
  loads the **edited** `config/jetson_production.yaml` and asserts
  `llm.backend == "anthropic"`, `llm.fallback_backend == "llama_cpp"`,
  `llm.model_path` resolves to the Phi-3 GGUF name. Pins the deployed config against
  schema drift (PR #110 style). *Only passes after §4.2 is applied — by design.*
- **Backwards-compat (regression), SEPARATE fixture**: an in-test minimal pre-#107
  `llm:` block (only `enabled` + `model_path` + local fields, **no** `backend`/
  `fallback_backend`) still loads and defaults to `backend="llama_cpp"` /
  `fallback_backend="none"` → single gateway, byte-identical to pre-#107.
- **Integration** (reuse/extend `tests/integration/test_anthropic_gateway_wiring.py`):
  `build_llm_gateway` on the production config returns a `FallbackLLMGateway`
  (anthropic primary + llama_cpp secondary). Mock the SDK; no network.
- **Probe unit test** (`tests/unit/test_translate_mission_cli.py`): the new CLI
  produces a clamped `GoalVector` and exit 0 with a mocked gateway; degraded-path
  exit semantics covered.
- **Config validation:** `python scripts/validate_configs.py --include-default`
  passes (verified safe without an API key — it only loads `Settings`, never builds
  a gateway or imports `anthropic`).

### 4.6 Runbook — `docs/runbooks/jetson-claude-pilot-deploy.md`

Operator-facing deploy sequence (the corrected ordering in §5), secret provisioning,
structured-log grep recipes (`anthropic_gateway_*`, fallback events), staged
validation, and rollback.

---

## 5. Change-set B — Live Deploy (`ian@mousedroid.local`)

> **Ordering rationale (revised after peer review).** Two facts from the device +
> compose force the order:
> 1. `docker-compose.jetson.yml` has **no bind-mount over site-packages**, so a
>    `docker exec … pip install` lands in the container's **writable layer** — it is
>    **wiped by `--force-recreate`** (and by a rebuild).
> 2. The new `ANTHROPIC_API_KEY` lives **only** in the external `env_file`
>    (`/etc/mousedroid/docker.env`); env is injected at **container creation**, so a
>    plain `up -d` may report "up-to-date" and leave it unread. Loading it needs
>    `--force-recreate`.
>
> Therefore: **recreate FIRST** (loads the new env + config + synced source), **then
> hot-install** `anthropic` into the recreated container, **then `restart`** (which
> re-runs the process so `start()` imports the SDK, *without* recreating — so the
> writable-layer install survives). `restart` ≠ `recreate`. The Dockerfile bake
> (§4.1) is what makes a *future* `--force-recreate`/rebuild safe.

Prerequisite (workstation): **push the deploy branch** so the rover can fetch it —
`git push -u origin feat/jetson-claude-pilot-deploy` (the rover clones from GitHub;
the branch is currently workstation-local only).

Ordered steps, each idempotent and reversible:

1. **Pre-flight + safe backup.**
   - Confirm `mousedroid` container healthy; record the rover's current branch:
     `git -C /opt/mousedroid rev-parse --abbrev-ref HEAD` (for deterministic rollback —
     currently `feat/jetson-rover-usbc-smoke` @ `f96b9c2`).
   - `git -C /opt/mousedroid status --porcelain` must be empty; abort/stash if dirty.
   - `sudo cp -a /etc/mousedroid/jetson_production.yaml /etc/mousedroid/jetson_production.yaml.bak.<ts>`
     **and validate the backup parses** (`validate_configs.py --config-dir /etc/mousedroid
     --include-default`) so rollback can't restore a config that crash-loops under
     `restart: unless-stopped`.
2. **Sync source.** `git -C /opt/mousedroid fetch origin` + checkout the deploy
   branch (a **tracked branch**, not a detached SHA, so rollback is deterministic).
   Verify `/opt/mousedroid/src/mousedroid/llm_gateway/anthropic_gateway.py` now exists
   on disk. (Live-verified: the container imports from `/opt/mousedroid/src`; the file
   is *present* now but only *loaded* on the next process start — step 5/7.)
3. **Deploy config.** `sudo` write the merged `jetson_production.yaml` to
   `/etc/mousedroid/`; **validate it parses** before trusting it.
4. **Provision key (operator).** Add `ANTHROPIC_API_KEY=sk-ant-…` to
   `/etc/mousedroid/docker.env` **using an editor** (not `echo >>`, which leaks the
   key into shell history); ensure the file is `chmod 600` root-owned. Claude prepares
   the exact line; the operator pastes the value. Never echoed to logs/chat.
5. **Recreate** to load env + config + source:
   `docker compose -f docker-compose.jetson.yml up -d --force-recreate`.
   → New container has the key in env and reads the new config; `anthropic` SDK is
   still absent, so the primary degrades and the **Phi-3 fallback serves** —
   **Stage-1 validation** (§6) runs here, proving off-network autonomy.
6. **Hot-install anthropic** into the recreated container:
   `docker exec mousedroid python3 -m pip install "anthropic>=0.40"`; verify
   `docker exec mousedroid python3 -c "import anthropic; print(anthropic.__version__)"`.
7. **Restart the process** (preserves the writable-layer SDK):
   `docker compose -f docker-compose.jetson.yml restart mousedroid`.
   → `start()` now imports `anthropic`; env (key) persists across `restart`; config
   re-read → cloud tier live. **Stage-2 validation** (§6).
8. **Durability note.** The hot install persists across `restart`/reboot but **not**
   across the next `--force-recreate`/rebuild — the §4.1 Dockerfile bake closes that
   gap for future image builds.

---

## 6. Validation (staged)

**Stage 1 — fallback path (after step-5 recreate; no SDK/key needed; proves
off-network autonomy):**

- Gateway builds as `FallbackLLMGateway`.
- `anthropic` absent → primary degrades cleanly (`anthropic_gateway_degraded`), and
  the **Phi-3 fallback returns a valid normalised `GoalVector`** (∈ [-1, 1]).
- Confirm via structlog events + the new `scripts/translate_mission.py` probe (§4.4).

**Stage 2 — cloud path (after step-6 install + step-7 restart, key provisioned):**

- Cloud tier serves a translation; `anthropic_gateway_*` shows success.
- Simulate WAN loss → fail over to Phi-3 → restore → cooldown re-probe self-heals
  (`anthropic_gateway_recovered`). **Note:** with `network_mode: host` (§3), the
  realistic way to simulate WAN loss is a host-level egress block / firewall rule to
  `api.anthropic.com`, not a Docker network toggle.
- Confirm the 30 Hz reactive loop health/telemetry is unaffected throughout.

**Always:** full local test suite green pre-push; CI green on the PR.

---

## 7. Logging & Debugging

- **Reuse PR #107's structured events** (no new ad-hoc logging): `anthropic_gateway_recovered`,
  `anthropic_gateway_slow`, `anthropic_gateway_degraded`, and the fallback composite's
  failover/restore events. Grep recipes documented in the runbook.
- **Operator dry-run probe:** `scripts/translate_mission.py` (NL mission → printed
  `GoalVector`, no motor command) lets the deliberative path be verified live without
  the (physically dead) ESP32. This is **new work** (§4.4) — verified that no such
  entry point ships in PR #107 — built in the `greet_intro.py` CLI shape.

---

## 8. Cross-cutting Requirements (acceptance criteria)

- **No hardcoded values:** every knob via the Pydantic `llm:` block or `docker.env`;
  fallback reuses the present Phi-3 `model_path`.
- **Backwards-compatible:** `backend`/`fallback_backend` defaults keep all existing
  YAML loading unchanged; regression test enforces it.
- **Reusable components:** lean on existing `build_llm_gateway`, `FallbackLLMGateway`,
  injection filter, and structured-log events; no reimplementation. The one new
  artifact (`scripts/translate_mission.py`, §4.4) is a thin CLI over `build_llm_gateway`
  in the established `greet_intro.py` shape — not a reimplementation of gateway logic.
- **Network egress:** cloud tier reaches `api.anthropic.com` via `network_mode: host`;
  no compose network change required.
- **Lint/type/format:** `python -m ruff check src/ tests/` (pinned 0.8.0, **not** bare
  `ruff`), `python -m ruff format --check`, `python -m mypy --strict src/mousedroid/`.
- **numpy:** PR #107's diff has no numpy surface; full-suite run guards against
  incidental numpy/runtime regressions from the config/dep change.
- **Full test suite:** `pytest -m "not hardware" --cov --cov-fail-under=85` locally;
  all CI stages on the PR; hardware-tier checks on the rover.

---

## 9. Risks & Rollback

| Risk | Mitigation |
|---|---|
| **Sequencing**: `--force-recreate` wipes the writable-layer `anthropic` install | Recreate **before** install; `restart` (not recreate) after install; bake into Dockerfile for future rebuilds (§5 rationale) |
| **Env stale**: plain `up -d` leaves new `docker.env` key unread | Use `up -d --force-recreate` for the env-loading step (§5.5) |
| Config edit breaks parse → crash-loop under `restart: unless-stopped` | Validate the **new** config parses before write **and** validate the `.bak` at backup time so rollback is known-good |
| `anthropic` hot-install lost on recreate/rebuild | Baked into `Dockerfile.jetson` (§4.1) for the next rebuild |
| No API key → cloud tier dead | By design degrades to Phi-3 fallback (Stage-1 validation proves this) |
| Deploy branch not on the rover's remote | Push to `origin` first (§5 prerequisite) |
| Detached HEAD / dirty tree in `/opt/mousedroid` | Pre-flight `git status --porcelain` gate + record `--abbrev-ref HEAD`; check out a **tracked branch**, not a SHA |
| Secret leaks into shell history | Edit `docker.env` via an editor (not `echo >>`); `chmod 600`; never echo the value |
| Egress to `api.anthropic.com` | `network_mode: host` (verified) → no Docker network config needed; only a genuinely off-network rover blocks it (the designed fallback trigger) |
| Disk pressure (14 GB free) | No GGUF download (Phi-3 reused); wheel + deps < ~50 MB; no image rebuild this pass |
| ESP32 dead → can't validate motion | Out of scope; deliberative path validated via `GoalVector` inspection (§4.4 probe) |

---

## 10. Rollback Procedure

1. Restore `/etc/mousedroid/jetson_production.yaml` from the validated `.bak.<ts>` copy.
2. `git -C /opt/mousedroid checkout <recorded-prior-branch>` (e.g. `feat/jetson-rover-usbc-smoke`, prior ref `f96b9c2`).
3. `docker compose -f docker-compose.jetson.yml up -d --force-recreate` (reload prior env + config + source).
4. (The `anthropic` hot-install is harmless to leave; it only activates when `backend=anthropic`.)

---

## 11. Peer-Review Record (objective, 2026-06-02)

Reviewed by two independent agents (a code-claims reviewer over the merged code, an
ops-mechanics reviewer over Dockerfile/compose) **plus direct live-device
verification**. Dispositions:

**Accepted — spec changed:**
- **HIGH — deploy sequencing.** `--force-recreate` wipes the writable-layer
  `anthropic` install; the plan originally installed *before* recreating → Stage-2
  would run without the SDK. Fixed: recreate → install → `restart` (§5).
- **HIGH — env reload.** Plain `up -d` can leave the new `docker.env` key unread;
  changed to `up -d --force-recreate` for the env-loading step (§5.5).
- **GAP — dry-run probe doesn't exist.** No NL→`GoalVector` CLI ships in PR #107;
  re-scoped from "reuse" to **new work** `scripts/translate_mission.py` + unit test
  (§4.4, §7).
- **MEDIUM — rollback safety.** Validate the `.bak` at backup time so rollback can't
  restore a crash-looping config; record prior branch; check out a tracked branch not
  a SHA; push the deploy branch to the rover's remote first (§5, §9).
- **MEDIUM — test fixtures.** Split the config-pinning test from the backwards-compat
  test (distinct fixtures) (§4.5).
- **LOW — secret hygiene / wording.** Edit `docker.env` via an editor + `chmod 600`;
  clarify "code present on checkout, loaded on restart" (§5).

**Confirmed sound (no change):**
- Editable import path == bind mount — **live-verified** (`import mousedroid` →
  `/opt/mousedroid/src`; `_editable_impl_mousedroid.pth`; WORKDIR `/opt/mousedroid`).
  `anthropic_gateway.py` confirmed absent on the live tree (sync genuinely needed).
- All proposed `llm:` field names/types/literals validate; `build_llm_gateway`
  produces the `FallbackLLMGateway(anthropic, llama_cpp)` composite; backwards-compat
  holds; `AnthropicLLMGateway` degrades (never raises) on missing SDK/key; `system_prompt`
  has a default; `validate_configs.py` is key-safe (loads `Settings` only).
- `network_mode: host` → egress unimpeded; hot-install lands in writable layer; 14 GB
  ample for the wheel.

**Reviewer finding rejected (false positive):**
- `n_batch: 32` was flagged as a regression "from default 512". It is **not** — `32`
  is the value already deployed in both the live and repo `jetson_production.yaml`;
  retained verbatim. (This prompted tightening §4.2 to a strict minimal diff and
  dropping the earlier draft's needless `max_tokens: 256` / `max_command_len` changes.)
