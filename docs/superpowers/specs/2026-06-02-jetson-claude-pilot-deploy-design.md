# Design: Deploy PR #107 (Claude LLM Gateway + Cloud→Local Failover) to the Jetson

- **Date:** 2026-06-02
- **Author:** Ian Cruickshank (with Claude Code)
- **Status:** Approved (design) — pending spec review
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

**Additive only.** All new fields have schema defaults, so the change is
backwards-compatible. Existing local-tier fields are **retained** for the fallback
`llama_cpp` tier (which reuses the same `LLMConfig` with only `backend` overridden —
verified in `factory._build_single_llm_gateway` / `build_llm_gateway`).

Target block:

```yaml
llm:
  enabled: true

  # --- Primary tier: Claude (Anthropic) ---
  backend: "anthropic"
  # No model hardcoded in the gateway; Haiku = lowest-latency on-rover default.
  model_name: "claude-haiku-4-5"
  # api_key intentionally absent — supplied via ANTHROPIC_API_KEY /
  # MOUSEDROID_LLM__API_KEY in /etc/mousedroid/docker.env (SecretStr).
  request_timeout_s: 20.0          # cloud round-trips are seconds, not ms
  # Cloud-appropriate; the 500 ms local default would spam anthropic_gateway_slow.
  latency_target_ms: 5000.0
  max_tokens: 256
  max_command_len: 512
  temperature: 0.1

  # --- Fallback tier: local Phi-3-mini via llama_cpp (off-network) ---
  fallback_backend: "llama_cpp"
  # REUSED: already-staged, F-006-GPU-tuned Phi-3-mini. llama_cpp reads model_path
  # and ignores model_name, so no fallback_model_name needed.
  model_path: "/opt/mousedroid/models/Phi-3-mini-4k-instruct-q4.gguf"
  context_length: 2048
  n_threads: 6
  n_gpu_layers: -1                 # F-006: offload all layers to iGPU
  n_batch: 32

  # --- Failover dynamics ---
  # After a transient WAN dropout, the composite serves locally and re-probes
  # Claude on this cadence; a successful re-probe clears the degrade.
  fallback_retry_cooldown_s: 30.0
```

> **Schema note (verified against `schema.py` @ `8c1ba09`):** The cooldown field
> is **`fallback_retry_cooldown_s`** (default `30.0`, `gt=0`). There is **no**
> `primary_recovery_interval_s` field — the PR #107 round-2 narrative used an
> earlier name that was consolidated before merge; putting it in YAML would fail
> validation. `fallback_model_name` is **not** set: the canonical
> `anthropic → llama_cpp` pairing needs no override (llama_cpp loads `model_path`,
> not `model_name`). `latency_target_ms`, `max_tokens`, `temperature`,
> `request_timeout_s` are shared by both tiers (one `LLMConfig`); the cloud values
> mean the local tier simply never logs "slow" (Phi-3 on GPU is well under 5 s) and
> ignores `request_timeout_s` (in-process). Matches the flat-block design of
> `config/jetson_claude_pilot.yaml`.

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

### 4.4 Tests (project tier discipline)

- **Regression / AQA** (`tests/regression/test_jetson_claude_pilot_config.py`):
  - `config/jetson_production.yaml` loads under the Pydantic `Settings`.
  - `llm.backend == "anthropic"`, `llm.fallback_backend == "llama_cpp"`,
    `llm.model_path` resolves to the Phi-3 GGUF name.
  - **Backwards-compat:** a minimal pre-#107 `llm:` block (no `backend`/
    `fallback_backend`) still loads, defaulting to `llama_cpp` / `none`.
  - Mirrors PR #110's config-pinning regression style.
- **Integration** (reuse/extend `tests/integration/test_anthropic_gateway_wiring.py`):
  - `build_llm_gateway` on the production config returns a `FallbackLLMGateway`
    composite (anthropic primary + llama_cpp secondary). Mock the SDK; no network.
- **Config validation:** `python scripts/validate_configs.py --include-default`
  passes with the updated overlay.

### 4.5 Runbook — `docs/runbooks/jetson-claude-pilot-deploy.md`

Operator-facing deploy sequence, secret provisioning, structured-log grep recipes
(`anthropic_gateway_*`, fallback events), staged validation, and rollback.

---

## 5. Change-set B — Live Deploy (`ian@mousedroid.local`)

Ordered, each step idempotent and reversible:

1. **Pre-flight + backup.** Confirm container healthy; `git -C /opt/mousedroid status`
   clean; `sudo cp /etc/mousedroid/jetson_production.yaml{,.bak.<ts>}`.
2. **Sync source.** In `/opt/mousedroid`: `git fetch` + checkout `feat/jetson-claude-pilot-deploy`
   (editable bind-mount → gateway code live immediately). Verify the new
   `src/mousedroid/llm_gateway/anthropic_gateway.py` is present.
3. **Hot-install anthropic.** `docker exec mousedroid python3 -m pip install "anthropic>=0.40"`;
   verify `python3 -c "import anthropic; print(anthropic.__version__)"`.
4. **Provision key (operator).** Add `ANTHROPIC_API_KEY=…` to
   `/etc/mousedroid/docker.env`. (Claude prepares the line; operator pastes value.)
5. **Deploy config.** `sudo` write the merged `jetson_production.yaml` to
   `/etc/mousedroid/` (validate it parses first).
6. **Restart.** `docker compose -f docker-compose.jetson.yml up -d` (picks up env +
   config; code already live via mount).
7. **Validate** — see §6.

---

## 6. Validation (staged)

**Stage 1 — fallback path (no key required; proves off-network autonomy):**

- Gateway builds as `FallbackLLMGateway`.
- With no/invalid key, primary degrades cleanly (`anthropic_gateway_degraded`),
  and the **Phi-3 fallback returns a valid normalised `GoalVector`** (∈ [-1, 1]).
- Confirm via structlog events + an operator dry-run translation probe (§7).

**Stage 2 — cloud path (after operator confirms key):**

- Cloud tier serves a translation; `anthropic_gateway_*` shows success.
- Simulate WAN loss (e.g. block egress) → fail over to Phi-3 → restore →
  cooldown re-probe self-heals (`anthropic_gateway_recovered`).
- Confirm the 30 Hz reactive loop health/telemetry is unaffected throughout.

**Always:** full local test suite green pre-push; CI green on the PR.

---

## 7. Logging & Debugging

- **Reuse PR #107's structured events** (no new ad-hoc logging): `anthropic_gateway_recovered`,
  `anthropic_gateway_slow`, `anthropic_gateway_degraded`, and the fallback composite's
  failover/restore events. Grep recipes documented in the runbook.
- **Operator dry-run probe:** a small, reusable mission-translation diagnostic
  (NL mission → printed `GoalVector`, no motor command) so the deliberative path can
  be verified live without the (physically dead) ESP32. Reuse existing tool/CLI
  patterns (e.g. the `greet_intro.py --dry-run` shape) rather than inventing a new one;
  confirm whether PR #107 already ships such an entry point before adding one.

---

## 8. Cross-cutting Requirements (acceptance criteria)

- **No hardcoded values:** every knob via the Pydantic `llm:` block or `docker.env`;
  fallback reuses the present Phi-3 `model_path`.
- **Backwards-compatible:** `backend`/`fallback_backend` defaults keep all existing
  YAML loading unchanged; regression test enforces it.
- **Reusable components:** lean on existing `build_llm_gateway`, `FallbackLLMGateway`,
  injection filter, and structured-log events; no reimplementation.
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
| Config edit breaks parse → container crash-loop | Validate parse before `sudo` write; timestamped `.bak`; rollback = restore `.bak` + restart |
| `anthropic` hot-install lost on container recreate | Baked into `Dockerfile.jetson` (§4.1) for the next rebuild |
| No API key → cloud tier dead | By design degrades to Phi-3 fallback (Stage-1 validation proves this) |
| Source checkout conflicts with local edits in `/opt/mousedroid` | Pre-flight `git status` gate; stash/abort if dirty |
| Disk pressure (14 GB free) | No GGUF download (Phi-3 reused); no image rebuild this pass |
| ESP32 dead → can't validate motion | Out of scope; deliberative path validated via `GoalVector` inspection |

---

## 10. Rollback Procedure

1. Restore `/etc/mousedroid/jetson_production.yaml` from the `.bak.<ts>` copy.
2. `git -C /opt/mousedroid checkout feat/jetson-rover-usbc-smoke` (prior ref `f96b9c2`).
3. `docker compose -f docker-compose.jetson.yml up -d`.
4. (anthropic hot-install is harmless to leave; it only activates when `backend=anthropic`.)
