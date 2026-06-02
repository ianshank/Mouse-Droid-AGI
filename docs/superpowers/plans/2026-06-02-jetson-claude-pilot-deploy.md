# Jetson Claude-Pilot Deploy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the merged PR #107 (Anthropic Claude LLM gateway + cloud→local failover) to the live Jetson rover — Claude as the deliberative mission-translation primary, the already-staged local Phi-3-mini as the off-network fallback — leaving the 30 Hz reactive control loop untouched.

**Architecture:** Two change-sets. **A (repo, branch `feat/jetson-claude-pilot-deploy`, PR base `claude/markdown-implementation-plan-aVJ2l`):** a non-fatal `anthropic` layer in `Dockerfile.jetson`, a minimal additive merge into the `llm:` block of `config/jetson_production.yaml`, a `docker.env.example` secret-doc, a new `scripts/translate_mission.py` operator probe, tests, and a runbook. **B (live deploy to `ian@mousedroid.local`):** push branch → backup → sync source → write config → operator key → `up -d --force-recreate` (Stage-1 fallback validation) → hot-install `anthropic` → `restart` (Stage-2 cloud validation). Ordering is dictated by two device facts: the SDK hot-install lands in the container's writable layer (wiped by recreate) and the API key lives only in the `env_file` (read at container creation).

**Tech Stack:** Python 3.10, Pydantic v2 `Settings`, `structlog`, `pytest`/`pytest-asyncio`, `ruff==0.8.0`, `mypy --strict`, Docker Compose, `anthropic>=0.40`, `llama-cpp-python`.

**Reference spec:** `docs/superpowers/specs/2026-06-02-jetson-claude-pilot-deploy-design.md` (peer-reviewed, §11 review record).

**Key invariants (CLAUDE.md):** no hardcoded values (all knobs via Pydantic/`docker.env`); backwards-compat (new fields keep defaults; existing YAML loads unchanged); reuse existing components (`build_llm_gateway`, `FallbackLLMGateway`, structured-log events); `mypy --strict` clean; full suite ≥85% coverage. **Lint MUST use `python -m ruff` (pinned 0.8.0), never bare `ruff`.**

---

## Verified API surface (use these exact names — do not guess)

- `GoalVector` dataclass fields: **`vx_target`, `vy_target`, `omega_target`** (floats in [-1,1]) — `src/mousedroid/llm_gateway/protocol.py:10`.
- `LLMGatewayProtocol`: `is_ready` (property), `async start() -> None`, `async translate_mission(nl_command: str) -> GoalVector`, `async stop() -> None`. `is_degraded` is a non-protocol attribute on concrete gateways (read via `getattr(gw, "is_degraded", False)`).
- `build_llm_gateway(cfg: Settings, *, injection_filter: PromptInjectionFilterProtocol | None = None) -> LLMGatewayProtocol` — `src/mousedroid/factory.py:793`. With `backend="anthropic"` + `fallback_backend="llama_cpp"` returns a `FallbackLLMGateway`.
- `FallbackLLMGateway(primary, secondary, *, retry_cooldown_s=30.0, clock=None)` — `src/mousedroid/llm_gateway/fallback_gateway.py:73`.
- `load_settings(*overlay_paths: Path, config_dir: Path | None = None) -> Settings` — `src/mousedroid/config/loader.py:58`.
- CLI template to mirror: `scripts/greet_intro.py` (structlog-to-stderr before importing `mousedroid`; `--config` append; exit codes 0/1/2).
- Existing tests that already guarantee backwards-compat (must stay green, do NOT duplicate): `tests/regression/test_pr107_backwards_compat.py`, `tests/regression/test_config_overlays_load.py`.

---

## PHASE A — Repo changes

### Task A0: Confirm branch & clean baseline

**Files:** none (state check).

- [ ] **Step 1: Verify branch and that PR #107 code is present**

Run:
```bash
cd "C:\Users\iansh\OneDrive\Documents\Gronk-Droid-Jetson-Nano"
git rev-parse --abbrev-ref HEAD
git log --oneline -1
ls src/mousedroid/llm_gateway/anthropic_gateway.py src/mousedroid/llm_gateway/fallback_gateway.py
```
Expected: branch `feat/jetson-claude-pilot-deploy`; HEAD is `08e2f15` (or later spec commit); both gateway files exist.

- [ ] **Step 2: Confirm working tree clean (spec commits only)**

Run: `git status --porcelain`
Expected: empty.

---

### Task A1: `anthropic` dependency layer in `Dockerfile.jetson`

**Files:**
- Modify: `Dockerfile.jetson` (insert after the `huggingface-hub` install in Stage 4)

- [ ] **Step 1: Add the non-fatal anthropic layer**

Edit `Dockerfile.jetson`. Find this line (end of Stage 4):
```dockerfile
RUN pip install --no-cache-dir "huggingface-hub>=0.20" diskcache || true
```
Insert immediately after it:
```dockerfile

# ---------------------------------------------------------------------------
# Stage 4b: Anthropic Claude SDK (cloud LLM tier — PR #107) — non-fatal.
# PR #107's gateway degrades to the local llama_cpp tier when this is absent,
# so a failed install must NOT break the build (matches the hardware/GCP
# layer policy). Pure-Python wheel — no native build, no OOM risk.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir "anthropic>=0.40" \
    || echo "WARNING: anthropic install failed (cloud Claude tier disabled; local fallback only)"
```

- [ ] **Step 2: Validate Dockerfile syntax (no build needed)**

Run: `docker compose -f docker-compose.jetson.yml config --quiet`
Expected: exit 0, no output (compose can parse the build context). If `docker` is unavailable on the workstation, skip — CI's docker stage validates this.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.jetson
git commit -m "build(jetson): non-fatal anthropic SDK layer for PR #107 cloud tier"
```

---

### Task A2: Merge anthropic + fallback into `config/jetson_production.yaml`

**Files:**
- Modify: `config/jetson_production.yaml` (the `llm:` block, ~lines 52-65)

- [ ] **Step 1: Apply the minimal additive diff**

Edit `config/jetson_production.yaml`. Replace the existing `llm:` block:
```yaml
llm:
  enabled: true
  model_path: "/opt/mousedroid/models/Phi-3-mini-4k-instruct-q4.gguf"
  context_length: 2048
  n_threads: 6
  # F-006: -1 offloads every layer to the iGPU so matmul stays on CUDA instead
  # of CPU (where Phi-3-mini-q4 runs at ~0.5 tok/s, blowing latency_target_ms).
  # Override per-host via MOUSEDROID_LLM__N_GPU_LAYERS=<int> in
  # /etc/mousedroid/docker.env (e.g. =0 on a CPU-only fallback host).
  n_gpu_layers: -1
  n_batch: 32
  max_tokens: 128
  temperature: 0.1
  latency_target_ms: 500.0
```
with:
```yaml
llm:
  enabled: true

  # --- Primary tier: Claude (Anthropic) — deliberative mission translation ---
  # PR #107. Cloud Claude translates NL missions -> GoalVector OUTSIDE the 30 Hz
  # hot loop; the deterministic RSSM->MCTS->ESP32 path is unchanged.
  backend: "anthropic"
  # No model id is hardcoded in the gateway. Haiku = lowest-latency on-rover
  # default for short velocity-goal translations; switch to a Sonnet id for
  # harder multi-step mission language.
  model_name: "claude-haiku-4-5"
  # api_key intentionally absent here — supply via ANTHROPIC_API_KEY or
  # MOUSEDROID_LLM__API_KEY in /etc/mousedroid/docker.env (SecretStr; never YAML).
  request_timeout_s: 20.0
  # Cloud round-trips are 1-5 s; the 500 ms local default would spam
  # anthropic_gateway_slow WARNINGs (code-reviewer PR #107 finding 3).
  latency_target_ms: 5000.0

  # --- Off-network fallback tier: local Phi-3-mini via llama_cpp ---
  # When api.anthropic.com is unreachable, the composite serves locally so the
  # rover stays autonomous. llama_cpp loads model_path (below) and ignores
  # model_name, so no fallback_model_name override is needed.
  fallback_backend: "llama_cpp"
  fallback_retry_cooldown_s: 30.0

  # --- Local-tier knobs (RETAINED verbatim — drive the llama_cpp fallback) ---
  model_path: "/opt/mousedroid/models/Phi-3-mini-4k-instruct-q4.gguf"
  context_length: 2048
  n_threads: 6
  # F-006: -1 offloads every layer to the iGPU so matmul stays on CUDA instead
  # of CPU. Override per-host via MOUSEDROID_LLM__N_GPU_LAYERS=<int> in
  # /etc/mousedroid/docker.env (e.g. =0 on a CPU-only fallback host).
  n_gpu_layers: -1
  n_batch: 32
  max_tokens: 128
  temperature: 0.1
```

- [ ] **Step 2: Verify it parses (no API key needed)**

Run: `python scripts/validate_configs.py --include-default`
Expected: exit 0; all overlays OK (the line for `jetson_production.yaml` shows OK).

- [ ] **Step 3: Verify the parsed values are what we intend**

Run:
```bash
python -c "from pathlib import Path; from mousedroid.config.loader import load_settings; s=load_settings(Path('config/jetson_production.yaml'), config_dir=Path('config')); print(s.llm.backend, s.llm.fallback_backend, s.llm.model_name, s.llm.fallback_retry_cooldown_s, s.llm.model_path)"
```
Expected: `anthropic llama_cpp claude-haiku-4-5 30.0 /opt/mousedroid/models/Phi-3-mini-4k-instruct-q4.gguf`

- [ ] **Step 4: Confirm existing backwards-compat tests still pass**

Run: `python -m pytest tests/regression/test_pr107_backwards_compat.py tests/regression/test_config_overlays_load.py -q`
Expected: all pass (these guarantee the new fields default safely and every overlay loads).

- [ ] **Step 5: Commit**

```bash
git add config/jetson_production.yaml
git commit -m "config(jetson): wire Claude primary + Phi-3 llama_cpp fallback (PR #107)"
```

---

### Task A3: Document the secret in `config/docker.env.example`

**Files:**
- Modify: `config/docker.env.example` (append a block at end)

- [ ] **Step 1: Append the commented placeholders**

Add to the end of `config/docker.env.example`:
```sh

# ---------------------------------------------------------------------------
# Claude (Anthropic) cloud LLM tier — PR #107
# ---------------------------------------------------------------------------
# Provide ONE of these to enable the cloud mission-translation tier. NEVER
# commit a real key. The Anthropic SDK reads ANTHROPIC_API_KEY natively; the
# schema-mapped override below is SecretStr-wrapped and kept out of YAML.
# When neither is set, the gateway degrades to the local Phi-3 fallback.
# ANTHROPIC_API_KEY=sk-ant-...
# MOUSEDROID_LLM__API_KEY=sk-ant-...
```

- [ ] **Step 2: Verify no real secret leaked**

Run: `git diff config/docker.env.example`
Expected: only commented placeholder lines (`# ANTHROPIC_API_KEY=sk-ant-...`), no actual key.

- [ ] **Step 3: Commit**

```bash
git add config/docker.env.example
git commit -m "docs(env): document ANTHROPIC_API_KEY provisioning for PR #107"
```

---

### Task A4: New operator probe `scripts/translate_mission.py` (TDD)

**Files:**
- Create: `scripts/translate_mission.py`
- Create: `tests/unit/test_translate_mission_cli.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/unit/test_translate_mission_cli.py`:
```python
"""Unit tests for scripts/translate_mission.py (the operator dry-run probe).

The probe loads Settings, builds the LLM gateway via the factory, translates a
single NL mission, prints the resulting GoalVector + degraded/tier state, and
exits. The gateway is mocked end-to-end so no network, API key, or GGUF is
needed. Mirrors the test style of the greeting CLI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.llm_gateway.protocol import GoalVector

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "translate_mission.py"


def _load_cli():
    """Import the script module by path (it lives in scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("translate_mission", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_gateway(vector: GoalVector, *, degraded: bool = False) -> MagicMock:
    gw = MagicMock()
    gw.is_ready = True
    gw.is_degraded = degraded
    gw.start = AsyncMock(return_value=None)
    gw.stop = AsyncMock(return_value=None)
    gw.translate_mission = AsyncMock(return_value=vector)
    return gw


def test_translate_prints_goalvector_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector(vx_target=0.4, vy_target=0.0, omega_target=-0.2))
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "patrol left then stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vx_target" in out and "0.4" in out
    gw.translate_mission.assert_awaited_once_with("patrol left then stop")
    gw.start.assert_awaited_once()
    gw.stop.assert_awaited_once()


def test_degraded_gateway_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    gw = _fake_gateway(GoalVector(), degraded=True)
    with (
        patch.object(cli, "load_settings", return_value=SimpleNamespace()),
        patch.object(cli, "build_llm_gateway", return_value=gw),
    ):
        rc = cli.main(["--mission", "stop"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "degraded" in out.lower()


def test_missing_mission_arg_exits_nonzero() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as exc:  # argparse error
        cli.main([])
    assert exc.value.code != 0
```

- [ ] **Step 2: Run the test — verify it fails (no script yet)**

Run: `python -m pytest tests/unit/test_translate_mission_cli.py -q`
Expected: FAIL / ERROR — `scripts/translate_mission.py` does not exist (spec load fails).

- [ ] **Step 3: Implement `scripts/translate_mission.py`**

Create `scripts/translate_mission.py`:
```python
#!/usr/bin/env python3
"""MSE-6 mission-translation dry-run probe (PR #107).

Translates a single natural-language mission into a normalised ``GoalVector``
via the deliberative LLM gateway — WITHOUT issuing any motor command — so the
Claude-primary / Phi-3-fallback path can be verified live on the rover even
when the ESP32 / drivetrain is detached.

It builds the gateway through the real factory (``build_llm_gateway``), so it
honours whatever ``llm:`` config the rover runs (cloud Claude when reachable,
local llama_cpp fallback when off-network). The served tier + degraded state
are printed so an operator can confirm which path answered.

Examples::

    # On the Jetson with the production overlay:
    MOUSEDROID_CONFIG=/etc/mousedroid/jetson_production.yaml \\
        python scripts/translate_mission.py --mission "patrol left then stop"

    # Dev box with an explicit overlay:
    python scripts/translate_mission.py \\
        --config config/jetson_production.yaml --mission "go forward slowly"

Exit codes:

* ``0`` — mission translated (prints the GoalVector + tier/degraded state).
* ``1`` — runtime / gateway failure.
* ``2`` — configuration error (config load failed).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make src/ importable when run from repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Force structlog to stderr BEFORE importing mousedroid so the import-time
# configure() in mousedroid.config.loader doesn't latch onto stdout — keeps
# the GoalVector print on stdout clean for piping. (Mirrors greet_intro.py.)
import structlog  # noqa: E402

structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO and above
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=False,
)

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.factory import build_llm_gateway  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402

_log = get_logger(__name__)

_EXIT_OK = 0
_EXIT_RUNTIME_ERROR = 1
_EXIT_CONFIG_ERROR = 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        required=True,
        help="Natural-language mission to translate into a GoalVector.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        type=Path,
        help=(
            "One or more YAML overlays (repeatable). When omitted, "
            "load_settings() uses default.yaml + the MOUSEDROID_CONFIG env "
            "overlay (same resolution as the orchestrator)."
        ),
    )
    return parser.parse_args(argv)


async def _run(mission: str, gateway: object) -> int:
    """Start the gateway, translate one mission, print the result, stop."""
    start = getattr(gateway, "start")
    stop = getattr(gateway, "stop")
    translate = getattr(gateway, "translate_mission")
    await start()
    try:
        vector = await translate(mission)
    finally:
        await stop()

    degraded = getattr(gateway, "is_degraded", False)
    tier = "local-fallback (degraded primary)" if degraded else "primary"
    # GoalVector is a dataclass — print its fields explicitly on stdout.
    print(
        f"mission={mission!r} tier={tier} degraded={degraded} "
        f"GoalVector(vx_target={vector.vx_target:.3f}, "
        f"vy_target={vector.vy_target:.3f}, "
        f"omega_target={vector.omega_target:.3f})"
    )
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    overlays = tuple(args.config) if args.config else ()
    try:
        settings = load_settings(*overlays)
    except Exception:  # noqa: BLE001 — operator probe: report, don't traceback-crash
        _log.exception("translate_mission_config_error", overlays=[str(p) for p in overlays])
        return _EXIT_CONFIG_ERROR

    try:
        gateway = build_llm_gateway(settings)
    except Exception:  # noqa: BLE001
        _log.exception("translate_mission_build_error")
        return _EXIT_RUNTIME_ERROR

    try:
        return asyncio.run(_run(args.mission, gateway))
    except Exception:  # noqa: BLE001
        _log.exception("translate_mission_runtime_error")
        return _EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test — verify it passes**

Run: `python -m pytest tests/unit/test_translate_mission_cli.py -q`
Expected: 3 passed.

- [ ] **Step 5: Lint + type the new files**

Run:
```bash
python -m ruff check scripts/translate_mission.py tests/unit/test_translate_mission_cli.py
python -m ruff format --check scripts/translate_mission.py tests/unit/test_translate_mission_cli.py
python -m mypy --strict src/mousedroid/   # script imports only; mypy target is src/
```
Expected: ruff "All checks passed"; format clean; mypy "Success". If ruff format reports a diff, run `python -m ruff format scripts/translate_mission.py tests/unit/test_translate_mission_cli.py` and re-check.

- [ ] **Step 6: Commit**

```bash
git add scripts/translate_mission.py tests/unit/test_translate_mission_cli.py
git commit -m "feat(tools): translate_mission.py dry-run GoalVector probe (PR #107 validation)"
```

---

### Task A5: Config-pinning regression test for `jetson_production.yaml`

**Files:**
- Create: `tests/regression/test_jetson_claude_pilot_config.py`

- [ ] **Step 1: Write the test**

Create `tests/regression/test_jetson_claude_pilot_config.py`:
```python
"""Regression: pin config/jetson_production.yaml's PR #107 LLM wiring.

Guards against silent drift of the deployed cloud-primary / local-fallback
wiring. (Backwards-compat of the LLMConfig DEFAULTS is covered separately by
tests/regression/test_pr107_backwards_compat.py — this file pins the PRODUCTION
overlay's explicit values.)
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_PROD = _CONFIG_DIR / "jetson_production.yaml"


def _load() -> Settings:
    return load_settings(_PROD, config_dir=_CONFIG_DIR)


def test_production_uses_anthropic_primary() -> None:
    assert _load().llm.backend == "anthropic"


def test_production_falls_back_to_llama_cpp() -> None:
    assert _load().llm.fallback_backend == "llama_cpp"


def test_production_model_name_is_a_claude_id() -> None:
    assert _load().llm.model_name.startswith("claude-")


def test_production_fallback_model_path_is_phi3() -> None:
    # The off-network fallback reuses the already-staged Phi-3-mini GGUF.
    assert "Phi-3-mini" in str(_load().llm.model_path)


def test_production_cooldown_positive() -> None:
    assert _load().llm.fallback_retry_cooldown_s > 0.0


def test_production_loads_without_api_key(monkeypatch) -> None:
    """Settings load must not require a key (the gateway degrades, not parse)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MOUSEDROID_LLM__API_KEY", raising=False)
    assert isinstance(_load(), Settings)
```

- [ ] **Step 2: Run the test — verify it passes**

Run: `python -m pytest tests/regression/test_jetson_claude_pilot_config.py -q`
Expected: 6 passed. (If it fails, Task A2 wasn't applied or has a typo.)

- [ ] **Step 3: Lint**

Run: `python -m ruff check tests/regression/test_jetson_claude_pilot_config.py && python -m ruff format --check tests/regression/test_jetson_claude_pilot_config.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_jetson_claude_pilot_config.py
git commit -m "test(regression): pin jetson_production.yaml PR #107 LLM wiring"
```

---

### Task A6: Integration assertion — factory builds the failover composite

**Files:**
- Create: `tests/integration/test_jetson_pilot_gateway_wiring.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_jetson_pilot_gateway_wiring.py`:
```python
"""Integration: build_llm_gateway on the production config yields the composite.

Asserts that the deployed jetson_production.yaml wiring (anthropic primary +
llama_cpp fallback) produces a FallbackLLMGateway through the real factory.
No network / API key / GGUF load — we only inspect the composite's structure,
not run inference.
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_llm_gateway
from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"


def test_production_config_builds_fallback_composite() -> None:
    settings = load_settings(_CONFIG_DIR / "jetson_production.yaml", config_dir=_CONFIG_DIR)
    gateway = build_llm_gateway(settings)
    assert isinstance(gateway, FallbackLLMGateway)
```

- [ ] **Step 2: Run it — verify pass**

Run: `python -m pytest tests/integration/test_jetson_pilot_gateway_wiring.py -q`
Expected: 1 passed. (`build_llm_gateway` lazily imports the anthropic SDK at `start()`, not at build time, so this works without the SDK installed.)

> If this errors because constructing `AnthropicLLMGateway` imports the SDK eagerly, fall back to asserting on `build_llm_gateway`'s branch via the existing faked-SDK harness in `tests/integration/test_anthropic_gateway_wiring.py` instead, and delete this file. (Verified in spec review: the SDK import is deferred to `start()`, so the simple assertion above is expected to pass.)

- [ ] **Step 3: Lint**

Run: `python -m ruff check tests/integration/test_jetson_pilot_gateway_wiring.py && python -m ruff format --check tests/integration/test_jetson_pilot_gateway_wiring.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_jetson_pilot_gateway_wiring.py
git commit -m "test(integration): production config builds anthropic+llama_cpp composite"
```

---

### Task A7: Deploy runbook

**Files:**
- Create: `docs/runbooks/jetson-claude-pilot-deploy.md`

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/jetson-claude-pilot-deploy.md`:
```markdown
# Runbook — Jetson Claude-Pilot Deploy (PR #107)

Deploy the Anthropic Claude mission-translation gateway + Phi-3 off-network
fallback to the rover. Design: `docs/superpowers/specs/2026-06-02-jetson-claude-pilot-deploy-design.md`.

## Prerequisites
- Deploy branch pushed to `origin` (`feat/jetson-claude-pilot-deploy`).
- An `ANTHROPIC_API_KEY` (`sk-ant-...`) to provision (cloud tier; fallback works without it).
- SSH: `ssh ian@mousedroid.local` (WiFi). Container `mousedroid` healthy.

## Deploy sequence (ordering matters — see design §5)
1. **Backup + validate** the live config:
   `sudo cp -a /etc/mousedroid/jetson_production.yaml /etc/mousedroid/jetson_production.yaml.bak.$(date +%Y%m%d_%H%M%S)`
   then validate it parses before trusting it as rollback.
2. **Sync source** (record current branch first for rollback):
   `git -C /opt/mousedroid rev-parse --abbrev-ref HEAD`
   `git -C /opt/mousedroid status --porcelain`  (must be empty)
   `git -C /opt/mousedroid fetch origin && git -C /opt/mousedroid checkout feat/jetson-claude-pilot-deploy`
   Verify: `ls /opt/mousedroid/src/mousedroid/llm_gateway/anthropic_gateway.py`
3. **Write config**: `sudo` write the merged `jetson_production.yaml`; validate parse.
4. **Provision key** (editor, NOT `echo >>` — avoids shell history):
   add `ANTHROPIC_API_KEY=sk-ant-...` to `/etc/mousedroid/docker.env`; `sudo chmod 600` it.
5. **Recreate** (loads env + config + source):
   `docker compose -f docker-compose.jetson.yml up -d --force-recreate`
   → Stage-1 validation (fallback; SDK still absent).
6. **Hot-install** the SDK: `docker exec mousedroid python3 -m pip install "anthropic>=0.40"`
7. **Restart** (preserves the writable-layer SDK; recreate would wipe it):
   `docker compose -f docker-compose.jetson.yml restart mousedroid`
   → Stage-2 validation (cloud).

## Validation
- Probe (no motors needed):
  `docker exec mousedroid python3 /opt/mousedroid/scripts/translate_mission.py --mission "patrol left then stop"`
  Stage-1 (no key): prints `tier=local-fallback (degraded primary)` + a GoalVector.
  Stage-2 (key set): prints `tier=primary` + a GoalVector.
- Structured-log grep recipes (docker logs / Loki):
  - `anthropic_gateway_degraded` — primary unreachable (expected off-network / no key).
  - `anthropic_gateway_recovered` — primary self-healed after cooldown re-probe.
  - `anthropic_gateway_slow` — cloud call exceeded latency_target_ms (tune if noisy).
- Confirm the 30 Hz loop / telemetry is unaffected throughout (Grafana dashboards).

## Rollback
1. Restore `/etc/mousedroid/jetson_production.yaml` from the validated `.bak.<ts>`.
2. `git -C /opt/mousedroid checkout <recorded-prior-branch>`.
3. `docker compose -f docker-compose.jetson.yml up -d --force-recreate`.

## Durability note
The hot-installed `anthropic` survives `restart`/reboot but NOT a future
`--force-recreate`/rebuild. The Dockerfile `Stage 4b` bake (this PR) ensures
future image builds include it.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/jetson-claude-pilot-deploy.md
git commit -m "docs(runbook): Jetson Claude-pilot deploy procedure (PR #107)"
```

---

### Task A8: Full local CI gate

**Files:** none (verification).

- [ ] **Step 1: Lint + format (pinned ruff via `python -m`)**

Run:
```bash
python -m ruff check src/ tests/ scripts/
python -m ruff format --check src/ tests/ scripts/
```
Expected: "All checks passed"; format clean. Fix any reported issues and re-run.

- [ ] **Step 2: Type check**

Run: `python -m mypy --strict src/mousedroid/`
Expected: `Success: no issues found`.

- [ ] **Step 3: Config validation**

Run: `python scripts/validate_configs.py --include-default`
Expected: exit 0; all overlays OK.

- [ ] **Step 4: Full test suite + coverage**

Run: `python -m pytest tests/ -m "not hardware and not slow" --cov=src/mousedroid --cov-fail-under=85 -q --import-mode=importlib`
Expected: all pass, 0 failures, coverage ≥ 85%. Note the new files: `test_translate_mission_cli.py`, `test_jetson_claude_pilot_config.py`, `test_jetson_pilot_gateway_wiring.py` all green.

- [ ] **Step 5: Push the branch (required before live deploy)**

```bash
git push -u origin feat/jetson-claude-pilot-deploy
```
Expected: branch published to `origin`.

---

### Task A9: Open the PR

**Files:** none.

- [ ] **Step 1: Open PR against the integration branch**

```bash
gh pr create --base claude/markdown-implementation-plan-aVJ2l \
  --head feat/jetson-claude-pilot-deploy \
  --title "deploy(jetson): activate PR #107 Claude gateway + Phi-3 failover" \
  --body "Deploys merged PR #107 to the rover. Non-fatal anthropic Dockerfile layer; minimal additive llm: merge into jetson_production.yaml (Claude primary + reused Phi-3 fallback); docker.env.example secret docs; new translate_mission.py probe; config-pinning + integration tests; runbook. Backwards-compat covered by existing test_pr107_backwards_compat.py. Spec: docs/superpowers/specs/2026-06-02-jetson-claude-pilot-deploy-design.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
Expected: PR URL printed.

- [ ] **Step 2: Wait for CI green**

Run: `gh pr checks --watch`
Expected: all required checks SUCCESS (lint×3, typecheck×3, test×3, config-validate, docker, etc.). If a check fails, fix on-branch and re-push before proceeding to Phase B.

---

## PHASE B — Live deploy to `ian@mousedroid.local`

> Run each SSH step and read its output before the next. Stop on any unexpected result. The probe in Task B5/B7 needs no motors (ESP32 may be dead). **Never paste the API key into a command that lands in shell history.**

### Task B1: Pre-flight + safe backup

- [ ] **Step 1: Confirm container healthy + record rollback state**

Run:
```bash
ssh ian@mousedroid.local 'docker ps --format "{{.Names}} {{.Status}}" | grep mousedroid; echo "BRANCH:"; git -C /opt/mousedroid rev-parse --abbrev-ref HEAD; echo "DIRTY:"; git -C /opt/mousedroid status --porcelain'
```
Expected: `mousedroid ... Up ... (healthy)`; BRANCH printed (record it — e.g. `feat/jetson-rover-usbc-smoke`); DIRTY empty. **If DIRTY is non-empty, STOP** and resolve (stash/commit) before continuing.

- [ ] **Step 2: Backup the live config and validate the backup parses**

Run:
```bash
ssh ian@mousedroid.local 'TS=$(date +%Y%m%d_%H%M%S); sudo cp -a /etc/mousedroid/jetson_production.yaml /etc/mousedroid/jetson_production.yaml.bak.$TS && echo "backed up .bak.$TS" && docker exec mousedroid python3 /opt/mousedroid/scripts/validate_configs.py --config-dir /etc/mousedroid --include-default; echo "VALIDATE_RC=$?"'
```
Expected: "backed up .bak.<ts>"; validator exit 0 (`VALIDATE_RC=0`). **If the validator is unavailable on the old source tree, instead load-check the backup after Task B2's source sync** — note: do not trust the `.bak` as rollback until it parses.

---

### Task B2: Sync source

- [ ] **Step 1: Fetch + checkout the deploy branch (tracked branch, not a SHA)**

Run:
```bash
ssh ian@mousedroid.local 'git -C /opt/mousedroid fetch origin && git -C /opt/mousedroid checkout feat/jetson-claude-pilot-deploy && git -C /opt/mousedroid rev-parse --abbrev-ref HEAD'
```
Expected: checkout succeeds; prints `feat/jetson-claude-pilot-deploy` (a branch name, not `HEAD`).

- [ ] **Step 2: Verify the gateway code is now present on disk**

Run: `ssh ian@mousedroid.local 'ls -l /opt/mousedroid/src/mousedroid/llm_gateway/anthropic_gateway.py /opt/mousedroid/src/mousedroid/llm_gateway/fallback_gateway.py /opt/mousedroid/scripts/translate_mission.py'`
Expected: all three files listed (present).

---

### Task B3: Deploy the merged config

- [ ] **Step 1: Copy the new config into place (sudo)**

Run:
```bash
ssh ian@mousedroid.local 'sudo cp /opt/mousedroid/config/jetson_production.yaml /etc/mousedroid/jetson_production.yaml && echo copied'
```
Expected: "copied".

- [ ] **Step 2: Validate the deployed config parses**

Run:
```bash
ssh ian@mousedroid.local 'docker exec mousedroid python3 /opt/mousedroid/scripts/validate_configs.py --config-dir /etc/mousedroid --include-default; echo RC=$?'
```
Expected: `RC=0`. **If non-zero, restore the `.bak` and STOP.**

---

### Task B4: Provision the API key (OPERATOR action)

- [ ] **Step 1: Operator adds the key via an editor (no shell history)**

Operator runs on the Jetson (interactively):
```bash
ssh ian@mousedroid.local
sudo nano /etc/mousedroid/docker.env      # add a line: ANTHROPIC_API_KEY=sk-ant-...
sudo chmod 600 /etc/mousedroid/docker.env
```
> Claude does NOT paste the key. Verify presence (name only, not value):
> `ssh ian@mousedroid.local 'sudo grep -c "^ANTHROPIC_API_KEY=" /etc/mousedroid/docker.env'` → expect `1`.

- [ ] **Step 2 (optional): proceed without the key**

If validating fallback-only first, skip Step 1 — Stage-1 (Task B5) still proves the Phi-3 failover. Provision the key later and re-run Task B7.

---

### Task B5: Recreate the container (loads env + config + source) — Stage-1 validation

- [ ] **Step 1: Force-recreate**

Run:
```bash
ssh ian@mousedroid.local 'cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml up -d --force-recreate mousedroid && sleep 5 && docker ps --format "{{.Names}} {{.Status}}" | grep mousedroid'
```
Expected: container recreated; `Up ... (health: starting|healthy)`.

- [ ] **Step 2: Stage-1 — fallback path serves (SDK absent → degrade to Phi-3)**

Run:
```bash
ssh ian@mousedroid.local 'docker exec mousedroid python3 /opt/mousedroid/scripts/translate_mission.py --mission "patrol left then stop"; echo RC=$?'
```
Expected: `RC=0`; stdout prints a `GoalVector(...)`. With no key/SDK, stderr/log shows `anthropic_gateway_degraded` and the line reports `tier=local-fallback (degraded primary)`. **This proves off-network autonomy.**

---

### Task B6: Hot-install the anthropic SDK

- [ ] **Step 1: Install into the recreated container**

Run:
```bash
ssh ian@mousedroid.local 'docker exec mousedroid python3 -m pip install "anthropic>=0.40" && docker exec mousedroid python3 -c "import anthropic; print(anthropic.__version__)"'
```
Expected: pip success; an anthropic version string (≥ 0.40).

---

### Task B7: Restart the process — Stage-2 validation (cloud)

> Only meaningful if the key was provisioned (Task B4). If fallback-only, skip to Task B8.

- [ ] **Step 1: Restart (preserves writable-layer SDK; recreate would wipe it)**

Run:
```bash
ssh ian@mousedroid.local 'cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml restart mousedroid && sleep 5 && docker ps --format "{{.Names}} {{.Status}}" | grep mousedroid'
```
Expected: container restarted; healthy.

- [ ] **Step 2: Stage-2 — cloud tier serves**

Run:
```bash
ssh ian@mousedroid.local 'docker exec mousedroid python3 /opt/mousedroid/scripts/translate_mission.py --mission "go forward slowly then turn right"; echo RC=$?'
```
Expected: `RC=0`; prints `tier=primary` + a GoalVector; logs show `anthropic_gateway` success (no `_degraded`).

- [ ] **Step 3: Failover + self-heal check (optional, host-level egress block)**

With `network_mode: host`, simulate WAN loss at the host:
```bash
ssh ian@mousedroid.local 'sudo iptables -A OUTPUT -d api.anthropic.com -j REJECT 2>/dev/null || echo "use your host firewall to block api.anthropic.com"; docker exec mousedroid python3 /opt/mousedroid/scripts/translate_mission.py --mission "stop"; sudo iptables -D OUTPUT -d api.anthropic.com -j REJECT 2>/dev/null || true'
```
Expected: during the block, `tier=local-fallback`; after removing the block + waiting `fallback_retry_cooldown_s` (30 s), the next call self-heals to `tier=primary` (`anthropic_gateway_recovered`). (`iptables -d <hostname>` resolves at insert time; acceptable for a one-shot probe.)

---

### Task B8: Post-deploy verification + record

- [ ] **Step 1: Confirm the reactive hot path is unaffected**

Run: `ssh ian@mousedroid.local 'docker logs --since 2m mousedroid 2>&1 | grep -iE "orchestrator|loop_hz|30" | tail -5'` and check the Grafana dashboard.
Expected: 30 Hz loop healthy; no new errors tied to the LLM path.

- [ ] **Step 2: Update project memory**

Update `C:\Users\iansh\.claude\projects\C--Users-iansh-OneDrive-Documents-Gronk-Droid-Jetson-Nano\memory\` with a new project memory file recording: PR #107 deployed to the rover on 2026-06-02, branch `feat/jetson-claude-pilot-deploy`, Claude-haiku primary + Phi-3 fallback, key in `/etc/mousedroid/docker.env`, anthropic hot-installed (re-install after any `--force-recreate` until the next image rebuild bakes it). Add the one-line pointer to `MEMORY.md`.

- [ ] **Step 3: Final status report**

Summarize to the user: Phase A PR + CI status, which validation stages passed (fallback / cloud), and any deferred item (e.g. image rebuild to bake `anthropic`, or key still to provision).

---

## Self-review notes (author)

- **Spec coverage:** Dockerfile layer (A1 ↔ spec §4.1); config merge (A2 ↔ §4.2); docker.env docs (A3 ↔ §4.3); probe (A4 ↔ §4.4/§7); config-pinning test (A5 ↔ §4.5); integration test (A6 ↔ §4.5); backwards-compat (covered by existing `test_pr107_backwards_compat.py`, verified in A2.4 — no new task needed); runbook (A7 ↔ §4.6); CI gate (A8 ↔ §8); deploy sequence B1-B8 ↔ §5/§6 with the post-review ordering (recreate→install→restart), `.bak` validation, secret hygiene, host-network egress, push-first.
- **No placeholders:** all code blocks complete; all commands have expected output.
- **Type/name consistency:** `vx_target/vy_target/omega_target`, `translate_mission`, `build_llm_gateway`, `FallbackLLMGateway(..., retry_cooldown_s=)`, `load_settings(*overlays, config_dir=)`, `is_degraded` via getattr — all match the verified API surface.
```text

