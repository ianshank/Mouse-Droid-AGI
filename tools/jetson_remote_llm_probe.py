"""F-006 remote-LLM verification: HTTP-side latency probe.

Mirror of ``tools/llm_latency_probe.py`` (PR #102) but for the
``openai_compatible`` HTTP backend introduced by Tier C2.3 / PR #99. PR
#102's live-Jetson run confirmed Phi-3-mini-q4 OOMs at any GPU offload
on the Orin Nano 8GB; this probe verifies the F-006 fix path (c): route
``LLMGateway`` at a host-PC Ollama instance over the USB-net bridge.

What it does:

1. Refuses unless ``cfg.llm.backend == "openai_compatible"`` (exit 3 —
   pointing the operator at ``llm_latency_probe.py`` for the local path).
2. Cold-pings ``{base_url}/v1/models`` to (a) surface a wrong host IP
   or "Ollama not listening on 0.0.0.0" faster than waiting for
   ``gateway.start()`` to time out, and (b) emit the list of models the
   server advertises so the operator can confirm the model_name they
   configured is actually loaded.
3. Builds the gateway via ``build_llm_gateway`` (so the same
   ``injection_filter`` wiring used by the orchestrator is exercised),
   calls ``start()`` then a single ``translate_mission(--mission)``,
   measures elapsed against ``cfg.llm.latency_target_ms``.
4. Snapshots tegrastats RAM before + after via ``tools/_jetson_helpers``
   so the operator can see whether the host's RAM stayed sane during
   the round-trip (in remote-LLM mode the Jetson's RAM should NOT move
   — that's the whole point of moving the LLM off-Jetson).

Structured-log events emitted (mirror PR #102 plus one new):

* ``probe_cfg`` — same cfg.llm fields PR #102 emits + backend/base_url/model_name.
* ``remote_llm_models_listed`` — NEW; cold-ping response payload.
* ``llm_gateway_load_failed`` — same shape as PR #102; catches HTTP
  transport errors (aiohttp.ClientError, asyncio.TimeoutError).
* ``tegrastats_before`` / ``tegrastats_after`` — RAM snapshots.
* ``llm_latency_result`` — same shape as PR #102.

Exit codes (identical contract to PR #102):

* ``0`` — translate_mission elapsed <= cfg.llm.latency_target_ms.
* ``1`` — elapsed > target (F-006 fix path (c) didn't deliver on this host).
* ``2`` — gateway / HTTP transport failed (load + start + translate).
* ``3`` — config error: llm disabled, wrong backend, etc.

Operator usage (inside the orchestrator container):

    docker exec mousedroid python3 /opt/mousedroid/tools/jetson_remote_llm_probe.py \\
        --config /etc/mousedroid/jetson_production.yaml \\
        --overlay /etc/mousedroid/jetson_production_remote_llm.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from pydantic import SecretStr
from tools._jetson_helpers import tegrastats_snapshot as _tegrastats_snapshot

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_injection_filter, build_llm_gateway
from mousedroid.logging.setup import get_logger

_log = get_logger("jetson_remote_llm_probe")

# Default mission text — same one PR #102's local probe uses, so the
# two probes' result-line outputs are directly comparable when an
# operator A/B-tests them against the same Ollama backend.
_DEFAULT_MISSION = "turn left slowly"

# OpenAI-compatible models endpoint (mirrors the constant in
# openai_compatible.py). Hard-pinned because it's part of the OpenAI
# REST contract, not an operator-tunable knob — the host portion comes
# from cfg.llm.base_url.
_MODELS_ENDPOINT_PATH = "/v1/models"

# Cold-ping timeout. Deliberately tight (3s default) because the
# USB-net bridge has <1ms RTT and Ollama's /v1/models is a trivial in-
# memory list — anything slower indicates the host PC is unreachable
# or Ollama is hung, both of which the operator wants to know
# immediately rather than after the gateway start timeout fires.
_COLD_PING_TIMEOUT_S = 3.0


async def _cold_ping_models(base_url: str, api_key: SecretStr | None) -> int:
    """Hit ``{base_url}/v1/models`` once, log the list, return exit code.

    Returns ``0`` on 200 + parseable JSON, ``2`` on any transport /
    parse error. Logs ``remote_llm_models_listed`` on success with the
    list of model IDs the server advertises so the operator can confirm
    the configured model_name is available.

    Never raises — wraps every aiohttp call in try/except so the probe's
    "exit code, not exception" contract holds.
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
    url = f"{base_url}{_MODELS_ENDPOINT_PATH}"
    t0 = time.monotonic()
    try:
        ping_timeout = aiohttp.ClientTimeout(total=_COLD_PING_TIMEOUT_S)
        async with (
            aiohttp.ClientSession() as session,
            session.get(url, headers=headers, timeout=ping_timeout) as resp,
        ):
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if resp.status != 200:
                _log.error(
                    "llm_gateway_load_failed",
                    stage="cold_ping",
                    url=url,
                    status=resp.status,
                    elapsed_ms=elapsed_ms,
                    hint=(
                        "Cold ping to /v1/models returned non-200. Check "
                        "that Ollama is running on the host PC and the "
                        "base_url IP is correct (look at "
                        "tools/jetson_remote_llm_probe --help for the "
                        "operator runbook reference)."
                    ),
                )
                return 2
            body = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        _log.error(
            "llm_gateway_load_failed",
            stage="cold_ping",
            url=url,
            elapsed_ms=elapsed_ms,
            error_type=type(exc).__name__,
            error=str(exc),
            hint=(
                "Cold ping failed before reaching Ollama. Common causes: "
                "host PC not reachable on the USB-net bridge (ping the IP "
                "from the Jetson host first), Ollama bound to 127.0.0.1 "
                "only (set OLLAMA_HOST=0.0.0.0:11434), or firewall "
                "blocking the bridge interface."
            ),
        )
        return 2

    model_ids = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
    _log.info(
        "remote_llm_models_listed",
        url=url,
        elapsed_ms=elapsed_ms,
        model_count=len(model_ids),
        models=model_ids,
    )
    return 0


async def _main(args: argparse.Namespace) -> int:
    overlay_paths: list[Path] = []
    if args.config:
        overlay_paths.append(Path(args.config))
    if args.overlay:
        overlay_paths.append(Path(args.overlay))
    cfg = load_settings(*overlay_paths)

    if not cfg.llm.enabled:
        _log.error("llm_gateway_disabled_in_cfg", config=args.config)
        return 3

    if cfg.llm.backend != "openai_compatible":
        _log.error(
            "llm_gateway_backend_mismatch",
            configured=cfg.llm.backend,
            expected="openai_compatible",
            hint=(
                "This probe is openai_compatible-specific. Use "
                "tools/llm_latency_probe.py for the llama_cpp backend."
            ),
        )
        return 3

    _log.info(
        "probe_cfg",
        backend=cfg.llm.backend,
        base_url=cfg.llm.base_url,
        model_name=cfg.llm.model_name,
        request_timeout_s=cfg.llm.request_timeout_s,
        latency_target_ms=cfg.llm.latency_target_ms,
        env_base_url_override=os.environ.get("MOUSEDROID_LLM__BASE_URL"),
        env_model_name_override=os.environ.get("MOUSEDROID_LLM__MODEL_NAME"),
        env_api_key_set=os.environ.get("MOUSEDROID_LLM__API_KEY") is not None,
    )

    snapshot_before = _tegrastats_snapshot()
    _log.info("tegrastats_before", **snapshot_before)

    ping_rc = await _cold_ping_models(cfg.llm.base_url, cfg.llm.api_key)
    if ping_rc != 0:
        return ping_rc

    injection_filter = build_injection_filter(cfg)
    gateway = build_llm_gateway(cfg, injection_filter=injection_filter)

    t_start = time.monotonic()
    try:
        await gateway.start()
    except Exception as exc:  # boundary catch — never raise out of probe
        cold_start_ms = (time.monotonic() - t_start) * 1000.0
        _log.error(
            "llm_gateway_load_failed",
            stage="gateway_start",
            cold_start_ms=cold_start_ms,
            error_type=type(exc).__name__,
            error=str(exc),
            hint=(
                "gateway.start() failed after a successful cold-ping. The "
                "host is reachable but the OpenAICompatibleLLMGateway "
                "couldn't initialise — most likely a configuration issue "
                "(invalid api_key, model_name not loaded on the server)."
            ),
        )
        return 2
    cold_start_ms = (time.monotonic() - t_start) * 1000.0
    _log.info("llm_start_complete", cold_start_ms=cold_start_ms)

    if not gateway.is_ready:
        _log.error(
            "llm_gateway_not_ready_after_start",
            cold_start_ms=cold_start_ms,
            hint="check llm_gateway_degraded_* events in earlier log lines",
        )
        return 2

    snapshot_after = _tegrastats_snapshot()
    _log.info("tegrastats_after", **snapshot_after)

    t_translate = time.monotonic()
    goal = await gateway.translate_mission(args.mission)
    elapsed_ms = (time.monotonic() - t_translate) * 1000.0

    passed = elapsed_ms <= cfg.llm.latency_target_ms
    _log.info(
        "llm_latency_result",
        elapsed_ms=elapsed_ms,
        target_ms=cfg.llm.latency_target_ms,
        passed=passed,
        goal_vx=goal.vx_target,
        goal_vy=goal.vy_target,
        goal_omega=goal.omega_target,
        mission=args.mission,
    )

    await gateway.stop()
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.jetson_remote_llm_probe",
        description=(
            "F-006 remote-LLM latency probe — verifies the openai_compatible "
            "HTTP backend against a host-PC Ollama instance over the USB-net "
            "bridge."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to the base config overlay YAML "
            "(e.g. /etc/mousedroid/jetson_production.yaml). "
            "Omit to use config/default.yaml only."
        ),
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help=(
            "Path to a SECOND overlay YAML applied on top of --config "
            "(e.g. /etc/mousedroid/jetson_production_remote_llm.yaml). "
            "Mirrors how the operator runbook layers the remote-LLM "
            "overlay onto the production overlay."
        ),
    )
    parser.add_argument(
        "--mission",
        default=_DEFAULT_MISSION,
        help="Mission text to translate (default: %(default)r).",
    )
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        _log.warning("probe_interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
