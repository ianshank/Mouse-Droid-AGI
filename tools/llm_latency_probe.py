"""F-006 verification: live-Jetson LLMGateway.translate_mission latency probe.

PR #101 flipped ``config/jetson_production.yaml`` from ``llm.n_gpu_layers: 0``
to ``-1`` (offload every layer to the iGPU). This probe verifies that change
actually delivers sub-``cfg.llm.latency_target_ms`` inference on the real
Jetson, not just in theory.

What it measures:

* Cold-start: time to ``LLMGateway.start()`` (model load + warm-up).
* Single-shot: ``translate_mission("turn left slowly")`` elapsed ms.
* (Optional) tegrastats GPU memory before/after model load — operator-side
  signal that GPU layers actually landed. If ``tegrastats`` is unavailable
  (non-Jetson host) the probe falls back gracefully and logs a WARN.

Structured-log fields emitted at INFO so operator dashboards can ingest:

* ``probe_cfg``: n_gpu_layers, n_threads, n_batch, latency_target_ms, model_path
* ``llama_model_metadata``: parsed n_gpu_layers reported by the loaded Llama
  instance (distinguishes "GPU offload silently failed" from "model is slow")
* ``tegrastats_before`` / ``tegrastats_after``: GPU RAM MB snapshot
* ``llm_latency_result``: elapsed_ms, target_ms, passed (bool)

Exit codes:

* ``0`` — translate_mission elapsed <= cfg.llm.latency_target_ms
* ``1`` — elapsed > target (F-006 not actually fixed on this host)
* ``2`` — gateway failed to start (degraded / missing model)
* ``3`` — config / argparse error

Operator usage (on the Jetson, inside the orchestrator container):

    docker exec mousedroid python3 /opt/mousedroid/tools/llm_latency_probe.py \\
        --config /etc/mousedroid/jetson_production.yaml

The probe is deliberately read-only: it builds an isolated LLMGateway via
the factory, runs one translate_mission, and exits. It does not touch the
running orchestrator's gateway or any other subsystem.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_injection_filter, build_llm_gateway
from mousedroid.logging.setup import get_logger

_log = get_logger("llm_latency_probe")

# Default mission text — short + deterministic enough that the parser path is
# exercised but inference time isn't dominated by token-count variance.
_DEFAULT_MISSION = "turn left slowly"

# tegrastats output line shape (Orin Nano):
#   RAM 2914/7619MB ...  GR3D_FREQ 0%@[306,...]  ... NVRGTX_FREQ ...
# We extract two numbers: used RAM (MB) and total RAM (MB). On non-Jetson hosts
# tegrastats is absent — the probe falls back to ``None`` for the snapshot
# fields. We deliberately don't try to use ``nvidia-smi`` because the Orin
# Nano's iGPU shares system RAM (UMA) — RAM is the right signal here, not
# discrete VRAM.
_TEGRASTATS_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")


def _tegrastats_snapshot(timeout_s: float = 2.0) -> dict[str, int | str | None]:
    """Capture one ``tegrastats`` line and parse the RAM usage.

    Returns a dict with keys ``ram_used_mb``, ``ram_total_mb``, ``raw_line``.
    All values are ``None`` when ``tegrastats`` is absent (non-Jetson host) or
    the parse fails. Never raises.
    """
    if shutil.which("tegrastats") is None:
        _log.warning("tegrastats_not_available", host_kind="non_jetson")
        return {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None}
    try:
        # ``--interval`` is in ms; ``--count`` exits after N samples.
        # S603/S607: fixed argv list, no shell, executable resolved via PATH
        # (the shutil.which check above guarantees presence). Operator-side
        # diagnostic tool — not exposed to untrusted input.
        result = subprocess.run(  # noqa: S603
            ["tegrastats", "--interval", "100", "--count", "1"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("tegrastats_invocation_failed", error=f"{type(exc).__name__}: {exc}")
        return {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None}

    raw_line = (result.stdout or "").strip().splitlines()[-1:] or [""]
    line = raw_line[0]
    match = _TEGRASTATS_RAM_RE.search(line)
    if match is None:
        _log.warning("tegrastats_parse_failed", raw_line=line[:120])
        return {"ram_used_mb": None, "ram_total_mb": None, "raw_line": line}
    return {
        "ram_used_mb": int(match.group(1)),
        "ram_total_mb": int(match.group(2)),
        "raw_line": line,
    }


def _llama_model_metadata(gateway: Any) -> dict[str, Any]:
    """Introspect the loaded llama-cpp Llama instance for the actual GPU layer count.

    Distinguishes "GPU offload silently failed" (cfg requests -1 but the model
    reports 0 layers offloaded) from "GPU offload worked, model just slow".

    Returns ``{"loaded": False}`` if the gateway is degraded or the Llama
    instance doesn't expose the introspection attribute (older llama-cpp
    versions). Never raises.
    """
    model = getattr(gateway, "_model", None)
    if model is None:
        return {"loaded": False, "reason": "gateway_model_is_none"}

    # llama-cpp-python exposes the resolved n_gpu_layers via several attribute
    # paths depending on version. Try them in priority order; fall back to
    # ``model_metadata`` which is the most-stable surface.
    candidates: list[tuple[str, Callable[[Any], Any]]] = [
        ("attr_n_gpu_layers", lambda m: getattr(m, "n_gpu_layers", None)),
        ("attr_model_n_gpu_layers", lambda m: getattr(m, "_n_gpu_layers", None)),
        (
            "model_params_n_gpu_layers",
            lambda m: getattr(getattr(m, "model_params", None), "n_gpu_layers", None),
        ),
    ]
    metadata: dict[str, Any] = {"loaded": True}
    for name, accessor in candidates:
        try:
            value = accessor(model)
        except Exception as exc:  # pragma: no cover — defensive across versions
            metadata[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if value is not None:
            metadata[name] = value
    metadata["model_class"] = type(model).__name__
    return metadata


async def _main(args: argparse.Namespace) -> int:
    config_paths = [Path(args.config)] if args.config else []
    cfg = load_settings(*config_paths)

    if not cfg.llm.enabled:
        _log.error("llm_gateway_disabled_in_cfg", config=args.config)
        return 3

    _log.info(
        "probe_cfg",
        n_gpu_layers=cfg.llm.n_gpu_layers,
        n_threads=cfg.llm.n_threads,
        n_batch=cfg.llm.n_batch,
        context_length=cfg.llm.context_length,
        latency_target_ms=cfg.llm.latency_target_ms,
        model_path=str(cfg.llm.model_path),
        env_n_gpu_layers_override=os.environ.get("MOUSEDROID_LLM__N_GPU_LAYERS"),
    )

    snapshot_before = _tegrastats_snapshot()
    _log.info("tegrastats_before", **snapshot_before)

    injection_filter = build_injection_filter(cfg)
    gateway = build_llm_gateway(cfg, injection_filter=injection_filter)

    t_start = time.monotonic()
    try:
        await gateway.start()
    except Exception as exc:
        # llama-cpp-python raises ValueError("Failed to load model from file: ...")
        # when CUDA-build is absent or the model doesn't fit at the configured
        # n_gpu_layers. The gateway only catches ImportError + OSError, so
        # everything else bubbles. Catch it here and emit a clean structured
        # log so the operator dashboard sees the load-failure signal — this
        # is the "GPU offload silently failed at load time" case the probe
        # is designed to surface (distinguishes from "loaded but slow").
        cold_start_ms = (time.monotonic() - t_start) * 1000.0
        _log.error(
            "llm_gateway_load_failed",
            cold_start_ms=cold_start_ms,
            error_type=type(exc).__name__,
            error=str(exc),
            n_gpu_layers=cfg.llm.n_gpu_layers,
            env_n_gpu_layers_override=os.environ.get("MOUSEDROID_LLM__N_GPU_LAYERS"),
            hint=(
                "Common causes: llama-cpp-python in container not built with CUDA "
                "support; model doesn't fit at n_gpu_layers=-1 on shared 8GB RAM "
                "(try smaller integer); model file corrupt."
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

    _log.info("llama_model_metadata", **_llama_model_metadata(gateway))

    # Measure the single-shot translate_mission elapsed.
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
        prog="python -m tools.llm_latency_probe",
        description="F-006 live-Jetson latency probe for LLMGateway.translate_mission.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a config overlay YAML (e.g. /etc/mousedroid/jetson_production.yaml). "
            "Omit to use config/default.yaml only."
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
