"""F-006 verification: live-Jetson LLMGateway.translate_mission latency probe.

PR #101 flipped ``config/jetson_production.yaml`` from ``llm.n_gpu_layers: 0``
to ``-1`` (offload every layer to the iGPU). This probe verifies that change
actually delivers sub-``cfg.llm.latency_target_ms`` inference on the real
Jetson, not just in theory.

What it measures:

* Cold-start: time to ``LLMGateway.start()`` (model load + warm-up).
* Per-iteration: ``translate_mission("turn left slowly")`` elapsed ms, sampled
  ``--iterations`` times (default 1). With >1 the probe emits a
  ``llm_latency_summary`` (p50/p95/p99) and gates on p95 instead of a single
  unlucky sample.
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
* ``4`` — (``--report-json`` only) the configured primary backend did not
  answer every iteration: a fallback served, so the latency numbers describe
  the fallback, not the model the overlay names

``--report-json PATH`` (F-059 WS-A) additionally writes a machine-readable run
report — the evidence artifact for retargeting the cloud model::

    {"model_requested": ..., "model_resolved": ..., "resolved_via": ...,
     "backend": ..., "served": "primary"|"fallback", "cold_start_ms": ...,
     "p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "iterations": ...,
     "neutral_replies": ..., "passed": ...}

``served`` answers the question a latency number cannot: with a cloud primary
and a local fallback, a request the cloud tier rejects (HTTP 400 for a
parameter the model does not take, an expired key, no route) is answered by the
fallback in a few hundred milliseconds and would otherwise read as a pass. It is
``"primary"`` only when the primary backend was ready and healthy after EVERY
iteration, so a run that failed over once and recovered still reads
``"fallback"``. ``passed`` is the latency gate AND ``served == "primary"``.
``neutral_replies`` counts the iterations whose goal came back all-zero. The
gateways answer an empty / unparseable reply with a neutral vector WITHOUT
degrading (the round trip worked), so ``served`` stays ``"primary"`` for a model
that answered with nothing usable - e.g. a Haiku 5.5 whose ``max_tokens`` went on
thinking. It is surfaced, not gated, because a neutral goal is the correct answer
to a "stop" mission; for the default mission anything above 0 needs a look, and a
run in which every reply was neutral logs ``llm_probe_every_reply_neutral``.
``model_resolved`` is what ``client.models.retrieve`` returns for the requested
id (``resolved_via: "models.retrieve"``); with no client to ask, or a failed
lookup, it is the configured id (``resolved_via: "config"``). Without the flag
nothing is written and the exit codes above (except ``4``) are unchanged.

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
import inspect
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Shared Jetson-side helper (promoted out of this file during the F-006
# remote-LLM sprint so jetson_remote_llm_probe can reuse it). Aliased to
# the prior private name so existing test patches at
# ``probe._tegrastats_snapshot`` keep working without test changes.
from tools._jetson_helpers import tegrastats_snapshot as _tegrastats_snapshot

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_injection_filter, build_llm_gateway
from mousedroid.logging.setup import get_logger
from mousedroid.validation.latency_stats import summarize

_log = get_logger("llm_latency_probe")

# Default mission text — short + deterministic enough that the parser path is
# exercised but inference time isn't dominated by token-count variance.
_DEFAULT_MISSION = "turn left slowly"

# NOTE: ``_tegrastats_snapshot`` and ``_TEGRASTATS_RAM_RE`` moved to
# ``tools/_jetson_helpers.py`` during the F-006 remote-LLM sprint. The
# module-level import above re-exports the function under the prior
# private name so existing test patches at ``probe._tegrastats_snapshot``
# keep working.

# Default sample count. Single source of truth for both the argparse default and
# the defensive fallback in ``_main`` (which is called directly in unit tests
# with hand-built Namespaces that may predate the flag). 1 == legacy single-shot.
_DEFAULT_ITERATIONS = 1

# ``--report-json`` vocabulary (F-059 WS-A). ``served`` says whether the
# configured primary backend answered; ``resolved_via`` says where
# ``model_resolved`` came from.
_SERVED_PRIMARY = "primary"
_SERVED_FALLBACK = "fallback"
_RESOLVED_VIA_RETRIEVE = "models.retrieve"
_RESOLVED_VIA_CONFIG = "config"

# Exit code: a fallback answered (reported only under ``--report-json``).
_EXIT_PRIMARY_NOT_SERVED = 4

# The ``GoalVector`` axes a "usable reply" must move at least one of.
_GOAL_AXES = ("vx_target", "vy_target", "omega_target")


def _health_flag(obj: Any, name: str) -> bool:
    """Read a boolean health flag that is a property (shipped gateways) or a method."""
    value = getattr(obj, name, False)
    if callable(value):
        value = value()
    return bool(value)


def _primary_of(gateway: Any) -> Any:
    """Return the tier a report vouches for: a composite's primary, else the gateway itself.

    ``FallbackLLMGateway`` keeps its cloud tier in ``_primary``; a bare gateway
    has no fallback, so it IS the primary.
    """
    return getattr(gateway, "_primary", gateway)


def _primary_served(primary: Any) -> bool:
    """True iff ``primary`` is ready and healthy, i.e. it (not a fallback) just answered.

    The shipped gateways never raise on a backend failure: they return a
    neutral vector and flip ``is_degraded``. After a call, a degraded or
    not-ready primary therefore means the composite served the command from its
    secondary (or nothing answered), so the flags are the signal — not the
    returned vector, which a legitimate "stop" also makes neutral.
    """
    return _health_flag(primary, "is_ready") and not _health_flag(primary, "is_degraded")


def _is_neutral(goal: Any) -> bool:
    """True iff every velocity axis of ``goal`` is zero (the gateways' "no usable reply").

    The shipped gateways answer an empty / non-JSON / non-object reply with a
    neutral ``GoalVector`` and DO NOT flip ``is_degraded`` - the round trip
    itself worked - so the degrade flags that drive ``served`` cannot see it.
    """
    return all(getattr(goal, axis, 0.0) == 0.0 for axis in _GOAL_AXES)


class _RunTracker:
    """Accumulates, across iterations, what the report needs beyond the latencies.

    * whether the primary answered every iteration - judging by the final state
      alone would call a run that failed over on iteration 1 and recovered after
      the cooldown "primary"; it was not;
    * how many replies were neutral (see :func:`_is_neutral`).
    """

    def __init__(self, primary: Any) -> None:
        self._primary = primary
        self._always_primary = True
        self.neutral_replies = 0

    def observe(self, goal: Any) -> None:
        """Record the outcome of the translate call that just returned ``goal``."""
        if not _primary_served(self._primary):
            self._always_primary = False
        if _is_neutral(goal):
            self.neutral_replies += 1

    @property
    def label(self) -> str:
        """``"primary"`` iff every observed iteration was answered by the primary."""
        return _SERVED_PRIMARY if self._always_primary else _SERVED_FALLBACK


async def _resolve_model(primary: Any, requested: str, timeout_s: float) -> tuple[str, str]:
    """Resolve the model id the API serves for ``requested``.

    Asks the primary's SDK client (``client.models.retrieve``) so the artifact
    records what the account was really served, not only what the YAML said.
    Falls back to the configured id (``resolved_via == "config"``) when there is
    no client to ask (a local gateway) or the lookup fails (offline, 404, auth):
    an unresolvable id must not turn a healthy latency run into a failure.

    Returns:
        ``(model_resolved, resolved_via)``.
    """
    models = getattr(getattr(primary, "_client", None), "models", None)
    retrieve = getattr(models, "retrieve", None)
    if retrieve is None:
        return requested, _RESOLVED_VIA_CONFIG
    try:
        info = retrieve(requested)
        if inspect.isawaitable(info):
            info = await asyncio.wait_for(info, timeout=timeout_s)
    except Exception as exc:
        _log.warning(
            "llm_probe_model_resolve_failed",
            model=requested,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return requested, _RESOLVED_VIA_CONFIG
    resolved = getattr(info, "id", None)
    if isinstance(resolved, str) and resolved.strip():
        return resolved, _RESOLVED_VIA_RETRIEVE
    return requested, _RESOLVED_VIA_CONFIG


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Write ``report`` as pretty-printed JSON (LF endings, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


async def _finish_report(
    report_path: Path,
    *,
    primary: Any,
    cfg: Any,
    served: str,
    cold_start_ms: float,
    summary: Any,
    iterations: int,
    neutral_replies: int,
    latency_ok: bool,
) -> int:
    """Resolve the model id, write the JSON report, and return the probe's exit code.

    ``passed`` is the latency gate AND ``served == "primary"``: a fallback-served
    run never reads as a pass (plan risk R10). ``neutral_replies`` is surfaced,
    not gated - a neutral goal is the right answer to a "stop" mission - but a run
    in which EVERY reply was neutral is logged loudly. Must run BEFORE
    ``gateway.stop()``, which drops the client ``models.retrieve`` needs.
    """
    requested = cfg.llm.model_name
    resolved, resolved_via = await _resolve_model(primary, requested, cfg.llm.request_timeout_s)
    primary_served = served == _SERVED_PRIMARY
    passed = latency_ok and primary_served
    report = {
        "model_requested": requested,
        "model_resolved": resolved,
        "resolved_via": resolved_via,
        "backend": cfg.llm.backend,
        "served": served,
        "cold_start_ms": cold_start_ms,
        "p50_ms": summary.p50_ms,
        "p95_ms": summary.p95_ms,
        "p99_ms": summary.p99_ms,
        "iterations": iterations,
        "neutral_replies": neutral_replies,
        "passed": passed,
    }
    await asyncio.to_thread(_write_report, report_path, report)
    _log.info(
        "llm_probe_report_written",
        path=str(report_path),
        served=served,
        passed=passed,
        neutral_replies=neutral_replies,
        model_requested=requested,
        model_resolved=resolved,
        resolved_via=resolved_via,
    )
    if iterations > 0 and neutral_replies == iterations:
        _log.warning(
            "llm_probe_every_reply_neutral",
            neutral_replies=neutral_replies,
            iterations=iterations,
            hint=(
                "every reply parsed to an all-zero goal: the backend answered but nothing "
                "usable came back (expected only for a 'stop' mission). Look for "
                "anthropic_gateway_translation_unparseable / anthropic_gateway_empty_content; "
                "Haiku 5.5 thinks by default and thinking tokens count toward max_tokens, "
                "so set llm.anthropic.thinking: disabled or raise max_tokens"
            ),
        )
    if not primary_served:
        _log.error(
            "llm_primary_not_served",
            model=requested,
            hint=(
                "a fallback answered, so the latency numbers do not describe the "
                "requested model; look for anthropic_gateway_request_failed / "
                "fallback_primary_to_secondary events earlier in this log"
            ),
        )
        return _EXIT_PRIMARY_NOT_SERVED
    return 0 if latency_ok else 1


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

    # Measure translate_mission elapsed over ``--iterations`` runs. iterations=1
    # preserves the legacy single-shot ``llm_latency_result`` event + gate (a
    # single elapsed <= target) byte-for-byte; >1 adds a percentile summary and
    # gates on p95 so a flaky tail (cloud round-trip variance / GPU contention)
    # is caught without failing on a single unlucky sample.
    samples_ms: list[float] = []
    # ``getattr`` fallback keeps ``_main`` robust when called directly with a
    # hand-built Namespace (unit tests) that predates the --iterations flag.
    iterations = getattr(args, "iterations", _DEFAULT_ITERATIONS)
    # ``--report-json`` is optional and absent from hand-built Namespaces; only
    # when it is given do we track which tier answered (legacy runs are unchanged).
    report_path: Path | None = getattr(args, "report_json", None)
    tracker = _RunTracker(_primary_of(gateway)) if report_path is not None else None
    for i in range(iterations):
        t_translate = time.monotonic()
        goal = await gateway.translate_mission(args.mission)
        elapsed_ms = (time.monotonic() - t_translate) * 1000.0
        samples_ms.append(elapsed_ms)
        if tracker is not None:
            tracker.observe(goal)
        _log.info(
            "llm_latency_result",
            iteration=i,
            elapsed_ms=elapsed_ms,
            target_ms=cfg.llm.latency_target_ms,
            passed=elapsed_ms <= cfg.llm.latency_target_ms,
            goal_vx=goal.vx_target,
            goal_vy=goal.vy_target,
            goal_omega=goal.omega_target,
            mission=args.mission,
        )

    summary = summarize(samples_ms)
    # Single-shot gates on the one sample (== p95); multi-shot gates on p95.
    gate_ms = summary.p95_ms
    passed = gate_ms <= cfg.llm.latency_target_ms
    _log.info(
        "llm_latency_summary",
        iterations=iterations,
        target_ms=cfg.llm.latency_target_ms,
        gate_ms=gate_ms,
        passed=passed,
        min_ms=summary.min_ms,
        mean_ms=summary.mean_ms,
        p50_ms=summary.p50_ms,
        p95_ms=summary.p95_ms,
        p99_ms=summary.p99_ms,
        max_ms=summary.max_ms,
        mission=args.mission,
    )

    exit_code = 0 if passed else 1
    if report_path is not None and tracker is not None:
        # Before ``stop()``: stopping drops the client ``models.retrieve`` needs.
        exit_code = await _finish_report(
            report_path,
            primary=_primary_of(gateway),
            cfg=cfg,
            served=tracker.label,
            cold_start_ms=cold_start_ms,
            summary=summary,
            iterations=iterations,
            neutral_replies=tracker.neutral_replies,
            latency_ok=passed,
        )
    await gateway.stop()
    return exit_code


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
    parser.add_argument(
        "--iterations",
        type=int,
        default=_DEFAULT_ITERATIONS,
        help=(
            "Number of translate_mission runs to sample. 1 (default) is the "
            "legacy single-shot gate; >1 emits a p50/p95/p99 summary and gates "
            "on p95 to absorb tail variance."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write a machine-readable run report to PATH (parent directories are "
            "created): model requested / resolved, which tier served (primary or "
            "fallback), cold start, p50/p95/p99, iterations, passed. With this flag "
            "a run in which a fallback answered exits 4 instead of passing on the "
            "fallback's latency. Default: write nothing."
        ),
    )
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        _log.warning("probe_interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
