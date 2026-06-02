#!/usr/bin/env python3
"""MSE-6 mission-translation dry-run probe (PR #107).

Translates a single natural-language mission into a normalised ``GoalVector``
via the deliberative LLM gateway — WITHOUT issuing any motor command — so the
Claude-primary / Phi-3-fallback path can be verified live on the rover even
when the ESP32 / drivetrain is detached.

It builds the gateway through the real factory (``build_llm_gateway``), so it
honours whatever ``llm:`` config the rover runs (cloud Claude when reachable,
local llama_cpp fallback when off-network). The tier that actually served and
the composite's degraded state are printed so an operator can confirm which
path answered.

Config resolution mirrors the orchestrator/greeting CLIs via
``resolve_runtime_config_paths``: explicit ``--config`` wins, else the
``MOUSEDROID_CONFIGS`` / ``MOUSEDROID_JETSON_CONFIGS`` CSV env vars, else the
single ``MOUSEDROID_CONFIG`` / ``MOUSEDROID_JETSON_CONFIG`` env var.

Examples::

    # On the Jetson with the production overlay (resolved from the env var):
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
import logging
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
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=False,
)

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.factory import build_llm_gateway  # noqa: E402
from mousedroid.llm_gateway.protocol import LLMGatewayProtocol  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402
from mousedroid.validation.runtime import resolve_runtime_config_paths  # noqa: E402

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
            "One or more YAML overlays (repeatable). When omitted, the config "
            "is resolved from MOUSEDROID_CONFIG(S)/MOUSEDROID_JETSON_CONFIG(S) "
            "env vars — the same resolution the orchestrator uses."
        ),
    )
    return parser.parse_args(argv)


def _describe_tier(gateway: object) -> str:
    """Best-effort label for which tier served the translation.

    Tolerant of the gateway's concrete shape (diagnostic only): a
    :class:`FallbackLLMGateway` composite exposes ``_primary``/``is_degraded``,
    while a single backend exposes only ``is_degraded``. ``is_degraded`` is not
    part of :class:`LLMGatewayProtocol`, so every read is guarded.

    * composite, both tiers down  -> ``"none — both tiers degraded"``
    * composite, primary degraded -> ``"secondary (local fallback)"``
    * composite, primary usable   -> ``"primary"``
    * single gateway, degraded    -> ``"degraded"``
    * single gateway, usable      -> ``"primary"``
    """
    composite_primary = getattr(gateway, "_primary", None)
    if composite_primary is not None:
        if bool(getattr(gateway, "is_degraded", False)):
            return "none — both tiers degraded"
        if bool(getattr(composite_primary, "is_degraded", False)):
            return "secondary (local fallback)"
        return "primary"
    return "degraded" if bool(getattr(gateway, "is_degraded", False)) else "primary"


async def _run(mission: str, gateway: LLMGatewayProtocol) -> int:
    """Start the gateway, translate one mission, print the result, stop.

    ``stop()`` is guaranteed even if ``start()`` or ``translate_mission()``
    raises — both are inside the single ``try`` whose ``finally`` calls stop.

    The serving tier is described BEFORE ``stop()`` runs: some gateways clear
    their degraded flag on stop (e.g. ``AnthropicLLMGateway.stop()``), which
    would otherwise make a secondary-served call misreport as ``primary``.
    """
    try:
        await gateway.start()
        vector = await gateway.translate_mission(mission)
        tier = _describe_tier(gateway)  # capture before stop() may clear degraded state
    finally:
        await gateway.stop()

    # GoalVector is a dataclass — print its fields explicitly on stdout.
    print(
        f"mission={mission!r} tier={tier} "
        f"GoalVector(vx_target={vector.vx_target:.3f}, "
        f"vy_target={vector.vy_target:.3f}, "
        f"omega_target={vector.omega_target:.3f})"
    )
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve config the same way the orchestrator does: explicit --config wins,
    # else the MOUSEDROID_CONFIG(S) env vars (so the probe matches production).
    config_paths = resolve_runtime_config_paths(args.config)
    try:
        settings = load_settings(*config_paths)
    except Exception:
        # Operator probe: report via structured logs + exit code rather than
        # dumping a traceback. (BLE001 is not enabled in this project.)
        _log.exception(
            "translate_mission_config_error",
            config_paths=[str(p) for p in config_paths],
        )
        return _EXIT_CONFIG_ERROR

    try:
        gateway = build_llm_gateway(settings)
    except Exception:
        _log.exception("translate_mission_build_error")
        return _EXIT_RUNTIME_ERROR

    try:
        return asyncio.run(_run(args.mission, gateway))
    except Exception:
        _log.exception("translate_mission_runtime_error")
        return _EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
