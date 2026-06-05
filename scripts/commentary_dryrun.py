#!/usr/bin/env python3
"""MSE-6 grounded-commentary dry-run probe.

Exercises the commentary composer + voice path WITHOUT motors or the orchestrator
loop: synthesise one :class:`CommentaryFacts` from CLI flags, compose a plain
line (template or LLM, per ``cfg.commentary.composer``), apply Rocky styling, and
speak it through the (mock under ``MOUSEDROID_MOCK_HARDWARE=true``) voice engine.
Lets an operator hear/verify the narration phrasing offline.

Config resolution mirrors ``scripts/ask_rover.py`` (explicit ``--config`` wins,
else ``MOUSEDROID_CONFIG(S)`` env vars).

Exit codes: ``0`` ok / ``1`` runtime error / ``2`` config error.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import structlog  # noqa: E402

structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=False,
)

from mousedroid.commentary.protocol import CommentaryFacts  # noqa: E402
from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.factory import build_commentary_composer, build_voice_engine  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402
from mousedroid.validation.runtime import resolve_runtime_config_paths  # noqa: E402
from mousedroid.voice.rocky import rocky_transform  # noqa: E402

_log = get_logger(__name__)

_EXIT_OK = 0
_EXIT_RUNTIME_ERROR = 1
_EXIT_CONFIG_ERROR = 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=None, type=Path)
    parser.add_argument("--novelty", type=float, default=None)
    parser.add_argument("--clearance-m", type=float, default=5.0)
    parser.add_argument("--audio-rms", type=float, default=0.0)
    parser.add_argument("--speed-mps", type=float, default=0.0)
    parser.add_argument("--battery-v", type=float, default=12.0)
    parser.add_argument("--emergency", action="store_true")
    parser.add_argument("--lidar-invalid", action="store_true")
    return parser.parse_args(argv)


def _facts_from_args(args: argparse.Namespace) -> CommentaryFacts:
    return CommentaryFacts(
        min_clearance_m=args.clearance_m,
        forward_distance_m=args.clearance_m,
        audio_rms=args.audio_rms,
        speed_mps=args.speed_mps,
        turn_rate=0.0,
        battery_v=args.battery_v,
        novelty=args.novelty,
        is_emergency=args.emergency,
        lidar_valid=not args.lidar_invalid,
        audio_valid=args.audio_rms > 0.0,
        timestamp=0.0,
    )


async def _run(args: argparse.Namespace, settings: object) -> int:
    composer = build_commentary_composer(settings, gateway=None)  # type: ignore[arg-type]
    if composer is None:
        _log.error("commentary_dryrun_no_composer")
        return _EXIT_RUNTIME_ERROR

    facts = _facts_from_args(args)
    plain = await composer.compose(facts)
    intensity = settings.commentary.excitement_intensity  # type: ignore[union-attr]
    styled = rocky_transform(plain, intensity=intensity) if plain else ""

    # Speaking is best-effort: a dev box without a speaker still shows the
    # composed text (the primary deliverable of this probe).
    samples = 0
    voice = build_voice_engine(settings)  # type: ignore[arg-type]
    if voice is not None and styled:
        await voice.start()
        try:
            samples, _peak = await voice.play_phrase(styled)
        finally:
            await voice.stop()
    elif voice is None:
        _log.info("commentary_dryrun_voice_unavailable", hint="speaker disabled; text-only")

    print(
        f"composer={type(composer).__name__} "
        f"plain={plain!r} styled={styled!r} samples={samples}"
    )
    return _EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings = load_settings(*resolve_runtime_config_paths(args.config))
    except Exception:
        _log.exception("commentary_dryrun_config_error")
        return _EXIT_CONFIG_ERROR
    if settings.commentary is None or not settings.commentary.enabled:
        _log.error("commentary_dryrun_disabled", hint="set commentary.enabled=true")
        return _EXIT_CONFIG_ERROR
    try:
        return asyncio.run(_run(args, settings))
    except Exception:
        _log.exception("commentary_dryrun_runtime_error")
        return _EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
