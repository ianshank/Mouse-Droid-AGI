#!/usr/bin/env python3
"""MSE-6 named-greeting one-shot CLI.

Plays the operator-configured greeting through the Rocky voice engine:
``greeting_excited`` pre-flourish chirp → operator-tunable inter-chirp
pause → rocky-transformed ``Hello {names}!`` message. Names live in the
YAML overlay (``config/greeting_pilot.yaml.example``) — no CLI name
override by design.

The OLED face controller is deliberately NOT driven here (operator's
current dev rover has no SSD1306 attached); the ``Greeter`` class
exposes a future extension point.

Examples::

    # Mock-hardware dry-run on a dev box (no real audio, just logs):
    python scripts/greet_intro.py --config config/greeting_pilot.yaml --dry-run

    # On the Jetson with the production overlay stacked underneath:
    MOUSEDROID_JETSON_CONFIGS=\\
        config/jetson_production.yaml,config/greeting_pilot.yaml \\
        python scripts/greet_intro.py

Exit codes:

* ``0`` — greeting played (or dry-run logged) successfully.
* ``1`` — runtime hardware / voice-engine failure.
* ``2`` — configuration error (greeting disabled, names empty,
  template missing ``{names}``, voice disabled, etc.).
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

# Force structlog to stderr BEFORE importing anything from mousedroid so the
# import-time configure() inside mousedroid.config.loader doesn't latch on to
# stdout (mirrors scripts/check_usbc_devices.py to keep --json paths clean
# for future structured-output additions).
import structlog  # noqa: E402

structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO and above
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=False,
)

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.factory import build_greeter  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402
from mousedroid.validation.runtime import resolve_runtime_config_paths  # noqa: E402

_log = get_logger(__name__)

_EXIT_OK = 0
_EXIT_RUNTIME_ERROR = 1
_EXIT_CONFIG_ERROR = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        type=Path,
        help=(
            "One or more YAML overlays (repeatable). When omitted, falls "
            "back to the resolve_runtime_config_paths() chain — same "
            "resolution path as the orchestrator + smoke wrappers."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Force mock_hardware=true so the script runs on a dev machine "
            "without touching real audio hardware. MockTTS records the "
            "synthesized text, MockSpeaker silently drops chunks, and the "
            "structured log shows the full greeting flow."
        ),
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    """Async entry — returns the desired process exit code."""
    config_paths = resolve_runtime_config_paths(args.config)
    settings = load_settings(*config_paths)

    if args.dry_run and not settings.mock_hardware:
        # ``model_copy`` keeps the override local to this process —
        # doesn't mutate the file-loaded settings the orchestrator
        # would see in another process.
        settings = settings.model_copy(update={"mock_hardware": True})
        _log.info("greet_intro_dry_run_enabled", original_mock_hardware=False)

    try:
        greeter = build_greeter(settings)
    except ValueError as exc:
        # build_greeter raises ValueError for: greeting None / disabled,
        # voice engine not buildable. These are operator configuration
        # errors — exit 2 so wrapper scripts can distinguish from a
        # runtime hardware failure (exit 1).
        _log.error("greet_intro_config_error", error=str(exc))
        return _EXIT_CONFIG_ERROR

    # Caller owns the voice-engine lifecycle (start before / stop after).
    # ``Greeter.voice_engine`` is a read-only property typed against the
    # protocol — no private-attribute access.
    voice = greeter.voice_engine
    try:
        await voice.start()
        try:
            await greeter.greet()
        finally:
            await voice.stop()
    except ValueError as exc:
        # Empty names list at runtime (slipped past schema validator).
        _log.error("greet_intro_runtime_value_error", error=str(exc))
        return _EXIT_CONFIG_ERROR
    except Exception as exc:  # top-level CLI error boundary
        _log.error(
            "greet_intro_runtime_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return _EXIT_RUNTIME_ERROR

    return _EXIT_OK


def main() -> int:
    """Synchronous entry point — wraps :func:`_run` in ``asyncio.run``."""
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
