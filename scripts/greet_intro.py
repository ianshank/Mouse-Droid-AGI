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
import logging
import sys
from pathlib import Path

# Make src/ importable when run from repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# structlog is imported at module scope but configured inside ``main()``
# so import-as-a-library callers (e.g. test collection that happens to
# import this script) don't get a global structlog override as a side
# effect. The configuration is still applied BEFORE any greet_intro
# work happens — see ``_configure_stderr_logging`` below.
import structlog  # noqa: E402

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.factory import build_greeter  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402
from mousedroid.validation.runtime import resolve_runtime_config_paths  # noqa: E402

# Round-3 review (Gemini): the module-level ``get_logger(__name__)``
# call was firing at import time, BEFORE :func:`main` runs
# :func:`_configure_stderr_logging`. ``cache_logger_on_first_use=False``
# (set inside the configure call) makes structlog re-bind on every
# emit, so a stale cached logger was unlikely — but tying the logger
# acquisition to ``_run()`` removes the ordering question entirely
# and matches the pattern used elsewhere in scripts/ that respect
# late-bound logging configuration.

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
    # Logger acquisition deferred from module scope so it picks up the
    # structlog config installed by :func:`_configure_stderr_logging`.
    _log = get_logger(__name__)
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


def _configure_stderr_logging(level: int = logging.INFO) -> None:
    """Pin structlog to JSON-on-stderr at the given level.

    Called from :func:`main` so importing this module as a library
    (e.g. during test collection) doesn't have the side effect of
    reconfiguring the importing process's structlog state. The
    orchestrator's structlog configuration is left untouched until a
    direct CLI invocation explicitly calls this.

    Args:
        level: Numeric log level (``logging.INFO`` by default). The
            named constant avoids the prior magic-number ``20`` and
            lets future operator-tools surface a ``--log-level`` flag
            by forwarding to this helper.
    """
    structlog.configure(
        processors=[structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def main() -> int:
    """Synchronous entry point — wraps :func:`_run` in ``asyncio.run``."""
    _configure_stderr_logging()
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
