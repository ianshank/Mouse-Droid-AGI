"""Print healthcheck env vars in shell-sourceable KEY=value form.

Invoked by the container entrypoint (``scripts/mousedroid_entrypoint.sh``)
to materialize the env file the healthcheck script reads. Kept separate
from :mod:`mousedroid.health.healthcheck_env` so that pure helper stays
import-side-effect-free and unit-testable in isolation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import structlog

from mousedroid.config.loader import load_settings
from mousedroid.health.healthcheck_env import derive_healthcheck_env


def _silence_logging() -> None:
    """Silence stdlib logging and structlog before the preflight load.

    The subsequent ``python -m mousedroid.main`` call configures logging
    properly. Without this, ``load_settings`` emits debug/info events
    through structlog's default chain (``PrintLoggerFactory`` → stdout),
    which would mix log lines with the KEY=VALUE output the entrypoint
    redirects into the env file. ``logging.disable`` alone is not
    sufficient because structlog writes directly to stdout, bypassing
    stdlib logging.
    """
    logging.disable(logging.CRITICAL)
    structlog.configure(
        processors=[],
        logger_factory=structlog.ReturnLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list excluding the program name. ``None`` means
            read from ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    _silence_logging()

    parser = argparse.ArgumentParser(
        description="Print Docker healthcheck env vars derived from Settings.",
    )
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    cfg = load_settings(*args.config)
    # ``_validate_path`` inside ``derive_healthcheck_env`` rejects values
    # containing characters that would break single-quoted shell strings,
    # so wrapping with single quotes here is safe to dot-source.
    for key, value in derive_healthcheck_env(cfg).items():
        sys.stdout.write(f"{key}='{value}'\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
