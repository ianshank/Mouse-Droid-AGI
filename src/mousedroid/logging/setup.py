"""Structured logging setup for MouseDroid.

Uses structlog with JSON renderer in production, console renderer in development.
All modules should use ``get_logger(__name__)`` — never ``print()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from mousedroid.config.schema import LoggingConfig

_configured: bool = False


def configure_logging(cfg: LoggingConfig) -> None:
    """Configure structlog for the given logging config.

    Args:
        cfg: Logging configuration with level and format.
    """
    global _configured

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if cfg.format == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            _level_to_int(cfg.level),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound with the given module name.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Bound structlog logger instance.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]


_LEVEL_MAP: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def _level_to_int(level: str) -> int:
    """Convert string log level to integer.

    Args:
        level: Log level name (case-insensitive).

    Returns:
        Integer log level.
    """
    return _LEVEL_MAP.get(level.upper(), 20)
