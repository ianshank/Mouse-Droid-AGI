"""Structured logging setup."""

from mousedroid.logging.setup import (
    EXTRA_KEY_PREFIX,
    RESERVED_LOG_KEYS,
    configure_logging,
    get_logger,
    safe_log_extra,
)

__all__ = [
    "EXTRA_KEY_PREFIX",
    "RESERVED_LOG_KEYS",
    "configure_logging",
    "get_logger",
    "safe_log_extra",
]
