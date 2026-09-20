# Logging Subsystem — Surface Contract

> Structured logging setup via structlog — JSON in production, console in development
> (``configure_logging``, ``get_logger``, ``safe_log_extra``).

## Invariants & Logging Rules

1. **Single Setup Entry**: ``configure_logging`` owns process-wide structlog configuration
   from ``LoggingConfig``; modules obtain loggers only via ``get_logger(__name__)``.
2. **Reserved-Key Safety**: ``safe_log_extra`` / ``EXTRA_KEY_PREFIX`` prevent caller keys
   colliding with reserved structlog event keys.
3. **Credential Redaction**: ``redaction.py`` strips URI credentials from values that reach
   log events.

## Key Files

- `setup.py::configure_logging` / `get_logger` / `safe_log_extra` — process setup and logger factory.
- `redaction.py` — URI credential redaction helpers.
- `tests/unit/logging/` — subsystem unit tests.
