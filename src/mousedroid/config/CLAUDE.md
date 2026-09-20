# Config Subsystem — Surface Contract

> Pydantic settings schema and YAML loader (``Settings``, ``load_settings``).

## Invariants & Config Rules

1. **Single Load Entry**: Runtime settings enter through ``load_settings`` in ``loader.py``;
   callers import ``Settings`` from ``mousedroid.config`` / ``mousedroid.config.schema``.
2. **Domain-Split Schema Package**: ``schema/`` re-exports the former flat schema surface
   unchanged; models live in domain modules (``hardware``, ``cognitive``, ``world_model``,
   ``learning``, ``reward_safety``, ``telemetry``, ``llm``, ``voice``, ``sim``, ``gcp_cloud``,
   ``arm``, ``training``, ``harness_mcp``, ``misc``, ``root``, ``_primitives``).
3. **Migration Helpers**: ``migration.py`` supports settings upgrades without breaking
   existing YAML on ``git pull``.

## Key Files

- `loader.py::load_settings` — YAML load, overlay merge, and ``Settings`` construction.
- `schema/` — domain-split Pydantic models with a stable re-export surface.
- `migration.py` — settings migration helpers.
- `tests/unit/config/` — subsystem unit tests.
