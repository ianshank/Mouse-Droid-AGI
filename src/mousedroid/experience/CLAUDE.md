# Experience Subsystem — Surface Contract

> LMDB-backed experience logging with versioned, serializable records
> (``ExperienceProtocol``; ``ExperienceLogger``, ``MouseDroidExperienceRecord``).

## Invariants & Experience Rules

1. **Protocol Surface**: Records implement ``ExperienceProtocol`` (``schema_version``,
   ``serialize``, ``deserialize``). Schema version is a constant — never silently changed.
2. **LMDB Logger**: ``ExperienceLogger`` writes records under ``ExperienceConfig.path`` with
   configured flush cadence and map size.
3. **Dataset Helpers**: ``dataset.py`` builds training datasets from stored experience.

## Key Files

- `protocol.py::ExperienceProtocol` — versioned record interface.
- `record.py::MouseDroidExperienceRecord` — concrete msgpack-serializable record.
- `logger.py::ExperienceLogger` — LMDB-backed writer.
- `dataset.py` — experience dataset construction helpers.
- `tests/unit/experience/` — subsystem unit tests.
