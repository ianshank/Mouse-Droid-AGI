# Telemetry Subsystem — Surface Contract (PR #115 Observability Pattern)

> REST endpoints, WebSocket live state streaming, and Prometheus metrics exposition
> on `/metrics` and `/api/v1/health`.

## Invariants & Metric Rules

1. **Config-Gated Metric Families**: Add `track_<area>: bool = True` in `MetricsConfig`.
   Histograms must register bucket fields in the single `@field_validator`.
2. **Pure-Add Render**: Gate metric exposition with `if count > 0:` or `if snapshot:`
   guards so unused metrics do not appear in `/metrics`.
3. **Keyword-Only Shared Registry**: Subsystem builders receive `metrics: MetricsRegistry | None = None`.
   Passing `None` must maintain byte-identical legacy behavior.
4. **Low-Cardinality Validated Labels**: Module-level `frozenset` defines allowed label values.
   Out-of-set values are dropped with a `_log.debug` event.
5. **Success-Path Recording Only**: Record metrics on success; never on cancellation or error paths.
6. **Promtool Golden Samples**: Every new metric family seeds `generate_metrics_sample()` for
   Prometheus validation tests.

## Key Files

- `server/` — REST and WebSocket server (`_lifecycle.py`, `_rest_handlers.py`,
  `_ws_handlers.py`, `_state.py`, `_models.py`, `_protocol.py` — split from a
  monolithic `server.py` per ADR-017).
- `metrics/` — In-memory lock-free `MetricsRegistry` and the Prometheus text
  format renderer (`registry.py`, `primitives.py`, plus one
  `_registry_<area>.py` mixin per metric family — split from a monolithic
  `metrics.py` per the same ADR).
- `tests/unit/telemetry/` & `tests/e2e/test_telemetry_server.py` — Test suites.
