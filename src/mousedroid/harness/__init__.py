"""Agent harness — deterministic exoskeleton around the orchestrator.

The harness layer provides task tracking, tick middleware, persistent
journalling, approval gates, and skill / sub-agent delegation. Every
component is opt-in via :class:`mousedroid.config.schema.HarnessConfig`
and ships a no-op default so the orchestrator's existing 30 Hz behaviour
is byte-identical when ``Settings.harness is None``.
"""

from __future__ import annotations
