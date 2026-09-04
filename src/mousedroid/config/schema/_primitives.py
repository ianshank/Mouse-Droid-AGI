"""Shared primitives for the config schema package.

Small building blocks reused across many domain modules: the ``PlatformType``
platform enum, the ``RangeF`` inclusive-range model, the public ``Literal``
type aliases that are the single source of truth for label values used
across config schemas and telemetry metric helpers, and the
``_settings_default_factory`` typing workaround used throughout the package
wherever a nested config model is supplied as a ``Field(default_factory=...)``.
"""

from __future__ import annotations

import enum
import sys
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import Self as Self
else:
    from typing_extensions import Self as Self

    class StrEnum(str, enum.Enum):
        """Backport of enum.StrEnum for Python 3.10."""


from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Public Literal type aliases — single source of truth for label values used
# across config schemas and telemetry metric helpers. Keeping these here
# (rather than inlining string literals at each call site) means a backend
# rename only needs to touch this module.
# ---------------------------------------------------------------------------

VLABackendLiteral = Literal["none", "mock", "distilled_onnx"]
"""VLA policy backend identifier. Source of truth: :class:`VLAConfig.backend`.

Includes ``"none"`` (the disabled default). For label values on metrics
that only fire from a *running* backend (e.g.
``mousedroid_vla_timeouts_total{mode}``) use the narrower
:data:`VLAActiveBackendLiteral` alias below."""

VLAActiveBackendLiteral = Literal["mock", "distilled_onnx"]
"""Subset of :data:`VLABackendLiteral` excluding ``"none"``.

Use this for any metric or callback where a value of ``"none"`` is
operationally impossible (the disabled backend cannot run inference, so
it cannot fire a timeout or emit a latency sample). Narrowing at the
call site prevents accidental cardinality growth from spurious
``{mode="none"}`` series."""

ESP32CommandSetLiteral = Literal["legacy", "waveshare_stock"]
"""ESP32 firmware command-set identifier. Source of truth:
:class:`ESP32Config.command_set`.

``"legacy"`` (the default) speaks the pre-F-025 private JSON protocol —
byte-identical to every deployment that predates the selector.
``"waveshare_stock"`` speaks the stock Waveshare ``General_Driver``
command set (``ugv_base_general``): ``CMD_ROS_CTRL`` velocity, a
``CMD_HEART_BEAT_SET`` chassis failsafe armed at connect, and battery /
wheel telemetry read from the ``FEEDBACK_BASE_INFO`` frame. The codec
dispatch lives in :mod:`mousedroid.comms.command_set`."""

ReplayOutcomeLiteral = Literal["ok", "schema_mismatch"]
"""LMDB replay-record deserialization outcome. Drives the
``mousedroid_replay_records_total{outcome}`` Prometheus counter labels.
``"ok"`` = record passed schema-version check; ``"schema_mismatch"`` =
record was skipped because its ``SCHEMA_VERSION`` differed from the
runtime constant in :mod:`mousedroid.experience.record`."""

TickPhaseLiteral = Literal[
    "sense",
    "safety",
    "world_model",
    "plan",
    "act",
    "learn",
    "telemetry",
    "post",
]
"""One phase of the orchestrator's sense-plan-act tick. Drives the
``mousedroid_tick_phase_ms{phase}`` histogram labels.

The phases tile the tick contiguously, so ``sum(phases)`` approximates the
whole-tick duration and an operator can find a regression by subtraction
rather than by guessing. They are *diagnostics only* — no phase timing has
emergency-stop authority, because phases do not sum exactly: every ``await``
hands control to the event loop, so scheduler delay lands between brackets
and every phase can be within budget while the tick as a whole overruns.

Typing the writer's parameter to this alias makes mypy reject a mistyped
phase at the call site; :data:`mousedroid.telemetry.metrics.primitives`
mirrors the same set as a runtime drop-guard for anything that reaches the
registry dynamically. A regression test asserts the two never diverge."""


def _settings_default_factory(factory: Any) -> Any:
    """Return nested settings factories unchanged.

    Pydantic accepts model classes directly as ``default_factory`` callables,
    while the current mypy stubs are stricter about the callable signature.
    This helper preserves runtime behaviour and keeps the workaround local.
    """
    return factory


class PlatformType(StrEnum):
    """Supported hardware platform types."""

    MOUSE_DROID = "mouse_droid"
    ROBOT_ARM = "robot_arm"


class StrictBaseModel(BaseModel):
    """Base for every config schema model — rejects unknown fields.

    Pydantic v2 defaults to ``extra="ignore"``, so a misspelled or stale
    field name in a constructor call (``LidarConfig(device_path=...)`` when
    the real field is ``serial_port``) is silently dropped instead of
    raising — at both runtime and in tests. This is a single audit-confirmed
    root cause: a test built a config with a wrong kwarg, Pydantic silently
    ignored it, and the resulting default masked a real bug in
    ``factory/autonomous.py`` for months. Every domain config model in this
    package should subclass this instead of ``BaseModel`` directly.

    A subclass needing looser behaviour may still override
    ``model_config`` for a specific key (Pydantic v2 merges parent and
    child ``model_config`` dicts rather than replacing them wholesale), but
    that should be a deliberate, reviewed exception — not the default.
    """

    model_config = ConfigDict(extra="forbid")


class RangeF(StrictBaseModel):
    """Inclusive ``[low, high]`` range for a randomly sampled float parameter."""

    low: float = Field(description="Inclusive lower bound")
    high: float = Field(description="Inclusive upper bound")

    @model_validator(mode="after")
    def _check_ordered(self) -> Self:
        if self.low > self.high:
            msg = f"RangeF.low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)
        return self
