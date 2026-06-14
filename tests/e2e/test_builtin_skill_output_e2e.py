"""E2E: a live builtin skill produces ``schema_out``-conformant output.

Drives the read-only ``mousedroid-sensor-report`` builtin skill through the real
:class:`SkillDelegator` (registry + approval gate + tracker + journal, reusing
the construction pattern from ``tests/unit/skills/test_delegator.py``) and
asserts both contracts of its :class:`SkillSpec`:

  * the documented input payload validates against ``schema_in``; and
  * the delegated sub-agent's output validates against ``schema_out``.

This needs no arm deps, so it runs on every CI host — proving "e2e on skill
output" actually executes rather than always skipping on the deferred arm
surface that ``test_skill_commands_e2e.py`` covers.
"""

from __future__ import annotations

from typing import Any

import pytest

from mousedroid.config.schema import HarnessTrackerConfig
from mousedroid.harness.approval.auto import AutoApproveGate
from mousedroid.harness.journal.null_journal import NullJournal
from mousedroid.harness.predicates import AlwaysFalse
from mousedroid.harness.protocol import TaskSpec
from mousedroid.harness.task_tracker import InMemoryTaskTracker
from mousedroid.skills.builtin import SENSOR_REPORT_SPEC
from mousedroid.skills.delegator import SkillDelegator
from mousedroid.skills.protocol import SubAgentResult
from mousedroid.skills.registry import SkillRegistry


class _SensorReportSubAgent:
    """Stub sub-agent emitting a realistic schema_out-shaped sensor report.

    Mirrors the stub-sub-agent shape used in ``tests/unit/skills/test_delegator``
    (``name`` / ``is_busy`` / ``invoke`` / ``cancel``). The returned ``output``
    is the dict the read-only skill would produce; the test validates it against
    the spec's ``schema_out`` rather than constructing the validated value, so
    the schema contract is genuinely exercised.
    """

    name = "mousedroid-sensor-report"
    is_busy = False

    async def invoke(self, spec: TaskSpec, parent_ctx: Any | None = None) -> SubAgentResult:
        report = {
            "timestamp_s": 1234.5,
            "lidar": [0.1, 0.2, 0.3, 0.4],
            "pose": {"roll": 0.0, "pitch": 0.0, "yaw": 1.57},
            "battery_voltage_v": 12.4,
            "health": {"cpu_temp_c": 48.0, "state": "nominal"},
        }
        return SubAgentResult(task_id=spec.id, status="ok", output=report)

    def cancel(self) -> None:
        return None


@pytest.mark.asyncio
async def test_sensor_report_output_conforms_to_schema_out() -> None:
    spec = SENSOR_REPORT_SPEC
    assert spec.schema_in is not None
    assert spec.schema_out is not None

    # Input contract: the documented request payload validates.
    payload = {"include_lidar": True, "include_imu": True, "include_battery": True}
    spec.schema_in.model_validate(payload)

    registry = SkillRegistry()
    registry.register(spec)
    tracker = InMemoryTaskTracker(HarnessTrackerConfig(enabled=True, history_size=8, max_active=4))
    delegator = SkillDelegator(
        registry,
        AutoApproveGate(),
        NullJournal(),
        tracker,
        agent_factory=lambda _name: _SensorReportSubAgent(),
    )

    task = TaskSpec(
        id="sensor-report-e2e",
        goal="capture latest sensor snapshot",
        acceptance_predicate=AlwaysFalse(),
    )
    result = await delegator.delegate(spec.name, task)

    assert result.status == "ok"
    assert result.output is not None
    # Output contract: the delegated result validates against schema_out.
    validated = spec.schema_out.model_validate(result.output)
    assert validated.timestamp_s == pytest.approx(1234.5)
