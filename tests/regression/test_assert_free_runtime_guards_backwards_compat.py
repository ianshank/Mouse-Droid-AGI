"""Backwards-compatibility regression tests for the assert-to-raise conversion.

Pins that replacing eight ``assert`` statements with explicit guards changed
*nothing* an existing caller could observe on a path that previously worked.
Three of the four conversion shapes carry a compatibility risk worth naming:

* ``mission_lifecycle`` gained a narrowing accessor, and its four call sites now
  read a bound local instead of ``self._mission``. Identical happy-path
  behaviour is the contract.
* ``experience_exporter._upload_shard`` binds ``bucket`` above its existing
  ``None`` guard, so the early-return contract must be untouched.
* ``pubsub_sink._publish`` gained a guard that did not exist before. It must
  *drop* rather than raise, matching this module's documented "the droid
  continues logging locally to LMDB regardless" philosophy — a raise here would
  be a new failure mode on a telemetry path.

The new ``MissionLifecycleStateError`` is additive: it subclasses
``RuntimeError``, so any pre-existing ``except RuntimeError`` handler keeps
catching, and ``__all__`` only grew.
"""

from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import ExperienceConfig, MissionConfig
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
    MissionLifecycleStateError,
)
from tests.unit.cloud.conftest import _make_gcp_cfg


def _cfg(**overrides: object) -> MissionConfig:
    base: dict[str, object] = {
        "replan_enabled": False,
        "success_threshold": 0.9,
        "stall_threshold": 0.1,
        "stall_window_ticks": 3,
        "max_replans_per_mission": 2,
    }
    base.update(overrides)
    return MissionConfig(**base)  # type: ignore[arg-type]


class _StubVLM:
    """Returns a queued score sequence, mirroring tests/unit/orchestrator's stub."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)

    def score(
        self,
        prev_obs: torch.Tensor,
        curr_obs: torch.Tensor,
        *,
        instruction: str | None = None,
    ) -> torch.Tensor:
        del prev_obs, curr_obs, instruction
        value = self._scores.pop(0) if self._scores else 0.0
        return torch.tensor([[float(value)]], dtype=torch.float32)


class TestMissionLifecycleHappyPathUnchanged:
    """The bound-local refactor must be invisible when a mission is active."""

    @pytest.mark.asyncio
    async def test_start_then_transition_to_succeeded(self) -> None:
        """Exercises ``_transition`` and ``_record_terminal_duration`` together.

        SUCCEEDED is terminal, so this drives both converted sites on one tick.
        """
        lifecycle = MissionLifecycle(_cfg(), vlm_progress=_StubVLM([0.95]))
        lifecycle.start_mission("m-1", "go to the door")
        obs = torch.zeros(1, 4)
        result = await lifecycle.tick(obs, obs)
        assert result.state == MissionLifecycleState.SUCCEEDED
        assert result.transitioned is True

    def test_start_records_running_state(self) -> None:
        lifecycle = MissionLifecycle(_cfg())
        lifecycle.start_mission("m-2", "patrol")
        assert lifecycle.current_state == MissionLifecycleState.RUNNING

    def test_explicit_fail_still_transitions(self) -> None:
        lifecycle = MissionLifecycle(_cfg())
        lifecycle.start_mission("m-3", "patrol")
        lifecycle.fail(reason="operator_abort")
        assert lifecycle.current_state == MissionLifecycleState.FAILED

    @pytest.mark.asyncio
    async def test_stall_path_reaches_transition_to_failed(self) -> None:
        """Drives ``_handle_stall`` -> ``_transition_to_failed``, the 4th site.

        ``replan_enabled`` with no replanner wired is the documented
        "llm_replan_unavailable" branch, which is the shortest route through
        both converted methods.
        """
        lifecycle = MissionLifecycle(
            _cfg(replan_enabled=True, stall_window_ticks=1),
            vlm_progress=_StubVLM([0.0, 0.0]),
        )
        lifecycle.start_mission("m-4", "patrol")
        obs = torch.zeros(1, 4)
        result = await lifecycle.tick(obs, obs)
        assert result.state == MissionLifecycleState.FAILED
        assert result.reason == "llm_replan_unavailable"

    def test_no_mission_started_is_still_a_refusal_not_a_crash_type(self) -> None:
        """Before: ``AssertionError`` (or ``AttributeError`` under -O). Now: typed.

        Either way the call was always a programming error, never a supported
        operation — so narrowing the type is additive for any caller that was
        not already depending on an ``AssertionError``.
        """
        lifecycle = MissionLifecycle(_cfg())
        with pytest.raises(RuntimeError):
            lifecycle._require_mission("transition")


class TestMissionLifecycleStateErrorIsAdditive:
    def test_subclasses_runtime_error(self) -> None:
        assert issubclass(MissionLifecycleStateError, RuntimeError)

    def test_all_only_grew(self) -> None:
        from mousedroid.orchestrator import mission_lifecycle

        pre_existing = {
            "MissionLifecycle",
            "MissionLifecycleState",
            "MissionReplannerProtocol",
            "MissionTickResult",
        }
        assert pre_existing <= set(mission_lifecycle.__all__)
        assert "MissionLifecycleStateError" in mission_lifecycle.__all__


class TestCloudSinkGuardsDropRatherThanRaise:
    """Telemetry-path guards must never introduce a new crash."""

    @pytest.mark.asyncio
    async def test_upload_shard_without_bucket_still_returns_false(self) -> None:
        from mousedroid.cloud.experience_exporter import CloudExperienceExporter

        exporter = CloudExperienceExporter(_make_gcp_cfg(), ExperienceConfig(path="/tmp/test_exp"))
        assert exporter._gcs_bucket is None
        assert await exporter._upload_shard(b"payload") is False

    @pytest.mark.asyncio
    async def test_publish_without_publisher_drops_without_raising(self) -> None:
        """The guard added to ``_publish`` returns; it must not raise."""
        from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

        sink = CloudTelemetrySink(_make_gcp_cfg())
        assert sink._publisher is None
        # No exception, and no attempt to touch the absent publisher.
        await sink._publish("projects/p/topics/t", b"data", {}, "telemetry")

    @pytest.mark.asyncio
    async def test_publish_without_publisher_still_counts_the_outcome(self) -> None:
        """Promoting the assert into a branch must not lose the error metric.

        Before: the stripped-under-``-O`` assert raised inside the closure, was
        caught by ``_publish``'s ``except Exception``, and recorded
        ``result="error"``. The new guard returns early, so it records through
        the same helper rather than skipping the metric block — otherwise the
        hardening would have traded a crash for a silent telemetry blind spot.
        """
        from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

        recorded: list[tuple[str, float]] = []

        class _CapturingMetrics:
            def inc_cloud_telemetry_publish(self, result: str) -> None:
                recorded.append((result, -1.0))

            def observe_cloud_telemetry_publish_latency_ms(self, value: float) -> None:
                recorded.append(("latency", value))

        sink = CloudTelemetrySink(_make_gcp_cfg())
        sink._metrics = _CapturingMetrics()  # type: ignore[assignment]
        await sink._publish("projects/p/topics/t", b"data", {}, "telemetry")

        assert ("error", -1.0) in recorded, f"outcome not counted: {recorded}"
        # Zero latency, because no publish was attempted — not a real timing.
        assert ("latency", 0.0) in recorded, f"latency not observed: {recorded}"

    @pytest.mark.asyncio
    async def test_publish_telemetry_before_start_still_a_noop(self) -> None:
        """The pre-existing caller-side guard is unchanged by the new inner one."""
        from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

        sink = CloudTelemetrySink(_make_gcp_cfg())
        await sink.publish_telemetry({"test": "data"})
        assert sink._publisher is None
