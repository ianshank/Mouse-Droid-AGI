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

import asyncio

import pytest
import torch

from mousedroid.config.schema import ExperienceConfig, MissionConfig
from mousedroid.interfaces.protocols import GoalVector
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

    @pytest.mark.asyncio
    async def test_in_flight_upload_survives_a_concurrent_close(self) -> None:
        """The stated rationale for binding above the guard, finally pinned.

        ``_upload_shard`` binds ``bucket`` before its ``None`` check so an upload
        already in flight completes against the bucket it validated, rather than
        raising when a concurrent ``close()`` nulls the attribute. Nothing
        asserted that — only the already-``None`` case was covered — so the
        refactor's whole justification was untested.
        """
        from mousedroid.cloud.experience_exporter import CloudExperienceExporter

        exporter = CloudExperienceExporter(_make_gcp_cfg(), ExperienceConfig(path="/tmp/test_exp"))
        uploaded: list[bytes] = []

        class _Blob:
            def upload_from_string(self, data: bytes) -> None:
                uploaded.append(data)

        class _Bucket:
            def blob(self, path: str) -> _Blob:
                del path
                # Simulate close() landing mid-upload: the attribute is nulled
                # after the guard passed but before the blob is used.
                exporter._gcs_bucket = None
                return _Blob()

        exporter._gcs_bucket = _Bucket()  # type: ignore[assignment]
        assert await exporter._upload_shard(b"payload") is True
        assert uploaded == [b"payload"], "in-flight upload did not complete"
        assert exporter._gcs_bucket is None, "the concurrent close did happen"


class TestOnnxWarmupGuardsRaiseByName:
    """The two ONNX conversions, which no other test reaches.

    Both files are at or near 0% in the CI coverage selection because
    ``onnxruntime``/``mujoco`` are absent from the measured environment, so the
    converted guards shipped unexecuted. These tests need no ONNX runtime: they
    stub ``warmup`` to a no-op, which is exactly the "warmup silently failed to
    produce a session" state the guards exist for.
    """

    def test_distilled_vla_onnx_raises_when_warmup_yields_no_session(self) -> None:
        from mousedroid.vla.policy import DistilledVLAOnnx, VLAObservation

        policy = DistilledVLAOnnx(model_path="/nonexistent/model.onnx", action_dim=3)
        # warmup() is where a real session would be created; make it a no-op so
        # `_session` stays None past the lazy-warmup branch.
        policy.warmup = lambda: None  # type: ignore[method-assign]
        observation = VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))

        with pytest.raises(RuntimeError, match="warmup"):
            policy.predict(observation)

    def test_distilled_vla_onnx_error_is_not_an_attribute_error(self) -> None:
        """Before the conversion, ``-O`` turned this into ``AttributeError``."""
        from mousedroid.vla.policy import DistilledVLAOnnx, VLAObservation

        policy = DistilledVLAOnnx(model_path="/nonexistent/model.onnx", action_dim=3)
        policy.warmup = lambda: None  # type: ignore[method-assign]
        observation = VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))

        with pytest.raises(RuntimeError) as excinfo:
            policy.predict(observation)
        assert not isinstance(excinfo.value, AttributeError)
        assert "session" in str(excinfo.value)

    def test_dual_stream_onnx_raises_when_warmup_yields_no_session(self) -> None:
        from mousedroid.config.schema import ModelConfig
        from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx

        engine = DualStreamRSSMOnnx(
            model_path="/nonexistent/model.onnx",
            cfg=ModelConfig(cfc_hidden_dim=8),
        )
        engine.warmup = lambda: None  # type: ignore[method-assign]
        assert engine._session is None
        # Reaching observe_step's guard is what matters; the packer runs after it.
        with pytest.raises(RuntimeError, match="warmup"):
            engine.observe_step(  # type: ignore[call-arg]
                observation=None,
                prev_action=torch.zeros(1, 2),
                h=torch.zeros(1, 8),
                z=torch.zeros(1, 4),
            )


# ---------------------------------------------------------------------------
# Mid-await mission swap (found by review of the _require_mission hold)
# ---------------------------------------------------------------------------
class _SwappingReplanner:
    """A replanner that starts a *new* mission while the old one awaits it.

    This is not a contrived race. ``submit_replan_request`` is an LLM call that
    can take seconds; ``start_mission`` is sync, replaces ``_mission``
    unconditionally, and is reachable from the aiohttp REST handler
    (``_handle_mission_post`` -> ``process_mission`` ->
    ``_start_mission_lifecycle_if_wired``) on the same event loop as the 30 Hz
    tick. Doing the swap inside the await is the deterministic way to land the
    interleaving a real REST request would land nondeterministically.
    """

    def __init__(self, lifecycle: MissionLifecycle, new_id: str) -> None:
        self._lifecycle = lifecycle
        self._new_id = new_id
        self.calls = 0

    async def submit_replan_request(
        self, *, mission_id: str, goal_text: str, last_progress: float
    ) -> GoalVector:
        del mission_id, goal_text, last_progress
        self.calls += 1
        await asyncio.sleep(0)
        self._lifecycle.start_mission(self._new_id, "a different goal")
        return GoalVector()


@pytest.mark.asyncio
async def test_replan_abandons_when_the_mission_is_swapped_mid_await() -> None:
    """A mission that arrives mid-replan must not inherit the replan.

    Two properties, and the second is the one that bites: the new mission must
    not be transitioned by a replan nobody requested for it (which would also
    mislabel ``mission_state_transitions_total``), and the abandoned mission's
    ``replan_count`` must not be incremented on a result that was discarded.
    """
    lifecycle = MissionLifecycle(_cfg())
    lifecycle.start_mission("m-old", "original goal")
    replanner = _SwappingReplanner(lifecycle, "m-new")
    lifecycle._replanner = replanner

    result = await lifecycle._handle_stall()

    assert replanner.calls == 1, "the replan must actually have been awaited"
    assert result.transitioned is False
    assert result.reason == "replan_abandoned_mission_changed"
    # The new mission is left exactly where start_mission put it: RUNNING, with
    # no replan attributed to it.
    current = lifecycle._mission
    assert current is not None
    assert current.mission_id == "m-new"
    assert current.replan_count == 0
    assert current.last_goal_vector is None
