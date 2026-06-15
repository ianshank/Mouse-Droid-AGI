"""Unit tests for the neutral ``common/onnx_session`` helpers.

These pin the shared ONNX session-lifecycle logic that
:class:`mousedroid.vla.policy.DistilledVLAOnnx` and
:class:`mousedroid.world_model.dual_stream_rssm_onnx.DualStreamRSSMOnnx`
both delegate to, so a regression in either wrapper's path is caught in
ONE place.

No real ``onnxruntime`` is required: a stub session / stub module is
injected via ``sys.modules`` (mirroring ``tests/unit/vla/test_distilled_onnx``).
We exercise:

* :func:`resolve_providers` — every branch of the provider intersection /
  CPU fallback / empty-result contract.
* :func:`run_session_with_zeros` — zero-filled feeds inspect the live
  graph's input metadata (dynamic dims collapse to 1) and the supplied
  output-names list is forwarded to ``session.run``.
* :func:`warmup_session` — missing-file raise, lazy ORT import, provider
  resolution, the configured warmup-pass count, the returned
  ``(session, active_providers)`` tuple, and that the supplied
  ``log_prefix`` is emitted on the structured warmup events.
* Module import-graph neutrality — importing ``common.onnx_session`` pulls
  in neither ``onnxruntime`` nor the ``vla`` / ``world_model`` packages.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog

from mousedroid.common.onnx_session import (
    resolve_providers,
    run_session_with_zeros,
    warmup_session,
)


# ----------------------------------------------------------------------
# Stub ORT session / module (mirrors tests/unit/vla/test_distilled_onnx)
# ----------------------------------------------------------------------
class _StubInput:
    def __init__(self, name: str, shape: tuple[int | str, ...]) -> None:
        self.name = name
        self.shape = list(shape)


class _StubSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``."""

    def __init__(
        self,
        model_path: str,
        *,
        providers: list[str],
        inputs: list[_StubInput] | None = None,
        outputs: list[Any] | None = None,
    ) -> None:
        self.model_path = model_path
        self.providers = providers
        self._inputs = inputs if inputs is not None else [_StubInput("h", (1, 4))]
        self._outputs = outputs if outputs is not None else [np.zeros((3,), dtype=np.float32)]
        self.run_calls: list[dict[str, Any]] = []

    def get_inputs(self) -> list[_StubInput]:
        return self._inputs

    def run(self, output_names: list[str], feeds: dict[str, Any]) -> list[Any]:
        self.run_calls.append({"outputs": output_names, "feeds": feeds})
        return self._outputs


class _StubORT:
    """Stub ``onnxruntime`` module."""

    def __init__(
        self,
        available: tuple[str, ...] = ("CPUExecutionProvider",),
        *,
        inputs: list[_StubInput] | None = None,
    ) -> None:
        self._available = available
        self._inputs = inputs
        self.sessions: list[_StubSession] = []

    def get_available_providers(self) -> list[str]:
        return list(self._available)

    def InferenceSession(  # noqa: N802 — matches ORT API
        self, model_path: str, providers: list[str]
    ) -> _StubSession:
        session = _StubSession(model_path, providers=providers, inputs=self._inputs)
        self.sessions.append(session)
        return session


@pytest.fixture
def stub_ort_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject a configurable stub ``onnxruntime`` into ``sys.modules``."""

    def _install(**kwargs: Any) -> _StubORT:
        ort = _StubORT(**kwargs)
        monkeypatch.setitem(sys.modules, "onnxruntime", ort)
        return ort

    return _install


@pytest.fixture
def fake_onnx_file(tmp_path: Path) -> Path:
    """Create an empty file that pretends to be an ONNX model."""
    path = tmp_path / "model.onnx"
    path.write_bytes(b"\x08\x07")  # arbitrary non-empty bytes
    return path


# ----------------------------------------------------------------------
# resolve_providers (pure function; no ORT)
# ----------------------------------------------------------------------
class TestResolveProviders:
    def test_intersects_in_requested_order(self) -> None:
        chosen = resolve_providers(
            requested=("CUDAExecutionProvider", "CPUExecutionProvider"),
            available=("CPUExecutionProvider", "CUDAExecutionProvider"),
        )
        assert chosen == ("CUDAExecutionProvider", "CPUExecutionProvider")

    def test_skips_unavailable(self) -> None:
        chosen = resolve_providers(
            requested=(
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ),
            available=("CPUExecutionProvider",),
        )
        assert chosen == ("CPUExecutionProvider",)

    def test_falls_back_to_cpu_when_intersection_empty(self) -> None:
        chosen = resolve_providers(
            requested=("CUDAExecutionProvider",),
            available=("CPUExecutionProvider",),
        )
        assert chosen == ("CPUExecutionProvider",)

    def test_returns_empty_when_nothing_available(self) -> None:
        chosen = resolve_providers(
            requested=("CUDAExecutionProvider",),
            available=(),
        )
        assert chosen == ()

    def test_returns_empty_when_only_non_cpu_available_and_unrequested(self) -> None:
        # Intersection empty AND CPU not present -> empty (let ORT raise).
        chosen = resolve_providers(
            requested=("TensorrtExecutionProvider",),
            available=("CUDAExecutionProvider",),
        )
        assert chosen == ()


# ----------------------------------------------------------------------
# run_session_with_zeros
# ----------------------------------------------------------------------
class TestRunSessionWithZeros:
    def test_feeds_zeros_for_each_input(self) -> None:
        session = _StubSession(
            "m.onnx",
            providers=["CPUExecutionProvider"],
            inputs=[_StubInput("h", (1, 4)), _StubInput("z", (1, 8))],
        )
        run_session_with_zeros(session, ["action"])
        assert len(session.run_calls) == 1
        feeds = session.run_calls[0]["feeds"]
        assert set(feeds) == {"h", "z"}
        assert feeds["h"].shape == (1, 4)
        assert feeds["z"].shape == (1, 8)
        assert feeds["h"].dtype == np.float32
        # Zero-filled.
        assert not feeds["h"].any()
        assert not feeds["z"].any()

    def test_dynamic_and_nonpositive_dims_collapse_to_one(self) -> None:
        # A symbolic batch dim ("batch") and a 0/negative dim collapse to 1.
        session = _StubSession(
            "m.onnx",
            providers=["CPUExecutionProvider"],
            inputs=[_StubInput("x", ("batch", 0, -1, 5))],
        )
        run_session_with_zeros(session, ["out"])
        feeds = session.run_calls[0]["feeds"]
        assert feeds["x"].shape == (1, 1, 1, 5)

    def test_empty_input_shape_yields_scalar(self) -> None:
        session = _StubSession(
            "m.onnx",
            providers=["CPUExecutionProvider"],
            inputs=[_StubInput("scalar_in", ())],
        )
        run_session_with_zeros(session, ["out"])
        feeds = session.run_calls[0]["feeds"]
        assert feeds["scalar_in"].shape == ()

    def test_output_names_forwarded_as_list(self) -> None:
        session = _StubSession("m.onnx", providers=["CPUExecutionProvider"])
        # Pass a tuple to prove the helper materialises it to a list for ORT.
        run_session_with_zeros(session, ("a", "b", "c"))
        assert session.run_calls[0]["outputs"] == ["a", "b", "c"]
        assert isinstance(session.run_calls[0]["outputs"], list)


# ----------------------------------------------------------------------
# warmup_session
# ----------------------------------------------------------------------
class TestWarmupSession:
    def test_missing_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            warmup_session(
                tmp_path / "no_such.onnx",
                ("CPUExecutionProvider",),
                1,
                ["action"],
                log_prefix="distilled_vla_onnx",
            )

    def test_returns_session_and_resolved_providers(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory(available=("CPUExecutionProvider",))
        session, active = warmup_session(
            fake_onnx_file,
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
            0,
            ["action"],
            log_prefix="distilled_vla_onnx",
        )
        assert active == ("CPUExecutionProvider",)
        assert session is ort.sessions[0]
        # Session constructed with exactly the resolved providers.
        assert ort.sessions[0].providers == ["CPUExecutionProvider"]

    def test_honors_explicit_provider_order(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        stub_ort_factory(
            available=(
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
                "TensorrtExecutionProvider",
            )
        )
        _session, active = warmup_session(
            fake_onnx_file,
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
            0,
            ["action"],
            log_prefix="dual_stream_rssm_onnx",
        )
        assert active == ("CUDAExecutionProvider", "CPUExecutionProvider")

    def test_runs_configured_number_of_warmup_passes(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory()
        warmup_session(
            fake_onnx_file,
            ("CPUExecutionProvider",),
            4,
            ["action"],
            log_prefix="distilled_vla_onnx",
        )
        assert len(ort.sessions[0].run_calls) == 4
        # Every warmup pass forwarded the supplied output-names list.
        for call in ort.sessions[0].run_calls:
            assert call["outputs"] == ["action"]

    def test_zero_warmup_iterations_skips_dummy_runs(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory()
        warmup_session(
            fake_onnx_file,
            ("CPUExecutionProvider",),
            0,
            ["action"],
            log_prefix="distilled_vla_onnx",
        )
        assert ort.sessions[0].run_calls == []

    def test_forwards_multi_output_names_to_warmup_run(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        # World-model wrapper passes a multi-name output list.
        ort = stub_ort_factory()
        output_names = ["new_h", "new_z", "obs_embed", "surprise"]
        warmup_session(
            fake_onnx_file,
            ("CPUExecutionProvider",),
            1,
            output_names,
            log_prefix="dual_stream_rssm_onnx",
        )
        assert ort.sessions[0].run_calls[0]["outputs"] == output_names

    @pytest.mark.parametrize("log_prefix", ["distilled_vla_onnx", "dual_stream_rssm_onnx"])
    def test_log_prefix_names_the_warmup_events(
        self,
        fake_onnx_file: Path,
        stub_ort_factory: Any,
        log_prefix: str,
    ) -> None:
        """The supplied prefix names the start/pass/complete structlog events.

        ``structlog.testing.capture_logs`` records emitted event dicts
        regardless of the configured renderer / logger factory (the repo's
        established pattern for asserting on structured events).
        """
        stub_ort_factory()
        with structlog.testing.capture_logs() as captured:
            warmup_session(
                fake_onnx_file,
                ("CPUExecutionProvider",),
                1,
                ["action"],
                log_prefix=log_prefix,
            )
        events = [entry.get("event", "") for entry in captured]
        assert f"{log_prefix}_warmup_start" in events
        assert f"{log_prefix}_warmup_pass" in events
        assert f"{log_prefix}_warmup_complete" in events


# ----------------------------------------------------------------------
# Import-graph neutrality
# ----------------------------------------------------------------------
def test_module_is_neutral_and_lazy() -> None:
    """``common.onnx_session`` imports neither ORT nor the wrapper packages.

    Run in a fresh subprocess so a stub ``onnxruntime`` injected by an
    earlier test cannot mask a regression. PYTHONPATH is propagated so the
    spawned interpreter resolves ``mousedroid`` (pytest's pythonpath does
    not survive the subprocess boundary).
    """
    import os

    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    extra = os.pathsep.join([str(repo / "src"), str(repo)])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extra + (os.pathsep + existing if existing else "")

    script = textwrap.dedent(
        """
        import sys
        assert 'onnxruntime' not in sys.modules
        import mousedroid.common.onnx_session  # noqa: F401
        for mod in ('onnxruntime', 'mousedroid.vla.policy',
                    'mousedroid.world_model.dual_stream_rssm_onnx'):
            assert mod not in sys.modules, (
                f'common.onnx_session must not import {mod} at module load'
            )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert (
        result.returncode == 0
    ), f"neutrality check failed:\nstdout={result.stdout}\nstderr={result.stderr}"
