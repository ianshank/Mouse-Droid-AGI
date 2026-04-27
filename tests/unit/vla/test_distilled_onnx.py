"""Unit tests for ``DistilledVLAOnnx`` and Phase 3b factory plumbing.

These tests do **not** require ``onnxruntime`` to be installed. We exercise:

* Construction-time validation (``action_dim``, ``confidence``,
  ``warmup_iterations``).
* ``VLAPolicyProtocol`` conformance.
* Provider-fallback resolution (pure function, no ORT needed).
* Module import-graph isolation: importing ``mousedroid.vla.policy``
  MUST NOT pull in ``onnxruntime``.
* Lazy warmup behavior with a stubbed ``onnxruntime`` injected via
  ``sys.modules`` — verifies that the requested → available provider
  intersection is honored, that warmup is idempotent, that
  shape-mismatched ONNX outputs raise, and that ``predict`` runs under
  ``torch.no_grad``.
* Factory-level ``build_vla_policy`` for ``backend='distilled_onnx'``,
  including the missing-file / no-repo error path and a stubbed
  HuggingFace download path.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mousedroid.config.schema import Settings, VLAConfig
from mousedroid.factory import build_vla_policy
from mousedroid.vla import (
    DEFAULT_ORT_PROVIDERS,
    DistilledVLAOnnx,
    VLAAction,
    VLAObservation,
    VLAPolicyProtocol,
)


# ----------------------------------------------------------------------
# Stub ORT session / module
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
        action_dim: int = 3,
        action_output_name: str = "action",
        h_input_name: str = "h",
        z_input_name: str = "z",
        latent_dim: int = 4,
        bad_output_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.model_path = model_path
        self.providers = providers
        self._action_dim = action_dim
        self._action_output_name = action_output_name
        self._h_input_name = h_input_name
        self._z_input_name = z_input_name
        self._latent_dim = latent_dim
        self._bad_output_shape = bad_output_shape
        self.run_calls: list[dict[str, Any]] = []

    def get_inputs(self) -> list[_StubInput]:
        return [
            _StubInput(self._h_input_name, (1, self._latent_dim)),
            _StubInput(self._z_input_name, (1, self._latent_dim)),
        ]

    def run(self, output_names: list[str], feeds: dict[str, Any]) -> list[Any]:
        self.run_calls.append({"outputs": output_names, "feeds": feeds})
        shape = self._bad_output_shape or (self._action_dim,)
        return [np.zeros(shape, dtype=np.float32)]


class _StubORT:
    """Stub ``onnxruntime`` module."""

    def __init__(
        self,
        available: tuple[str, ...] = ("CPUExecutionProvider",),
        **session_kwargs: Any,
    ) -> None:
        self._available = available
        self._session_kwargs = session_kwargs
        self.sessions: list[_StubSession] = []

    def get_available_providers(self) -> list[str]:
        return list(self._available)

    def InferenceSession(  # noqa: N802 — matches ORT API
        self, model_path: str, providers: list[str]
    ) -> _StubSession:
        session = _StubSession(model_path, providers=providers, **self._session_kwargs)
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
# Construction validation
# ----------------------------------------------------------------------
class TestDistilledVLAConstruction:
    def test_rejects_zero_action_dim(self, fake_onnx_file: Path) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=0)

    def test_rejects_negative_action_dim(self, fake_onnx_file: Path) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=-2)

    def test_rejects_confidence_above_one(self, fake_onnx_file: Path) -> None:
        with pytest.raises(ValueError, match="confidence"):
            DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, confidence=1.1)

    def test_rejects_confidence_below_zero(self, fake_onnx_file: Path) -> None:
        with pytest.raises(ValueError, match="confidence"):
            DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, confidence=-0.01)

    def test_rejects_negative_warmup(self, fake_onnx_file: Path) -> None:
        with pytest.raises(ValueError, match="warmup_iterations"):
            DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, warmup_iterations=-1)

    def test_default_providers_used_when_none(self, fake_onnx_file: Path) -> None:
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3)
        assert policy.active_providers == ()  # not warmed yet
        # internal default exposed via public constant — sanity
        assert DEFAULT_ORT_PROVIDERS[0] == "TensorrtExecutionProvider"
        assert DEFAULT_ORT_PROVIDERS[-1] == "CPUExecutionProvider"

    def test_satisfies_protocol(self, fake_onnx_file: Path) -> None:
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3)
        assert isinstance(policy, VLAPolicyProtocol)

    def test_default_name_is_telemetry_safe(self, fake_onnx_file: Path) -> None:
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3)
        assert policy.name == "distilled_vla_onnx"

    def test_custom_name_propagates(self, fake_onnx_file: Path) -> None:
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, name="smolvla_v0")
        assert policy.name == "smolvla_v0"


# ----------------------------------------------------------------------
# Provider resolution (pure function; no ORT)
# ----------------------------------------------------------------------
class TestProviderResolution:
    def test_intersects_in_requested_order(self) -> None:
        chosen = DistilledVLAOnnx._resolve_providers(
            requested=("CUDAExecutionProvider", "CPUExecutionProvider"),
            available=("CPUExecutionProvider", "CUDAExecutionProvider"),
        )
        assert chosen == ("CUDAExecutionProvider", "CPUExecutionProvider")

    def test_skips_unavailable(self) -> None:
        chosen = DistilledVLAOnnx._resolve_providers(
            requested=DEFAULT_ORT_PROVIDERS,
            available=("CPUExecutionProvider",),
        )
        assert chosen == ("CPUExecutionProvider",)

    def test_falls_back_to_cpu_when_intersection_empty(self) -> None:
        chosen = DistilledVLAOnnx._resolve_providers(
            requested=("CUDAExecutionProvider",),
            available=("CPUExecutionProvider",),
        )
        assert chosen == ("CPUExecutionProvider",)

    def test_returns_empty_when_nothing_available(self) -> None:
        chosen = DistilledVLAOnnx._resolve_providers(
            requested=("CUDAExecutionProvider",),
            available=(),
        )
        assert chosen == ()


# ----------------------------------------------------------------------
# Import-graph isolation
# ----------------------------------------------------------------------
class TestImportGraphIsolation:
    def test_policy_module_does_not_import_onnxruntime(self) -> None:
        """``import mousedroid.vla.policy`` MUST NOT pull in onnxruntime.

        Run in a fresh subprocess so any prior test that injected a stub
        ``onnxruntime`` into the parent ``sys.modules`` cannot mask a
        regression here.
        """
        script = textwrap.dedent(
            """
            import sys
            assert 'onnxruntime' not in sys.modules
            import mousedroid.vla.policy  # noqa: F401
            assert 'onnxruntime' not in sys.modules, (
                'mousedroid.vla.policy must not import onnxruntime at module load'
            )
            assert 'transformers' not in sys.modules, (
                'mousedroid.vla.policy must not import transformers at module load'
            )
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"isolation check failed:\nstdout={result.stdout}\n" f"stderr={result.stderr}"
        )


# ----------------------------------------------------------------------
# Warmup + predict (with stubbed ORT)
# ----------------------------------------------------------------------
class TestWarmup:
    def test_missing_model_raises(self, tmp_path: Path) -> None:
        policy = DistilledVLAOnnx(model_path=tmp_path / "no_such.onnx", action_dim=3)
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            policy.warmup()

    def test_resolves_providers_against_available(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory(available=("CPUExecutionProvider",))
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3)
        policy.warmup()
        assert policy.active_providers == ("CPUExecutionProvider",)
        # The session was constructed with exactly the resolved providers.
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
        policy = DistilledVLAOnnx(
            model_path=fake_onnx_file,
            action_dim=3,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        policy.warmup()
        assert policy.active_providers == (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        )

    def test_runs_configured_number_of_warmup_passes(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory()
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, warmup_iterations=4)
        policy.warmup()
        assert len(ort.sessions[0].run_calls) == 4

    def test_zero_warmup_iterations_skips_dummy_runs(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory()
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, warmup_iterations=0)
        policy.warmup()
        assert ort.sessions[0].run_calls == []

    def test_warmup_is_idempotent(self, fake_onnx_file: Path, stub_ort_factory: Any) -> None:
        ort = stub_ort_factory()
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, warmup_iterations=1)
        policy.warmup()
        policy.warmup()  # second call must be a no-op
        assert len(ort.sessions) == 1


class TestPredict:
    def test_lazy_warmup_on_first_predict(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory()
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, warmup_iterations=0)
        obs = VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))
        action = policy.predict(obs)
        assert isinstance(action, VLAAction)
        assert action.action.shape == (3,)
        assert action.action.dtype == torch.float32
        # Warmup must have created the session lazily.
        assert len(ort.sessions) == 1

    def test_predict_returns_configured_confidence(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        stub_ort_factory()
        policy = DistilledVLAOnnx(
            model_path=fake_onnx_file,
            action_dim=3,
            warmup_iterations=0,
            confidence=0.42,
        )
        action = policy.predict(VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4)))
        assert action.confidence == pytest.approx(0.42)

    def test_runs_under_no_grad(self, fake_onnx_file: Path, stub_ort_factory: Any) -> None:
        stub_ort_factory()
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, warmup_iterations=0)
        obs = VLAObservation(
            h=torch.zeros(1, 4, requires_grad=True),
            z=torch.zeros(1, 4, requires_grad=True),
        )
        action = policy.predict(obs)
        assert action.action.requires_grad is False

    def test_shape_mismatch_raises(self, fake_onnx_file: Path, stub_ort_factory: Any) -> None:
        # Stub returns shape (5,) but we configured action_dim=3.
        stub_ort_factory(bad_output_shape=(5,))
        policy = DistilledVLAOnnx(model_path=fake_onnx_file, action_dim=3, warmup_iterations=0)
        with pytest.raises(ValueError, match="action_dim"):
            policy.predict(VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4)))

    def test_passes_h_and_z_through_named_inputs(
        self, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        ort = stub_ort_factory(
            available=("CPUExecutionProvider",),
            h_input_name="latent_h",
            z_input_name="latent_z",
        )
        policy = DistilledVLAOnnx(
            model_path=fake_onnx_file,
            action_dim=3,
            warmup_iterations=0,
            h_input_name="latent_h",
            z_input_name="latent_z",
        )
        policy.predict(VLAObservation(h=torch.ones(1, 4), z=torch.ones(1, 4) * 2))
        feeds = ort.sessions[0].run_calls[0]["feeds"]
        assert "latent_h" in feeds
        assert "latent_z" in feeds


# ----------------------------------------------------------------------
# Factory wiring
# ----------------------------------------------------------------------
def _make_cfg(**vla: object) -> Settings:
    cfg = Settings(mock_hardware=True)
    cfg.vla = VLAConfig(**vla)  # type: ignore[arg-type]
    return cfg


class TestBuildVLAPolicyDistilledOnnx:
    def test_missing_file_and_no_repo_raises_value_error(self, tmp_path: Path) -> None:
        cfg = _make_cfg(
            backend="distilled_onnx",
            cache_dir=str(tmp_path),
            model_filename="missing.onnx",
            model_repo_id=None,
        )
        with pytest.raises(ValueError, match="distilled_onnx model not found"):
            build_vla_policy(cfg)

    def test_existing_local_file_is_used_without_download(
        self, tmp_path: Path, fake_onnx_file: Path, stub_ort_factory: Any
    ) -> None:
        # Place the file under tmp_path with the configured filename.
        target = tmp_path / "model.onnx"
        target.write_bytes(fake_onnx_file.read_bytes())
        stub_ort_factory()  # in case predict ever called
        cfg = _make_cfg(
            backend="distilled_onnx",
            cache_dir=str(tmp_path),
            model_filename="model.onnx",
            model_repo_id=None,
        )
        policy = build_vla_policy(cfg)
        assert policy is not None
        assert isinstance(policy, VLAPolicyProtocol)

    def test_download_path_is_invoked_when_file_absent(self, tmp_path: Path) -> None:
        cfg = _make_cfg(
            backend="distilled_onnx",
            cache_dir=str(tmp_path),
            model_filename="model.onnx",
            model_repo_id="acme/vla",
        )

        def _fake_download(
            *,
            repo_id: str,
            filenames: list[str],
            cache_dir: Path,
            local_dir: Path | None = None,
        ) -> bool:
            assert repo_id == "acme/vla"
            assert filenames == ["model.onnx"]
            # Factory must request flat layout via ``local_dir`` so the
            # subsequent ``model_path.is_file()`` check sees the file.
            assert local_dir is not None
            assert Path(local_dir) == Path(cache_dir)
            (Path(local_dir) / "model.onnx").write_bytes(b"\x08\x07")
            return True

        with patch(
            "mousedroid.utils.weights_manager.download_weights_from_huggingface",
            side_effect=_fake_download,
        ) as patched:
            policy = build_vla_policy(cfg)
        assert policy is not None
        assert patched.call_count == 1

    def test_download_failure_raises_value_error(self, tmp_path: Path) -> None:
        cfg = _make_cfg(
            backend="distilled_onnx",
            cache_dir=str(tmp_path),
            model_filename="model.onnx",
            model_repo_id="acme/vla",
        )
        with (
            patch(
                "mousedroid.utils.weights_manager.download_weights_from_huggingface",
                return_value=False,
            ),
            pytest.raises(ValueError, match="failed to download"),
        ):
            build_vla_policy(cfg)

    def test_propagates_provider_and_io_names(self, tmp_path: Path, fake_onnx_file: Path) -> None:
        target = tmp_path / "model.onnx"
        target.write_bytes(fake_onnx_file.read_bytes())
        cfg = _make_cfg(
            backend="distilled_onnx",
            cache_dir=str(tmp_path),
            model_filename="model.onnx",
            providers=["CPUExecutionProvider"],
            h_input_name="hh",
            z_input_name="zz",
            action_output_name="aa",
            warmup_iterations=0,
            confidence=0.5,
        )
        policy = build_vla_policy(cfg)
        assert isinstance(policy, DistilledVLAOnnx)
        assert policy._requested_providers == ("CPUExecutionProvider",)
        assert policy._h_input_name == "hh"
        assert policy._z_input_name == "zz"
        assert policy._action_output_name == "aa"
        assert policy._confidence == pytest.approx(0.5)


# Sanity: importing the package doesn't accidentally import onnxruntime
# in the *current* process either (separate from the subprocess test).
def test_module_level_imports_have_not_loaded_onnxruntime() -> None:
    # Force re-import of the package and assert the heavy deps are absent.
    sys.modules.pop("onnxruntime", None)
    sys.modules.pop("transformers", None)
    importlib.import_module("mousedroid.vla.policy")
    assert "onnxruntime" not in sys.modules
    assert "transformers" not in sys.modules
