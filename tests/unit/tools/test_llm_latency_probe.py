"""F-006 verification: ``tools/llm_latency_probe.py`` unit tests.

The probe is operator-facing (intended for ``docker exec`` on the Jetson)
but the dispatch + result-reporting paths are pure Python and unit-testable
against a stub gateway. Mocks the factory + Llama model so no real LLM
download / inference happens in CI.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import the probe module by path so the test doesn't depend on a ``tools``
# package install. The probe file lives at <repo_root>/tools/llm_latency_probe.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROBE_PATH = _REPO_ROOT / "tools" / "llm_latency_probe.py"


def _import_probe() -> Any:
    """Load the probe module via importlib so tests don't need a tools/ package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("llm_latency_probe", _PROBE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_latency_probe"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe() -> Any:
    return _import_probe()


@pytest.fixture
def stub_goal_vector() -> Any:
    from mousedroid.llm_gateway.protocol import GoalVector

    return GoalVector(vx_target=0.0, vy_target=0.0, omega_target=-0.5)


@pytest.fixture
def stub_cfg() -> Any:
    """Build a stub Settings-like object with cfg.llm.enabled=True so the probe runs."""
    cfg = MagicMock()
    cfg.llm.enabled = True
    cfg.llm.n_gpu_layers = -1
    cfg.llm.n_threads = 6
    cfg.llm.n_batch = 32
    cfg.llm.context_length = 2048
    cfg.llm.latency_target_ms = 500.0
    cfg.llm.model_path = "/fake/phi-3-mini.gguf"
    return cfg


def test_tegrastats_snapshot_returns_none_keys_when_binary_absent(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tegrastats binary (non-Jetson host) → all keys ``None``, never raises."""
    monkeypatch.setattr(probe.shutil, "which", lambda _name: None)
    result = probe._tegrastats_snapshot()
    assert result == {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None}


def test_tegrastats_snapshot_parses_ram_line(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realistic tegrastats line → parsed ram_used_mb + ram_total_mb."""
    fake_line = "RAM 2914/7619MB (lfb 12x4MB) SWAP 0/3810MB GR3D_FREQ 0%@[306,...]"
    monkeypatch.setattr(probe.shutil, "which", lambda _name: "/usr/bin/tegrastats")

    fake_completed = subprocess.CompletedProcess(
        args=["tegrastats", "--interval", "100", "--count", "1"],
        returncode=0,
        stdout=fake_line + "\n",
        stderr="",
    )
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: fake_completed)

    result = probe._tegrastats_snapshot()
    assert result["ram_used_mb"] == 2914
    assert result["ram_total_mb"] == 7619
    assert "RAM 2914/7619MB" in str(result["raw_line"])


def test_tegrastats_snapshot_handles_timeout_gracefully(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tegrastats hang → TimeoutExpired caught, all keys None, no raise."""
    monkeypatch.setattr(probe.shutil, "which", lambda _name: "/usr/bin/tegrastats")

    def _raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="tegrastats", timeout=2.0)

    monkeypatch.setattr(probe.subprocess, "run", _raise_timeout)
    result = probe._tegrastats_snapshot()
    assert result == {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None}


def test_llama_model_metadata_returns_not_loaded_when_model_is_none(probe: Any) -> None:
    """Degraded gateway (model=None) → ``loaded=False`` + reason; never raises."""
    stub_gateway = MagicMock(spec=[])
    stub_gateway._model = None
    result = probe._llama_model_metadata(stub_gateway)
    assert result["loaded"] is False
    assert result["reason"] == "gateway_model_is_none"


def test_llama_model_metadata_extracts_attr_n_gpu_layers(probe: Any) -> None:
    """When the Llama instance exposes ``n_gpu_layers`` → metadata picks it up."""
    fake_llama = MagicMock()
    fake_llama.n_gpu_layers = 32
    fake_llama.__class__.__name__ = "Llama"
    stub_gateway = MagicMock(spec=[])
    stub_gateway._model = fake_llama
    result = probe._llama_model_metadata(stub_gateway)
    assert result["loaded"] is True
    assert result["attr_n_gpu_layers"] == 32


@pytest.mark.asyncio
async def test_main_returns_0_when_elapsed_under_target(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_goal_vector: Any,
    stub_cfg: Any,
) -> None:
    """translate_mission elapsed <= cfg.llm.latency_target_ms → exit 0."""
    stub_gateway = MagicMock()
    stub_gateway.is_ready = True
    stub_gateway.start = AsyncMock()
    stub_gateway.stop = AsyncMock()
    # AsyncMock returns immediately → elapsed ~0 ms, far below default 500 ms.
    stub_gateway.translate_mission = AsyncMock(return_value=stub_goal_vector)
    stub_gateway._model = MagicMock(n_gpu_layers=-1)

    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: stub_gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)

    args = probe.argparse.Namespace(config=None, mission="turn left slowly")
    rc = await probe._main(args)
    assert rc == 0
    stub_gateway.start.assert_awaited_once()
    stub_gateway.translate_mission.assert_awaited_once_with("turn left slowly")
    stub_gateway.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_returns_1_when_elapsed_exceeds_target(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_goal_vector: Any,
    stub_cfg: Any,
) -> None:
    """Slow translate_mission → exit 1 (F-006 not actually fixed on this host)."""
    stub_gateway = MagicMock()
    stub_gateway.is_ready = True
    stub_gateway.start = AsyncMock()
    stub_gateway.stop = AsyncMock()
    stub_gateway._model = MagicMock(n_gpu_layers=0)

    async def _slow_translate(_mission: str) -> Any:
        # Sleep just past the default 500 ms target. The test takes <1 s.
        await asyncio.sleep(0.6)
        return stub_goal_vector

    stub_gateway.translate_mission = _slow_translate

    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: stub_gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)

    args = probe.argparse.Namespace(config=None, mission="turn left slowly")
    rc = await probe._main(args)
    assert rc == 1


@pytest.mark.asyncio
async def test_main_returns_2_when_gateway_start_raises_load_failure(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
) -> None:
    """llama-cpp-python ``ValueError: Failed to load model from file`` is caught.

    Live-Jetson finding (F-006 verification): when ``n_gpu_layers=-1`` is set
    but llama-cpp-python is built without CUDA (or the model doesn't fit at
    full offload), ``Llama.__init__`` raises ``ValueError`` which the
    ``LLMGateway.start()`` doesn't catch (it only handles ImportError +
    OSError). The probe must catch this cleanly + emit
    ``llm_gateway_load_failed`` + exit 2 so dashboards see the GPU-offload-
    failed-at-load signal as distinct from the "loaded but slow" path.
    """
    stub_gateway = MagicMock()
    stub_gateway.is_ready = False
    stub_gateway.start = AsyncMock(
        side_effect=ValueError(
            "Failed to load model from file: /opt/mousedroid/models/Phi-3-mini-q4.gguf"
        ),
    )
    stub_gateway.stop = AsyncMock()

    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: stub_gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)

    args = probe.argparse.Namespace(config=None, mission="turn left slowly")
    rc = await probe._main(args)
    assert rc == 2
    # stop() must NOT be called when start raised — operator runbook signal.
    stub_gateway.stop.assert_not_called()


@pytest.mark.asyncio
async def test_main_returns_2_when_gateway_degraded(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
) -> None:
    """Gateway fails to start (degraded) → exit 2 + clear log."""
    stub_gateway = MagicMock()
    stub_gateway.is_ready = False
    stub_gateway.start = AsyncMock()
    stub_gateway.stop = AsyncMock()

    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: stub_gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)

    args = probe.argparse.Namespace(config=None, mission="turn left slowly")
    rc = await probe._main(args)
    assert rc == 2
    # stop() should NOT be called when start failed — operator runbook signal.
    stub_gateway.stop.assert_not_called()


def test_main_cli_returns_3_when_llm_disabled_in_cfg(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cfg.llm.enabled=False`` → exit 3 (config error, not latency failure)."""

    class _StubLLMCfg:
        enabled = False

    class _StubCfg:
        llm = _StubLLMCfg()

    monkeypatch.setattr(probe, "load_settings", lambda *_paths: _StubCfg())
    rc = probe.main([])
    assert rc == 3


def test_main_cli_help_flag_exits_zero(probe: Any) -> None:
    """`--help` exits 0 + prints argparse usage (no crash)."""
    with pytest.raises(SystemExit) as excinfo:
        probe.main(["--help"])
    assert excinfo.value.code == 0
