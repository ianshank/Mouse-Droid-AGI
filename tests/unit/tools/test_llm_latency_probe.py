"""F-006 verification: ``tools/llm_latency_probe.py`` unit tests.

The probe is operator-facing (intended for ``docker exec`` on the Jetson)
but the dispatch + result-reporting paths are pure Python and unit-testable
against a stub gateway. Mocks the factory + Llama model so no real LLM
download / inference happens in CI.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
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
    # _tegrastats_snapshot was promoted to tools/_jetson_helpers in the
    # f006-remote-llm sprint; patch the new location.
    from tools import _jetson_helpers

    monkeypatch.setattr(_jetson_helpers.shutil, "which", lambda _name: None)
    result = probe._tegrastats_snapshot()
    assert result == {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None}


def test_tegrastats_snapshot_parses_ram_line(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realistic tegrastats line → parsed ram_used_mb + ram_total_mb."""
    fake_line = "RAM 2914/7619MB (lfb 12x4MB) SWAP 0/3810MB GR3D_FREQ 0%@[306,...]"
    from tools import _jetson_helpers

    monkeypatch.setattr(_jetson_helpers.shutil, "which", lambda _name: "/usr/bin/tegrastats")

    fake_completed = subprocess.CompletedProcess(
        args=["tegrastats", "--interval", "100", "--count", "1"],
        returncode=0,
        stdout=fake_line + "\n",
        stderr="",
    )
    monkeypatch.setattr(_jetson_helpers.subprocess, "run", lambda *a, **kw: fake_completed)

    result = probe._tegrastats_snapshot()
    assert result["ram_used_mb"] == 2914
    assert result["ram_total_mb"] == 7619
    assert "RAM 2914/7619MB" in str(result["raw_line"])


def test_tegrastats_snapshot_handles_timeout_gracefully(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tegrastats hang → TimeoutExpired caught, all keys None, no raise."""
    from tools import _jetson_helpers

    monkeypatch.setattr(_jetson_helpers.shutil, "which", lambda _name: "/usr/bin/tegrastats")

    def _raise_timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="tegrastats", timeout=2.0)

    monkeypatch.setattr(_jetson_helpers.subprocess, "run", _raise_timeout)
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


@pytest.mark.asyncio
async def test_main_samples_multiple_iterations_and_gates_on_p95(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_goal_vector: Any,
    stub_cfg: Any,
) -> None:
    """``--iterations N`` runs translate_mission N times and gates on p95."""
    stub_gateway = MagicMock()
    stub_gateway.is_ready = True
    stub_gateway.start = AsyncMock()
    stub_gateway.stop = AsyncMock()
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

    args = probe.argparse.Namespace(config=None, mission="turn left slowly", iterations=3)
    rc = await probe._main(args)
    assert rc == 0
    assert stub_gateway.translate_mission.await_count == 3


def test_main_cli_rejects_zero_iterations(probe: Any) -> None:
    """``--iterations 0`` is a usage error (must sample at least once)."""
    with pytest.raises(SystemExit) as excinfo:
        probe.main(["--iterations", "0"])
    assert excinfo.value.code == 2


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


# --------------------------------------------------------------------------- #
# F-059 WS-A - ``--report-json``: which tier served, and which model it was
#
# Driven through the REAL ``FallbackLLMGateway`` composite and the REAL
# ``AnthropicLLMGateway`` (faked SDK, no network), so "served" is derived from
# the same degrade semantics production uses rather than from a hand-set flag.
# --------------------------------------------------------------------------- #
_REPORT_KEYS = {
    "model_requested",
    "model_resolved",
    "resolved_via",
    "backend",
    "served",
    "cold_start_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "iterations",
    "passed",
}
_GOOD_REPLY = '{"vx": 0.0, "vy": 0.0, "omega": 0.5}'


class _FakeModels:
    """``client.models`` - ``retrieve`` is how the probe resolves the served model id."""

    def __init__(self, *, resolved: str | None = None, exc: Exception | None = None) -> None:
        self.requested: list[str] = []
        self._resolved = resolved
        self._exc = exc

    async def retrieve(self, model_id: str) -> Any:
        self.requested.append(model_id)
        if self._exc is not None:
            raise self._exc
        return types.SimpleNamespace(id=self._resolved)


class _FakeMessages:
    """``client.messages`` - fails on the call indices in ``fail_on`` (0-based)."""

    def __init__(self, *, fail_on: frozenset[int] = frozenset(), always_fail: bool = False) -> None:
        self.calls = 0
        self._fail_on = fail_on
        self._always_fail = always_fail

    async def create(self, **_kwargs: Any) -> Any:
        index = self.calls
        self.calls += 1
        if self._always_fail or index in self._fail_on:
            raise RuntimeError("503 overloaded")
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=_GOOD_REPLY)])


class _LocalSecondary:
    """A healthy local fallback that answers a recognisably different goal."""

    is_ready = True
    is_degraded = False

    def __init__(self) -> None:
        self.calls = 0

    async def start(self) -> None:
        return None

    async def translate_mission(self, _command: str) -> Any:
        from mousedroid.llm_gateway.protocol import GoalVector

        self.calls += 1
        return GoalVector(vx_target=0.25)

    async def stop(self) -> None:
        return None


def _composite(
    messages: _FakeMessages,
    models: _FakeModels | None = None,
    *,
    clock: Any = None,
    cooldown_s: float = 30.0,
    primary_enabled: bool = True,
) -> tuple[Any, _LocalSecondary]:
    """A real cloud-primary / local-secondary composite over a faked SDK."""
    from mousedroid.config.schema import LLMConfig
    from mousedroid.llm_gateway.anthropic_gateway import AnthropicLLMGateway
    from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.messages = messages
            self.models = models if models is not None else _FakeModels(resolved="claude-haiku-5-5")

    cfg = LLMConfig(backend="anthropic", model_name="claude-haiku-5-5", enabled=primary_enabled)
    primary = AnthropicLLMGateway(cfg, sdk=types.SimpleNamespace(AsyncAnthropic=_Client))
    secondary = _LocalSecondary()
    gateway = FallbackLLMGateway(primary, secondary, retry_cooldown_s=cooldown_s, clock=clock)
    return gateway, secondary


def _wire_probe(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
    gateway: Any,
) -> None:
    stub_cfg.llm.model_name = "claude-haiku-5-5"
    stub_cfg.llm.backend = "anthropic"
    stub_cfg.llm.request_timeout_s = 5.0
    stub_cfg.llm.latency_target_ms = 5000.0
    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)


def _report_args(probe: Any, report: Path, iterations: int = 1) -> Any:
    return probe.argparse.Namespace(
        config=None, mission="turn left slowly", iterations=iterations, report_json=report
    )


@pytest.mark.asyncio
async def test_probe_report_marks_primary_when_the_cloud_tier_answers(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    models = _FakeModels(resolved="claude-haiku-5-5")
    gateway, secondary = _composite(_FakeMessages(), models)
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "haiku-5-5.json"

    rc = await probe._main(_report_args(probe, report_path, iterations=3))

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["served"] == "primary"
    assert report["passed"] is True
    assert report["iterations"] == 3
    assert report["model_requested"] == "claude-haiku-5-5"
    assert report["model_resolved"] == "claude-haiku-5-5"
    assert report["resolved_via"] == "models.retrieve"
    assert report["backend"] == "anthropic"
    assert models.requested == ["claude-haiku-5-5"]
    assert secondary.calls == 0


@pytest.mark.asyncio
async def test_probe_report_marks_fallback_when_primary_degraded(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    """R10: a 400 / outage must not read as a pass because the local tier answered quickly."""
    gateway, secondary = _composite(_FakeMessages(always_fail=True))
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "haiku-5-5.json"

    rc = await probe._main(_report_args(probe, report_path, iterations=2))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["served"] == "fallback"
    assert report["passed"] is False, "a fallback-served run must never be reported as a pass"
    assert secondary.calls == 2  # the local tier really did answer
    assert rc == probe._EXIT_PRIMARY_NOT_SERVED


@pytest.mark.asyncio
async def test_probe_report_remembers_an_early_fallback_even_if_the_primary_recovers(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    """Iteration 0 fails over, the cooldown elapses, later iterations are cloud again.

    Judging by the final state alone would call this run "primary"; it was not.
    """
    ticks = iter(range(0, 10_000, 100))  # the composite's clock jumps 100 s per read
    gateway, secondary = _composite(
        _FakeMessages(fail_on=frozenset({0})),
        clock=lambda: float(next(ticks)),
        cooldown_s=30.0,
    )
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "report.json"

    rc = await probe._main(_report_args(probe, report_path, iterations=3))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert secondary.calls == 1  # only iteration 0 was served locally
    assert gateway._primary.is_degraded is False  # ...and the primary is healthy again
    assert report["served"] == "fallback"
    assert report["passed"] is False
    assert rc == probe._EXIT_PRIMARY_NOT_SERVED


@pytest.mark.asyncio
async def test_probe_report_marks_fallback_when_the_primary_never_became_ready(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    """A disabled / never-started primary is not DEGRADED, but it did not serve either."""
    gateway, secondary = _composite(_FakeMessages(), primary_enabled=False)
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "report.json"

    rc = await probe._main(_report_args(probe, report_path))

    assert gateway._primary.is_degraded is False  # the trap: healthy-looking, but not serving
    assert gateway._primary.is_ready is False
    assert secondary.calls == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["served"] == "fallback"
    assert rc == probe._EXIT_PRIMARY_NOT_SERVED


@pytest.mark.asyncio
async def test_probe_report_has_exactly_the_documented_shape(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    gateway, _ = _composite(_FakeMessages())
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "report.json"

    await probe._main(_report_args(probe, report_path, iterations=4))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == _REPORT_KEYS
    assert report["served"] in {"primary", "fallback"}
    assert report["resolved_via"] in {"models.retrieve", "config"}
    for key in ("cold_start_ms", "p50_ms", "p95_ms", "p99_ms"):
        assert isinstance(report[key], float), key
        assert report[key] >= 0.0, key
    assert report["p50_ms"] <= report["p95_ms"] <= report["p99_ms"]
    assert isinstance(report["iterations"], int)
    assert isinstance(report["passed"], bool)
    assert report_path.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.asyncio
async def test_probe_report_falls_back_to_the_config_id_when_retrieve_fails(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    models = _FakeModels(exc=RuntimeError("404 not_found_error"))
    gateway, _ = _composite(_FakeMessages(), models)
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "report.json"

    rc = await probe._main(_report_args(probe, report_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_resolved"] == "claude-haiku-5-5"
    assert report["resolved_via"] == "config"
    assert rc == 0  # a failed id lookup does not fail an otherwise healthy run


@pytest.mark.asyncio
async def test_probe_report_records_the_resolved_id_when_it_differs_from_the_request(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    models = _FakeModels(resolved="claude-haiku-5-5-20261001")
    gateway, _ = _composite(_FakeMessages(), models)
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "report.json"

    await probe._main(_report_args(probe, report_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_requested"] == "claude-haiku-5-5"
    assert report["model_resolved"] == "claude-haiku-5-5-20261001"
    assert report["resolved_via"] == "models.retrieve"


@pytest.mark.asyncio
async def test_probe_report_uses_the_config_id_when_the_gateway_has_no_client(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
    stub_goal_vector: Any,
    tmp_path: Path,
) -> None:
    """A local (llama-cpp) gateway has no Anthropic client to ask."""
    gateway = MagicMock(spec=["is_ready", "is_degraded", "start", "stop", "translate_mission"])
    gateway.is_ready = True
    gateway.is_degraded = False
    gateway.start = AsyncMock()
    gateway.stop = AsyncMock()
    gateway.translate_mission = AsyncMock(return_value=stub_goal_vector)
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    stub_cfg.llm.backend = "llama_cpp"
    report_path = tmp_path / "report.json"

    rc = await probe._main(_report_args(probe, report_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["resolved_via"] == "config"
    assert report["model_resolved"] == report["model_requested"]
    assert report["backend"] == "llama_cpp"
    assert report["served"] == "primary"  # nothing to fall back to: the gateway IS the primary
    assert rc == 0


@pytest.mark.asyncio
async def test_probe_report_creates_missing_parent_directories(
    probe: Any, monkeypatch: pytest.MonkeyPatch, stub_cfg: Any, tmp_path: Path
) -> None:
    gateway, _ = _composite(_FakeMessages())
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    report_path = tmp_path / "reports" / "llm_latency" / "2026-10-19" / "haiku-5-5.json"

    await probe._main(_report_args(probe, report_path))

    assert report_path.is_file()


@pytest.mark.asyncio
async def test_probe_without_the_flag_writes_no_report_and_keeps_legacy_exit_codes(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
    tmp_path: Path,
) -> None:
    """No ``--report-json``: nothing is written and a fallback-served run is NOT re-judged."""
    gateway, secondary = _composite(_FakeMessages(always_fail=True))
    _wire_probe(probe, monkeypatch, stub_cfg, gateway)
    monkeypatch.chdir(tmp_path)

    args = probe.argparse.Namespace(config=None, mission="turn left slowly")
    rc = await probe._main(args)

    assert rc == 0  # legacy behaviour: latency under target is the whole verdict
    assert secondary.calls == 1
    assert list(tmp_path.iterdir()) == []


def test_main_cli_parses_report_json_as_a_path(probe: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def _fake_main(args: Any) -> int:
        seen["report_json"] = args.report_json
        return 0

    monkeypatch.setattr(probe, "_main", _fake_main)
    rc = probe.main(["--iterations", "2", "--report-json", "out/report.json"])
    assert rc == 0
    assert seen["report_json"] == Path("out/report.json")


def test_main_cli_report_json_defaults_to_none(probe: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def _fake_main(args: Any) -> int:
        seen["report_json"] = args.report_json
        return 0

    monkeypatch.setattr(probe, "_main", _fake_main)
    probe.main([])
    assert seen["report_json"] is None
