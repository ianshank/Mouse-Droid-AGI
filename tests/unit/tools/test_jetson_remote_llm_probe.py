"""F-006 remote-LLM verification: ``tools/jetson_remote_llm_probe.py`` unit tests.

Mirror the structure of ``tests/unit/tools/test_llm_latency_probe.py`` (PR
#102). The probe is operator-facing (intended for ``docker exec`` on the
Jetson) but the dispatch + cold-ping + result-reporting paths are pure
Python and unit-testable against a stub gateway + mocked aiohttp.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from pydantic import SecretStr

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROBE_PATH = _REPO_ROOT / "tools" / "jetson_remote_llm_probe.py"


def _import_probe() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("jetson_remote_llm_probe", _PROBE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["jetson_remote_llm_probe"] = module
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
    """Build a stub Settings-like object configured for openai_compatible backend."""
    cfg = MagicMock()
    cfg.llm.enabled = True
    cfg.llm.backend = "openai_compatible"
    cfg.llm.base_url = "http://192.168.55.100:11434"
    cfg.llm.model_name = "phi3:mini"
    cfg.llm.api_key = None
    cfg.llm.request_timeout_s = 60.0
    cfg.llm.latency_target_ms = 750.0
    return cfg


def _aiohttp_get_async_ctx(*, status: int, json_body: dict[str, Any]) -> MagicMock:
    """Build the nested async-context-manager mock chain aiohttp.get() returns."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_body)
    get_ctx = MagicMock()
    get_ctx.__aenter__ = AsyncMock(return_value=resp)
    get_ctx.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get = MagicMock(return_value=get_ctx)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=None)
    return session_ctx


@pytest.mark.asyncio
async def test_cold_ping_succeeds_and_lists_models(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cold-ping 200 + JSON {data: [{id: phi3:mini}, ...]} → rc=0 + log lists models."""
    fake_body = {"data": [{"id": "phi3:mini"}, {"id": "qwen2.5:1.5b"}]}
    monkeypatch.setattr(
        probe.aiohttp,
        "ClientSession",
        MagicMock(return_value=_aiohttp_get_async_ctx(status=200, json_body=fake_body)),
    )
    rc = await probe._cold_ping_models("http://host:11434", None)
    assert rc == 0


@pytest.mark.asyncio
async def test_cold_ping_connection_error_returns_2(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aiohttp.ClientConnectorError on cold-ping → rc=2 + llm_gateway_load_failed."""

    class _BoomSession:
        async def __aenter__(self) -> Any:
            raise aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError(111, "Connection refused"),
            )

        async def __aexit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(probe.aiohttp, "ClientSession", MagicMock(return_value=_BoomSession()))
    rc = await probe._cold_ping_models("http://unreachable:11434", None)
    assert rc == 2


def _aiohttp_get_async_ctx_raising_json(*, status: int, exc: BaseException) -> MagicMock:
    """aiohttp ctx whose response is 200 but ``resp.json()`` raises ``exc``.

    Models the real-world case where the host answers with a 200 but a
    non-JSON / malformed body (HTML error page from a reverse proxy,
    truncated stream, a non-Ollama service on the configured port).
    """
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(side_effect=exc)
    get_ctx = MagicMock()
    get_ctx.__aenter__ = AsyncMock(return_value=resp)
    get_ctx.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.get = MagicMock(return_value=get_ctx)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=None)
    return session_ctx


@pytest.mark.asyncio
async def test_cold_ping_non_json_content_type_returns_2(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """200 + ``resp.json()`` raising ContentTypeError → rc=2, no exception escapes.

    Also asserts the structured ``cold_ping_parse`` diagnostic is emitted so
    an operator can tell a parse failure (host answered, wrong shape) apart
    from a transport failure (host unreachable).
    """
    content_type_error = aiohttp.ContentTypeError(
        request_info=MagicMock(),
        history=(),
        message="Attempt to decode JSON with unexpected mimetype: text/html",
    )
    monkeypatch.setattr(
        probe.aiohttp,
        "ClientSession",
        MagicMock(
            return_value=_aiohttp_get_async_ctx_raising_json(status=200, exc=content_type_error)
        ),
    )
    rc = await probe._cold_ping_models("http://host:11434", None)
    assert rc == 2
    # structlog renders to stdout via the console renderer in this process.
    out = capsys.readouterr().out
    assert "cold_ping_parse" in out
    assert "llm_gateway_load_failed" in out


@pytest.mark.asyncio
async def test_cold_ping_malformed_json_body_returns_2(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 + ``resp.json()`` raising ValueError (truncated JSON) → rc=2, no raise."""
    monkeypatch.setattr(
        probe.aiohttp,
        "ClientSession",
        MagicMock(
            return_value=_aiohttp_get_async_ctx_raising_json(
                status=200,
                exc=ValueError("Expecting value: line 1 column 1 (char 0)"),
            )
        ),
    )
    rc = await probe._cold_ping_models("http://host:11434", None)
    assert rc == 2


@pytest.mark.asyncio
async def test_cold_ping_non_dict_json_body_returns_2(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 + valid JSON that is a bare list (not the {data:[...]} shape) → rc=2."""
    monkeypatch.setattr(
        probe.aiohttp,
        "ClientSession",
        MagicMock(
            return_value=_aiohttp_get_async_ctx(status=200, json_body=["phi3:mini"])  # type: ignore[arg-type]
        ),
    )
    rc = await probe._cold_ping_models("http://host:11434", None)
    assert rc == 2


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
    stub_gateway.translate_mission = AsyncMock(return_value=stub_goal_vector)

    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: stub_gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)
    monkeypatch.setattr(probe, "_cold_ping_models", AsyncMock(return_value=0))
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )

    args = probe.argparse.Namespace(config=None, overlay=None, mission="turn left slowly")
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
    """translate_mission elapsed > target → exit 1."""
    stub_cfg.llm.latency_target_ms = 50.0  # tight target

    async def _slow_translate(_mission: str) -> Any:
        await asyncio.sleep(0.1)  # 100ms > 50ms target
        return stub_goal_vector

    stub_gateway = MagicMock()
    stub_gateway.is_ready = True
    stub_gateway.start = AsyncMock()
    stub_gateway.stop = AsyncMock()
    stub_gateway.translate_mission = _slow_translate

    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: stub_gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)
    monkeypatch.setattr(probe, "_cold_ping_models", AsyncMock(return_value=0))
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )

    args = probe.argparse.Namespace(config=None, overlay=None, mission="turn left slowly")
    rc = await probe._main(args)
    assert rc == 1


@pytest.mark.asyncio
async def test_main_returns_2_when_cold_ping_fails(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
) -> None:
    """Cold-ping failure short-circuits before gateway.start() is called."""
    stub_gateway_build = MagicMock()  # should NOT be called
    monkeypatch.setattr(probe, "build_llm_gateway", stub_gateway_build)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)
    monkeypatch.setattr(probe, "_cold_ping_models", AsyncMock(return_value=2))
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )

    args = probe.argparse.Namespace(config=None, overlay=None, mission="turn left slowly")
    rc = await probe._main(args)
    assert rc == 2
    stub_gateway_build.assert_not_called()  # critical short-circuit


@pytest.mark.asyncio
async def test_main_returns_2_when_gateway_start_raises(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
) -> None:
    """gateway.start() raising any exception → exit 2 + structured log."""
    stub_gateway = MagicMock()
    stub_gateway.start = AsyncMock(
        side_effect=aiohttp.ClientError("upstream auth rejected"),
    )
    stub_gateway.stop = AsyncMock()

    monkeypatch.setattr(probe, "build_llm_gateway", lambda _cfg, **_kw: stub_gateway)
    monkeypatch.setattr(probe, "build_injection_filter", lambda _cfg: MagicMock())
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)
    monkeypatch.setattr(probe, "_cold_ping_models", AsyncMock(return_value=0))
    monkeypatch.setattr(
        probe,
        "_tegrastats_snapshot",
        lambda: {"ram_used_mb": None, "ram_total_mb": None, "raw_line": None},
    )

    args = probe.argparse.Namespace(config=None, overlay=None, mission="turn left slowly")
    rc = await probe._main(args)
    assert rc == 2
    stub_gateway.stop.assert_not_called()


def test_main_cli_returns_3_when_llm_disabled(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
) -> None:
    """``cfg.llm.enabled=False`` → exit 3."""
    stub_cfg.llm.enabled = False
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)
    rc = probe.main([])
    assert rc == 3


def test_main_cli_returns_3_when_backend_is_llama_cpp(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_cfg: Any,
) -> None:
    """``cfg.llm.backend != 'openai_compatible'`` → exit 3 + clear warning.

    This probe is openai_compatible-specific; the operator wanting the
    local llama_cpp path should use tools/llm_latency_probe.py instead.
    """
    stub_cfg.llm.backend = "llama_cpp"
    monkeypatch.setattr(probe, "load_settings", lambda *_paths: stub_cfg)
    rc = probe.main([])
    assert rc == 3


def test_main_cli_help_flag_exits_zero(probe: Any) -> None:
    """``--help`` exits 0 + prints argparse usage."""
    with pytest.raises(SystemExit) as excinfo:
        probe.main(["--help"])
    assert excinfo.value.code == 0


@pytest.mark.asyncio
async def test_api_key_forwarded_to_cold_ping_as_bearer_but_never_logged(
    probe: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``api_key`` reaches the HTTP layer as Authorization header BUT never appears in logs.

    Regression for PR #99 review finding: api_key is SecretStr and must
    only resolve via get_secret_value() inside the HTTP layer.
    """
    fake_body = {"data": [{"id": "phi3:mini"}]}
    session_ctx = _aiohttp_get_async_ctx(status=200, json_body=fake_body)
    # Capture the headers actually passed to session.get
    captured_headers: dict[str, str] = {}

    async def _entered() -> Any:
        return session_ctx.__aenter__._mock_wraps  # type: ignore[attr-defined]

    real_session = MagicMock()

    def _get(url: str, *, headers: dict[str, str], timeout: Any) -> Any:
        captured_headers.update(headers)
        resp_ctx = MagicMock()
        resp_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(status=200, json=AsyncMock(return_value=fake_body)),
        )
        resp_ctx.__aexit__ = AsyncMock(return_value=None)
        return resp_ctx

    real_session.get = _get
    real_session_ctx = MagicMock()
    real_session_ctx.__aenter__ = AsyncMock(return_value=real_session)
    real_session_ctx.__aexit__ = AsyncMock(return_value=None)

    monkeypatch.setattr(
        probe.aiohttp,
        "ClientSession",
        MagicMock(return_value=real_session_ctx),
    )

    secret_key = SecretStr("super-secret-token-xyz")
    with caplog.at_level("INFO"):
        rc = await probe._cold_ping_models("http://host:11434", secret_key)

    assert rc == 0
    assert captured_headers.get("Authorization") == "Bearer super-secret-token-xyz"
    # Critical: the secret must NOT appear in any captured log message.
    assert all("super-secret-token-xyz" not in record.message for record in caplog.records)
