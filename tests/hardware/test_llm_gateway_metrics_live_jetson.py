"""Hardware: live LLM-gateway observability on the rover (server + wiring).

Two complementary checks for the PR #115 metric families
(``mousedroid_llm_tokens_total`` / ``…_llm_gateway_latency_ms`` /
``…_llm_gateway_served_total`` / ``…_llm_latency_budget_exceeded_total``):

* **Test A** scrapes the *live* running telemetry server's ``/metrics`` endpoint
  (auth-exempt) and asserts it is a healthy Prometheus surface. It does NOT
  assert the families are populated — on production there is no HTTP mission
  ingress (``openclaw`` is disabled), so nothing drives them over the wire.
* **Test B** drives the gateway **in-process** through the real orchestrator
  wiring — ``build_orchestrator(...).process_mission("navigate to the
  cantina")`` (a guaranteed-UNKNOWN command that the rule parser misses, so it
  reaches Claude) — and asserts the orchestrator's shared
  :class:`MetricsRegistry` now renders the families. This proves the exact
  ``build_orchestrator -> build_llm_gateway(metrics=…)`` wiring on real Claude,
  needing neither ``openclaw`` nor the HTTP server.
* **Test C** (optional) exercises the HTTP ``POST /api/v1/mission`` ingress, but
  skips whenever ``openclaw`` is disabled (the production default).

Complements ``test_llm_gateway_observability_jetson.py`` (which builds a bare
gateway); this module validates the orchestrator-level wiring + live endpoint.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from tests._jetson_hardware import is_jetson_host, load_jetson_runtime_settings

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings

aiohttp = pytest.importorskip("aiohttp")

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not is_jetson_host(), reason="Jetson-only hardware test"),
]

# Four PR #115 family base names (namespace prefix applied at render time).
_LLM_FAMILIES = (
    "llm_tokens_total",
    "llm_gateway_latency_ms",
    "llm_gateway_served_total",
    "llm_latency_budget_exceeded_total",
)

# Guaranteed-UNKNOWN command: the rule parser (stop/forward/turn/strafe/patrol/
# avoid) does not match it, so process_mission routes it to the LLM gateway.
_UNKNOWN_COMMAND = "navigate to the cantina"


def _telemetry_base_url() -> str:
    """Return the live telemetry base URL (env override, localhost default)."""
    return os.getenv("MOUSEDROID_TELEMETRY_URL", "http://127.0.0.1:8080").rstrip("/")


# --------------------------------------------------------------------------- #
# Test A — the live /metrics endpoint is a healthy Prometheus surface
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_live_metrics_endpoint_is_healthy() -> None:
    """GET the running server's metrics_path → 200 + Prometheus exposition."""
    cfg = load_jetson_runtime_settings()
    metrics_path = cfg.telemetry.metrics_path
    url = f"{_telemetry_base_url()}{metrics_path}"

    timeout = aiohttp.ClientTimeout(total=5.0)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(url) as resp,
        ):
            status = resp.status
            body = await resp.text()
    except aiohttp.ClientError as exc:  # server not running / unreachable
        pytest.skip(f"telemetry server unreachable at {url}: {exc}")

    assert status == 200, f"/metrics returned {status}"
    namespace = cfg.metrics.namespace
    assert namespace in body, f"expected metric namespace {namespace!r} in /metrics body"
    # Prometheus exposition line-shape: at least one non-comment metric sample.
    sample_lines = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
    assert sample_lines, "no Prometheus samples rendered"


# --------------------------------------------------------------------------- #
# Test B — a real mission populates the #115 families via orchestrator wiring
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inprocess_mission_populates_metric_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_orchestrator -> process_mission(UNKNOWN) -> orch._metrics has families.

    Validates the production wiring (shared registry threaded into the gateway)
    against live Claude, with the dead ESP32 mocked out. Requires
    ``ANTHROPIC_API_KEY`` so the cloud primary can answer.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — cloud primary cannot answer")

    from mousedroid.factory import build_orchestrator
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    # Validate-around the functionally-dead ESP32 (MockESP32Driver, resilience
    # wrapper intact) so the orchestrator builds without the drivetrain.
    monkeypatch.setenv("MOUSEDROID_ESP32__ENABLED", "false")
    cfg: Settings = load_jetson_runtime_settings()

    # build_orchestrator returns ``object`` (protocol-DI invariant); narrow to
    # the concrete type so the wiring attributes are accessible + type-checked.
    orch = build_orchestrator(cfg)
    assert isinstance(orch, MouseDroidOrchestrator)
    if orch._metrics is None:
        pytest.fail("build_orchestrator produced no MetricsRegistry")
    if orch._llm_gateway is None:
        pytest.skip("LLM gateway disabled in this config (cfg.llm.enabled=false)")

    await orch._llm_gateway.start()
    try:
        await orch.process_mission(_UNKNOWN_COMMAND)
    finally:
        await orch._llm_gateway.stop()

    out = orch._metrics.render_prometheus()
    ns = cfg.metrics.namespace

    # At minimum, the gateway path must have fired *some* #115 family — proves
    # the registry threaded by build_orchestrator is the one the gateway writes.
    fired = [f for f in _LLM_FAMILIES if f"{ns}_{f}" in out]
    assert fired, f"no #115 family populated after a real mission; rendered:\n{out}"

    # When the cloud primary succeeded, latency + token usage are recorded.
    if f"{ns}_llm_gateway_latency_ms_count" in out:
        assert f"{ns}_llm_tokens_total{{model=" in out, "latency recorded but tokens missing"

    # The composite (cfg.llm.fallback_backend != 'none') always records a served
    # outcome for the answering tier.
    from mousedroid.llm_gateway.fallback_gateway import FallbackLLMGateway

    if isinstance(orch._llm_gateway, FallbackLLMGateway):
        assert f"{ns}_llm_gateway_served_total{{" in out, "composite served counter missing"


# --------------------------------------------------------------------------- #
# Test C — HTTP ingress population (optional; skips when openclaw disabled)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_http_mission_populates_metrics_when_openclaw_enabled() -> None:
    """POST /api/v1/mission -> live /metrics increments (only if openclaw on)."""
    cfg = load_jetson_runtime_settings()
    if cfg.openclaw is None or not cfg.openclaw.enabled:
        pytest.skip("openclaw mission ingress disabled (production default)")

    token = os.getenv("MOUSEDROID_TELEMETRY_TOKEN")
    if not token:
        pytest.skip("no telemetry token — POST /api/v1/mission requires auth")

    base = _telemetry_base_url()
    metrics_url = f"{base}{cfg.telemetry.metrics_path}"
    mission_url = f"{base}/api/v1/mission"
    headers = {"Authorization": f"Bearer {token}"}
    ns = cfg.metrics.namespace
    timeout = aiohttp.ClientTimeout(total=30.0)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                mission_url,
                json={"nl_command": _UNKNOWN_COMMAND},
                headers=headers,
            ) as post_resp:
                post_status = post_resp.status
            async with session.get(metrics_url) as get_resp:
                body = await get_resp.text()
    except aiohttp.ClientError as exc:
        pytest.skip(f"telemetry server unreachable: {exc}")

    assert post_status == 202, f"mission POST returned {post_status}"
    assert f"{ns}_llm_gateway_served_total{{" in body, "served counter not populated via HTTP"
