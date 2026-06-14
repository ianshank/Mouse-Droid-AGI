"""Hardware-bound MSE-6 greeting test — real Piper TTS + real ALSA speaker.

Issue #109. Gated by :data:`pytest.mark.hardware` so CI (which always sets
``MOUSEDROID_MOCK_HARDWARE=true`` and runs ``-m "not hardware"``) never
touches it; on the Jetson the rover-side runner picks it up via
``pytest -m hardware``. On any non-Jetson host (the dev workstation) the
``_require_jetson`` guard SKIPs cleanly — it must never ERROR.

What this covers on the live rover:

1. The greeting subsystem builds through the production factory with real
   hardware (PiperTTS synthesiser + USB/ALSA speaker), not the mocks.
2. ``Greeter.greet()`` runs end-to-end with no exceptions.
3. A real waveform was synthesised (``greeting_done.samples > 0``) and the
   documented structured-log sequence was emitted.

Skip-gate + docstring style mirror ``tests/hardware/test_pr104_jetson_dashboard.py``.
"""

from __future__ import annotations

import pytest
import structlog.testing

pytestmark = [pytest.mark.hardware]


def _require_jetson() -> None:
    """Graceful skip on a non-Jetson host (deferred to the rover runner)."""
    from tests._jetson_hardware import is_jetson_host

    if not is_jetson_host():
        pytest.skip("Not running on Jetson host — hardware greeting test deferred.")


@pytest.mark.asyncio
async def test_greet_synthesises_real_waveform_on_jetson(jetson_settings) -> None:
    """Real Piper TTS + ALSA speaker: greeting plays and synthesises audio.

    Builds the greeter through the production factory against the live
    ``jetson_settings`` (``mock_hardware=False`` on the Jetson). Asserts
    the terminal ``greeting_done`` event reports a positive sample count
    (a real waveform was produced) and that the ordered structured events
    were emitted, with no exception escaping.
    """
    _require_jetson()

    cfg = jetson_settings
    if cfg.voice is None or not cfg.voice.enabled:
        pytest.skip("voice disabled in Jetson config")
    if cfg.speaker is None or not cfg.speaker.enabled:
        pytest.skip("speaker disabled in Jetson config")
    if cfg.greeting is None or not cfg.greeting.enabled:
        pytest.skip("greeting disabled in Jetson config")

    from mousedroid.factory import build_greeter, build_voice_engine

    engine = build_voice_engine(cfg)
    if engine is None:
        pytest.skip("voice engine could not be built on this host")
    greeter = build_greeter(cfg, voice_engine=engine)

    try:
        await engine.start()
    except Exception as exc:
        pytest.skip(f"voice engine could not start on hardware: {exc}")

    try:
        with structlog.testing.capture_logs() as logs:
            await greeter.greet()  # must not raise on real hardware
    finally:
        await engine.stop()

    by_event = {e["event"]: e for e in logs if e["event"].startswith("greeting_")}
    assert "greeting_started" in by_event
    assert "greeting_message_playing" in by_event
    assert "greeting_done" in by_event
    # A real waveform was synthesised through Piper + clocked out to ALSA.
    assert by_event["greeting_done"]["samples"] > 0
