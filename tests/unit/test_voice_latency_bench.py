"""Unit tests for the benchmark_voice_latency CLI script."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    personality: str = "rocky",
    model_path: str | None = "/models/rocky.onnx",
    personality_to_model_map: dict[str, str] | None = None,
) -> Settings:
    data: dict[str, Any] = {
        "mock_hardware": True,
        "voice": {
            "enabled": True,
            "personality": personality,
            "tts_model_path": model_path,
        },
    }
    if personality_to_model_map is not None:
        data["voice"]["personality_to_model_map"] = personality_to_model_map
    return Settings.model_validate(data)


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    """Default CLI args have sensible values."""
    import scripts.benchmark_voice_latency as bench

    args = bench._parse_args([])
    assert args.personalities == ["rocky"]
    assert args.n_warmup == 3
    assert args.n_iter == 20
    assert args.p95_target_ms == pytest.approx(500.0)


def test_parse_args_custom_values() -> None:
    """Explicit CLI args are parsed correctly."""
    import scripts.benchmark_voice_latency as bench

    args = bench._parse_args(
        [
            "--personalities",
            "rocky",
            "friendly",
            "--n-warmup",
            "1",
            "--n-iter",
            "5",
            "--p95-target-ms",
            "250",
        ]
    )
    assert args.personalities == ["rocky", "friendly"]
    assert args.n_warmup == 1
    assert args.n_iter == 5
    assert args.p95_target_ms == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


def test_percentile_p50() -> None:
    """p50 of [1, 2, 3, 4, 5] ≈ 3."""
    import scripts.benchmark_voice_latency as bench

    assert bench._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)


def test_percentile_p95_all_same() -> None:
    """p95 of identical values equals that value."""
    import scripts.benchmark_voice_latency as bench

    assert bench._percentile([10.0] * 20, 95) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _report
# ---------------------------------------------------------------------------


def test_report_pass_when_p95_under_target(capsys: pytest.CaptureFixture[str]) -> None:
    """_report returns True when p95 is within target."""
    import scripts.benchmark_voice_latency as bench

    latencies = [100.0] * 20  # p95 = 100 ms
    passed = bench._report("rocky", latencies, p95_target_ms=200.0)
    assert passed is True
    captured = capsys.readouterr()
    assert "PASS" in captured.out


def test_report_fail_when_p95_over_target(capsys: pytest.CaptureFixture[str]) -> None:
    """_report returns False when p95 exceeds target."""
    import scripts.benchmark_voice_latency as bench

    latencies = [600.0] * 20  # p95 = 600 ms
    passed = bench._report("rocky", latencies, p95_target_ms=500.0)
    assert passed is False
    captured = capsys.readouterr()
    assert "FAIL" in captured.out


def test_report_skip_on_empty_latencies(capsys: pytest.CaptureFixture[str]) -> None:
    """_report returns True (skip) when latencies list is empty."""
    import scripts.benchmark_voice_latency as bench

    passed = bench._report("rocky", [], p95_target_ms=500.0)
    assert passed is True
    captured = capsys.readouterr()
    assert "SKIP" in captured.out


# ---------------------------------------------------------------------------
# _benchmark_personality — mocked PiperTTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_benchmark_personality_returns_latency_list() -> None:
    """_benchmark_personality returns a list of float latencies when model exists."""
    import scripts.benchmark_voice_latency as bench

    fake_audio = np.zeros(22050, dtype=np.float32)
    mock_tts_cls = MagicMock()
    mock_tts_inst = MagicMock()
    mock_tts_inst.synthesize = AsyncMock(return_value=fake_audio)
    mock_tts_inst.stop = MagicMock()
    mock_tts_cls.return_value = mock_tts_inst

    cfg = _make_settings(personality="rocky", model_path="/models/rocky.onnx")

    with patch("mousedroid.voice.tts.PiperTTS", mock_tts_cls):
        # Patch where _benchmark_personality imports PiperTTS at runtime.
        latencies = await bench._benchmark_personality(
            personality="rocky",
            cfg=cfg,
            n_warmup=1,
            n_iter=3,
        )

    # Expect 3 timed latency samples
    assert len(latencies) == 3
    assert all(isinstance(v, float) for v in latencies)


@pytest.mark.asyncio
async def test_benchmark_personality_empty_when_no_model(tmp_path: Path) -> None:
    """_benchmark_personality returns [] when no model path is resolved."""
    import scripts.benchmark_voice_latency as bench

    cfg = _make_settings(personality="rocky", model_path=None)
    latencies = await bench._benchmark_personality(
        personality="rocky",
        cfg=cfg,
        n_warmup=1,
        n_iter=3,
    )
    assert latencies == []
