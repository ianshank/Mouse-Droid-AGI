"""Smoke-pass: run_preflight returns OK on mock_hardware within 5s."""

from __future__ import annotations

import time

import pytest

from mousedroid.config.schema import Settings
from mousedroid.validation.preflight import PreflightStatus, run_preflight

pytestmark = pytest.mark.smoke


@pytest.mark.asyncio
async def test_preflight_completes_under_5s_on_mock_hardware() -> None:
    cfg = Settings(mock_hardware=True)
    deadline = time.monotonic() + 5.0
    report = await run_preflight(cfg)
    assert report.overall_status == PreflightStatus.OK
    assert time.monotonic() < deadline, "preflight budget exceeded"
