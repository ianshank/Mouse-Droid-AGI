"""Smoke-pass: Hailo-8 accelerator probe + mock-fallback + minimal inference.

Skipped on non-Jetson hosts via :func:`is_jetson_host`. Skipped at module
level when the optional ``hailort`` package isn't installed (Hailo is an
optional dep — operators wire it only on rovers carrying the Hailo-8
M.2 accelerator).

Covers:

* ``build_hailo_runtime(cfg)`` returns a runtime instance on Jetson
  with Hailo enabled, or ``None`` when ``cfg.hailo.enabled=False``.
* ``MockHailoRuntime`` (constructed when ``cfg.mock_hardware=True``)
  exposes ``infer_sync`` returning zero-filled output of the expected
  shape after ``await start()``.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tests._jetson_hardware import is_jetson_host, load_jetson_runtime_settings

pytestmark = pytest.mark.hardware

# Hailo is an optional dep; skip the whole module if hailort isn't
# installed (operator-driven gate).
pytest.importorskip("hailort")

if not is_jetson_host():
    pytest.skip(
        "Hailo accelerator smoke tests require Jetson hardware",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def settings() -> object:
    """Load the Jetson runtime settings (real-hardware mode)."""
    return load_jetson_runtime_settings()


def test_build_hailo_runtime_returns_none_when_disabled(settings: object) -> None:
    """``cfg.hailo.enabled=False`` (or missing cfg.hailo) returns ``None``."""
    import copy

    from mousedroid.factory import build_hailo_runtime

    s = copy.deepcopy(settings)
    if s.hailo is None:  # type: ignore[attr-defined]
        # The disabled-by-omission path — the factory shortcuts to None.
        assert build_hailo_runtime(s) is None  # type: ignore[arg-type]
        return
    s.hailo.enabled = False  # type: ignore[attr-defined]
    assert build_hailo_runtime(s) is None  # type: ignore[arg-type]


def test_mock_hailo_runtime_built_when_mock_hardware_enabled(settings: object) -> None:
    """``mock_hardware=True`` + ``hailo.enabled=True`` returns ``MockHailoRuntime``."""
    import copy

    from mousedroid.factory import build_hailo_runtime
    from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime

    s = copy.deepcopy(settings)
    if s.hailo is None:  # type: ignore[attr-defined]
        pytest.skip("cfg.hailo missing in runtime settings")
    s.mock_hardware = True  # type: ignore[attr-defined]
    s.hailo.enabled = True  # type: ignore[attr-defined]

    rt = build_hailo_runtime(s)  # type: ignore[arg-type]
    assert isinstance(rt, MockHailoRuntime)


def test_mock_hailo_runtime_infer_returns_zero_filled_output(settings: object) -> None:
    """After ``start()``, ``infer_sync('yolo', input)`` returns zero-filled output.

    Reuses the canonical ``DEFAULT_OUTPUT_SHAPES["yolo"]`` shape
    declared on ``MockHailoRuntime`` — pins the contract without
    hardcoding the shape literal here.
    """
    import copy

    from mousedroid.factory import build_hailo_runtime
    from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime

    s = copy.deepcopy(settings)
    if s.hailo is None:  # type: ignore[attr-defined]
        pytest.skip("cfg.hailo missing in runtime settings")
    s.mock_hardware = True  # type: ignore[attr-defined]
    s.hailo.enabled = True  # type: ignore[attr-defined]

    rt = build_hailo_runtime(s)  # type: ignore[arg-type]
    assert isinstance(rt, MockHailoRuntime)
    asyncio.run(rt.start())

    # Use a sentinel input shape — MockHailoRuntime ignores input shape
    # and returns the configured output shape for the named model.
    fake_input = np.zeros((1, 3, 224, 224), dtype=np.uint8)
    output = rt.infer_sync("yolo", fake_input)
    assert output.dtype == np.float32
    # Compare against the class-level default to avoid hardcoding (1, 25200, 85).
    assert output.shape == MockHailoRuntime.DEFAULT_OUTPUT_SHAPES["yolo"]
    assert np.all(output == 0.0)
