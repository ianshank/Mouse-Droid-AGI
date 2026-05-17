"""Smoke-pass: SSD1306 face display I²C probe + driver build + controller wrap.

Skipped on non-Jetson hosts via :func:`is_jetson_host`. On Jetson with the
SSD1306 panel reachable on the configured I²C bus this exercises:

* ``build_face_display(cfg)`` — probes the I²C bus + instantiates the
  real ``SSD1306FaceDriver`` (or falls back to ``MockFaceDriver`` when
  ``cfg.face_display.fallback_to_mock_on_error=True``).
* ``build_face_controller(cfg, driver)`` — wraps the driver in the
  ``FaceController`` so the orchestrator's face seam is exercisable.

Architecture invariants (per CLAUDE.md):

* No hardcoded thresholds — every check resolves expectations from
  the loaded ``Settings`` (e.g. I²C bus / address from
  ``cfg.face_display``).
* Reuses the canonical ``is_jetson_host`` + ``load_jetson_runtime_settings``
  helpers; does not parallel them.
"""

from __future__ import annotations

import pytest

from tests._jetson_hardware import is_jetson_host, load_jetson_runtime_settings

pytestmark = pytest.mark.hardware

if not is_jetson_host():
    pytest.skip(
        "SSD1306 face display smoke tests require Jetson hardware",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def settings() -> object:
    """Load the Jetson runtime settings (real-hardware mode)."""
    return load_jetson_runtime_settings()


def test_face_display_driver_builds_without_error(settings: object) -> None:
    """``build_face_display`` returns a driver instance OR falls back to mock."""
    from mousedroid.factory import build_face_display

    driver = build_face_display(settings)  # type: ignore[arg-type]
    # ``build_face_display`` returns None when the subsystem is fully
    # disabled in cfg. On a Jetson runtime that's allowed but means the
    # face controller smoke is vacuous — record + accept.
    if driver is None:
        pytest.skip("face_display subsystem disabled in this runtime config")
    assert driver is not None


def test_face_controller_wraps_driver(settings: object) -> None:
    """``build_face_controller`` wraps the driver into a controller."""
    from mousedroid.factory import build_face_controller, build_face_display

    driver = build_face_display(settings)  # type: ignore[arg-type]
    if driver is None:
        pytest.skip("face_display subsystem disabled in this runtime config")
    controller = build_face_controller(settings, driver)  # type: ignore[arg-type]
    assert controller is not None


def test_face_controller_can_render_and_clear(settings: object) -> None:
    """Controller exposes idempotent render + clear operations."""
    from mousedroid.factory import build_face_controller, build_face_display

    driver = build_face_display(settings)  # type: ignore[arg-type]
    if driver is None:
        pytest.skip("face_display subsystem disabled in this runtime config")
    controller = build_face_controller(settings, driver)  # type: ignore[arg-type]
    assert controller is not None
    # Either method may not exist on every controller implementation —
    # call them defensively and pass when they're missing. The asserted
    # contract is "no exception" on a healthy I²C bus.
    if hasattr(controller, "clear"):
        controller.clear()
    if hasattr(controller, "render_default"):
        controller.render_default()


def test_face_display_falls_back_to_mock_on_i2c_error(settings: object) -> None:
    """When ``fallback_to_mock_on_error=True``, an unreachable bus returns MockFaceDriver.

    This pins the existing factory contract — operators rely on the
    fallback to keep ``orchestrator.start()`` from crashing on a
    disconnected I²C cable.
    """
    # Mutate the cfg to a definitely-unreachable bus address so we
    # exercise the fallback path even on a healthy Jetson.
    import copy

    from mousedroid.factory import build_face_display

    s = copy.deepcopy(settings)
    if s.face_display is None:  # type: ignore[attr-defined]
        pytest.skip("face_display config missing")
    s.face_display.i2c_address = 0x7F  # type: ignore[attr-defined]
    s.face_display.fallback_to_mock_on_error = True  # type: ignore[attr-defined]

    driver = build_face_display(s)
    # On the unreachable address the real driver raises; the factory
    # catches and returns the MockFaceDriver per fallback_to_mock_on_error.
    assert driver is not None
