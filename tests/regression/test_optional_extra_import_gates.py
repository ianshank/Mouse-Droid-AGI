"""Optional-extra import gates in the test suite.

Pillow ships in the ``[telemetry]`` extra, not the bare ``[dev]`` one. Four
test modules used to reach Pillow with no ``pytest.importorskip`` gate, so a
``pip install -e ".[dev]"`` checkout produced 31 collection/assert failures
that look like real defects and are not — CI only stays green because the
``test`` job happens to install ``[dev,telemetry,mcp]``.

The rest of the suite already uses ``pytest.importorskip`` for exactly this
(``mujoco``, ``faiss``, ``ncps``, ``aiohttp``, ``mlflow``, ``onnx``, …). These
pins keep the convention mechanical instead of remembered.

The scan is a source-text pin on purpose: it needs no import of the module
under inspection (which is the very thing that would blow up without the
extra installed) and it stays readable in a diff.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]

#: Modules that exercise Pillow *indirectly* — they never name ``PIL`` but
#: call a renderer that lazily imports it, so the textual scan below cannot
#: discover them. Listed explicitly rather than inferred.
_INDIRECT_PILLOW_MODULES = (
    "unit/hardware/display/test_expressions.py",
    "unit/hardware/display/test_ssd1306_face_driver.py",
)

#: ``import PIL`` / ``from PIL import …`` at any indentation. ``TYPE_CHECKING``
#: blocks are excluded by the caller (they never execute at runtime).
_PIL_IMPORT_RE = re.compile(r"^\s*(?:from PIL(?:\.\w+)*\s+import|import PIL\b)", re.MULTILINE)

#: A gate for the same distribution, however it is spelled.
_PIL_GATE_RE = re.compile(r"""importorskip\(\s*["']PIL["']""")


def _test_modules() -> list[Path]:
    """Every test module in the suite, excluding this one."""
    return sorted(p for p in _TESTS_ROOT.rglob("test_*.py") if p != Path(__file__))


def _uses_pillow_at_runtime(source: str) -> bool:
    """True when the module imports Pillow outside a ``TYPE_CHECKING`` block."""
    runtime_source = "\n".join(line for line in source.splitlines() if "TYPE_CHECKING" not in line)
    return bool(_PIL_IMPORT_RE.search(runtime_source))


def _pillow_importing_modules() -> list[Path]:
    """Test modules that name Pillow in a runtime import."""
    return [p for p in _test_modules() if _uses_pillow_at_runtime(p.read_text())]


def test_scan_finds_the_pillow_importers() -> None:
    """Guard the guard: an empty scan would make every pin below vacuous."""
    found = _pillow_importing_modules()
    assert found, "no test module imports PIL — the regex or the layout drifted"


@pytest.mark.parametrize("module", _pillow_importing_modules(), ids=lambda p: p.name)
def test_pillow_importing_modules_carry_a_gate(module: Path) -> None:
    """Every runtime ``PIL`` import in the suite sits behind an importorskip."""
    source = module.read_text()
    assert _PIL_GATE_RE.search(source), (
        f"{module.relative_to(_TESTS_ROOT)} imports Pillow at runtime with no "
        f'pytest.importorskip("PIL", reason=...) gate. Pillow ships in the '
        f"[telemetry] extra, so this module fails rather than skips on a bare "
        f'pip install -e ".[dev]" checkout.'
    )


@pytest.mark.parametrize("relative_path", _INDIRECT_PILLOW_MODULES)
def test_indirect_pillow_modules_carry_a_gate(relative_path: str) -> None:
    """Renderer-driven modules gate too, even though they never name PIL.

    ``hardware/display/expressions.py`` imports Pillow lazily inside
    ``_new_canvas``; the import is invisible to a textual scan but the failure
    it produces without the extra is identical.
    """
    module = _TESTS_ROOT / relative_path
    assert module.exists(), f"{relative_path} moved — update _INDIRECT_PILLOW_MODULES"
    assert _PIL_GATE_RE.search(module.read_text()), (
        f"{relative_path} renders through Pillow with no importorskip gate"
    )


def test_pillow_degrade_is_logged_not_silent() -> None:
    """``_encode_camera_frame_jpeg`` leaves a breadcrumb when Pillow is absent.

    Without the log, ``verify_sensors.py --save-frame`` silently writes no
    file and the operator diagnoses a dead camera instead of a missing extra.
    """
    runtime_src = (
        _TESTS_ROOT.parent / "src" / "mousedroid" / "validation" / "runtime.py"
    ).read_text()
    assert "camera_jpeg_encode_skipped_no_pillow" in runtime_src
