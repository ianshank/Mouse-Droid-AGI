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

import ast
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

#: A gate for the same distribution, however it is spelled.
_PIL_GATE_RE = re.compile(r"""importorskip\(\s*["']PIL["']""")

#: Distribution whose imports must be gated. Matched against the dotted
#: module root, so ``PIL``, ``PIL.Image`` and ``from PIL.ImageDraw import x``
#: all resolve to the same package.
_GATED_PACKAGE = "PIL"


def _test_modules() -> list[Path]:
    """Every test module in the suite, excluding this one."""
    return sorted(p for p in _TESTS_ROOT.rglob("test_*.py") if p != Path(__file__))


def _is_type_checking_guard(node: ast.stmt) -> bool:
    """True for ``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:``."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _type_only_nodes(tree: ast.Module) -> set[int]:
    """Identity set of every node inside an ``if TYPE_CHECKING:`` body.

    Only ``node.body`` is collected: an ``else:`` under ``TYPE_CHECKING`` is
    the *runtime* branch, so its imports still need a gate.
    """
    type_only: set[int] = set()
    for node in ast.walk(tree):
        if not _is_type_checking_guard(node) or not isinstance(node, ast.If):
            continue
        for body_node in node.body:
            for child in ast.walk(body_node):
                type_only.add(id(child))
    return type_only


def _imports_gated_package(node: ast.AST) -> bool:
    """True when ``node`` is an import of :data:`_GATED_PACKAGE`."""
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == _GATED_PACKAGE for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module.split(".")[0] == _GATED_PACKAGE
    return False


def _uses_pillow_at_runtime(source: str) -> bool:
    """True when the module imports Pillow outside a ``TYPE_CHECKING`` block.

    Walks the AST rather than filtering lines. A line-based filter drops only
    the ``if TYPE_CHECKING:`` line itself and leaves the imports in its
    *body* to match, which would wrongly demand a runtime skip gate on a
    module whose Pillow imports are type-only.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a broken module fails elsewhere
        return False

    type_only = _type_only_nodes(tree)
    return any(
        _imports_gated_package(node) and id(node) not in type_only for node in ast.walk(tree)
    )


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


def test_pil_font_union_is_an_explicit_type_alias() -> None:
    """``PILFont`` must be annotated ``TypeAlias``, not left as a bare union.

    With Pillow absent, mypy resolves both operands to ``Any``, the union
    collapses to a plain variable, and every annotation using it fails with
    "not valid as a type" — three ``mypy --strict`` errors that look like
    defects in the change under review and are not.
    """
    source = (
        _TESTS_ROOT.parent / "src" / "mousedroid" / "hardware" / "display" / "expressions.py"
    ).read_text()
    assert "PILFont: TypeAlias = " in source, (
        "PILFont lost its explicit TypeAlias annotation — mypy --strict will "
        "report three phantom errors on any checkout without the [telemetry] extra"
    )


def test_pillow_degrade_is_logged_not_silent() -> None:
    """``_encode_camera_frame_jpeg`` leaves a breadcrumb when Pillow is absent.

    Without the log, ``verify_sensors.py --save-frame`` silently writes no
    file and the operator diagnoses a dead camera instead of a missing extra.
    """
    runtime_src = (
        _TESTS_ROOT.parent / "src" / "mousedroid" / "validation" / "runtime" / "_camera.py"
    ).read_text()
    assert "camera_jpeg_encode_skipped_no_pillow" in runtime_src


def test_type_checking_only_imports_do_not_require_a_gate() -> None:
    """A type-only Pillow import is not a runtime dependency.

    Guards the guard's precision: a line-based filter drops only the
    ``if TYPE_CHECKING:`` line and leaves the indented import matching, which
    would demand an ``importorskip`` on a module that never touches Pillow at
    runtime.
    """
    type_only = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image
"""
    assert _uses_pillow_at_runtime(type_only) is False


def test_runtime_import_beside_a_type_checking_block_still_counts() -> None:
    """The exemption is scoped to the guarded suite, not the whole module."""
    mixed = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

from PIL import ImageDraw
"""
    assert _uses_pillow_at_runtime(mixed) is True


def test_else_branch_of_a_type_checking_guard_counts_as_runtime() -> None:
    """``else:`` under ``TYPE_CHECKING`` is the branch that actually runs."""
    else_branch = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
else:
    import PIL
"""
    assert _uses_pillow_at_runtime(else_branch) is True


def test_function_local_import_counts_as_runtime() -> None:
    """Most gated Pillow imports in this suite are inside a test function."""
    fn_local = """
def test_thing():
    from PIL import Image
"""
    assert _uses_pillow_at_runtime(fn_local) is True
