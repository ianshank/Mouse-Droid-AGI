# tests/unit/common/test_imports.py
"""Unit tests for the optional-dependency probes in ``mousedroid.common.imports``.

Pins the distinction between :func:`module_available` (spec presence) and
:func:`module_importable` (a real, guarded import) — the latter is what the
camera/Hailo backend auto-selection relies on so a spec-present-but-import-fails
package (missing native deps) does not get wrongly selected.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from mousedroid.common.imports import module_available, module_importable


def test_module_available_true_for_stdlib() -> None:
    assert module_available("json") is True


def test_module_available_false_for_missing() -> None:
    assert module_available("definitely_not_a_real_module_xyz") is False


def test_module_importable_true_for_stdlib() -> None:
    assert module_importable("json") is True


def test_module_importable_false_for_missing() -> None:
    assert module_importable("definitely_not_a_real_module_xyz") is False


def test_module_importable_true_for_injected_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake already in ``sys.modules`` (test-injected) counts as importable."""
    fake = ModuleType("fake_injected_pkg")
    monkeypatch.setitem(sys.modules, "fake_injected_pkg", fake)
    assert module_importable("fake_injected_pkg") is True


def test_module_importable_false_when_spec_present_but_import_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core bug #1 contract: a discoverable spec whose import FAILS is NOT importable.

    ``module_available`` returns True (the meta-path finder yields a spec), but
    ``module_importable`` must return False because the actual import raises —
    mirroring picamera2/hailo_platform with missing native bindings.
    """
    name = "spec_present_import_fails_pkg"
    monkeypatch.delitem(sys.modules, name, raising=False)

    import importlib.abc
    import importlib.machinery

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec: object) -> None:
            return None

        def exec_module(self, module: ModuleType) -> None:
            msg = "native dependency missing"
            raise ImportError(msg)

    class _Finder:
        def find_spec(
            self, fullname: str, _path: object = None, _target: object = None
        ) -> object | None:
            if fullname == name:
                return importlib.machinery.ModuleSpec(name, _Loader())
            return None

    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        assert module_available(name) is True
        assert module_importable(name) is False
    finally:
        sys.meta_path.remove(finder)
