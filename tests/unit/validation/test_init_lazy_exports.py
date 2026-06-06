"""Unit tests for the lazy (:pep:`562`) re-exports in ``validation/__init__``.

In-process counterpart to ``tests/regression/test_validation_import_decoupling``
(which guards numpy-purity in a subprocess). These cover both ``__getattr__``
branches so the resolution + AttributeError paths are exercised under coverage.
"""

from __future__ import annotations

import pytest

import mousedroid.validation as validation


def test_lazy_name_resolves_to_runtime_symbol() -> None:
    resolved = validation.resolve_runtime_config_paths
    assert callable(resolved)
    assert resolved.__name__ == "resolve_runtime_config_paths"


def test_all_names_are_resolvable() -> None:
    for name in validation.__all__:
        assert getattr(validation, name) is not None


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        validation.does_not_exist  # noqa: B018 — attribute access triggers __getattr__
