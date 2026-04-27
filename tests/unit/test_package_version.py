"""Version resolution tests for source-tree execution paths."""

from __future__ import annotations

import importlib.metadata as metadata
import runpy
from pathlib import Path


def test_version_falls_back_when_package_metadata_is_missing(
    monkeypatch,
) -> None:
    """Source-tree imports should still work when dist-info metadata is absent."""

    def raise_package_not_found(_distribution: str) -> str:
        raise metadata.PackageNotFoundError()

    monkeypatch.setattr(metadata, "version", raise_package_not_found)

    module_globals = runpy.run_path(Path("src/mousedroid/__init__.py"))

    assert module_globals["__version__"] == "0+unknown"
