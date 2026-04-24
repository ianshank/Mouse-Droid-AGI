#!/usr/bin/env python3
"""Validate canonical Settings import identity before test execution.

This catches module alias drift (for example, duplicate imports under
``mousedroid.config.schema`` and ``src.mousedroid.config.schema``) that can
produce Pydantic ``is_instance_of`` validation errors under coverage.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Final

CANONICAL_MODULE: Final[str] = "mousedroid.config.schema"


def _module_path(module: ModuleType) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return None
    return Path(module_file).resolve()


def _schema_alias_modules() -> list[str]:
    alias_suffix = ".mousedroid.config.schema"
    return sorted(
        name
        for name in sys.modules
        if name.endswith(alias_suffix) and name != CANONICAL_MODULE
    )


def main() -> int:
    try:
        schema_module = importlib.import_module(CANONICAL_MODULE)
        from mousedroid.config.schema import Settings

        imported_settings = schema_module.Settings
        if Settings is not imported_settings:
            print("Settings identity mismatch between canonical imports")
            print(f"  from-import id={id(Settings)}")
            print(f"  module attr id={id(imported_settings)}")
            return 1

        aliases = _schema_alias_modules()
        if aliases:
            print("Duplicate schema module aliases loaded:")
            for alias in aliases:
                print(f"  - {alias}")
            return 1

        settings = Settings(mock_hardware=True)
        if not isinstance(settings, Settings):
            print("Settings instance is not recognized as Settings type")
            return 1

        path = _module_path(schema_module)
        print("Settings identity check passed")
        print(f"  module={CANONICAL_MODULE}")
        if path is not None:
            print(f"  path={path}")
        return 0
    except Exception:
        print("Settings identity check failed with exception")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
