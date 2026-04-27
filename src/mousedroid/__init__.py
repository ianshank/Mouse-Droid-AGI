"""MouseDroidAGI — Star Wars MSE-6 Agentic World Model on Jetson Orin Nano."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("mousedroid")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
