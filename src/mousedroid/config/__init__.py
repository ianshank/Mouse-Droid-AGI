"""Configuration schema and YAML loader."""

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings

__all__ = [
    "Settings",
    "load_settings",
]
