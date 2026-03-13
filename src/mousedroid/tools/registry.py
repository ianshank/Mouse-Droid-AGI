"""Tool registry — registration and dispatch for MouseDroid tools.

.. deprecated::
    This module is deprecated and will be removed in a future release.
    Use :mod:`mousedroid.common.tools.registry` instead.
"""

from __future__ import annotations

import warnings

from mousedroid.common.tools.registry import (
    ToolRegistry,
    ToolSpec,
    create_default_registry,
)

__all__ = ["ToolRegistry", "ToolSpec", "create_default_registry"]


def __getattr__(name: str) -> object:
    """Emit deprecation warning on any attribute access."""
    warnings.warn(
        f"mousedroid.tools.registry.{name} is deprecated. "
        "Use mousedroid.common.tools.registry instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Re-export everything from canonical module
    import mousedroid.common.tools.registry as _canonical

    return getattr(_canonical, name)
