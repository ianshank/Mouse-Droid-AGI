"""Shared torch device resolution.

A single canonical resolver so every device-agnostic surface (drift training,
the comparison + spike scripts, and any future torch entry point) agrees on
what ``None`` / ``"auto"`` mean. Keeps the CUDA-when-available default in one
place instead of re-deriving ``torch.cuda.is_available()`` inline at each call
site.
"""

from __future__ import annotations

import torch

#: Sentinel accepted (in addition to ``None``) for "pick the best device".
_AUTO = "auto"


def resolve_device(spec: str | torch.device | None) -> torch.device:
    """Resolve a device spec to a concrete :class:`torch.device`.

    Args:
        spec: ``None`` or ``"auto"`` selects CUDA when available, else CPU.
            A concrete ``"cpu"`` / ``"cuda"`` / ``"cuda:N"`` string or a
            :class:`torch.device` is honoured verbatim (an explicit ``"cuda"``
            on a host without CUDA fails loudly downstream — the caller asked
            for it).

    Returns:
        The resolved :class:`torch.device`.
    """
    if spec is None or spec == _AUTO:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


__all__ = ["resolve_device"]
