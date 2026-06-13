"""Regression: cap inline type:ignore / noqa debt in src/.

Update the budgets DOWN as the purge lands; never up without justification.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"
# Budgets = the MEASURED post-purge residual of load-bearing suppressions
# (untyped 3rd-party boundaries, Pydantic A00x field shadows) PLUS the handful
# of prose mentions of the literal tokens inside explanatory comments/docstrings.
# Expected NON-ZERO. May only ratchet DOWN, never up without a documented reason.
#
# The ignore budget (9) = 6 live torch untyped-call directives
#   (Tensor.backward x4, torch.jit.trace/.load x2) + 3 prose mentions
#   (safety/projector.py, telemetry/server.py, voice/greeting.py).
# The lint-waiver budget (19) = 18 live directives (torch.nn.functional N812 x4,
#   InjectionRejected N818, non-crypto random S311 x3, fixed-http urllib
#   S310 x4, Pydantic Field default paths S108/S104 x4, Argus socket S108,
#   watchdog systemd-notify S603/S607) + 1 prose mention
#   (common/imports.py docstring).
_MAX_TYPE_IGNORE = 9
_MAX_NOQA = 19


def _count(token: str) -> int:
    return sum(
        line.count(token)
        for p in _SRC.rglob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
    )


def test_type_ignore_within_budget() -> None:
    assert _count("type: ignore") <= _MAX_TYPE_IGNORE


def test_noqa_within_budget() -> None:
    assert _count("noqa") <= _MAX_NOQA
