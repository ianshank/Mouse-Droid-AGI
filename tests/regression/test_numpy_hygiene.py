"""Regression: no deprecated NumPy aliases creep back into src/."""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"
_BANNED = re.compile(r"\bnp\.(float|int|bool|object|str|NaN|complex)\b")


def test_no_deprecated_numpy_aliases() -> None:
    offenders = [
        f"{p}:{i}"
        for p in _SRC.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if _BANNED.search(line)
    ]
    assert offenders == [], "deprecated numpy aliases:\n" + "\n".join(offenders)
