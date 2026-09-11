"""Security: the optional [isaac] extra must not ship secrets."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN = ("api_key", "password", "secret", "token", "credential")


def test_isaac_extra_has_no_secret_literals() -> None:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    start = text.find("isaac = [")
    assert start != -1
    end = text.find("]", start)
    block = text[start:end].lower()
    for needle in _FORBIDDEN:
        assert needle not in block, f"forbidden token {needle!r} in [isaac] extra"
