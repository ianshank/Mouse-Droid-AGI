"""Structural regression tests for ``docs/playbooks/*``.

The four Phase 4 failure-mode playbooks (esp32, gpio, replay, bringup) plus
the Phase 11 ``promtool-install.md`` follow a shared three-section contract
so operators can navigate them consistently. The pre-existing
``camera-fail.md`` / ``lidar-fail.md`` / ``voice-fail.md`` predate this
contract and use slightly different heading text — they are intentionally
exempt to avoid retrofitting unrelated docs in this branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAYBOOKS_DIR = _REPO_ROOT / "docs" / "playbooks"

_REQUIRED_SECTIONS: tuple[str, ...] = (
    "## What This Covers",
    "## First Checks",
    "## Cross-Reference",
)

# Phase 4 deliverables — pin their presence so a future PR can't silently
# drop one of the four new failure-mode runbooks. The pinned set ALSO
# defines the exact files that must obey the section contract; older
# playbooks (camera, lidar, voice) are exempt by design.
_NEW_PLAYBOOKS_PINNED: tuple[str, ...] = (
    "esp32-fail",
    "gpio-fail",
    "replay-fail",
    "bringup-fail",
    # promtool-install.md uses the same contract (Phase 11), keep it pinned
    # so it can't drift either.
    "promtool-install",
)


def _structured_playbooks() -> list[Path]:
    """Return only the playbooks that follow the Phase 4 section contract."""
    return [
        _PLAYBOOKS_DIR / f"{name}.md"
        for name in _NEW_PLAYBOOKS_PINNED
        if (_PLAYBOOKS_DIR / f"{name}.md").is_file()
    ]


@pytest.mark.parametrize("path", _structured_playbooks(), ids=lambda p: p.name)
def test_structured_playbook_has_required_sections(path: Path) -> None:
    """Phase 4 + promtool-install playbooks obey the navigation contract."""
    content = path.read_text(encoding="utf-8")
    missing = [section for section in _REQUIRED_SECTIONS if section not in content]
    assert not missing, f"{path.name} missing sections: {missing}"


@pytest.mark.parametrize("name", _NEW_PLAYBOOKS_PINNED)
def test_pinned_playbook_present(name: str) -> None:
    """Pin existence of the structured playbooks so future PRs can't drop them."""
    assert (_PLAYBOOKS_DIR / f"{name}.md").is_file(), (
        f"{name}.md missing — Phase 4 / Phase 11 deliverable"
    )


def test_playbooks_directory_not_empty() -> None:
    """Sanity: the structural tests above are not vacuously passing."""
    assert sorted(_PLAYBOOKS_DIR.glob("*.md")), "docs/playbooks/ has no .md files at all"
