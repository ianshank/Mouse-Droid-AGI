"""Structure pins for ``docs/runbooks/`` (gap-analysis follow-up).

Light-touch analog of ``test_playbooks_structure.py``: runbooks legitimately
vary in internal structure (unlike the templated playbooks), so this pins
only (a) the inventory — a rename/delete of an operator runbook must be a
conscious red-test decision, because systemd units and other docs point at
these paths by name — and (b) minimal hygiene: non-empty with an H1 title.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RUNBOOKS_DIR = Path(__file__).resolve().parents[2] / "docs" / "runbooks"

# The operator-facing inventory. Additions are free (directory sweep below);
# removals/renames must update this list consciously - external pointers
# (mousedroid-trend.service Documentation=, NEXT_STEPS, C4 docs) use these names.
_PINNED_RUNBOOKS = (
    "claude-code-on-jetson.md",
    "claude-workforce-hooks.md",
    "history-purge.md",
    "jetson-alayaworld-spike.md",
    "jetson-claude-pilot-deploy.md",
    "jetson-full-bringup.md",
    "jetson-full-validation.md",
    "jetson-on-device-learning.md",
    "jetson-rover-smoke.md",
    "mlflow-local-ui.md",
    "secret-scanning.md",
)


@pytest.mark.parametrize("name", _PINNED_RUNBOOKS)
def test_pinned_runbook_exists(name: str) -> None:
    assert (_RUNBOOKS_DIR / name).is_file(), (
        f"runbook {name} missing - external pointers (systemd Documentation=, "
        "NEXT_STEPS, C4 docs) reference it by name"
    )


def test_every_runbook_is_nonempty_with_h1_title() -> None:
    runbooks = sorted(_RUNBOOKS_DIR.glob("*.md"))
    assert runbooks, "docs/runbooks/ unexpectedly empty"
    for runbook in runbooks:
        text = runbook.read_text(encoding="utf-8").strip()
        assert text, f"{runbook.name} is empty"
        assert text.startswith("# "), f"{runbook.name} must open with an H1 title"


def test_trend_service_documentation_pointer_resolves() -> None:
    """mousedroid-trend.service points operators at a runbook - keep it real."""
    service = _RUNBOOKS_DIR.parents[1] / "scripts" / "mousedroid-trend.service"
    text = service.read_text(encoding="utf-8")
    referenced = [name for name in _PINNED_RUNBOOKS if name in text]
    assert referenced, "the trend service must reference a pinned runbook"
    for name in referenced:
        assert (_RUNBOOKS_DIR / name).is_file()
