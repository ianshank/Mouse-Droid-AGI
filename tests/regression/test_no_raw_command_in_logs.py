"""Regression: the OpenClaw mission dispatcher must never log raw nl_command.

Greps the dispatcher source for any structlog call site that passes
``nl_command=`` as a structured field. The dispatcher's contract is to
log only ``command_hash`` (sha256 prefix) so user-supplied mission text
is kept out of the log buffer (defence-in-depth against accidental
token / PII leakage through OpenClaw channels).

Scoped to the new modules added in this PR — not the wider codebase —
so existing pre-OpenClaw call sites that legitimately log NL commands
(e.g. the LLM gateway) are not perturbed.
"""

from __future__ import annotations

import re
from pathlib import Path

# Files under inspection. Listed explicitly so the regression cannot
# accidentally pass by missing a new module — if a future module is
# added that also handles channel-driven NL ingress, it must be added
# here too.
_GUARDED_PATHS = (
    "src/mousedroid/orchestrator/mission_dispatcher.py",
    "src/mousedroid/security/injection_filter.py",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_dispatcher_never_logs_raw_nl_command() -> None:
    pattern = re.compile(r"_log\.(info|warning|debug|error)\([^)]*nl_command\s*=", re.DOTALL)
    offenders: list[tuple[str, int, str]] = []
    for rel in _GUARDED_PATHS:
        path = _project_root() / rel
        assert path.exists(), f"guarded path missing: {rel}"
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append((rel, line, match.group(0)))
    assert not offenders, (
        "Found raw nl_command logged in guarded files: "
        f"{offenders}. Use command_hash (sha256 prefix) instead."
    )
