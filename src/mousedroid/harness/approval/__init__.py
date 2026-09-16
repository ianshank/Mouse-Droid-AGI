"""Approval gate implementations for the harness.

The two names below were declared in ``__all__`` without being imported, so
``from mousedroid.harness.approval import *`` raised ``AttributeError`` — a
package whose advertised facade did not exist. This is incident 6 in
``.claude/skills/module-split-consistency-sweep/SKILL.md``, recurring in a
package the ADR-017 facade fix did not cover.

Binding them (rather than dropping ``__all__``) keeps the declaration honest and
is additive in both directions: every existing consumer imports the submodule
path directly and is unaffected, while the star-import and the attribute access
the facade always claimed to support now work. Neither module pulls a heavy
dependency — both reach only ``config.schema``, ``logging.setup`` and
``security.injection_filter`` — so the eager import costs nothing a caller of
this package was not already paying.
"""

from __future__ import annotations

from mousedroid.harness.approval.openclaw_gate import OpenClawSafetyGate
from mousedroid.harness.approval.sandbox_gate import SandboxPolicyGate

__all__ = [
    "OpenClawSafetyGate",
    "SandboxPolicyGate",
]
