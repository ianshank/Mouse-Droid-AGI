"""Claude Code hook implementations and their shared, reusable primitives.

This package is deliberately standalone: it never imports the ``mousedroid``
runtime package, so an edit-time hook cannot be slowed down (or broken) by the
robot runtime's heavy dependency graph (torch, faiss, lmdb).

Layout:

* :mod:`tools.claude_hooks.paths` — repo-root resolution + glob matching.
* :mod:`tools.claude_hooks.logging_setup` — structured logging to **stderr**
  (stdout is reserved for the hook protocol), structlog when available with a
  dependency-free fallback.
* :mod:`tools.claude_hooks.config` — :class:`WorkforceConfig`, the single
  source of truth for every threshold, gate key, glob and budget.
* :mod:`tools.claude_hooks.hookio` — hook stdin/stdout protocol helpers.
* :mod:`tools.claude_hooks.secret_scan` — PreToolUse edit-time secret scan.
* :mod:`tools.claude_hooks.freeze_gate` — PreToolUse capability freeze gate.
* :mod:`tools.claude_hooks.post_edit_check` — PostToolUse advisory checks.

Every module is import-safe (no side effects at import time) so the AQA
regression test can introspect them without running a hook.
"""

from __future__ import annotations

from tools.claude_hooks.config import (
    AgentsConfig,
    CoverageConfig,
    DocsConfig,
    EvidenceConfig,
    FreezeConfig,
    PostEditConfig,
    SecretScanConfig,
    WorkforceConfig,
    WorktreeConfig,
    load_config,
)
from tools.claude_hooks.logging_setup import get_logger
from tools.claude_hooks.paths import (
    path_matches_any,
    resolve_repo_root,
    to_repo_relative,
)

__all__ = [
    "AgentsConfig",
    "CoverageConfig",
    "DocsConfig",
    "EvidenceConfig",
    "FreezeConfig",
    "PostEditConfig",
    "SecretScanConfig",
    "WorkforceConfig",
    "WorktreeConfig",
    "get_logger",
    "load_config",
    "path_matches_any",
    "resolve_repo_root",
    "to_repo_relative",
]
