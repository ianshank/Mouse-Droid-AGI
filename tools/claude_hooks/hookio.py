"""Claude Code hook stdin/stdout protocol helpers.

A hook receives a JSON payload on stdin and communicates its decision on
**stdout**; anything else it wants to say belongs on stderr. This module owns
both directions so the individual hooks stay policy-only.

Two deliberate safety choices:

* **Allow is silent.** Emitting ``permissionDecision: "allow"`` would *bypass*
  the user's normal permission prompt. A hook that merely has no objection must
  therefore exit 0 without writing to stdout, letting the standard flow proceed.
  Only :func:`emit_deny` writes a decision.
* **Field names are probed, not assumed.** Payload shapes differ across tools
  (``Write`` carries ``content``; ``Edit`` carries ``new_string``; ``MultiEdit``
  nests ``edits``) and have varied across releases (``file_path`` vs ``path``),
  so candidate keys are tried in order rather than hardcoded to one spelling.
"""

from __future__ import annotations

import json
import sys
from typing import IO, Any

#: Exit code meaning "hook completed"; the decision (if any) is on stdout.
EXIT_OK = 0

#: Candidate keys holding the edit target path, most specific first.
_PATH_KEYS: tuple[str, ...] = ("file_path", "path", "notebook_path", "filePath")

#: Candidate keys holding pending content for single-edit tools.
_CONTENT_KEYS: tuple[str, ...] = ("content", "new_string", "new_str", "newString")

#: Key holding a list of edits for multi-edit tools.
_EDITS_KEY = "edits"


def read_payload(stream: IO[str] | None = None) -> dict[str, Any]:
    """Read and parse the hook payload from ``stream``.

    Args:
        stream: Input stream. Defaults to :data:`sys.stdin`.

    Returns:
        The parsed payload, or an empty mapping when stdin is empty or does not
        contain a JSON object. A malformed payload must never crash a hook — a
        crash on ``PreToolUse`` is indistinguishable from a policy failure.
    """
    source = sys.stdin if stream is None else stream
    try:
        raw = source.read()
    except (OSError, ValueError):
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_name(payload: dict[str, Any]) -> str:
    """Return the invoking tool's name, or an empty string when absent."""
    value = payload.get("tool_name") or payload.get("toolName") or ""
    return value if isinstance(value, str) else ""


def tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the tool-input mapping, tolerating either spelling or absence."""
    for key in ("tool_input", "toolInput"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def extract_target_path(payload: dict[str, Any]) -> str | None:
    """Return the filesystem path the tool call targets.

    Args:
        payload: The parsed hook payload.

    Returns:
        The first non-empty string found under a known path key, else ``None``.
    """
    data = tool_input(payload)
    for key in _PATH_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def extract_pending_content(payload: dict[str, Any]) -> str:
    """Return the content this tool call would write.

    Handles single-value tools (``Write``/``Edit``) and multi-edit payloads,
    whose fragments are concatenated so one scan covers every hunk.

    Args:
        payload: The parsed hook payload.

    Returns:
        The pending content, or an empty string when the payload carries none.
    """
    data = tool_input(payload)
    fragments: list[str] = []

    for key in _CONTENT_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value:
            fragments.append(value)

    edits = data.get(_EDITS_KEY)
    if isinstance(edits, list):
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            for key in _CONTENT_KEYS:
                value = edit.get(key)
                if isinstance(value, str) and value:
                    fragments.append(value)

    return "\n".join(fragments)


def emit_deny(
    reason: str,
    *,
    event_name: str = "PreToolUse",
    stream: IO[str] | None = None,
) -> int:
    """Write a deny decision to stdout and return the process exit code.

    Args:
        reason: Operator-facing explanation, surfaced by Claude Code.
        event_name: Hook event name echoed back in the payload.
        stream: Output stream. Defaults to :data:`sys.stdout`.

    Returns:
        :data:`EXIT_OK` — the decision itself lives in the JSON payload, so the
        process exits cleanly.
    """
    out = sys.stdout if stream is None else stream
    payload = {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    out.write(json.dumps(payload))
    out.flush()
    return EXIT_OK


def emit_allow() -> int:
    """Return the exit code for "no objection", writing nothing to stdout.

    Writing an explicit ``allow`` would suppress the user's normal permission
    prompt, so silence is the correct and safer signal.
    """
    return EXIT_OK
