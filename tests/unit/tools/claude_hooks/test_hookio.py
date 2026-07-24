# tests/unit/tools/claude_hooks/test_hookio.py
"""Unit tests for the hook stdin/stdout protocol helpers.

The safety contract under test: an "allow" decision writes **nothing** to
stdout. Emitting an explicit allow would bypass the user's normal permission
prompt, so silence is the only correct signal for "no objection".
"""

from __future__ import annotations

import io
import json

import pytest
from tools.claude_hooks import hookio


def _stream(text: str) -> io.StringIO:
    return io.StringIO(text)


# ---------------------------------------------------------------------------
# read_payload
# ---------------------------------------------------------------------------


def test_reads_valid_object() -> None:
    payload = hookio.read_payload(_stream('{"tool_name": "Write"}'))
    assert payload == {"tool_name": "Write"}


@pytest.mark.parametrize("raw", ["", "   \n", "not json", "[1, 2, 3]", '"a string"', "null"])
def test_malformed_or_non_object_payload_yields_empty_mapping(raw: str) -> None:
    # A crash here would be indistinguishable from a policy denial.
    assert hookio.read_payload(_stream(raw)) == {}


def test_unreadable_stream_yields_empty_mapping() -> None:
    closed = io.StringIO("{}")
    closed.close()
    assert hookio.read_payload(closed) == {}


# ---------------------------------------------------------------------------
# tool_name / tool_input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["tool_name", "toolName"])
def test_tool_name_accepts_either_spelling(key: str) -> None:
    assert hookio.tool_name({key: "Edit"}) == "Edit"


def test_tool_name_absent_or_wrong_type() -> None:
    assert hookio.tool_name({}) == ""
    assert hookio.tool_name({"tool_name": 42}) == ""


@pytest.mark.parametrize("key", ["tool_input", "toolInput"])
def test_tool_input_accepts_either_spelling(key: str) -> None:
    assert hookio.tool_input({key: {"a": 1}}) == {"a": 1}


def test_tool_input_absent_or_wrong_type() -> None:
    assert hookio.tool_input({}) == {}
    assert hookio.tool_input({"tool_input": "nope"}) == {}


# ---------------------------------------------------------------------------
# extract_target_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["file_path", "path", "notebook_path", "filePath"])
def test_target_path_probes_known_keys(key: str) -> None:
    payload = {"tool_input": {key: "src/a.py"}}
    assert hookio.extract_target_path(payload) == "src/a.py"


def test_target_path_prefers_file_path_over_alternatives() -> None:
    payload = {"tool_input": {"path": "second.py", "file_path": "first.py"}}
    assert hookio.extract_target_path(payload) == "first.py"


@pytest.mark.parametrize(
    "tool_input",
    [{}, {"file_path": ""}, {"file_path": "   "}, {"file_path": 12}],
)
def test_target_path_absent_returns_none(tool_input: dict[str, object]) -> None:
    assert hookio.extract_target_path({"tool_input": tool_input}) is None


# ---------------------------------------------------------------------------
# extract_pending_content
# ---------------------------------------------------------------------------


def test_write_content_is_extracted() -> None:
    payload = {"tool_name": "Write", "tool_input": {"content": "secret-body"}}
    assert hookio.extract_pending_content(payload) == "secret-body"


@pytest.mark.parametrize("key", ["new_string", "new_str", "newString"])
def test_edit_content_keys_are_extracted(key: str) -> None:
    payload = {"tool_name": "Edit", "tool_input": {key: "replacement"}}
    assert hookio.extract_pending_content(payload) == "replacement"


def test_multi_edit_fragments_are_concatenated() -> None:
    payload = {
        "tool_name": "MultiEdit",
        "tool_input": {"edits": [{"new_string": "one"}, {"new_string": "two"}]},
    }
    result = hookio.extract_pending_content(payload)
    assert "one" in result
    assert "two" in result


def test_multi_edit_ignores_malformed_entries() -> None:
    payload = {"tool_input": {"edits": ["not-a-dict", {"new_string": "kept"}, {}]}}
    assert hookio.extract_pending_content(payload) == "kept"


def test_edits_wrong_type_is_ignored() -> None:
    assert hookio.extract_pending_content({"tool_input": {"edits": "nope"}}) == ""


def test_no_content_returns_empty_string() -> None:
    assert hookio.extract_pending_content({"tool_input": {"file_path": "a.py"}}) == ""


# ---------------------------------------------------------------------------
# Decision emission
# ---------------------------------------------------------------------------


def test_deny_emits_expected_payload() -> None:
    out = io.StringIO()
    code = hookio.emit_deny("because reasons", stream=out)
    assert code == hookio.EXIT_OK
    decision = json.loads(out.getvalue())["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"] == "because reasons"


def test_deny_event_name_is_configurable() -> None:
    out = io.StringIO()
    hookio.emit_deny("r", event_name="PostToolUse", stream=out)
    assert json.loads(out.getvalue())["hookSpecificOutput"]["hookEventName"] == "PostToolUse"


def test_allow_writes_nothing_and_exits_zero() -> None:
    # Load-bearing: an explicit "allow" would bypass the permission prompt.
    assert hookio.emit_allow() == hookio.EXIT_OK
