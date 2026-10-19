"""F-059 WS-B / WS-D: ``tools/agent_reviewer`` - the read-only Agent-SDK peer reviewer.

The reviewer is a cloud agent that spends real money and reads this repository,
so what is pinned here is the *containment*, not the review quality:

* the tool allowlist / denylist, checked against ``MCPConfig`` and the literal
  actuation names - no actuation tool, no write tool, no web tool, ever;
* the PreToolUse Bash guard - only ``git diff|log|show`` and
  ``python -m pytest ... --collect-only`` get through, and the options that make
  those read-only commands write or read outside the repo do not;
* the MCP config - built from ``.mcp.json``, mock hardware forced, nothing else
  of the file (the ``github`` server, any rover host, any ``${...}`` pull from the
  parent environment) carried over;
* the prompt builder - front-matter stripped, house output shape appended, and a
  refusal for ``*.env`` / ``docker.env*`` files or key-shaped text;
* the ledger - the cap is checked BEFORE the SDK is touched, the cost comes from
  ``ResultMessage.total_cost_usd``, and a run whose cost is unknown is charged the
  worst case rather than zero;
* ``--dry-run`` - builds everything, constructs no client, touches no network.

Everything runs against a temporary git repository and a fake ``claude_agent_sdk``
module injected into ``sys.modules`` - no network, no API key, no credit. The few
tests that read the REAL SDK's surface (option names, ``ResultMessage`` fields)
``importorskip`` it: they are the pin that the names used here exist, and they run
wherever the ``agent_reviewer`` extra is installed.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
import re
import shutil
import subprocess
import sys
import types
import typing
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from tools import credit_ledger
from tools.agent_reviewer import ledger as reviewer_ledger
from tools.agent_reviewer import policy, prompt, run_review
from tools.claude_hooks.config import DEFAULT_CONFIG_RELPATH, WorkforceConfig, load_config
from tools.claude_hooks.secret_scan import ScanOutcome

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = _REPO_ROOT / "tools" / "agent_reviewer"

# Built by concatenation so this file never contains a key-shaped literal (the
# repo's secret scan reads test files too).
_FAKE_KEY = "sk-ant-" + "api03-" + "Z" * 40

_WORKFORCE_YAML = """\
credit:
    caps_usd:
        A: 10
        B: 60
        C: 15
    ledger: reports/credit/ledger.jsonl
agent_reviewer:
    model: opus
    max_turns: 40
"""


# --------------------------------------------------------------------------- #
# Fixtures and fakes
# --------------------------------------------------------------------------- #
def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _write(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


@pytest.fixture(scope="session")
def _template_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A git repo (two commits) holding copies of the reviewer's real inputs.

    Built once per session; each test gets its own copy (``git init`` is the slow part).
    """
    root = tmp_path_factory.mktemp("template") / "repo"
    root.mkdir()
    for rel in (".claude/agents/peer-reviewer.md", ".mcp.json", ".gitleaks.toml"):
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO_ROOT / rel, dest)
    _write(root, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML)
    _git(root, "init", "-q")
    _commit(root, "base")
    _write(root, "src_change.py", "x = 1\n")
    _commit(root, "head")
    return root


@pytest.fixture
def repo(_template_repo: Path, tmp_path: Path) -> Path:
    """A private, writable copy of the template repo."""
    root = tmp_path / "repo"
    shutil.copytree(_template_repo, root)
    return root


@pytest.fixture
def cfg(repo: Path) -> WorkforceConfig:
    return load_config(repo_root=repo)


class _FakeSdk(types.ModuleType):
    """A stand-in ``claude_agent_sdk`` that records calls and replays canned messages."""

    def __init__(
        self,
        messages: list[Any] | None = None,
        raise_after: BaseException | None = None,
    ) -> None:
        super().__init__("claude_agent_sdk")
        self.__version__ = "0.0.0-fake"
        self.queries: list[dict[str, Any]] = []
        self.messages = messages
        self.raise_after = raise_after

        class ClaudeAgentOptions:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        @dataclasses.dataclass
        class HookMatcher:
            matcher: str | None = None
            hooks: list[Any] = dataclasses.field(default_factory=list)
            timeout: float | None = None

        @dataclasses.dataclass
        class TextBlock:
            text: str

        @dataclasses.dataclass
        class AssistantMessage:
            content: list[Any]
            model: str = "claude-fake-1"

        @dataclasses.dataclass
        class ResultMessage:
            subtype: str = "success"
            duration_ms: int = 10
            duration_api_ms: int = 5
            is_error: bool = False
            num_turns: int = 3
            session_id: str = "s"
            stop_reason: str | None = "end_turn"
            total_cost_usd: float | None = 1.25
            usage: dict[str, Any] | None = None
            result: str | None = "REVIEW BODY"
            model_usage: dict[str, Any] | None = None
            terminal_reason: str | None = "completed"

        self.ClaudeAgentOptions = ClaudeAgentOptions
        self.HookMatcher = HookMatcher
        self.TextBlock = TextBlock
        self.AssistantMessage = AssistantMessage
        self.ResultMessage = ResultMessage
        self.ClaudeSDKClient = MagicMock(name="ClaudeSDKClient")
        self.query = MagicMock(name="query", side_effect=self._query)

    def _query(self, *, prompt: str, options: Any = None, transport: Any = None) -> Any:
        self.queries.append({"prompt": prompt, "options": options})
        messages = self.messages
        if messages is None:
            messages = [self.AssistantMessage([self.TextBlock("working")]), self.ResultMessage()]
        raise_after = self.raise_after

        async def stream() -> Any:
            for message in messages:
                yield message
            if raise_after is not None:
                raise raise_after

        return stream()


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeSdk:
    sdk = _FakeSdk()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    return sdk


def _install_sdk(monkeypatch: pytest.MonkeyPatch, sdk: _FakeSdk) -> _FakeSdk:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    return sdk


class _FakeAnthropic(types.ModuleType):
    """A stand-in ``anthropic`` whose ``models.retrieve`` is scripted."""

    def __init__(
        self,
        model_id: Any = "claude-resolved-1",
        error: Exception | None = None,
        delay_s: float = 0.0,
        close: str = "async",
    ) -> None:
        super().__init__("anthropic")
        self.constructed: list[dict[str, Any]] = []
        self.retrieved: list[str] = []
        outer = self

        class _Models:
            async def retrieve(self, model: str) -> Any:
                outer.retrieved.append(model)
                if delay_s:
                    await asyncio.sleep(delay_s)
                if error is not None:
                    raise error
                return SimpleNamespace(id=model_id)

        class AsyncAnthropic:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                outer.constructed.append({"args": args, "kwargs": kwargs})
                self.models = _Models()

        async def async_close(_self: Any) -> None:
            return None

        async def failing_close(_self: Any) -> None:
            raise RuntimeError("socket already closed")

        def sync_close(_self: Any) -> None:
            return None

        closers = {"async": async_close, "raises": failing_close, "sync": sync_close}
        if close in closers:  # "missing": the client has no close() at all
            AsyncAnthropic.close = closers[close]
        self.AsyncAnthropic = AsyncAnthropic


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees a real key or a real Anthropic SDK unless it installs a fake."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)  # `import anthropic` -> ImportError


def _no_scanner(_text: str) -> ScanOutcome:
    return ScanOutcome(clean=True, available=False, detail="gitleaks not found")


def _argv(repo: Path, *extra: str, cap: str = "10") -> list[str]:
    return [
        "--repo-root",
        str(repo),
        "--base",
        "HEAD~1",
        "--head",
        "HEAD",
        "--cap-usd",
        cap,
        *extra,
    ]


def _live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)


def _ledger_rows(repo: Path) -> list[dict[str, Any]]:
    path = repo / "reports/credit/ledger.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# policy - the allowlist and the denylist
# --------------------------------------------------------------------------- #
def test_allowlist_is_the_planned_set() -> None:
    """The allowlist is pinned literally: widening it is a reviewed edit of this test."""
    assert policy.ALLOWED == (
        "Read",
        "Grep",
        "Glob",
        "Bash(git diff *)",
        "Bash(git log *)",
        "Bash(git show *)",
        "Bash(python -m pytest * --collect-only*)",
        "mcp__mousedroid__system_info",
        "mcp__mousedroid__lidar_diagnostics",
        "mcp__mousedroid__esp32_diagnostics",
    )
    assert policy.DENIED == (
        "Write",
        "Edit",
        "MultiEdit",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
    )


def test_allowlist_has_no_actuation_or_write_tool() -> None:
    """WS-D: no ``MCPConfig.actuation_tools`` name, no write tool, no web tool."""
    from mousedroid.config.schema.harness_mcp import MCPConfig

    actuation = set(MCPConfig().actuation_tools) | {
        "set_velocity",
        "emergency_stop",
        "translate_nl_mission",
        "export_experience",
        "tensorrt_compile",
        "calibrate_ultrasonic",
    }
    forbidden = actuation | set(policy.DENIED)
    for rule in policy.ALLOWED:
        name = rule.split("__")[-1] if rule.startswith("mcp__") else rule.split("(")[0]
        assert name not in forbidden, f"{rule!r} grants a forbidden tool"
    # The built-in tools the session may even see are the read-only four.
    assert set(policy.BASE_TOOLS) == {"Read", "Grep", "Glob", "Bash"}
    assert not set(policy.BASE_TOOLS) & forbidden


def test_policy_actuation_list_covers_the_server_defaults() -> None:
    """A tool added to ``MCPConfig.actuation_tools`` upstream must reach the denylist here."""
    from mousedroid.config.schema.harness_mcp import MCPConfig

    assert set(MCPConfig().actuation_tools) <= set(policy.ACTUATION_TOOLS)


def test_every_mcp_allow_rule_names_the_one_mock_server() -> None:
    mcp_rules = [rule for rule in policy.ALLOWED if rule.startswith("mcp__")]
    assert mcp_rules
    assert all(rule.startswith(f"mcp__{policy.MCP_SERVER}__") for rule in mcp_rules)


def test_disallowed_tools_denies_writes_web_subagents_actuation_and_secret_reads() -> None:
    denied = policy.disallowed_tools()
    for tool in (*policy.DENIED, *policy.SUBAGENT_TOOLS):
        assert tool in denied
    for tool in policy.ACTUATION_TOOLS:
        assert f"mcp__{policy.MCP_SERVER}__{tool}" in denied
    for glob in policy.SECRET_PATH_GLOBS:
        assert f"Read(**/{glob})" in denied
    assert not set(denied) & set(policy.ALLOWED), "a rule cannot be both allowed and denied"


@pytest.mark.parametrize(
    "path",
    [
        "config/docker.env",
        "DOCKER.ENV",
        "a\\b\\docker.env.local",
        "x/prod.env",
        ".env",
        "sub/.env.production",
        "docker.environment.md",
    ],
)
def test_secret_path_reason_matches_by_basename_on_every_platform(path: str) -> None:
    assert policy.secret_path_reason(path)


@pytest.mark.parametrize(
    "path", ["README.md", "config/default.yaml", "src/environment.py", "env/x.py"]
)
def test_secret_path_reason_leaves_ordinary_files_alone(path: str) -> None:
    assert policy.secret_path_reason(path) is None


@pytest.mark.parametrize(
    "text",
    [
        _FAKE_KEY,
        "hf_" + "a" * 34,
        "AKIA" + "A" * 16,
        "ghp_" + "b" * 36,
        "github_pat_" + "c" * 60,
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
    ],
)
def test_credential_shape_reason_flags_real_key_shapes(text: str) -> None:
    reason = policy.credential_shape_reason(f"prefix {text} suffix")
    assert reason
    assert text not in reason, "the reason never echoes the secret"


@pytest.mark.parametrize(
    "text", ["sk-ant-...", "sk-ant-xyz", "sk-ant-test", "hf_token", "AKIA is a prefix"]
)
def test_credential_shape_reason_leaves_documented_placeholders_alone(text: str) -> None:
    assert policy.credential_shape_reason(text) is None


# --------------------------------------------------------------------------- #
# policy - the Bash guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "command",
    [
        "git diff HEAD~1",
        "git diff origin/main...HEAD --stat",
        "git diff -- tools/agent_reviewer/policy.py",
        "git diff --no-ext-diff --no-textconv HEAD",
        "git log --oneline -n 5",
        'git log --format="%h %s" -n 3',
        "git log -p -S needle -- tools",
        "git show HEAD:README.md",
        "git show --stat abc1234",
        # glob and brace characters are fine when quoted: they are literal then
        "git diff HEAD~1 -- '*.py'",
        'git log -- "tools/**"',
        "git show 'HEAD@{1}'",
        "python -m pytest tests --collect-only -q",
        "python -m pytest tests/unit/tools/test_x.py::test_y --collect-only",
        'python -m pytest tests -k "not slow" --collect-only -q',
        'python -m pytest -m "not hardware" --collect-only tests',
        "python -m pytest tests/unit --co -qq --no-header",
    ],
)
def test_bash_hook_allows_read_only_commands(command: str) -> None:
    assert policy.check_bash_command(command) is None


@pytest.mark.parametrize(
    "command",
    [
        # the commands the brief names
        "rm -rf build",
        "rm x",
        "git push",
        "git push origin main",
        "pytest tests",
        "pytest --collect-only",
        "git diff > f",
        "git log; rm x",
        # other mutations and other tools
        "git commit -m x",
        "git checkout main",
        "git reset --hard",
        "git stash",
        "git status",
        "ls",
        "cat README.md",
        "python -c 'print(1)'",
        "python script.py",
        "python -m pip install requests",
        "python3 -m pytest tests --collect-only",
        # chaining, redirection, substitution
        "git diff >> f",
        "git diff 2>&1",
        "git diff | cat",
        "git diff; rm x",
        "git log && rm x",
        "git log || rm x",
        "git diff & rm x",
        "git diff $(rm x)",
        "git diff `rm x`",
        "git diff\nrm x",
        "git diff\r\nrm x",
        "git diff $HOME",
        "git diff ${HOME}",
        "git diff <(echo x)",
        "git diff 'a\\b'",
        "cd tools && git diff",
        "GIT_EXTERNAL_DIFF=x git diff",
        # unquoted glob / brace characters: the shell would expand them into options
        "git diff *",
        "git diff -- *.py",
        "git diff {--output=x,y}",
        "git diff --{output,stat}=x",
        "git diff [a-z]*",
        "git diff ?",
        "python -m pytest tests/*.py --collect-only",
        # git options that write, run programs, or read outside the repo
        "git diff --output=out.txt",
        "git diff --output out.txt",
        "git log --output=out.txt",
        "git show --output=out.txt HEAD",
        "git diff --no-index a b",
        "git diff --ext-diff",
        "git diff --textconv",
        "git diff -Oorder.txt",
        "git -C other diff",
        "git --no-pager diff",
        "git -c core.pager=x diff",
        "git diff ~/notes",
        "git diff /etc/passwd",
        "git show C:/Users/x/file",
        # pytest options that load code, write files or leave the repo
        "python -m pytest tests",
        "python -m pytest tests -q",
        "python -m pytest -p plugin --collect-only",
        "python -m pytest --junitxml=r.xml --collect-only",
        "python -m pytest --basetemp=x --collect-only",
        "python -m pytest -c cfg.ini --collect-only",
        "python -m pytest -o addopts= --collect-only",
        "python -m pytest tests -k --collect-only",
        "python -m pytest tests --collect-only -k",
        "python -m pytest tests -k -p --collect-only",
        "python -m pytest tests -m --junitxml=r.xml --collect-only",
        "python -m pytest --rootdir=/ --collect-only",
        "python -m pytest ../elsewhere --collect-only",
        "python -m pytest /abs/tests --collect-only",
        "python -m pytest tests/../../x --collect-only",
        # secret-bearing paths
        "git show HEAD:config/docker.env",
        "git show HEAD:.env",
        "git diff -- prod.env",
        "git log -p -- docker.env.local",
        # unparseable / empty
        "git log --grep='unterminated",
        "",
        "   ",
    ],
)
def test_bash_hook_denies_non_readonly(command: str) -> None:
    reason = policy.check_bash_command(command)
    assert reason, f"{command!r} must be denied"


@pytest.mark.parametrize("command", [None, 123, ["git", "diff"], {"command": "git diff"}])
def test_bash_guard_denies_a_missing_or_non_string_command(command: object) -> None:
    assert policy.check_bash_command(command)


def test_bash_guard_denies_an_oversized_command() -> None:
    assert policy.check_bash_command("git log " + "a" * 100_000)


async def test_bash_hook_returns_a_deny_payload_and_never_an_allow() -> None:
    denied = await policy.deny_non_readonly_bash(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, "tu1", {"signal": None}
    )
    output = denied["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"]

    # "Allow is silent": an explicit allow would bypass the permission rules.
    allowed = await policy.deny_non_readonly_bash(
        {"tool_name": "Bash", "tool_input": {"command": "git diff HEAD~1"}}, "tu2", {"signal": None}
    )
    assert allowed == {}


async def test_bash_hook_fails_closed_on_malformed_input() -> None:
    bad_inputs: list[Any] = [
        None,
        {},
        {"tool_name": "Bash"},
        {"tool_name": "Bash", "tool_input": None},
        {"tool_name": "Bash", "tool_input": "git diff"},
    ]
    for bad in bad_inputs:
        result = await policy.deny_non_readonly_bash(bad, None, None)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny", bad


async def test_bash_hook_fails_closed_when_the_guard_itself_crashes() -> None:
    class Exploding(dict[str, Any]):
        def get(self, *_args: Any) -> Any:
            raise RuntimeError("boom")

    result = await policy.deny_non_readonly_bash(
        Exploding(tool_name="Bash", tool_input={"command": "git diff"}), "tu", {"signal": None}
    )
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "RuntimeError" in output["permissionDecisionReason"]


async def test_bash_hook_leaves_other_tools_to_the_permission_rules() -> None:
    result = await policy.deny_non_readonly_bash(
        {"tool_name": "Read", "tool_input": {"file_path": "README.md"}}, "tu", {"signal": None}
    )
    assert result == {}


# --------------------------------------------------------------------------- #
# policy - the MCP server config
# --------------------------------------------------------------------------- #
def test_mcp_config_forces_mock_hardware(repo: Path, cfg: WorkforceConfig) -> None:
    servers = policy.build_mcp_servers(repo, cfg.agent_reviewer)
    assert list(servers) == [policy.MCP_SERVER], "only the mock hardware server, never `github`"
    server = servers[policy.MCP_SERVER]
    assert server["type"] == "stdio"
    assert server["env"]["MOUSEDROID_MOCK_HARDWARE"] == "true"
    assert server["env"]["MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS"] == "false"
    assert not any(key in server for key in ("url", "headers", "host"))
    json.dumps(servers)  # the SDK passes it to the CLI as JSON


def test_mcp_config_matches_the_checked_in_mcp_json(repo: Path, cfg: WorkforceConfig) -> None:
    declared = json.loads((repo / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"][
        "mousedroid"
    ]
    server = policy.build_mcp_servers(repo, cfg.agent_reviewer)["mousedroid"]
    assert server["command"] == declared["command"]
    assert server["args"] == declared["args"]
    for key, value in declared["env"].items():
        if key.startswith("MOUSEDROID_MCP__") and key not in policy.FORCED_MCP_ENV:
            assert server["env"][key] == value


def test_mcp_config_overrides_a_hostile_mcp_json(tmp_path: Path) -> None:
    root = tmp_path / "r"
    cfg = load_config(repo_root=root)  # no workforce.yaml -> defaults
    body = {
        "mcpServers": {
            "github": {"command": "npx", "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "x"}},
            "mousedroid": {
                "command": "python",
                "args": ["-m", "mousedroid", "--config", "config/default.yaml"],
                "env": {
                    "MOUSEDROID_MOCK_HARDWARE": "false",
                    "MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS": "true",
                    "MOUSEDROID_MCP__TRANSPORT": "stdio",
                    "MOUSEDROID_ESP32__WIFI_HOST": "192.0.2.10",
                    "GITHUB_TOKEN": "t",
                },
            },
        }
    }
    _write(root, ".mcp.json", json.dumps(body))
    server = policy.build_mcp_servers(root, cfg.agent_reviewer)["mousedroid"]
    assert server["env"] == {
        "MOUSEDROID_MCP__TRANSPORT": "stdio",
        "MOUSEDROID_MOCK_HARDWARE": "true",
        "MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS": "false",
    }


@pytest.mark.parametrize("value", ["${ANTHROPIC_API_KEY}", "${HOME:-/tmp}", "x${Y}z"])
def test_mcp_config_refuses_a_value_that_pulls_from_the_parent_environment(
    tmp_path: Path, value: str
) -> None:
    root = tmp_path / "r"
    cfg = load_config(repo_root=root)
    body = {
        "mcpServers": {"mousedroid": {"command": "python", "env": {"MOUSEDROID_MCP__X": value}}}
    }
    _write(root, ".mcp.json", json.dumps(body))
    with pytest.raises(policy.PolicyViolationError):
        policy.build_mcp_servers(root, cfg.agent_reviewer)


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        json.dumps({}),
        json.dumps({"mcpServers": {}}),
        json.dumps({"mcpServers": {"mousedroid": "nope"}}),
        json.dumps({"mcpServers": {"mousedroid": {"args": []}}}),
        json.dumps({"mcpServers": {"mousedroid": {"command": "python", "args": "x"}}}),
        json.dumps({"mcpServers": {"mousedroid": {"command": "python", "args": [1]}}}),
        json.dumps({"mcpServers": {"mousedroid": {"command": "python", "env": {"A": 1}}}}),
        json.dumps({"mcpServers": {"mousedroid": {"command": "python", "env": []}}}),
    ],
)
def test_mcp_config_rejects_a_malformed_mcp_json(tmp_path: Path, body: str) -> None:
    root = tmp_path / "r"
    cfg = load_config(repo_root=root)
    _write(root, ".mcp.json", body)
    with pytest.raises(policy.PolicyViolationError):
        policy.build_mcp_servers(root, cfg.agent_reviewer)


def test_mcp_config_rejects_a_missing_mcp_json(tmp_path: Path) -> None:
    cfg = load_config(repo_root=tmp_path)
    with pytest.raises(policy.PolicyViolationError):
        policy.build_mcp_servers(tmp_path, cfg.agent_reviewer)


# --------------------------------------------------------------------------- #
# policy - the SDK options
# --------------------------------------------------------------------------- #
def test_reviewer_options_are_closed_and_read_only(repo: Path, cfg: WorkforceConfig) -> None:
    options = policy.reviewer_options(
        cfg=cfg.agent_reviewer,
        repo_root=repo,
        model="opus",
        system_prompt="SYS",
        cap_usd=7.5,
    )
    # A closed set: a new SDK option is a deliberate, reviewed addition.
    assert set(options) == {
        "model",
        "system_prompt",
        "tools",
        "allowed_tools",
        "disallowed_tools",
        "permission_mode",
        "max_turns",
        "max_budget_usd",
        "cwd",
        "mcp_servers",
        "strict_mcp_config",
        "setting_sources",
        "env",
        "verbatim_prompts",
    }
    assert options["permission_mode"] == "dontAsk"
    assert options["tools"] == list(policy.BASE_TOOLS)
    assert options["allowed_tools"] == list(policy.ALLOWED)
    assert options["disallowed_tools"] == policy.disallowed_tools()
    assert options["max_turns"] == cfg.agent_reviewer.max_turns
    assert options["max_budget_usd"] == 7.5
    assert options["cwd"] == str(repo)
    assert options["strict_mcp_config"] is True
    assert options["setting_sources"] == [], "no user/project settings: the allowlist is the policy"
    assert options["verbatim_prompts"] is True
    assert options["system_prompt"] == "SYS"
    assert options["model"] == "opus"
    assert options["env"]["MOUSEDROID_MOCK_HARDWARE"] == "true"
    assert options["env"]["PYTEST_ADDOPTS"] == "-p no:cacheprovider"
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert list(options["mcp_servers"]) == [policy.MCP_SERVER]


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf")])
def test_reviewer_options_refuse_an_unbounded_budget(
    repo: Path, cfg: WorkforceConfig, cap: float
) -> None:
    with pytest.raises(policy.PolicyViolationError):
        policy.reviewer_options(
            cfg=cfg.agent_reviewer, repo_root=repo, model="opus", system_prompt="s", cap_usd=cap
        )


def test_options_names_exist_in_the_real_sdk(repo: Path, cfg: WorkforceConfig) -> None:
    """The pin that every option name used here is a real ``ClaudeAgentOptions`` field."""
    sdk = pytest.importorskip("claude_agent_sdk")
    options = policy.reviewer_options(
        cfg=cfg.agent_reviewer, repo_root=repo, model="opus", system_prompt="s", cap_usd=5.0
    )
    fields = {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
    assert set(options) | {"hooks"} <= fields
    matcher = sdk.HookMatcher(
        matcher=policy.BASH_HOOK_MATCHER, hooks=[policy.deny_non_readonly_bash]
    )
    built = sdk.ClaudeAgentOptions(**options, hooks={"PreToolUse": [matcher]})
    assert built.permission_mode in typing.get_args(sdk.PermissionMode)
    assert built.permission_mode == policy.PERMISSION_MODE


def test_the_ledger_reads_real_result_message_fields() -> None:
    sdk = pytest.importorskip("claude_agent_sdk")
    fields = {f.name for f in dataclasses.fields(sdk.ResultMessage)}
    assert {
        "subtype",
        "is_error",
        "num_turns",
        "total_cost_usd",
        "usage",
        "model_usage",
        "result",
        "terminal_reason",
    } <= fields


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
def test_prompt_has_no_frontmatter_and_names_output_shape(repo: Path, cfg: WorkforceConfig) -> None:
    text = prompt.build(repo_root=repo, cfg=cfg, scanner=_no_scanner)
    assert not text.lstrip().startswith("---")
    for key in ("name: peer-reviewer", "tools: Read, Grep, Glob, Bash", "description: >"):
        assert key not in text, "front-matter leaked into the prompt"
    # The agent body is there ...
    assert "adversarial peer reviewer" in text
    # ... and so is the house output shape, last, with its sections in order.
    shape = text[text.index("House output shape") :]
    assert text.endswith(shape)
    markers = ("Verdict table", "Defects", "Survives", "Load-bearing pins", "Verdict:")
    positions = [shape.index(marker) for marker in markers]
    assert positions == sorted(positions)
    for tag in ("[Certain]", "[Likely]", "[Guessing]"):
        assert tag in text
    for verdict in ("APPROVE", "REQUEST_CHANGES", "NEEDS_CONTEXT"):
        assert verdict in text


def test_prompt_states_when_mcp_diagnostics_count_as_evidence(
    repo: Path, cfg: WorkforceConfig
) -> None:
    text = prompt.build(repo_root=repo, cfg=cfg, scanner=_no_scanner)
    assert "live_driver" in text
    assert "probe" in text
    assert "mock" in text
    assert "unconfigured" in text
    assert "esp32_diagnostics" in text  # the stub the sentence is written around


def test_prompt_states_it_is_unattended(repo: Path, cfg: WorkforceConfig) -> None:
    text = prompt.build(repo_root=repo, cfg=cfg, scanner=_no_scanner)
    assert "non-interactive" in text
    assert "NEEDS_CONTEXT" in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("---\nname: x\ntools: Read\n---\nBody\n", "Body\n"),
        ("---\r\nname: x\r\n---\r\nBody\r\n", "Body\r\n"),
        ("No front matter\n", "No front matter\n"),
        ("---\nname: x\n---\n\n\nBody", "Body"),
    ],
)
def test_strip_front_matter(raw: str, expected: str) -> None:
    assert prompt.strip_front_matter(raw) == expected


def test_strip_front_matter_refuses_an_unterminated_block() -> None:
    with pytest.raises(prompt.PromptRefusedError):
        prompt.strip_front_matter("---\nname: x\nBody with no closing fence\n")


@pytest.mark.parametrize(
    "rel",
    [
        "config/docker.env",
        "config/docker.env.local",
        "docker.env.production",
        ".env",
        ".env.local",
        "deploy/prod.env",
        "a/b/secrets.env",
    ],
)
def test_prompt_builder_refuses_env_files(repo: Path, cfg: WorkforceConfig, rel: str) -> None:
    # An innocuous body: the refusal is by NAME, before the content is even read.
    _write(repo, rel, "harmless = true\n")
    with pytest.raises(prompt.PromptRefusedError, match="secret"):
        prompt.build(repo_root=repo, cfg=cfg, extra_files=[rel], scanner=_no_scanner)
    with pytest.raises(prompt.PromptRefusedError, match="secret"):
        prompt.read_includable(rel, repo_root=repo, cfg=cfg)


def test_prompt_builder_refuses_the_agent_file_when_it_is_an_env_file(repo: Path) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML + "    agent_file: config/docker.env\n")
    _write(repo, "config/docker.env", "---\nname: x\n---\nbody\n")
    bad = load_config(repo_root=repo)
    with pytest.raises(prompt.PromptRefusedError):
        prompt.build(repo_root=repo, cfg=bad, scanner=_no_scanner)


def test_prompt_builder_refuses_key_shaped_text(repo: Path, cfg: WorkforceConfig) -> None:
    _write(repo, "notes/context.md", f"deploy with {_FAKE_KEY} then restart\n")
    with pytest.raises(prompt.PromptRefusedError) as excinfo:
        prompt.build(repo_root=repo, cfg=cfg, extra_files=["notes/context.md"], scanner=_no_scanner)
    assert _FAKE_KEY not in str(excinfo.value), "the refusal must never echo the secret"


@pytest.mark.parametrize(
    "secret",
    [
        "hf_" + "a" * 34,
        "AKIA" + "A" * 16,
        "ghp_" + "b" * 36,
        "-----BEGIN " + "RSA PRIVATE KEY-----",
    ],
)
def test_prompt_builder_refuses_other_credential_shapes(
    repo: Path, cfg: WorkforceConfig, secret: str
) -> None:
    _write(repo, "notes/context.md", f"token {secret}\n")
    with pytest.raises(prompt.PromptRefusedError):
        prompt.build(repo_root=repo, cfg=cfg, extra_files=["notes/context.md"], scanner=_no_scanner)


def test_prompt_builder_refuses_text_matching_a_gitleaks_toml_regex(
    repo: Path, cfg: WorkforceConfig
) -> None:
    """The regexes in ``.gitleaks.toml`` are read at run time, not copied into code."""
    patterns = prompt.gitleaks_regexes(repo, cfg)
    assert patterns, "the repo's .gitleaks.toml declares allowlist regexes"
    assert any(p.search("test-api-key-abc123") for p in patterns)
    _write(repo, "notes/fixture.md", "uses test-api-key-abc123 as a bearer\n")
    with pytest.raises(prompt.PromptRefusedError):
        prompt.build(repo_root=repo, cfg=cfg, extra_files=["notes/fixture.md"], scanner=_no_scanner)


def test_gitleaks_regexes_tolerate_a_missing_or_odd_config(tmp_path: Path) -> None:
    cfg = load_config(repo_root=tmp_path)
    assert prompt.gitleaks_regexes(tmp_path, cfg) == []  # no .gitleaks.toml
    _write(tmp_path, ".gitleaks.toml", "this is = = not toml")
    with pytest.raises(prompt.PromptRefusedError):
        prompt.gitleaks_regexes(tmp_path, cfg)
    _write(tmp_path, ".gitleaks.toml", '[allowlist]\nregexes = ["(unclosed"]\n')
    with pytest.raises(prompt.PromptRefusedError):
        prompt.gitleaks_regexes(tmp_path, cfg)
    _write(tmp_path, ".gitleaks.toml", '[[rules]]\nid = "x"\nregex = "tok_[0-9]+"\n')
    assert [p.pattern for p in prompt.gitleaks_regexes(tmp_path, cfg)] == ["tok_[0-9]+"]
    _write(tmp_path, ".gitleaks.toml", '[[allowlists]]\nregexes = ["fake_[a-z]+", 3]\n')
    assert [p.pattern for p in prompt.gitleaks_regexes(tmp_path, cfg)] == ["fake_[a-z]+"]


def test_prompt_builder_accepts_ordinary_context_files(repo: Path, cfg: WorkforceConfig) -> None:
    _write(repo, "notes/plan.md", "Task 5.2: read-only reviewer.\n")
    text = prompt.build(repo_root=repo, cfg=cfg, extra_files=["notes/plan.md"], scanner=_no_scanner)
    assert "Task 5.2: read-only reviewer." in text


def test_prompt_builder_refuses_paths_outside_the_repo(repo: Path, cfg: WorkforceConfig) -> None:
    outside = repo.parent / "outside.md"
    outside.write_text("hello\n", encoding="utf-8")
    with pytest.raises(prompt.PromptRefusedError):
        prompt.read_includable(outside, repo_root=repo, cfg=cfg)
    with pytest.raises(prompt.PromptRefusedError):
        prompt.read_includable("../outside.md", repo_root=repo, cfg=cfg)


def test_prompt_builder_reports_a_missing_or_undecodable_file(
    repo: Path, cfg: WorkforceConfig
) -> None:
    with pytest.raises(prompt.PromptRefusedError):
        prompt.read_includable("no/such/file.md", repo_root=repo, cfg=cfg)
    (repo / "bin.dat").write_bytes(b"\xff\xfe\x00\x01 not utf-8 \xc3\x28")
    with pytest.raises(prompt.PromptRefusedError):
        prompt.read_includable("bin.dat", repo_root=repo, cfg=cfg)


def test_gitleaks_scanner_findings_refuse_the_prompt(repo: Path, cfg: WorkforceConfig) -> None:
    def dirty(_text: str) -> ScanOutcome:
        return ScanOutcome(clean=False, available=True, detail="1 leak")

    with pytest.raises(prompt.PromptRefusedError, match="gitleaks"):
        prompt.build(repo_root=repo, cfg=cfg, scanner=dirty)


def test_gitleaks_absent_is_tolerated_unless_strict(repo: Path) -> None:
    lenient = load_config(repo_root=repo)
    assert prompt.build(repo_root=repo, cfg=lenient, scanner=_no_scanner)
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML + "secret_scan:\n    strict: true\n")
    strict = load_config(repo_root=repo)
    with pytest.raises(prompt.PromptRefusedError, match="strict"):
        prompt.build(repo_root=repo, cfg=strict, scanner=_no_scanner)


def test_default_scanner_delegates_to_the_workforce_secret_scan(
    repo: Path, cfg: WorkforceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def fake_scan(
        content: str, config: WorkforceConfig, *, repo_root: Path, target: str | None = None
    ) -> ScanOutcome:
        seen.append(repo_root)
        return ScanOutcome(clean=True, available=True)

    monkeypatch.setattr("tools.agent_reviewer.prompt.secret_scan.scan_content", fake_scan)
    assert prompt.build(repo_root=repo, cfg=cfg)
    assert seen == [repo]


def test_user_prompt_pins_shas_and_lists_the_opening_items() -> None:
    text = prompt.build_user_prompt(
        base_ref="origin/main",
        base_sha="a" * 40,
        head_ref="feat/x",
        head_sha="b" * 40,
        worktree_sha="b" * 40,
        opening=("R3-1", "R3-2"),
    )
    assert "a" * 40 in text
    assert "b" * 40 in text
    assert "R3-1" in text
    assert "R3-2" in text
    assert f"git diff {'a' * 40}...{'b' * 40}" in text


def test_user_prompt_says_when_the_working_tree_is_not_the_head() -> None:
    text = prompt.build_user_prompt(
        base_ref="origin/main",
        base_sha="a" * 40,
        head_ref="feat/x",
        head_sha="b" * 40,
        worktree_sha="c" * 40,
        opening=(),
    )
    assert "c" * 40 in text
    assert f"git show {'b' * 40}:" in text
    assert "Opening" not in text


# --------------------------------------------------------------------------- #
# ledger adapter
# --------------------------------------------------------------------------- #
def test_ledger_refuses_over_cap(repo: Path, cfg: WorkforceConfig) -> None:
    ledger_path = repo / "reports/credit/ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({"ws": "B", "cost_usd": 55.0}) + "\n", encoding="utf-8")
    with pytest.raises(credit_ledger.CapExceeded):
        reviewer_ledger.precheck(cfg, 6.0, repo_root=repo)
    # Exactly up to the cap is allowed, and the pre-check never writes.
    before = ledger_path.read_bytes()
    headroom = reviewer_ledger.precheck(cfg, 5.0, repo_root=repo)
    assert headroom.ws == "B"
    assert headroom.remaining_usd == pytest.approx(0.0)
    assert ledger_path.read_bytes() == before


def _result(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "subtype": "success",
        "is_error": False,
        "total_cost_usd": 1.5,
        "usage": None,
        "model_usage": None,
        "terminal_reason": "completed",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_accounting_takes_the_cost_from_result_message_total_cost_usd() -> None:
    acc = reviewer_ledger.account_run(_result(), started=True, cap_usd=10.0, error=None)
    assert acc.cost_usd == 1.5
    assert acc.cost_source == "ResultMessage.total_cost_usd"
    assert acc.status == "ok"


@pytest.mark.parametrize("cost", [None, float("nan"), float("inf"), -1.0, "1.5", True])
def test_accounting_charges_the_worst_case_when_the_cost_is_unknown(cost: object) -> None:
    acc = reviewer_ledger.account_run(
        _result(total_cost_usd=cost), started=True, cap_usd=10.0, error=None
    )
    assert acc.cost_usd == 10.0
    assert acc.cost_source == "cap_assumed"


@pytest.mark.parametrize(
    "result",
    [
        _result(total_cost_usd=0.0, usage={"input_tokens": 5}),
        _result(total_cost_usd=0.0, model_usage={"claude-a": {"outputTokens": 1}}),
        _result(total_cost_usd=0.0, subtype="error_during_execution", is_error=True),
    ],
)
def test_accounting_does_not_believe_a_zero_the_evidence_contradicts(
    result: SimpleNamespace,
) -> None:
    """After a crash the CLI zeroes its cost fields; tokens beside a $0 mean it is unknown."""
    acc = reviewer_ledger.account_run(result, started=True, cap_usd=10.0, error=None)
    assert (acc.cost_usd, acc.cost_source) == (10.0, "cap_assumed")


def test_accounting_believes_a_zero_when_nothing_was_spent() -> None:
    refused = _result(total_cost_usd=0.0, is_error=True, terminal_reason="api_error")
    acc = reviewer_ledger.account_run(refused, started=True, cap_usd=10.0, error=None)
    assert (acc.cost_usd, acc.cost_source) == (0.0, "ResultMessage.total_cost_usd")
    assert acc.status == "error_api"


def test_accounting_charges_nothing_for_a_run_that_never_started() -> None:
    acc = reviewer_ledger.account_run(
        None, started=False, cap_usd=10.0, error=FileNotFoundError("claude")
    )
    assert acc.cost_usd == 0.0
    assert acc.cost_source == "not_started"
    assert acc.status == "error:FileNotFoundError"


def test_accounting_charges_the_worst_case_for_a_started_run_with_no_result() -> None:
    acc = reviewer_ledger.account_run(None, started=True, cap_usd=10.0, error=RuntimeError("boom"))
    assert (acc.cost_usd, acc.cost_source) == (10.0, "cap_assumed")
    assert acc.status == "error:RuntimeError"


def test_accounting_names_an_interrupted_and_a_timed_out_run() -> None:
    interrupted = reviewer_ledger.account_run(
        None, started=True, cap_usd=5.0, error=KeyboardInterrupt()
    )
    assert interrupted.status == "interrupted"
    timed_out = reviewer_ledger.account_run(
        None, started=True, cap_usd=5.0, error=asyncio.TimeoutError()
    )
    assert timed_out.status == "timeout"
    cancelled = reviewer_ledger.account_run(
        None, started=True, cap_usd=5.0, error=asyncio.CancelledError()
    )
    assert cancelled.status == "interrupted"


def test_accounting_without_result_or_error_is_an_error() -> None:
    acc = reviewer_ledger.account_run(None, started=True, cap_usd=5.0, error=None)
    assert acc.status == "error:no_result"
    assert acc.cost_usd == 5.0


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (_result(subtype="error_max_turns", is_error=True), "error_max_turns"),
        (_result(subtype="error_max_budget_usd", is_error=True), "error_max_budget_usd"),
        (_result(subtype="error_during_execution", is_error=True), "error_during_execution"),
        (_result(is_error=True, terminal_reason="api_error"), "error_api"),
        (_result(is_error=True, terminal_reason=None), "error"),
    ],
)
def test_accounting_status_follows_the_result_subtype(result: SimpleNamespace, status: str) -> None:
    acc = reviewer_ledger.account_run(result, started=True, cap_usd=10.0, error=None)
    assert acc.status == status
    assert acc.cost_source == "ResultMessage.total_cost_usd"  # an error result still carries cost


def test_accounting_sums_tokens_across_models_and_prefers_whole_tree_usage() -> None:
    model_usage = {
        "claude-a": {
            "inputTokens": 100,
            "outputTokens": 10,
            "cacheReadInputTokens": 7,
            "cacheCreationInputTokens": 3,
        },
        "claude-b": {"inputTokens": 1, "outputTokens": 2, "cacheReadInputTokens": 3},
    }
    usage = {"input_tokens": 1, "output_tokens": 1}
    acc = reviewer_ledger.account_run(
        _result(model_usage=model_usage, usage=usage), started=True, cap_usd=10.0, error=None
    )
    # cache-creation tokens are newly processed prompt tokens: folded into input.
    assert (acc.input_tokens, acc.output_tokens, acc.cache_read_tokens) == (104, 12, 10)


def test_accounting_falls_back_to_usage_and_tolerates_junk() -> None:
    usage = {
        "input_tokens": 50,
        "output_tokens": 5,
        "cache_read_input_tokens": 9,
        "cache_creation_input_tokens": 1,
    }
    acc = reviewer_ledger.account_run(_result(usage=usage), started=True, cap_usd=1.0, error=None)
    assert (acc.input_tokens, acc.output_tokens, acc.cache_read_tokens) == (51, 5, 9)
    junk = reviewer_ledger.account_run(
        _result(usage={"input_tokens": "x", "output_tokens": -4, "cache_read_input_tokens": True}),
        started=True,
        cap_usd=1.0,
        error=None,
    )
    assert (junk.input_tokens, junk.output_tokens, junk.cache_read_tokens) == (0, 0, 0)
    none = reviewer_ledger.account_run(_result(), started=True, cap_usd=1.0, error=None)
    assert (none.input_tokens, none.output_tokens, none.cache_read_tokens) == (0, 0, 0)
    # model_usage with no usable entry falls through to the main-loop usage
    unusable = reviewer_ledger.account_run(
        _result(model_usage={"claude-a": 5}, usage={"input_tokens": 7}),
        started=True,
        cap_usd=1.0,
        error=None,
    )
    assert (unusable.input_tokens, unusable.output_tokens, unusable.cache_read_tokens) == (7, 0, 0)


def test_record_run_appends_one_row_of_the_planned_shape(repo: Path, cfg: WorkforceConfig) -> None:
    acc = reviewer_ledger.account_run(_result(), started=True, cap_usd=10.0, error=None)
    row = reviewer_ledger.record_run(
        cfg,
        accounting=acc,
        run_id="wsb-0001",
        target="diff HEAD~1...HEAD @abc1234",
        model_requested="opus",
        model_resolved="claude-resolved-1",
        resolved_via="models.retrieve",
        artifact="reports/agent_reviewer/2026-10-19/wsb-0001.md",
        repo_root=repo,
    )
    lines = (repo / "reports/credit/ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == row
    assert list(row) == list(credit_ledger.LEDGER_FIELDS)
    assert row["ws"] == "B"
    assert row["cost_usd"] == 1.5
    assert row["cap_usd"] == 60.0
    assert row["cumulative_ws_usd"] == 1.5
    assert row["cost_source"] == "ResultMessage.total_cost_usd"
    assert row["status"] == "ok"


# --------------------------------------------------------------------------- #
# run_review - the CLI (dry run)
# --------------------------------------------------------------------------- #
def test_dry_run_constructs_no_client(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    anthropic = _FakeAnthropic()
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    _live_env(monkeypatch)

    code = run_review.main(_argv(repo, "--opening", "R3-1,R3-2", "--dry-run"))

    assert code == 0
    fake_sdk.ClaudeSDKClient.assert_not_called()
    fake_sdk.query.assert_not_called()
    assert anthropic.constructed == [], "no Anthropic client either: the lookup is a real-run step"
    assert not (repo / "reports").exists(), "a dry run writes no ledger row and no artifact"

    out = capsys.readouterr()
    text = out.out
    assert "dry-run" in text
    assert "model_requested: opus" in text
    assert "model_resolved: opus" in text
    assert "resolved_via: config" in text
    assert "max_turns: 40" in text
    assert "permission_mode: dontAsk" in text
    assert "cap_usd: 10.0" in text
    for rule in policy.ALLOWED:
        assert rule in text
    for tool in policy.DENIED:
        assert tool in text
    assert re.search(r"system_prompt_chars: \d+", text)
    assert "R3-1" in text
    assert "wsb-0001" in text
    assert "ANTHROPIC_API_KEY: set" in text, "presence only"
    assert _FAKE_KEY not in text + out.err, "a key is never printed"


def test_dry_run_reports_the_sdk_it_validated_against(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "sdk: claude-agent-sdk 0.0.0-fake" in out
    assert "ANTHROPIC_API_KEY: not set" in out


def test_dry_run_works_without_the_sdk_installed(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert "not installed" in capsys.readouterr().out


def test_dry_run_fails_when_the_sdk_rejects_an_option(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sdk = _FakeSdk()

    class Strict:
        def __init__(self, *, model: str) -> None:  # any other option name -> TypeError
            self.model = model

    sdk.ClaudeAgentOptions = Strict
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert "rejected" in capsys.readouterr().err


def test_dry_run_refuses_when_the_real_run_would_exceed_the_cap(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_review.main(_argv(repo, "--dry-run", cap="61"))
    assert code == run_review.EXIT_CAP
    assert "cap" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("cap", ["0", "-1", "nan", "inf", "abc"])
def test_cli_rejects_a_cap_that_does_not_bound_spend(repo: Path, cap: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_review.main(_argv(repo, "--dry-run", cap=cap))
    assert excinfo.value.code == 2


@pytest.mark.parametrize("ref", ["-x", "--output=f", "a b", "a;b", "a$(b)", "a`b`", "", "a\nb"])
def test_cli_rejects_a_ref_that_could_inject_an_option_or_a_command(repo: Path, ref: str) -> None:
    argv = [
        "--repo-root",
        str(repo),
        "--base",
        ref,
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
        "--dry-run",
    ]
    with pytest.raises(SystemExit) as excinfo:
        run_review.main(argv)
    assert excinfo.value.code == 2


def test_cli_rejects_an_unknown_ref(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [
        "--repo-root",
        str(repo),
        "--base",
        "no-such-ref",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
        "--dry-run",
    ]
    assert run_review.main(argv) == run_review.EXIT_REFUSED
    assert "no-such-ref" in capsys.readouterr().err


def test_cli_rejects_identical_base_and_head(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "--repo-root",
        str(repo),
        "--base",
        "HEAD",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
        "--dry-run",
    ]
    assert run_review.main(argv) == run_review.EXIT_REFUSED
    assert "same commit" in capsys.readouterr().err


def test_cli_rejects_an_unknown_head(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--repo-root", str(repo), "--base", "HEAD~1", "--head", "no-such-head"]
    assert run_review.main([*argv, "--cap-usd", "5", "--dry-run"]) == run_review.EXIT_REFUSED
    assert "no-such-head" in capsys.readouterr().err


def test_cli_explains_a_missing_git(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def no_git(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(run_review.asyncio, "create_subprocess_exec", no_git)
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert "git is required" in capsys.readouterr().err


def test_a_hung_git_is_killed_and_the_ref_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    killed: list[bool] = []

    class HungProcess:
        returncode = None

        async def communicate(self) -> Any:
            await asyncio.sleep(30)

        def kill(self) -> None:
            killed.append(True)

        async def wait(self) -> int:
            return 0

    async def fake_exec(*_args: Any, **_kwargs: Any) -> HungProcess:
        return HungProcess()

    monkeypatch.setattr(run_review.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(run_review, "_GIT_TIMEOUT_S", 0.05)
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert killed == [True]
    assert "does not name a commit" in capsys.readouterr().err


def test_a_damaged_ledger_refuses_the_dry_run(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, "reports/credit/ledger.jsonl", "this is not json\n")
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert "ledger" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("opening", ["R3-1,;", "a b", "R3-1,$(x)", "x" * 200, ",,"])
def test_cli_rejects_a_malformed_opening_list(repo: Path, opening: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_review.main(_argv(repo, "--opening", opening, "--dry-run"))
    assert excinfo.value.code == 2


def test_cli_has_no_option_that_takes_a_credential() -> None:
    parser = run_review.build_parser()
    flags = [flag for action in parser._actions for flag in action.option_strings]
    assert flags
    assert not [f for f in flags if re.search(r"key|token|secret|password", f, re.IGNORECASE)], (
        flags
    )


def test_cli_refuses_an_artifact_path_outside_the_repo(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run_review.main(_argv(repo, "--dry-run", "--out", str(tmp_path / "elsewhere.md")))
    assert code == run_review.EXIT_REFUSED
    assert "outside" in capsys.readouterr().err
    assert run_review.main(_argv(repo, "--dry-run", "--out", "../x.md")) == run_review.EXIT_REFUSED


def test_cli_refuses_to_overwrite_an_existing_artifact(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, "reports/mine.md", "keep me\n")
    code = run_review.main(_argv(repo, "--dry-run", "--out", "reports/mine.md"))
    assert code == run_review.EXIT_REFUSED
    assert "exists" in capsys.readouterr().err
    assert (repo / "reports/mine.md").read_text(encoding="utf-8") == "keep me\n"


def test_run_ids_continue_from_the_artifacts_already_written(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "agent_reviewer"
    assert run_review.next_run_id(root) == "wsb-0001"
    _write(tmp_path, "reports/agent_reviewer/2026-10-18/wsb-0001.md", "x")
    _write(tmp_path, "reports/agent_reviewer/2026-10-19/wsb-0007.md", "x")
    _write(tmp_path, "reports/agent_reviewer/2026-10-19/notes.md", "x")
    assert run_review.next_run_id(root) == "wsb-0008"


# --------------------------------------------------------------------------- #
# run_review - the live path (fake SDK)
# --------------------------------------------------------------------------- #
def test_a_live_run_writes_the_artifact_and_one_ledger_row(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sdk = _FakeSdk()
    sdk.messages = [
        sdk.AssistantMessage([sdk.TextBlock("thinking out loud")], model="claude-served-9"),
        sdk.ResultMessage(
            total_cost_usd=2.5,
            usage={"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 50},
            result="## Verdict\nAPPROVE\n",
            model_usage={"claude-served-9": {"inputTokens": 1000, "outputTokens": 200}},
            num_turns=7,
        ),
    ]
    _install_sdk(monkeypatch, sdk)
    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropic(model_id="claude-resolved-1"))
    _live_env(monkeypatch)

    code = run_review.main(_argv(repo, "--opening", "R3-1"))

    assert code == 0
    # The SDK was driven exactly once, with the read-only options.
    assert len(sdk.queries) == 1
    options = sdk.queries[0]["options"]
    assert options.model == "claude-resolved-1"
    assert options.permission_mode == "dontAsk"
    assert options.max_budget_usd == 10.0
    assert options.max_turns == 40
    assert options.setting_sources == []
    (matcher,) = options.hooks["PreToolUse"]
    assert matcher.matcher == "Bash"
    assert matcher.hooks == [policy.deny_non_readonly_bash]
    assert "R3-1" in sdk.queries[0]["prompt"]
    # One ledger row, from ResultMessage.total_cost_usd.
    (row,) = _ledger_rows(repo)
    assert row["ws"] == "B"
    assert row["run_id"] == "wsb-0001"
    assert row["model_requested"] == "opus"
    assert row["model_resolved"] == "claude-resolved-1"
    assert row["resolved_via"] == "models.retrieve"
    assert row["cost_usd"] == 2.5
    assert row["cost_source"] == "ResultMessage.total_cost_usd"
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_tokens"]) == (1000, 200, 0)
    assert row["status"] == "ok"
    assert row["cumulative_ws_usd"] == 2.5
    # The artifact the row points at carries the review and the provenance.
    assert re.fullmatch(r"reports/agent_reviewer/\d{4}-\d{2}-\d{2}/wsb-0001\.md", row["artifact"])
    artifact = repo / row["artifact"]
    body = artifact.read_text(encoding="utf-8")
    assert "APPROVE" in body
    assert "claude-resolved-1" in body
    assert "models.retrieve" in body
    assert "claude-served-9" in body, "the ids the API actually served are recorded"
    assert _git(repo, "rev-parse", "HEAD") in body
    assert _git(repo, "rev-parse", "HEAD~1") in body
    assert "2.5" in body
    assert "\r" not in body
    assert _FAKE_KEY not in body + json.dumps(row)
    assert row["artifact"] in capsys.readouterr().out


def test_a_live_run_passes_the_alias_to_the_cli_when_the_lookup_fails(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    anthropic = _FakeAnthropic(error=RuntimeError("404 not found"))
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == 0
    assert anthropic.retrieved == ["opus"]
    assert fake_sdk.queries[0]["options"].model == "opus"
    (row,) = _ledger_rows(repo)
    assert (row["model_resolved"], row["resolved_via"]) == ("opus", "config")


def test_a_live_run_without_the_anthropic_sdk_uses_the_configured_alias(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_env(monkeypatch)  # `anthropic` is blocked by the autouse fixture
    assert run_review.main(_argv(repo)) == 0
    (row,) = _ledger_rows(repo)
    assert (row["model_resolved"], row["resolved_via"]) == ("opus", "config")


def test_the_api_key_is_never_handed_to_a_client_or_the_options(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    anthropic = _FakeAnthropic()
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == 0
    (constructed,) = anthropic.constructed
    assert "api_key" not in constructed["kwargs"], "the client reads the key from the environment"
    assert _FAKE_KEY not in repr(fake_sdk.queries[0]["options"].__dict__)
    captured = capsys.readouterr()
    assert _FAKE_KEY not in captured.out + captured.err


def test_a_run_without_an_api_key_stops_before_the_sdk_and_the_ledger(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_review.main(_argv(repo)) == run_review.EXIT_REFUSED
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    fake_sdk.query.assert_not_called()
    assert _ledger_rows(repo) == []


def test_a_run_over_the_cap_is_refused_before_any_client_is_touched(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    anthropic = _FakeAnthropic()
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    _live_env(monkeypatch)
    ledger_path = repo / "reports/credit/ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({"ws": "B", "cost_usd": 55.0}) + "\n", encoding="utf-8")

    assert run_review.main(_argv(repo, cap="10")) == run_review.EXIT_CAP

    fake_sdk.query.assert_not_called()
    fake_sdk.ClaudeSDKClient.assert_not_called()
    assert anthropic.constructed == []
    assert anthropic.retrieved == []
    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 1, "no row appended"
    assert "cap" in capsys.readouterr().err.lower()


def test_a_damaged_ledger_stops_a_live_run_before_the_sdk(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(repo, "reports/credit/ledger.jsonl", "this is not json\n")
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_REFUSED
    fake_sdk.query.assert_not_called()
    assert "ledger" in capsys.readouterr().err.lower()


def test_a_run_without_the_sdk_installed_explains_the_extra(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_REFUSED
    assert "agent_reviewer" in capsys.readouterr().err
    assert _ledger_rows(repo) == []


def test_an_error_result_is_recorded_with_its_cost_and_the_exit_code_fails(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _FakeSdk()
    sdk.messages = [
        sdk.AssistantMessage([sdk.TextBlock("partial findings")]),
        sdk.ResultMessage(
            subtype="error_max_turns", is_error=True, total_cost_usd=3.0, result=None, num_turns=40
        ),
    ]
    # query() raises AFTER yielding the final result on an error result.
    sdk.raise_after = RuntimeError("Reached maximum number of turns")
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED

    (row,) = _ledger_rows(repo)
    assert row["status"] == "error_max_turns"
    assert row["cost_usd"] == 3.0
    assert row["cost_source"] == "ResultMessage.total_cost_usd"
    body = (repo / row["artifact"]).read_text(encoding="utf-8")
    assert "error_max_turns" in body
    assert "partial findings" in body


def test_a_crash_before_any_message_costs_nothing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _FakeSdk(messages=[], raise_after=FileNotFoundError("claude cli missing"))
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    (row,) = _ledger_rows(repo)
    assert (row["cost_usd"], row["cost_source"]) == (0.0, "not_started")
    assert row["status"] == "error:FileNotFoundError"


def test_a_crash_mid_run_is_charged_the_worst_case(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _FakeSdk()
    sdk.messages = [sdk.AssistantMessage([sdk.TextBlock("partial analysis")])]
    sdk.raise_after = RuntimeError("process died")
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    (row,) = _ledger_rows(repo)
    assert (row["cost_usd"], row["cost_source"]) == (10.0, "cap_assumed")
    assert "partial analysis" in (repo / row["artifact"]).read_text(encoding="utf-8")


def test_an_interrupt_is_recorded_and_becomes_an_exit_code(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _FakeSdk()
    sdk.messages = [sdk.AssistantMessage([sdk.TextBlock("so far")])]
    sdk.raise_after = KeyboardInterrupt()
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_INTERRUPTED
    (row,) = _ledger_rows(repo)
    assert row["status"] == "interrupted"
    assert row["cost_usd"] == 10.0


def test_a_stalled_run_times_out_and_is_charged_the_worst_case(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Long enough that the first message always lands before the timeout, even on a slow host.
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML + "    run_timeout_s: 0.5\n")
    sdk = _FakeSdk()

    def hang(**_kwargs: Any) -> Any:
        async def stream() -> Any:
            yield sdk.AssistantMessage([sdk.TextBlock("started")])
            await asyncio.sleep(30)

        return stream()

    sdk.query = MagicMock(side_effect=hang)
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    (row,) = _ledger_rows(repo)
    assert row["status"] == "timeout"
    assert (row["cost_usd"], row["cost_source"]) == (10.0, "cap_assumed")


def test_unknown_cost_is_charged_the_cap_not_zero(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _FakeSdk()
    sdk.messages = [sdk.ResultMessage(total_cost_usd=None)]
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == 0
    (row,) = _ledger_rows(repo)
    assert (row["cost_usd"], row["cost_source"]) == (10.0, "cap_assumed")


def test_output_that_looks_like_a_credential_is_withheld(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sdk = _FakeSdk()
    sdk.messages = [sdk.ResultMessage(result=f"found {_FAKE_KEY} in a config")]
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED

    (row,) = _ledger_rows(repo)
    assert row["status"] == "output_withheld"
    assert row["cost_usd"] == 1.25, "the money was spent either way"
    body = (repo / row["artifact"]).read_text(encoding="utf-8")
    assert _FAKE_KEY not in body
    assert "withheld" in body.lower()
    captured = capsys.readouterr()
    assert _FAKE_KEY not in captured.out + captured.err


def test_a_cancelled_run_is_recorded_as_interrupted(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _FakeSdk()
    sdk.messages = [sdk.AssistantMessage([sdk.TextBlock("so far")])]
    sdk.raise_after = asyncio.CancelledError()
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_INTERRUPTED
    (row,) = _ledger_rows(repo)
    assert (row["status"], row["cost_usd"]) == ("interrupted", 10.0)


def test_output_that_gitleaks_flags_is_withheld(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def scan(
        content: str, config: WorkforceConfig, *, repo_root: Path, target: str | None = None
    ) -> ScanOutcome:
        calls.append(content)
        # The prompt is scanned first and is clean; the review output is what gets flagged.
        return ScanOutcome(clean=len(calls) == 1, available=True, detail="1 leak (redacted)")

    monkeypatch.setattr("tools.agent_reviewer.prompt.secret_scan.scan_content", scan)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    assert len(calls) == 2
    (row,) = _ledger_rows(repo)
    assert row["status"] == "output_withheld"
    assert "REVIEW BODY" not in (repo / row["artifact"]).read_text(encoding="utf-8")


def test_a_credential_screen_that_breaks_withholds_the_output(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr("tools.agent_reviewer.run_review._output_refusal", boom)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    (row,) = _ledger_rows(repo)
    assert row["status"] == "output_withheld"
    assert "REVIEW BODY" not in (repo / row["artifact"]).read_text(encoding="utf-8")


def test_an_artifact_that_cannot_be_written_still_records_the_run(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(repo, "reports/agent_reviewer", "a file where the artifact directory should be")
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    (row,) = _ledger_rows(repo)
    assert row["artifact"] == "(not written)"
    assert row["status"] == "ok", "the review ran; only its file is missing"
    assert row["cost_usd"] == 1.25
    assert "could not be written" in capsys.readouterr().err


def test_a_bug_in_rendering_the_artifact_still_records_the_run(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("bad template")

    monkeypatch.setattr("tools.agent_reviewer.run_review._render_artifact", boom)
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    (row,) = _ledger_rows(repo)
    assert row["artifact"] == "(not written)"
    assert "artifact step failed" in capsys.readouterr().err


def test_the_working_tree_head_is_told_to_the_reviewer_when_it_is_not_the_review_head(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(repo, "later.py", "y = 2\n")
    _commit(repo, "later")  # the working tree is now one commit past the head under review
    _live_env(monkeypatch)
    argv = ["--repo-root", str(repo), "--base", "HEAD~2", "--head", "HEAD~1", "--cap-usd", "5"]
    assert run_review.main(argv) == 0
    prompt_text = fake_sdk.queries[0]["prompt"]
    assert _git(repo, "rev-parse", "HEAD") in prompt_text, "the working tree's commit"
    assert f"git show {_git(repo, 'rev-parse', 'HEAD~1')}:" in prompt_text


def test_a_failed_ledger_write_is_loud_and_prints_the_row(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _live_env(monkeypatch)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise credit_ledger.LedgerError("disk full")

    monkeypatch.setattr("tools.agent_reviewer.ledger.credit_ledger.record", boom)
    assert run_review.main(_argv(repo)) == run_review.EXIT_FAILED
    err = capsys.readouterr().err
    assert "LEDGER ROW NOT WRITTEN" in err
    assert "wsb-0001" in err
    assert '"cost_usd"' in err


# --------------------------------------------------------------------------- #
# model resolution
# --------------------------------------------------------------------------- #
async def test_resolve_model_uses_models_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropic(model_id="claude-x-1"))
    resolved = await run_review.resolve_model("opus", timeout_s=5.0)
    assert resolved == ("claude-x-1", "models.retrieve")


@pytest.mark.parametrize("model_id", [None, "", "  ", 7, "--dangerously-skip-permissions", "a b"])
async def test_resolve_model_ignores_an_unusable_answer(
    monkeypatch: pytest.MonkeyPatch, model_id: object
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropic(model_id=model_id))
    assert await run_review.resolve_model("opus", timeout_s=5.0) == ("opus", "config")


async def test_resolve_model_gives_up_after_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropic(delay_s=30.0))
    assert await run_review.resolve_model("opus", timeout_s=0.05) == ("opus", "config")


@pytest.mark.parametrize("close", ["missing", "raises", "sync"])
async def test_resolve_model_does_not_depend_on_closing_the_client_cleanly(
    monkeypatch: pytest.MonkeyPatch, close: str
) -> None:
    monkeypatch.setitem(
        sys.modules, "anthropic", _FakeAnthropic(model_id="claude-x-1", close=close)
    )
    resolved = await run_review.resolve_model("opus", timeout_s=1.0)
    assert resolved == ("claude-x-1", "models.retrieve")


async def test_resolve_model_survives_a_client_that_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeAnthropic()

    class Broken:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("no credentials")

    fake.AsyncAnthropic = Broken
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert await run_review.resolve_model("opus", timeout_s=1.0) == ("opus", "config")


# --------------------------------------------------------------------------- #
# single source of truth, and what the package may import
# --------------------------------------------------------------------------- #
def test_workforce_caps_are_the_single_source(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_turns``, the model alias, the caps and the ledger path all come from YAML."""
    # (1) max_turns and the model alias follow workforce.yaml.
    _write(
        repo,
        DEFAULT_CONFIG_RELPATH,
        "credit:\n    caps_usd:\n        B: 4\n    ledger: out/spend.jsonl\n"
        "agent_reviewer:\n    model: sonnet\n    max_turns: 7\n",
    )
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo, cap="3")) == 0
    options = fake_sdk.queries[0]["options"]
    assert options.max_turns == 7
    assert options.model == "sonnet"
    # (2) the cap and the ledger path are the YAML's, too.
    lines = (repo / "out/spend.jsonl").read_text(encoding="utf-8").splitlines()
    (row,) = [json.loads(line) for line in lines]
    assert row["cap_usd"] == 4.0
    assert not (repo / "reports/credit/ledger.jsonl").exists()
    assert run_review.main(_argv(repo, cap="3")) == run_review.EXIT_CAP, "1.25 + 3 > 4"


def test_the_checked_in_workforce_yaml_states_the_planned_values() -> None:
    cfg = load_config(repo_root=_REPO_ROOT)
    assert cfg.agent_reviewer.max_turns == 40
    assert cfg.agent_reviewer.model == "opus"
    assert cfg.credit.caps_usd["B"] == 60.0


def _package_sources() -> Iterator[Path]:
    yield from sorted(_PACKAGE_DIR.glob("*.py"))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def test_no_cap_turn_count_or_model_literal_lives_in_the_package() -> None:
    """The numbers and the alias are configuration; a literal here would shadow the YAML."""
    sources = list(_package_sources())
    names = {p.name for p in sources}
    assert names >= {"__init__.py", "policy.py", "prompt.py", "ledger.py", "run_review.py"}
    offenders: list[str] = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or id(node) in skip:
                continue
            value = node.value
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value in (40, 60)
            ):
                offenders.append(f"{path.name}:{node.lineno} numeric {value!r}")
            if isinstance(value, str) and re.search(r"claude-(opus|sonnet|haiku)-\d", value):
                offenders.append(f"{path.name}:{node.lineno} model id {value!r}")
            if isinstance(value, str) and value in ("opus", "sonnet", "haiku"):
                offenders.append(f"{path.name}:{node.lineno} model alias {value!r}")
    assert not offenders, offenders


def test_the_package_never_imports_the_runtime_or_the_sdk_at_module_scope() -> None:
    """Importing the reviewer must not need the SDK (CI has none) or ``mousedroid`` (torch)."""
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module scope only; TYPE_CHECKING blocks are nested
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in {"claude_agent_sdk", "anthropic", "mousedroid"}, (
                    path.name,
                    name,
                )


def test_the_package_has_no_assert_and_no_key_literal() -> None:
    """`assert` is stripped under PYTHONOPTIMIZE; the key is read from the environment only."""
    key_shape = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")
    for path in _package_sources():
        source = path.read_text(encoding="utf-8")
        assert not [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assert)], path.name
        assert not key_shape.search(source), path.name
