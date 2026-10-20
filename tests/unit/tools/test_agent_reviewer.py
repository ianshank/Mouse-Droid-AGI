"""F-059 WS-B / WS-D: ``tools/agent_reviewer`` - the read-only Agent-SDK peer reviewer.

The reviewer is a cloud agent that spends real money and reads this repository,
so what is pinned here is the *containment*, not the review quality:

* the tool allowlist / denylist, checked against ``MCPConfig`` and the literal
  actuation names - no actuation tool, no write tool, no web tool, ever;
* the PreToolUse Bash guard - only ``git diff|log|show`` as one plain command gets
  through (no interpreter, no pytest), pathspecs are literal, ``git show <rev>:<path>``
  names only a file tracked at the base ref, and the options that make those
  read-only commands write, run a program, read outside the repo or search every
  ref do not;
* the PreToolUse read guard - ``Read``, ``Grep`` and ``Glob`` resolve their path (symlinks
  and junctions followed) and stay inside the checkout under review; ``.git``, secret-file
  names and escaping or secret-matching globs are refused; it fails closed;
* the MCP entry - built in code from constants (``sys.executable -m mousedroid``, a
  fixed minimal environment, mock hardware forced), with ``.mcp.json`` read from the
  BASE ref only to confirm it; the reviewer's instructions and the files ``git show``
  may name come from the base ref too, the environment added for the CLI is an
  allowlist that never carries a credential, and the run is refused when the server
  would import the reviewed branch's own ``mousedroid``;
* the prompt builder - front-matter stripped, house output shape appended, and a
  refusal for ``*.env`` / ``docker.env*`` files or key-shaped text;
* the ledger - the cap is checked BEFORE the SDK is touched, the cost comes from
  ``ResultMessage.total_cost_usd``, and a run whose cost is unknown is charged the
  worst case rather than zero; the file is the CONTROLLER's (never the head's, never a
  link, never outside its root) and the spend is floored by the base ref's copy;
* what the reviewer's own children never get (credentials), what it refuses before any
  spend (root files named like programs, prompts past ``max_prompt_chars``);
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
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import typing
from collections.abc import Iterator, Mapping
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

# The checkout the reviewer's own code runs from, as the CLI computes it by default (the
# autouse ``controller_root`` fixture swaps in a scratch directory for every test).
_REAL_CONTROLLER_ROOT = run_review._controller_root

# Built by concatenation so this file never contains a key-shaped literal (the
# repo's secret scan reads test files too).
_FAKE_KEY = "sk-ant-" + "api03-" + "Z" * 40

# The commits the Bash guard pins (the resolved base and head of a review) and one full
# commit id it does not: 40 hex digits each.
_BASE_SHA = "a1" * 20
_HEAD_SHA = "b2" * 20
_OTHER_SHA = "c3" * 20
_REVISIONS = frozenset({_BASE_SHA, _HEAD_SHA})

# Where the MCP server's configuration would be materialised: absolute, outside any checkout.
_CONFIG_PATH = Path(tempfile.gettempdir()) / "agent-reviewer-test" / "mcp-config.yaml"

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
    for rel in (
        ".claude/agents/peer-reviewer.md",
        ".mcp.json",
        ".gitleaks.toml",
        policy.MCP_CONFIG_RELPATH,
    ):
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


@pytest.fixture(autouse=True)
def controller_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The checkout holding the spend ledger is a scratch directory, never this repository.

    A test that does not say ``--ledger-root`` lands here, so none of them can write a row
    into the real ledger of the tree they run in.
    """
    root = tmp_path / "controller"
    root.mkdir()
    monkeypatch.setattr(run_review, "_controller_root", lambda: root)
    return root


def _no_scanner(_text: str) -> ScanOutcome:
    return ScanOutcome(clean=True, available=False, detail="gitleaks not found")


def _argv(repo: Path, *extra: str, cap: str = "10") -> list[str]:
    """The command line of a review of ``repo``; its ledger lives in ``repo`` unless overridden."""
    return [
        "--repo-root",
        str(repo),
        "--ledger-root",
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


def _argv_with_base_config(repo: Path, yaml_text: str, *extra: str, cap: str = "10") -> list[str]:
    """Commit ``yaml_text`` as the workforce config of a NEW base, put a change above it.

    The reviewer reads its configuration from the base ref, so a test that wants a different
    configuration has to commit it there.
    """
    _write(repo, DEFAULT_CONFIG_RELPATH, yaml_text)
    _commit(repo, "the base's workforce config")
    # Distinct content each time, so a test may call this more than once on one repository.
    _write(repo, "later_change.py", f"z = {_git(repo, 'rev-list', '--count', 'HEAD')}\n")
    _commit(repo, "the change under review")
    return _argv(repo, *extra, cap=cap)


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
# Files "tracked at the base ref" for the guard tests: the only files `git show
# <rev>:<path>` may name.
_BASE_PATHS = frozenset(
    {"README.md", ".mcp.json", "tools/agent_reviewer/policy.py", "src/mousedroid/__init__.py"}
)


def _pinned(command: str) -> str:
    """Spell out the placeholders: BASESHA and HEADSHA are the pinned commits, OTHERSHA is not."""
    return (
        command.replace("BASESHA", _BASE_SHA)
        .replace("HEADSHA", _HEAD_SHA)
        .replace("OTHERSHA", _OTHER_SHA)
    )


def _check(command: object) -> str | None:
    text = _pinned(command) if isinstance(command, str) else command
    return policy.check_bash_command(text, base_paths=_BASE_PATHS, revisions=_REVISIONS)


@pytest.mark.parametrize(
    "command",
    [
        "git diff HEADSHA~1",
        "git diff BASESHA...HEADSHA --stat",
        "git diff -- tools/agent_reviewer/policy.py",
        "git diff HEADSHA~1 -- tools/agent_reviewer/policy.py README.md",
        "git diff --no-ext-diff --no-textconv HEADSHA",
        "git log --oneline -n5",
        'git log --format="%h %s" -n3',
        "git log -p -Sneedle -- tools",
        "git log -Sa:b HEADSHA",
        "git show HEADSHA:README.md",
        "git show BASESHA:tools/agent_reviewer/policy.py",
        "git show --stat BASESHA",
        "git show HEADSHA~1",
        # a pinned commit, with ancestry suffixes, alone or in a range
        "git show HEADSHA",
        "git show HEADSHA^",
        "git show HEADSHA^2",
        "git show HEADSHA~",
        "git show HEADSHA~3^2~1",
        "git diff BASESHA HEADSHA -- README.md",
        "git log --oneline -n5 BASESHA..HEADSHA",
        'git log --format="%H %s" -n3 BASESHA..HEADSHA',
        "git log -p -Sneedle BASESHA..HEADSHA -- tools",
        "git log --abbrev-commit BASESHA..HEADSHA",
        "git log ^BASESHA HEADSHA",
        "git log BASESHA..",
        "git log ..HEADSHA",
        "git log --stat -n3",
        "git diff -pw HEADSHA",
        "git log -pn3 --oneline",
        "git diff -U3 HEADSHA",
        "git show BASESHA:README.md",
        "git show HEADSHA~2:tools/agent_reviewer/policy.py",
        # no revision at all
        "git diff",
        "git diff --cached --stat",
        "git log -n3",
    ],
)
def test_bash_hook_allows_read_only_commands(command: str) -> None:
    assert _check(command) is None


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
        # pytest in every form: collection imports the head's conftest.py and test modules,
        # so no interpreter is on the list at all (Ruling 49)
        "python -m pytest tests --collect-only -q",
        "python -m pytest tests --collect-only",
        "python3 -m pytest tests --collect-only",
        "python -m pytest @argfile --collect-only",
        "python -m pytest @/etc/hosts --collect-only",
        "python -m pytest -c x.ini --collect-only",
        "python -m pytest -p plugin --collect-only",
        "python -m pytest --rootdir=/ --confcutdir=/ --collect-only",
        "python -m pytest --noconftest tests --collect-only",
        "python -m pytest -o addopts= --collect-only",
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
        # git options that write, run programs, or read outside the repo
        "git diff --output=out.txt",
        "git diff --output out.txt",
        "git log --output=out.txt",
        "git show --output=out.txt HEADSHA",
        "git diff --no-index a b",
        "git diff --ext-diff",
        "git diff --textconv",
        "git diff -Oorder.txt",
        "git diff -O/etc/passwd",
        "git diff -pO/etc/passwd",
        "git log -pOorder.txt",
        "git diff -uO../secret",
        "git show --stat -pO/etc/hosts HEADSHA",
        "git log -SOrange",
        "git -C other diff",
        "git --no-pager diff",
        "git -c core.pager=x diff",
        "git diff ~/notes",
        "git diff /etc/passwd",
        # history-wide forms: every ref, every reflog entry, every glob
        "git log --all",
        "git log --all -p",
        "git log --reflog",
        "git log --branches",
        "git log --remotes",
        "git log --tags",
        "git log --glob=refs/heads/x",
        "git log --walk-reflogs",
        "git log -g",
        "git log --alternate-refs",
        "git log --indexed-objects",
        "git log --refl",
        "git log --remote",
        "git show --all",
        # pathspecs after `--` are literal paths: no globs, no :magic, nothing dash-led
        "git diff HEADSHA~1 -- '*.py'",
        'git log -- "tools/**"',
        "git diff -- 'conf*/*.env'",
        "git diff -- ':(glob)**/*.env'",
        "git diff -- ':!docs'",
        "git diff -- :/",
        "git diff -- :",
        "git diff -- -x",
        "git diff -- --output=x",
        "git diff -- '[a-z]*'",
        "git diff -- 'a?c'",
        "git diff -- ~",
        "git diff -- ..",
        "git diff -- ../x",
        "git diff -- a -- b",
        "git diff -- /abs/path",
        # ... and so are the words before `--` that git might read as paths
        "git diff 'conf*/*.env'",
        "git diff HEADSHA '*.py'",
        "git show 'HEADSHA@{1}'",
        # <rev>:<path> only names a non-secret file tracked at the base ref
        "git show HEADSHA:config/docker.env",
        "git show HEADSHA:.env",
        "git show HEADSHA:added_by_the_head.py",
        "git show HEADSHA:",
        "git show HEADSHA:tools/",
        "git show HEADSHA:README.md:x",
        "git show :0:README.md",
        "git show :/fix",
        "git show HEADSHA:./README.md",
        "git show HEADSHA:../x",
        "git show C:/Users/x/file",
        "git show 'HEADSHA@{1}:README.md'",
        "git show -- HEADSHA:README.md",
        # secret-bearing paths, named any way
        "git diff -- prod.env",
        "git log -p -- docker.env.local",
        "git log --grep=docker.env",
        # a revision must be a pinned commit: a bare blob or tree id, an abbreviation, a ref
        # name or a commit the task did not pin could print an object whatever its path
        "git show OTHERSHA",
        "git show OTHERSHA:README.md",
        "git diff OTHERSHA HEADSHA",
        "git diff BASESHA...OTHERSHA",
        "git log BASESHA..HEADSHA..OTHERSHA",
        "git log BASESHA..HEADSHA..BASESHA",
        "git log BASESHA...HEADSHA...BASESHA",
        "git show deadbeef",
        "git show abc1234",
        "git show abc1234:README.md",
        f"git show {_BASE_SHA[:7]}",
        f"git show {_BASE_SHA[:39]}",
        f"git show {_BASE_SHA.upper()}",
        f"git show {_HEAD_SHA}0",
        "git show HEAD",
        "git show main",
        "git show v1.0",
        "git show @",
        "git log origin/main..HEADSHA",
        "git diff main HEADSHA",
        "git show HEADSHA^{tree}",
        "git show HEADSHA^{blob}",
        "git show HEADSHA^!",
        "git log HEADSHA^@",
        "git log HEADSHA^-1",
        "git show HEADSHA~x",
        "git show HEADSHA^^^x",
        "git log ^^BASESHA HEADSHA",
        "git log ^ HEADSHA",
        "git log ....HEADSHA",
        # an option's value is read as a revision when it is a separate word
        "git log -n 5",
        "git log -S needle",
        "git log --grep needle",
        "git log --author someone",
        # options that print blob ids, which a later call could try to use as a revision
        "git log --raw",
        "git show --raw HEADSHA",
        "git diff --raw BASESHA HEADSHA",
        "git log --abbrev=7",
        "git log --abbrev",
        "git diff --abbr BASESHA HEADSHA",
        # unparseable / empty
        "git log --grep='unterminated",
        "",
        "   ",
    ],
)
def test_bash_hook_denies_non_readonly(command: str) -> None:
    reason = _check(command)
    assert reason, f"{command!r} must be denied"


def test_pytest_is_not_on_the_allowlist_and_not_in_the_guard() -> None:
    """Ruling 49: ``--collect-only`` still imports the head's conftest.py with the key in reach."""
    assert not [rule for rule in policy.ALLOWED if "pytest" in rule or "python" in rule]
    for argfile in ("@args.txt", "@/abs/args.txt", "@-"):
        assert _check(f"python -m pytest {argfile} --collect-only")


@pytest.mark.parametrize("command", [None, 123, ["git", "diff"], {"command": "git diff"}])
def test_bash_guard_denies_a_missing_or_non_string_command(command: object) -> None:
    assert policy.check_bash_command(command)


def test_bash_guard_denies_an_oversized_command() -> None:
    assert policy.check_bash_command("git log " + "a" * 100_000)


def test_a_bundled_orderfile_option_is_refused_with_a_reason_that_says_why() -> None:
    reason = str(_check("git diff -pO/etc/passwd"))
    assert "capital O" in reason
    assert "-pO/etc/passwd" in reason


def test_refusals_tell_the_model_what_to_do_instead() -> None:
    """A refusal that only says "no" sends the model round in circles; these say what works."""
    assert "option" in str(_check("git diff -- -x"))
    assert "literal" in str(_check("git diff -- '*.py'"))
    assert "tracked at the base" in str(_check("git show HEADSHA:added_by_the_head.py"))
    assert "git diff <base>...<head>" in str(_check("git show HEADSHA:added_by_the_head.py"))
    assert "after `--`" in str(_check("git diff '*.py'"))
    assert "git diff" in str(_check("rm -rf build"))
    # A refused revision names the commits that would have been accepted.
    reason = str(_check("git show abc1234"))
    assert _BASE_SHA in reason
    assert _HEAD_SHA in reason
    assert "-n5" in reason, "and says how to give an option its value"


def test_rev_path_is_refused_when_no_base_files_are_known() -> None:
    """A guard built with no base paths cannot vouch for any ``<rev>:<path>``."""
    command = _pinned("git show HEADSHA:README.md")
    assert policy.check_bash_command(command, revisions=_REVISIONS)
    assert (
        policy.check_bash_command(command, base_paths={"README.md"}, revisions=_REVISIONS) is None
    )


@pytest.mark.parametrize("word", ["/etc/passwd", "~/notes", "C:/Users/x", "C:\\Users\\x", "a/../b"])
def test_the_path_screen_stands_behind_the_literal_path_grammar(word: str) -> None:
    """Defence in depth: should the grammar ever be loosened, an absolute or escaping path
    is still refused."""
    assert policy._path_token_reason(word)
    assert policy._path_token_reason("tools/agent_reviewer/policy.py") is None


def test_a_guard_with_no_pinned_commit_accepts_no_revision() -> None:
    """The default guard knows no commit, so it can only run revision-free commands."""
    assert policy.check_bash_command(_pinned("git show HEADSHA")) is not None
    assert "no pinned commit" in str(policy.check_bash_command(_pinned("git show HEADSHA")))
    assert policy.check_bash_command("git diff --stat") is None


def test_only_the_pinned_commits_are_revisions() -> None:
    revisions = {_BASE_SHA}
    assert policy.check_bash_command(f"git show {_BASE_SHA}~2", revisions=revisions) is None
    assert policy.check_bash_command(f"git show {_HEAD_SHA}", revisions=revisions)


def test_a_sha256_commit_id_is_accepted_when_it_is_pinned() -> None:
    pinned = "d4" * 32
    assert policy.check_bash_command(f"git show {pinned}~1", revisions={pinned}) is None
    assert policy.check_bash_command(f"git show {pinned[:63]}", revisions={pinned})


def test_a_secret_named_file_tracked_at_the_base_is_still_refused() -> None:
    paths = {"config/docker.env.example", "README.md"}
    command = _pinned("git show HEADSHA:config/docker.env.example")
    assert policy.check_bash_command(command, base_paths=paths, revisions=_REVISIONS)


async def test_bash_hook_returns_a_deny_payload_and_never_an_allow() -> None:
    denied = await policy.deny_non_readonly_bash(
        {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, "tu1", {"signal": None}
    )
    output = denied["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert output["permissionDecisionReason"]

    # "Allow is silent": an explicit allow would approve the call on its own.
    allowed = await policy.deny_non_readonly_bash(
        {"tool_name": "Bash", "tool_input": {"command": f"git diff {_HEAD_SHA}~1"}},
        "tu2",
        {"signal": None},
        revisions=_REVISIONS,
    )
    assert allowed == {}


async def test_the_registered_guard_is_bound_to_the_base_refs_facts() -> None:
    guard = policy.BashGuard(base_paths=_BASE_PATHS, revisions=_REVISIONS)
    call = {"tool_name": "Bash", "tool_input": {"command": _pinned("git show HEADSHA:README.md")}}
    assert await guard(call, "tu", {"signal": None}) == {}
    other = {"tool_name": "Bash", "tool_input": {"command": _pinned("git show HEADSHA:added.py")}}
    denied = await guard(other, "tu", {"signal": None})
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    unpinned = {"tool_name": "Bash", "tool_input": {"command": "git show abc1234"}}
    denied = await guard(unpinned, "tu", {"signal": None})
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert guard.check(_pinned("git show HEADSHA:README.md")) is None
    assert guard.check(_pinned("git show HEADSHA:added.py"))
    # The default guard knows no base file and no commit at all.
    assert policy.BashGuard().check(_pinned("git show HEADSHA:README.md"))


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
# policy - Read, Grep and Glob stay inside the checkout
# --------------------------------------------------------------------------- #
_READ_FIELDS = (("Read", "file_path"), ("Grep", "path"), ("Glob", "path"))


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A small checkout and, beside it, a directory holding a file that must stay unread."""
    root = tmp_path / "checkout"
    _write(root, "src/app.py", "x = 1\n")
    _write(root, "docs/notes.txt", "notes\n")
    _write(root, "README.md", "readme\n")
    _write(tmp_path / "elsewhere", "secret.txt", "NOT FOR THE MODEL\n")
    return root


def _read(checkout: Path, tool: str, **tool_input: Any) -> str | None:
    return policy.ReadGuard(root=str(checkout)).check(tool, tool_input)


def _says(reason: str | None, fragment: str, context: object = "") -> None:
    """Assert a refusal was given and that its reason mentions ``fragment``."""
    assert reason is not None, context
    assert fragment in reason, (context, reason)


def _link_dir(link: Path, target: Path) -> None:
    """Make ``link`` an alias of the directory ``target``: a symlink, or a junction on Windows."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":  # a junction needs no privilege
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if done.returncode == 0:
            return
    pytest.skip("this account can create neither a symbolic link nor a junction")


def test_a_file_inside_the_checkout_may_be_read(checkout: Path) -> None:
    assert _read(checkout, "Read", file_path=str(checkout / "src" / "app.py")) is None
    assert _read(checkout, "Read", file_path="src/app.py") is None, "relative: from the checkout"
    assert _read(checkout, "Read", file_path="README.md") is None
    assert _read(checkout, "Grep", pattern="x", path=str(checkout / "src")) is None
    assert _read(checkout, "Grep", pattern="x") is None, "no path: the checkout itself"
    assert _read(checkout, "Grep", pattern="x", path="src", glob="*.py") is None
    assert _read(checkout, "Glob", pattern="src/**/*.py") is None
    assert _read(checkout, "Glob", pattern="*.md", path=str(checkout)) is None
    assert _read(checkout, "Read", file_path=str(checkout / "not" / "there.py")) is None


@pytest.mark.parametrize(("tool", "field"), _READ_FIELDS)
def test_a_path_outside_the_checkout_is_refused(checkout: Path, tool: str, field: str) -> None:
    elsewhere = checkout.parent / "elsewhere"
    targets = [
        str(elsewhere / "secret.txt"),
        str(elsewhere),
        str(checkout.parent),
        str(Path(checkout.anchor)),
        "../elsewhere/secret.txt",
        "..",
        "src/../../elsewhere",
    ]
    for target in targets:
        _says(_read(checkout, tool, **{field: target}), "outside the checkout", (tool, target))


@pytest.mark.parametrize(("tool", "field"), _READ_FIELDS)
@pytest.mark.parametrize(
    "raw", ["~", "~/.bashrc", "$HOME/.bashrc", "${HOME}/x", "%USERPROFILE%/.bashrc"]
)
def test_a_path_the_shell_would_expand_is_refused(
    checkout: Path, tool: str, field: str, raw: str
) -> None:
    _says(_read(checkout, tool, **{field: raw}), "expanded", (tool, raw))


@pytest.mark.parametrize(
    "raw", ["src/app.py::$DATA", "C:/Windows/win.ini", "src/app.py:stream", "a:b"]
)
def test_a_data_stream_or_drive_path_is_refused(checkout: Path, raw: str) -> None:
    assert _read(checkout, "Read", file_path=raw)


def test_a_sibling_whose_name_starts_with_the_checkouts_is_outside(checkout: Path) -> None:
    sibling = checkout.parent / (checkout.name + "-evil")
    _write(sibling, "x.txt", "x\n")
    assert _read(checkout, "Read", file_path=str(sibling / "x.txt"))
    assert _read(checkout, "Grep", pattern="x", path=str(sibling))


def test_an_alias_inside_the_checkout_that_leads_outside_is_refused(checkout: Path) -> None:
    """A symlink (or a Windows junction) is followed before the path is judged."""
    _link_dir(checkout / "alias", checkout.parent / "elsewhere")
    _says(_read(checkout, "Read", file_path="alias/secret.txt"), "outside the checkout")
    assert _read(checkout, "Read", file_path=str(checkout / "alias" / "secret.txt"))
    assert _read(checkout, "Grep", pattern="NOT", path="alias")
    assert _read(checkout, "Glob", pattern="*.txt", path="alias")


def test_an_alias_that_stays_inside_the_checkout_is_fine(checkout: Path) -> None:
    _link_dir(checkout / "docs_alias", checkout / "docs")
    assert _read(checkout, "Read", file_path="docs_alias/notes.txt") is None


def test_a_checkout_reached_through_an_alias_still_reads_its_own_files(
    checkout: Path, tmp_path: Path
) -> None:
    """The root is resolved too: a working directory that is itself an alias is inside itself."""
    _link_dir(tmp_path / "rootlink", checkout)
    guard = policy.ReadGuard(root=str(tmp_path / "rootlink"))
    assert guard.check("Read", {"file_path": "src/app.py"}) is None
    assert guard.check("Read", {"file_path": str(tmp_path / "rootlink" / "src" / "app.py")}) is None
    assert guard.check("Read", {"file_path": str(checkout / "src" / "app.py")}) is None
    _says(guard.check("Read", {"file_path": "../elsewhere/secret.txt"}), "outside the checkout")


def test_a_file_symlink_that_leads_outside_the_checkout_is_refused(checkout: Path) -> None:
    link = checkout / "docs" / "innocent.txt"
    try:
        os.symlink(checkout.parent / "elsewhere" / "secret.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("this account may not create symbolic links")
    _says(_read(checkout, "Read", file_path="docs/innocent.txt"), "outside the checkout")


@pytest.mark.parametrize(
    "name",
    ["prod.env", ".env", ".env.production", "docker.env", "docker.env.rover", "config/local.env"],
)
def test_a_secret_named_file_inside_the_checkout_is_refused(checkout: Path, name: str) -> None:
    _write(checkout, name, "KEY=value\n")
    for tool, field in _READ_FIELDS:
        _says(_read(checkout, tool, **{field: name}), "secret-file pattern", (tool, name))
    assert _read(checkout, "Read", file_path=str(checkout / name))


def test_a_secret_name_is_judged_after_an_alias_is_followed(checkout: Path) -> None:
    """An innocent-looking name that points at a secret-named file is still a secret."""
    _write(checkout, "config/prod.env", "KEY=value\n")
    _link_dir(checkout / "innocent", checkout / "config")
    assert _read(checkout, "Read", file_path="innocent/prod.env")
    assert _read(checkout, "Read", file_path="innocent/other.txt") is None


def test_git_internals_are_not_readable_files(checkout: Path) -> None:
    _write(checkout, ".git/config", "[core]\n")
    for rel in (".git/config", ".GIT/config", ".git", "src/../.git/config"):
        _says(_read(checkout, "Read", file_path=rel), ".git", rel)
    assert _read(checkout, "Grep", pattern="core", path=".git")
    assert _read(checkout, "Read", file_path=".github/workflows/ci.yml") is None
    assert _read(checkout, "Read", file_path="docs/.gitignore") is None


@pytest.mark.parametrize(
    "pattern",
    [
        "/etc/*",
        "C:/Users/*",
        "C:\\Users\\*",
        "~/*",
        "../*",
        "src/../../*",
        "**/../x",
        "*.env",
        ".env*",
        "docker.env*",
        "**/.env.local",
        "**/*.{env,py}",
        "*.{py,{env,md}}",
        "docker.env?",
        ".env?",
        ".env?x",
        "?env",
    ],
)
def test_a_glob_that_leaves_the_checkout_or_names_secrets_is_refused(
    checkout: Path, pattern: str
) -> None:
    assert _read(checkout, "Glob", pattern=pattern)
    assert _read(checkout, "Grep", pattern="x", glob=pattern)


@pytest.mark.parametrize(
    "pattern",
    [
        "*.py",
        "**/*.py",
        "src/**/*.{py,pyi}",
        "docs/*.txt",
        "*.md",
        "**/*_env.py",
        "environment.py",
        "?.py",
        "data?.json",
    ],
)
def test_an_ordinary_glob_is_fine(checkout: Path, pattern: str) -> None:
    assert _read(checkout, "Glob", pattern=pattern) is None
    assert _read(checkout, "Grep", pattern="x", glob=pattern) is None


def test_brace_expansion_is_bounded() -> None:
    """Four groups of ten alternatives would be ten thousand patterns; the screen caps them."""
    assert len(policy._brace_variants("{0,1,2,3,4,5,6,7,8,9}" * 4)) <= 64
    assert sorted(policy._brace_variants("a.{b,c}")) == ["a.b", "a.c"]
    assert policy._brace_variants("plain") == ["plain"]


@pytest.mark.parametrize("glob", [7, ["*.py"], "a\x00b", "*" * 5000])
def test_a_malformed_glob_is_refused(checkout: Path, glob: object) -> None:
    assert _read(checkout, "Glob", pattern=glob)
    assert _read(checkout, "Grep", pattern="x", glob=glob)


@pytest.mark.parametrize(
    "tool_input",
    [
        None,
        "src/app.py",
        ["src/app.py"],
        {},
        {"file_path": ""},
        {"file_path": "  "},
        {"file_path": 7},
        {"file_path": None},
        {"file_path": "a\x00b"},
        {"file_path": "a\nb"},
        {"file_path": "x" * 5000},
    ],
)
def test_a_malformed_read_request_is_refused(checkout: Path, tool_input: object) -> None:
    assert policy.ReadGuard(root=str(checkout)).check("Read", tool_input)


@pytest.mark.parametrize("tool", ["Grep", "Glob"])
@pytest.mark.parametrize("path", [7, "", "  ", "a\x00b"])
def test_a_grep_or_glob_with_an_unusable_path_is_refused(
    checkout: Path, tool: str, path: object
) -> None:
    assert _read(checkout, tool, pattern="x", path=path)


@pytest.mark.parametrize("tool", ["Write", "Bash", "WebFetch", "", None, 7])
def test_only_read_tools_are_judged_by_the_read_guard(checkout: Path, tool: object) -> None:
    assert policy.ReadGuard(root=str(checkout)).check(tool, {"file_path": "src/app.py"})


def test_every_allowed_file_tool_is_under_the_read_guard() -> None:
    """Adding a bare tool to the allowlist without extending the guard is a reviewed edit."""
    bare = [rule for rule in policy.ALLOWED if "(" not in rule and not rule.startswith("mcp__")]
    assert bare == ["Read", "Grep", "Glob"]
    for tool in bare:
        assert re.fullmatch(policy.READ_HOOK_MATCHER, tool), tool
    for tool in ("Bash", "Write", "Edit", "Task", "WebFetch", "mcp__mousedroid__system_info"):
        assert not re.fullmatch(policy.READ_HOOK_MATCHER, tool), tool
    assert policy.READ_HOOK_MATCHER != policy.BASH_HOOK_MATCHER


async def test_the_read_hook_returns_a_deny_payload_and_never_an_allow(checkout: Path) -> None:
    guard = policy.ReadGuard(root=str(checkout))
    leaving = {
        "tool_name": "Read",
        "tool_input": {"file_path": str(checkout.parent / "elsewhere" / "secret.txt")},
    }
    denied = await guard(leaving, "tu1", {"signal": None})
    output = denied["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert "outside the checkout" in output["permissionDecisionReason"]

    # "Allow is silent": an explicit allow would approve the call on its own.
    staying = {"tool_name": "Read", "tool_input": {"file_path": "src/app.py"}}
    assert await guard(staying, "tu2", {"signal": None}) == {}
    grep = {"tool_name": "Grep", "tool_input": {"pattern": "x", "path": "src"}}
    assert await guard(grep, "tu3", {"signal": None}) == {}
    # The root is the one the guard was built for, not the process's working directory.
    absolute = {"tool_name": "Read", "tool_input": {"file_path": str(checkout / "src" / "app.py")}}
    assert await guard(absolute, "tu4", {"signal": None}) == {}


async def test_the_read_hook_judges_grep_and_glob_too(checkout: Path) -> None:
    guard = policy.ReadGuard(root=str(checkout))
    for name, tool_input in (
        ("Grep", {"pattern": "x", "path": str(checkout.parent)}),
        ("Grep", {"pattern": "x", "glob": "*.env"}),
        ("Glob", {"pattern": "*", "path": ".."}),
        ("Glob", {"pattern": "/etc/*"}),
    ):
        result = await guard({"tool_name": name, "tool_input": tool_input}, None, None)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny", (name, tool_input)


async def test_the_read_hook_leaves_other_tools_to_the_permission_rules(checkout: Path) -> None:
    guard = policy.ReadGuard(root=str(checkout))
    for name in ("Bash", "Write", "mcp__mousedroid__system_info"):
        assert await guard({"tool_name": name, "tool_input": {}}, None, None) == {}


async def test_the_read_hook_fails_closed_on_malformed_input(checkout: Path) -> None:
    guard = policy.ReadGuard(root=str(checkout))
    bad_inputs: list[Any] = [
        None,
        {},
        "Read",
        5,
        [],
        {"tool_name": "Read"},
        {"tool_name": "Read", "tool_input": None},
        {"tool_name": "Read", "tool_input": "src/app.py"},
        {"tool_name": "Read", "tool_input": {}},
    ]
    for bad in bad_inputs:
        result = await guard(bad, None, None)
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny", bad


async def test_the_read_hook_fails_closed_when_the_guard_itself_crashes(checkout: Path) -> None:
    class Exploding(dict[str, Any]):
        def get(self, *_args: Any) -> Any:
            raise RuntimeError("boom")

    result = await policy.ReadGuard(root=str(checkout))(
        Exploding(tool_name="Read", tool_input={}), "tu", {"signal": None}
    )
    output = result["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    assert "RuntimeError" in output["permissionDecisionReason"]


# --------------------------------------------------------------------------- #
# policy - the MCP server config, the process environment and the git environment
# --------------------------------------------------------------------------- #
def _mcp_json(**overrides: Any) -> str:
    """A ``.mcp.json`` whose ``mousedroid`` entry is the pinned one, with fields overridden."""
    entry: dict[str, Any] = {
        "command": "python",
        "args": list(policy.MCP_DECLARED_ARGS),
        "env": {"MOUSEDROID_MOCK_HARDWARE": "${MOUSEDROID_MOCK_HARDWARE:-true}"},
    }
    entry.update(overrides)
    entry = {key: value for key, value in entry.items() if value is not _ABSENT}
    return json.dumps({"mcpServers": {"mousedroid": entry}})


_ABSENT = object()


def _build(mcp_json: str | None = None) -> dict[str, dict[str, Any]]:
    return policy.build_mcp_servers(mcp_json or _mcp_json(), config_path=_CONFIG_PATH)


def test_mcp_config_forces_mock_hardware() -> None:
    servers = _build()
    assert list(servers) == [policy.MCP_SERVER], "only the mock hardware server, never `github`"
    server = servers[policy.MCP_SERVER]
    assert server["type"] == "stdio"
    assert server["env"]["MOUSEDROID_MOCK_HARDWARE"] == "true"
    assert server["env"]["MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS"] == "false"
    assert not any(key in server for key in ("url", "headers", "host"))
    json.dumps(servers)  # the SDK passes it to the CLI as JSON


def test_mcp_entry_is_built_in_code_from_constants() -> None:
    server = _build()["mousedroid"]
    assert server["command"] == sys.executable, (
        "an absolute path: no search of the head's directory"
    )
    assert server["args"] == ["-I", "-c", policy.MCP_BOOTSTRAP, str(_CONFIG_PATH)]
    assert server["env"] == dict(policy.MCP_ENV)


def test_the_mcp_server_is_started_isolated_from_the_working_directory() -> None:
    args = _build()["mousedroid"]["args"]
    assert args[0] == "-I", "isolated: no cwd, no PYTHON* variable, no user site on sys.path"
    assert "-m" not in args, "`-m` would put the checkout first on sys.path before Python 3.11"


def test_the_bootstrap_is_one_inert_line() -> None:
    """It travels through the CLI's JSON config and a Windows command line."""
    assert "\n" not in policy.MCP_BOOTSTRAP
    assert '"' not in policy.MCP_BOOTSTRAP
    assert policy.MCP_MODULE in policy.MCP_BOOTSTRAP


def test_the_bootstrap_runs_the_module_from_the_configs_directory(tmp_path: Path) -> None:
    """Run the real bootstrap against a stand-in ``mousedroid`` and look at what it saw."""
    site = tmp_path / "site"
    package = site / policy.MCP_MODULE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'argv': sys.argv, 'name': __name__}))\n",
        encoding="utf-8",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config = scratch / "mcp-config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    done = subprocess.run(
        [sys.executable, "-c", policy.MCP_BOOTSTRAP, str(config)],
        cwd=checkout,
        env={**os.environ, "PYTHONPATH": str(site)},  # not -I here: the stand-in must import
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    seen = json.loads(done.stdout)
    assert Path(seen["cwd"]).resolve() == scratch.resolve(), "outside the checkout"
    assert seen["argv"][1:] == ["--config", str(config)]
    assert seen["name"] == "__main__"


def test_isolated_mode_keeps_the_working_directory_and_pythonpath_off_the_import_path(
    tmp_path: Path,
) -> None:
    """The property the launch relies on, checked on this interpreter."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "colorsys.py").write_text("PLANTED = True\n", encoding="utf-8")
    probe = "import colorsys; print(getattr(colorsys, 'PLANTED', False))"
    env = {**os.environ, "PYTHONPATH": str(checkout)}

    plain = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    isolated = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert plain.stdout.strip() == "True", "without -I the checkout shadows the standard library"
    assert isolated.stdout.strip() == "False", "-I keeps both the cwd and PYTHONPATH out"


def test_mcp_launch_args_need_an_absolute_config_path() -> None:
    with pytest.raises(policy.PolicyViolationError, match="absolute"):
        policy.mcp_launch_args("config/default.yaml")
    assert policy.mcp_launch_args(_CONFIG_PATH)[-1] == str(_CONFIG_PATH)


def test_mcp_env_carries_no_credential_and_blanks_the_ones_the_cli_inherits() -> None:
    env = policy.MCP_ENV
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "MOUSEDROID_LLM__API_KEY"):
        assert env[name] == "", f"{name} must be blanked for the MCP server"
    assert not [name for name, value in env.items() if value and "${" in value]
    assert not [name for name in env if name.startswith("PYTHON")], "-I ignores them anyway"


def test_the_pinned_invocation_matches_the_checked_in_mcp_json() -> None:
    """The constants are a copy of trunk's ``.mcp.json``; this fails the day the two drift."""
    declared = json.loads((_REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"][
        "mousedroid"
    ]
    assert declared["command"] in policy.MCP_DECLARED_COMMANDS
    assert tuple(declared["args"]) == policy.MCP_DECLARED_ARGS
    assert declared["args"][-1] == policy.MCP_CONFIG_RELPATH
    for key, value in declared["env"].items():
        if key.startswith("MOUSEDROID_MCP__"):
            assert policy.MCP_ENV[key] == value, key
    policy.confirm_mcp_declaration((_REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"), "real")


def test_nothing_in_the_declaring_file_is_passed_through() -> None:
    """Whatever the file says about env, extra servers or hosts, the entry is the same."""
    hostile = json.dumps(
        {
            "mcpServers": {
                "github": {"command": "npx", "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "x"}},
                "mousedroid": {
                    "command": "python",
                    "args": list(policy.MCP_DECLARED_ARGS),
                    "env": {
                        "MOUSEDROID_MOCK_HARDWARE": "false",
                        "MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS": "true",
                        "MOUSEDROID_ESP32__WIFI_HOST": "192.0.2.10",
                        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
                        "GITHUB_TOKEN": "t",
                    },
                },
            }
        }
    )
    servers = _build(hostile)
    assert list(servers) == ["mousedroid"]
    assert servers["mousedroid"]["env"] == dict(policy.MCP_ENV)


@pytest.mark.parametrize("command", ["python", "python3", sys.executable])
def test_mcp_declaration_may_name_the_interpreter_three_ways(command: str) -> None:
    assert _build(_mcp_json(command=command))["mousedroid"]["command"] == sys.executable


@pytest.mark.parametrize(
    "entry",
    [
        {"command": "./python"},
        {"command": "evil.exe"},
        {"command": "C:\\Users\\x\\python.exe"},
        {"command": "/usr/bin/python"},
        {"command": "npx"},
        {"command": "python "},
        {"command": "PYTHON"},
        {"command": ["python"]},
        {"command": 7},
        {"command": _ABSENT},
        {"args": [*policy.MCP_DECLARED_ARGS, "--extra"]},
        {"args": list(policy.MCP_DECLARED_ARGS[:-1])},
        {"args": list(reversed(policy.MCP_DECLARED_ARGS))},
        {"args": ["-c", "import os"]},
        {"args": ["-m", "mousedroid", "--config", "elsewhere.yaml"]},
        {"args": "-m mousedroid"},
        {"args": _ABSENT},
        {"type": "http"},
        {"url": "https://example.invalid/mcp"},
    ],
)
def test_mcp_declaration_that_does_not_match_the_pin_is_refused(entry: dict[str, Any]) -> None:
    with pytest.raises(policy.PolicyViolationError):
        _build(_mcp_json(**entry))


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        json.dumps({}),
        json.dumps({"mcpServers": {}}),
        json.dumps({"mcpServers": []}),
        json.dumps({"mcpServers": {"mousedroid": "nope"}}),
        json.dumps({"mcpServers": {"github": {"command": "npx"}}}),
    ],
)
def test_mcp_declaration_that_is_malformed_or_lacks_the_server_is_refused(body: str) -> None:
    with pytest.raises(policy.PolicyViolationError):
        _build(body)


def test_mcp_needs_an_interpreter_to_run_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy.sys, "executable", "")
    with pytest.raises(policy.PolicyViolationError):
        _build()


def test_process_env_is_an_allowlist_and_never_carries_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ruling 49: no credential is ever copied into the environment the reviewer adds."""
    secrets = {
        "ANTHROPIC_API_KEY": _FAKE_KEY,
        "ANTHROPIC_AUTH_TOKEN": "t1",
        "MOUSEDROID_LLM__API_KEY": "t2",
        "GITHUB_TOKEN": "t3",
        "HF_TOKEN": "t4",
        "AWS_SECRET_ACCESS_KEY": "t5",
        "PYTHONPATH": "/hostile",
        "GIT_EXTERNAL_DIFF": "/hostile/diff",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/me")
    monkeypatch.setenv("LANG", "C.UTF-8")

    for env in (policy.process_env(os.environ), policy.process_env()):
        assert set(env) == {*policy.PROCESS_ENV_PASSTHROUGH, *policy.FORCED_PROCESS_ENV}
        assert not set(secrets) & set(env)
        assert _FAKE_KEY not in json.dumps(env)
        assert (env["PATH"], env["HOME"], env["LANG"]) == ("/usr/bin", "/home/me", "C.UTF-8")
        assert env["GIT_LITERAL_PATHSPECS"] == "1"
        assert env["MOUSEDROID_MOCK_HARDWARE"] == "true"


def test_process_env_omits_what_the_parent_does_not_have() -> None:
    env = policy.process_env({"ANTHROPIC_API_KEY": _FAKE_KEY})
    assert env == dict(policy.FORCED_PROCESS_ENV)
    assert "ANTHROPIC_API_KEY" not in env


def _make_program(directory: Path, name: str) -> Path:
    """Create an executable file called ``name`` (with the extension Windows needs)."""
    directory.mkdir(parents=True, exist_ok=True)
    program = directory / (name + (".exe" if os.name == "nt" else ""))
    program.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    program.chmod(0o755)
    return program


def test_executables_are_found_on_path_and_never_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``shutil.which`` and ``CreateProcess`` try the working directory first; this does not."""
    checkout = tmp_path / "checkout"
    planted = _make_program(checkout, "reviewer-tool")
    monkeypatch.chdir(checkout)
    bin_dir = tmp_path / "bin"
    env = {"PATH": os.pathsep.join(["", ".", "relative", str(bin_dir)])}
    assert policy.resolve_executable("reviewer-tool", env) is None, "only the cwd has it"
    real = _make_program(bin_dir, "reviewer-tool")
    found = policy.resolve_executable("reviewer-tool", env)
    assert found is not None
    assert Path(found) == real
    assert Path(found) != planted


def test_executable_lookup_skips_relative_and_empty_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _make_program(tmp_path / "rel", "reviewer-tool")
    env = {"PATH": os.pathsep.join(["rel", "", "./rel"])}
    assert policy.resolve_executable("reviewer-tool", env) is None
    assert policy.resolve_executable("reviewer-tool", {}) is None
    assert policy.resolve_executable("reviewer-tool", {"PATH": ""}) is None


def test_executable_lookup_accepts_only_an_absolute_name_with_a_directory(tmp_path: Path) -> None:
    program = _make_program(tmp_path, "reviewer-tool")
    assert policy.resolve_executable(str(program), {"PATH": ""}) == str(program)
    assert policy.resolve_executable("sub/reviewer-tool", {"PATH": str(tmp_path)}) is None
    assert policy.resolve_executable(str(tmp_path / "missing"), {"PATH": ""}) is None


def test_executable_lookup_ignores_a_file_that_is_not_executable(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows has no execute bit: any file with the right extension counts")
    plain = tmp_path / "bin" / "reviewer-tool"
    plain.parent.mkdir()
    plain.write_text("data\n", encoding="utf-8")
    plain.chmod(0o644)
    assert policy.resolve_executable("reviewer-tool", {"PATH": str(plain.parent)}) is None


def test_executable_lookup_uses_pathext_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "reviewer-tool.EXE"
    exe.write_text("MZ\n", encoding="utf-8")
    exe.chmod(0o755)
    env = {"PATH": str(bin_dir), "PATHEXT": ".COM;.EXE;.BAT"}
    with monkeypatch.context() as patch:  # only around the calls: os.name is global
        patch.setattr(policy.os, "name", "nt")
        bare = policy.resolve_executable("reviewer-tool", env)
        spelled = policy.resolve_executable("reviewer-tool.EXE", env)
        other = policy.resolve_executable("reviewer-tool.txt", env)
    assert bare is not None
    assert spelled is not None
    assert other is None


def test_git_runs_with_literal_pathspecs_and_without_index_writes() -> None:
    assert policy.GIT_ENV["GIT_LITERAL_PATHSPECS"] == "1"
    assert policy.GIT_ENV["GIT_OPTIONAL_LOCKS"] == "0"
    assert policy.FORCED_PROCESS_ENV["GIT_LITERAL_PATHSPECS"] == "1"


def test_the_cli_is_told_not_to_look_in_the_working_directory_for_programs() -> None:
    """Windows tries the working directory - the checkout under review - before PATH."""
    assert policy.process_env({})["NoDefaultCurrentDirectoryInExePath"] == "1"
    options_env = policy.process_env(
        {"PATH": "/usr/bin", "NoDefaultCurrentDirectoryInExePath": "0"}
    )
    assert options_env["NoDefaultCurrentDirectoryInExePath"] == "1", "forced, not inherited"


# --------------------------------------------------------------------------- #
# policy - credentials the reviewer's own children never see, and planted programs
# --------------------------------------------------------------------------- #
def test_scrubbed_environ_drops_every_variable_that_can_hold_a_credential() -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "LANG": "C",
        "SYSTEMROOT": "C:\\Windows",
        "ANTHROPIC_API_KEY": "k1",
        "anthropic_api_key": "k2",
        "ANTHROPIC_AUTH_TOKEN": "t1",
        "CLAUDE_CODE_OAUTH_TOKEN": "t2",
        "MOUSEDROID_LLM__API_KEY": "k3",
        "OPENAI_API_KEY": "k4",
        "GITHUB_TOKEN": "t3",
        "HF_TOKEN": "t4",
        "DB_PASSWORD": "p1",
        "CLIENT_SECRET": "s1",
    }
    assert policy.scrubbed_environ(environ) == {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "LANG": "C",
        "SYSTEMROOT": "C:\\Windows",
    }
    assert "ANTHROPIC_API_KEY" in environ, "the caller's mapping is left alone"


def test_scrubbed_environ_reads_the_process_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("REVIEWER_TEST_MARKER", "kept")
    scrubbed = policy.scrubbed_environ()
    assert "ANTHROPIC_API_KEY" not in scrubbed
    assert scrubbed["REVIEWER_TEST_MARKER"] == "kept"
    assert os.environ["ANTHROPIC_API_KEY"] == _FAKE_KEY, "the process's own environment is intact"


def test_the_credential_names_include_the_ones_the_reviewer_is_known_to_hold() -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "MOUSEDROID_LLM__API_KEY"):
        assert name in policy.CREDENTIAL_ENV_NAMES
    assert policy.scrubbed_environ(dict.fromkeys(policy.CREDENTIAL_ENV_NAMES, "v")) == {}


@pytest.mark.parametrize(
    "name",
    [
        "git",
        "git.exe",
        "git.cmd",
        "GIT.BAT",
        "bash",
        "bash.exe",
        "sh.exe",
        "claude",
        "claude.cmd",
        "rg",
        "rg.exe",
        "node.exe",
        "npm.cmd",
        "npx",
        "gitleaks.exe",
        "cmd.exe",
        "powershell.ps1",
        "python",
        "python3",
        "python3.12.exe",
        "pythonw.exe",
        "git.exe.",
        "git.exe ",
    ],
)
def test_a_root_file_named_like_a_program_the_reviewer_starts_is_refused(name: str) -> None:
    reason = policy.planted_program_reason(["README.md", name])
    _says(reason, repr(name))
    _says(reason, "working directory")


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        ".gitignore",
        "gitattributes",
        "git-notes.txt",
        "gitx.exe",
        "pythonic.py",
        "python_requirements.txt",
        "nodes.py",
        "rg.md",
        "bash_completion.sh",
        "Makefile",
        "pyproject.toml",
        "setup.py",
        "src",
    ],
)
def test_other_root_file_names_are_not_mistaken_for_programs(name: str) -> None:
    assert policy.planted_program_reason([name]) is None


def test_no_names_is_no_planted_program() -> None:
    assert policy.planted_program_reason([]) is None


@pytest.mark.parametrize(
    "program",
    [
        "git",
        "bash",
        "sh",
        "claude",
        "rg",
        "node",
        "npm",
        "npx",
        "gitleaks",
        "cmd",
        "powershell",
        "pwsh",
        "python",
        "python3",
        "python3.12",
        "pythonw",
    ],
)
@pytest.mark.parametrize(
    "extension",
    [
        "",
        ".exe",
        ".com",
        ".bat",
        ".cmd",
        ".ps1",
        ".vbs",
        ".vbe",
        ".js",
        ".jse",
        ".wsf",
        ".wsh",
        ".msc",
        ".lnk",
        ".scr",
    ],
)
def test_every_program_name_with_every_windows_extension_is_refused(
    program: str, extension: str
) -> None:
    assert policy.planted_program_reason([program + extension])


# --------------------------------------------------------------------------- #
# policy - the SDK options
# --------------------------------------------------------------------------- #
def test_reviewer_options_are_closed_and_read_only(
    repo: Path, cfg: WorkforceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    options = policy.reviewer_options(
        cfg=cfg.agent_reviewer,
        repo_root=repo,
        model="opus",
        system_prompt="SYS",
        cap_usd=7.5,
        mcp_json=_mcp_json(),
        mcp_config_path=_CONFIG_PATH,
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
    assert list(options["mcp_servers"]) == [policy.MCP_SERVER]
    # The CLI process: literal pathspecs, mock hardware, no pytest knobs, no credential.
    env = options["env"]
    assert env == policy.process_env()
    assert env["GIT_LITERAL_PATHSPECS"] == "1"
    assert env["MOUSEDROID_MOCK_HARDWARE"] == "true"
    assert "ANTHROPIC_API_KEY" not in env
    assert "PYTEST_ADDOPTS" not in env


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf")])
def test_reviewer_options_refuse_an_unbounded_budget(
    repo: Path, cfg: WorkforceConfig, cap: float
) -> None:
    with pytest.raises(policy.PolicyViolationError):
        policy.reviewer_options(
            cfg=cfg.agent_reviewer,
            repo_root=repo,
            model="opus",
            system_prompt="s",
            cap_usd=cap,
            mcp_json=_mcp_json(),
            mcp_config_path=_CONFIG_PATH,
        )


def test_reviewer_options_refuse_an_mcp_declaration_that_does_not_match_the_pin(
    repo: Path, cfg: WorkforceConfig
) -> None:
    with pytest.raises(policy.PolicyViolationError):
        policy.reviewer_options(
            cfg=cfg.agent_reviewer,
            repo_root=repo,
            model="opus",
            system_prompt="s",
            cap_usd=5.0,
            mcp_json=_mcp_json(command="./evil"),
            mcp_config_path=_CONFIG_PATH,
        )


def test_options_names_exist_in_the_real_sdk(repo: Path, cfg: WorkforceConfig) -> None:
    """The pin that every option name used here is a real ``ClaudeAgentOptions`` field."""
    sdk = pytest.importorskip("claude_agent_sdk")
    options = policy.reviewer_options(
        cfg=cfg.agent_reviewer,
        repo_root=repo,
        model="opus",
        system_prompt="s",
        cap_usd=5.0,
        mcp_json=_mcp_json(),
        mcp_config_path=_CONFIG_PATH,
    )
    fields = {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
    assert set(options) | {"hooks"} <= fields
    guard = policy.BashGuard(base_paths=_BASE_PATHS, revisions=_REVISIONS)
    matchers = [
        sdk.HookMatcher(matcher=policy.BASH_HOOK_MATCHER, hooks=[guard]),
        sdk.HookMatcher(matcher=policy.READ_HOOK_MATCHER, hooks=[policy.ReadGuard(str(repo))]),
    ]
    built = sdk.ClaudeAgentOptions(**options, hooks={"PreToolUse": matchers})
    assert [m.matcher for m in built.hooks["PreToolUse"]] == ["Bash", "Read|Grep|Glob"]
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
    """The regexes in ``.gitleaks.toml`` are the policy's, not copied into code."""
    toml = (repo / ".gitleaks.toml").read_text(encoding="utf-8")
    patterns = prompt.gitleaks_regexes(toml)
    assert patterns, "the repo's .gitleaks.toml declares allowlist regexes"
    assert any(p.search("test-api-key-abc123") for p in patterns)
    _write(repo, "notes/fixture.md", "uses test-api-key-abc123 as a bearer\n")
    with pytest.raises(prompt.PromptRefusedError):
        prompt.build(
            repo_root=repo,
            cfg=cfg,
            extra_files=["notes/fixture.md"],
            gitleaks_toml=toml,
            scanner=_no_scanner,
        )


def test_the_secret_policy_is_the_text_given_never_the_working_trees_copy(
    repo: Path, cfg: WorkforceConfig
) -> None:
    """A head that can rewrite ``.gitleaks.toml`` must not be able to switch the screen off."""
    _write(repo, ".gitleaks.toml", '[[rules]]\nid = "x"\nregex = "ZZZ_[0-9]+"\n')
    _write(repo, "notes/context.md", "carries ZZZ_123 here\n")
    # The working tree's policy is ignored: nothing is given, so nothing extra is screened ...
    text = prompt.build(
        repo_root=repo, cfg=cfg, extra_files=["notes/context.md"], scanner=_no_scanner
    )
    assert "ZZZ_123" in text
    # ... and the policy that IS given (the base ref's) is the one that applies.
    with pytest.raises(prompt.PromptRefusedError):
        prompt.build(
            repo_root=repo,
            cfg=cfg,
            extra_files=["notes/context.md"],
            gitleaks_toml='[[rules]]\nid = "x"\nregex = "ZZZ_[0-9]+"\n',
            scanner=_no_scanner,
        )


def test_gitleaks_regexes_tolerate_a_missing_or_odd_config() -> None:
    assert prompt.gitleaks_regexes(None) == []  # the base has no .gitleaks.toml
    with pytest.raises(prompt.PromptRefusedError):
        prompt.gitleaks_regexes("this is = = not toml")
    with pytest.raises(prompt.PromptRefusedError):
        prompt.gitleaks_regexes('[allowlist]\nregexes = ["(unclosed"]\n')
    toml = '[[rules]]\nid = "x"\nregex = "tok_[0-9]+"\n'
    assert [p.pattern for p in prompt.gitleaks_regexes(toml)] == ["tok_[0-9]+"]
    toml = '[[allowlists]]\nregexes = ["fake_[a-z]+", 3]\n'
    assert [p.pattern for p in prompt.gitleaks_regexes(toml)] == ["fake_[a-z]+"]


def test_a_refusal_names_the_policy_file_it_could_not_read() -> None:
    with pytest.raises(prompt.PromptRefusedError, match=r"custom/policy\.toml"):
        prompt.gitleaks_regexes("= not toml", source="custom/policy.toml")


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


def _fake_gitleaks(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Pretend ``gitleaks`` is installed and record what the workforce wrapper is handed."""
    seen: list[dict[str, Any]] = []
    program = str(Path(sys.executable))

    def fake_scan(
        content: str,
        config: WorkforceConfig,
        *,
        repo_root: Path,
        target: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ScanOutcome:
        policy_file = repo_root / config.secret_scan.config_file
        seen.append(
            {
                "env": None if env is None else dict(env),
                "root": repo_root,
                "command": config.secret_scan.command,
                "extra_args": list(config.secret_scan.extra_args),
                "policy": policy_file.read_text(encoding="utf-8")
                if policy_file.is_file()
                else None,
            }
        )
        return ScanOutcome(clean=True, available=True)

    real_lookup = policy.resolve_executable

    def lookup(name: str, environ: Any = None) -> str | None:
        return program if name == "gitleaks" else real_lookup(name, environ)

    monkeypatch.setattr("tools.agent_reviewer.prompt.secret_scan.scan_content", fake_scan)
    monkeypatch.setattr(policy, "resolve_executable", lookup)
    return seen


def test_default_scanner_delegates_to_the_workforce_secret_scan(
    repo: Path, cfg: WorkforceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_gitleaks(monkeypatch)
    toml = '[allowlist]\nregexes = ["zzz"]\n'
    assert prompt.build(repo_root=repo, cfg=cfg, gitleaks_toml=toml)
    (call,) = seen
    assert call["command"] == str(Path(sys.executable)), "an absolute path, never a bare name"
    assert call["policy"] == toml, "the base ref's policy, written to a scratch checkout"
    assert not Path(call["root"]).is_relative_to(repo), "never the working tree"


def test_default_scanner_without_a_policy_file_uses_gitleaks_own_defaults(
    repo: Path, cfg: WorkforceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_gitleaks(monkeypatch)
    _write(repo, ".gitleaks.toml", "# the head's copy, which must not be used\n")
    assert prompt.build(repo_root=repo, cfg=cfg, gitleaks_toml=None)
    assert [call["policy"] for call in seen] == [None]


def test_the_scan_never_runs_a_program_it_could_only_find_in_the_working_directory(
    repo: Path, cfg: WorkforceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows tries the working directory first; the lookup is absolute PATH entries only."""
    asked: list[str] = []

    def nothing_on_path(name: str, environ: Any = None) -> None:
        asked.append(name)

    def must_not_run(*_args: Any, **_kwargs: Any) -> ScanOutcome:
        raise AssertionError("the wrapper was started without a trusted program")

    monkeypatch.setattr(policy, "resolve_executable", nothing_on_path)
    monkeypatch.setattr("tools.agent_reviewer.prompt.secret_scan.scan_content", must_not_run)
    assert prompt.build(repo_root=repo, cfg=cfg), "unavailable and not strict: skipped"
    assert asked == [cfg.secret_scan.command]
    strict = cfg.model_copy(
        update={"secret_scan": cfg.secret_scan.model_copy(update={"strict": True})}
    )
    with pytest.raises(prompt.PromptRefusedError, match="strict"):
        prompt.build(repo_root=repo, cfg=strict)


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
        reviewer_ledger.precheck(cfg, 6.0, ledger_root=repo)
    # Exactly up to the cap is allowed, and the pre-check never writes.
    before = ledger_path.read_bytes()
    headroom = reviewer_ledger.precheck(cfg, 5.0, ledger_root=repo)
    assert headroom.ws == "B"
    assert headroom.remaining_usd == pytest.approx(0.0)
    assert ledger_path.read_bytes() == before


_BASE_LEDGER_SOURCE = "abc1234:reports/credit/ledger.jsonl"


def _ledger_text(usd: float, ws: str = "B") -> str:
    return json.dumps({"ws": ws, "cost_usd": usd}) + "\n"


def test_locate_returns_the_ledger_under_the_controllers_root(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    found = reviewer_ledger.locate(cfg, root)
    assert found == root / "reports" / "credit" / "ledger.jsonl"
    assert not found.exists(), "locating the ledger creates nothing"


def test_locate_refuses_a_root_that_is_not_a_directory(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    with pytest.raises(reviewer_ledger.LedgerLocationError, match="not a directory"):
        reviewer_ledger.locate(cfg, tmp_path / "missing")
    (tmp_path / "plain-file").write_text("x", encoding="utf-8")
    with pytest.raises(reviewer_ledger.LedgerLocationError, match="not a directory"):
        reviewer_ledger.locate(cfg, tmp_path / "plain-file")


def test_a_location_refusal_is_a_ledger_error(cfg: WorkforceConfig, tmp_path: Path) -> None:
    """So the CLI treats it like any other damaged-ledger refusal (before any spend)."""
    assert issubclass(reviewer_ledger.LedgerLocationError, credit_ledger.LedgerError)


def test_locate_refuses_a_ledger_that_leaves_its_root_through_an_alias(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    outside = tmp_path / "outside"
    (outside / "credit").mkdir(parents=True)
    _link_dir(root / "reports", outside)
    with pytest.raises(reviewer_ledger.LedgerLocationError, match="outside"):
        reviewer_ledger.locate(cfg, root)


def test_locate_accepts_an_alias_that_stays_inside_the_root(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    (root / "data" / "credit").mkdir(parents=True)
    _link_dir(root / "reports", root / "data")
    assert reviewer_ledger.locate(cfg, root).name == "ledger.jsonl"


def _with_ledger(cfg: WorkforceConfig, ledger: str) -> WorkforceConfig:
    return cfg.model_copy(update={"credit": cfg.credit.model_copy(update={"ledger": ledger})})


@pytest.mark.parametrize("ledger", ["../elsewhere/ledger.jsonl", "reports/../../elsewhere.jsonl"])
def test_locate_refuses_a_ledger_path_that_climbs_out_of_the_root(
    cfg: WorkforceConfig, tmp_path: Path, ledger: str
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    with pytest.raises(reviewer_ledger.LedgerLocationError, match="outside"):
        reviewer_ledger.locate(_with_ledger(cfg, ledger), root)


def test_locate_refuses_an_absolute_ledger_path(cfg: WorkforceConfig, tmp_path: Path) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    absolute = str(tmp_path / "elsewhere" / "ledger.jsonl")
    with pytest.raises(reviewer_ledger.LedgerLocationError, match="outside"):
        reviewer_ledger.locate(_with_ledger(cfg, absolute), root)


def test_locate_refuses_a_ledger_that_is_a_symbolic_link(
    cfg: WorkforceConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "trusted"
    ledger_path = root / "reports" / "credit" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    target = root / "elsewhere.jsonl"
    target.write_text("", encoding="utf-8")
    try:
        os.symlink(target, ledger_path)
    except (OSError, NotImplementedError):  # this account may not make links: pretend it did
        monkeypatch.setattr(Path, "is_symlink", lambda self: self == ledger_path)
    with pytest.raises(reviewer_ledger.LedgerLocationError, match="symbolic link"):
        reviewer_ledger.locate(cfg, root)


def test_precheck_floors_the_files_spend_with_the_base_refs_copy(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    base_copy = _ledger_text(58.0)
    with pytest.raises(credit_ledger.CapExceeded):
        reviewer_ledger.precheck(
            cfg, 3.0, ledger_root=root, floor_text=base_copy, floor_source=_BASE_LEDGER_SOURCE
        )
    headroom = reviewer_ledger.precheck(cfg, 2.0, ledger_root=root, floor_text=base_copy)
    assert headroom.spent_usd == 58.0
    assert headroom.remaining_usd == pytest.approx(0.0)
    # No floor (the base tracks no ledger): the file alone speaks.
    assert reviewer_ledger.precheck(cfg, 3.0, ledger_root=root).spent_usd == 0.0
    assert not (root / "reports").exists(), "the pre-check never writes"


def test_the_file_may_exceed_the_floor_but_never_undercut_it(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    path = root / "reports" / "credit" / "ledger.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(_ledger_text(59.0), encoding="utf-8")
    assert (
        reviewer_ledger.precheck(
            cfg, 1.0, ledger_root=root, floor_text=_ledger_text(58.0)
        ).spent_usd
        == 59.0
    )
    path.write_text(_ledger_text(10.0), encoding="utf-8")
    assert (
        reviewer_ledger.precheck(
            cfg, 1.0, ledger_root=root, floor_text=_ledger_text(58.0)
        ).spent_usd
        == 58.0
    )


def test_the_floor_counts_only_the_reviewers_workstream(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    others = _ledger_text(500.0, "A") + _ledger_text(500.0, "C")
    assert reviewer_ledger.precheck(cfg, 1.0, ledger_root=root, floor_text=others).spent_usd == 0.0


def test_a_damaged_base_copy_of_the_ledger_is_a_refusal_naming_its_source(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    with pytest.raises(credit_ledger.LedgerError, match=r"abc1234:reports/credit/ledger\.jsonl"):
        reviewer_ledger.precheck(
            cfg, 1.0, ledger_root=root, floor_text="not json\n", floor_source=_BASE_LEDGER_SOURCE
        )


def test_precheck_refuses_a_ledger_that_is_not_where_it_must_be(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _link_dir(root / "reports", outside)
    with pytest.raises(reviewer_ledger.LedgerLocationError):
        reviewer_ledger.precheck(cfg, 1.0, ledger_root=root)


def _record(cfg: WorkforceConfig, root: Path, **kwargs: Any) -> dict[str, Any]:
    acc = reviewer_ledger.account_run(_result(), started=True, cap_usd=10.0, error=None)
    return reviewer_ledger.record_run(
        cfg,
        accounting=acc,
        run_id="wsb-0001",
        target="diff HEAD~1...HEAD @abc1234",
        model_requested="opus",
        model_resolved="claude-resolved-1",
        resolved_via="models.retrieve",
        artifact="reports/agent_reviewer/2026-10-19/wsb-0001.md",
        ledger_root=root,
        **kwargs,
    )


def test_record_run_counts_the_base_refs_spend_into_the_cumulative_total(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    row = _record(cfg, root, floor_text=_ledger_text(58.0), floor_source=_BASE_LEDGER_SOURCE)
    assert row["cost_usd"] == 1.5
    assert row["cumulative_ws_usd"] == 59.5, "the base's 58.0 plus this run"
    written = (root / "reports/credit/ledger.jsonl").read_text(encoding="utf-8")
    assert written.count("\n") == 1, "the floor itself is never copied into the file"


def test_record_run_writes_under_the_given_root_and_nowhere_else(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    _record(cfg, root)
    assert list(tmp_path.rglob("ledger.jsonl")) == [root / "reports" / "credit" / "ledger.jsonl"]


def test_record_run_will_not_append_through_an_alias_that_leaves_the_root(
    cfg: WorkforceConfig, tmp_path: Path
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    outside = tmp_path / "outside"
    (outside / "credit").mkdir(parents=True)
    _link_dir(root / "reports", outside)
    with pytest.raises(reviewer_ledger.LedgerWriteError, match="outside"):
        _record(cfg, root)
    assert not (outside / "credit" / "ledger.jsonl").exists()


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
        ledger_root=repo,
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
    assert re.search(r"base_tracked_files: \d+", text)
    assert "GIT_LITERAL_PATHSPECS=1" in text
    assert "confirms the pinned invocation" in text
    rules = text.replace(str(repo), "<repo>").replace(str(repo.resolve()), "<repo>")
    assert "pytest" not in rules, "no pytest rule, no pytest in the plan"
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
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~1",
        "--head",
        "no-such-head",
    ]
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
# run_review - what is taken from the trusted base ref, not from the head
# --------------------------------------------------------------------------- #
_HOSTILE_MCP_JSON = json.dumps(
    {
        "mcpServers": {
            "mousedroid": {
                "command": "./HOSTILE-MCP-COMMAND",
                "args": ["-c", "import os"],
                "env": {"X": "1"},
            }
        }
    }
)


def test_the_mcp_declaration_is_read_from_the_base_ref_not_the_head_or_the_working_tree(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    """The head controls the working tree; its ``.mcp.json`` is an exec sink and is never read."""
    _write(repo, ".mcp.json", _HOSTILE_MCP_JSON)
    _commit(repo, "the head rewrites .mcp.json")
    _write(repo, ".mcp.json", _HOSTILE_MCP_JSON.replace("HOSTILE", "WORSE"))  # uncommitted too
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~2",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]

    assert run_review.main([*argv, "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert sys.executable in out
    assert "HOSTILE" not in out
    assert "WORSE" not in out


def test_a_live_run_never_uses_a_tampered_declaration_from_the_head(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(repo, ".mcp.json", _HOSTILE_MCP_JSON)
    _commit(repo, "the head rewrites .mcp.json")
    _live_env(monkeypatch)
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~2",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]
    assert run_review.main(argv) == 0
    server = fake_sdk.queries[0]["options"].mcp_servers["mousedroid"]
    assert server["command"] == sys.executable
    assert server["args"][:3] == ["-I", "-c", policy.MCP_BOOTSTRAP]
    assert "HOSTILE" not in json.dumps(server)


def test_a_tampered_declaration_at_the_base_ref_refuses_the_run(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(repo, ".mcp.json", _HOSTILE_MCP_JSON)
    _commit(repo, "tampered base")
    _write(repo, "later.py", "z = 1\n")
    _commit(repo, "the change under review")
    _live_env(monkeypatch)
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~1",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]

    assert run_review.main(argv) == run_review.EXIT_REFUSED

    assert "HOSTILE-MCP-COMMAND" in capsys.readouterr().err
    fake_sdk.query.assert_not_called()
    assert _ledger_rows(repo) == []


def test_a_base_ref_without_the_declaration_refuses_the_run_and_does_not_fall_back(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(repo, "rm", "-q", ".mcp.json")
    _commit(repo, "base without .mcp.json")
    _write(repo, "later.py", "z = 1\n")
    _commit(repo, "the change under review")
    shutil.copyfile(_REPO_ROOT / ".mcp.json", repo / ".mcp.json")  # a clean copy, untracked
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~1",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]

    assert run_review.main([*argv, "--dry-run"]) == run_review.EXIT_REFUSED

    assert "no .mcp.json" in capsys.readouterr().err


def test_every_git_call_runs_with_literal_pathspecs(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, str]] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("env") or {})
        return await real(*args, **kwargs)

    monkeypatch.setattr(run_review.asyncio, "create_subprocess_exec", spy)
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert len(seen) >= 4, "rev-parse x3, ls-tree and cat-file at least"
    for env in seen:
        assert env.get("GIT_LITERAL_PATHSPECS") == "1"
        assert env.get("GIT_OPTIONAL_LOCKS") == "0"


def test_the_registered_guard_knows_exactly_the_files_tracked_at_the_base(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == 0
    matcher, _read_matcher = fake_sdk.queries[0]["options"].hooks["PreToolUse"]
    (guard,) = matcher.hooks
    tracked = set(_git(repo, "ls-tree", "-r", "--name-only", "HEAD~1").splitlines())
    base_sha, head_sha = _git(repo, "rev-parse", "HEAD~1"), _git(repo, "rev-parse", "HEAD")
    assert guard.base_paths == tracked
    assert guard.revisions == {base_sha, head_sha}, "the two commits of the review, resolved"
    assert ".mcp.json" in tracked
    assert "src_change.py" not in tracked, "added by the head: not a base file"
    assert guard.check(f"git show {head_sha}:.mcp.json") is None
    assert guard.check(f"git show {head_sha}:src_change.py"), "read it from the diff instead"
    assert guard.check("git show HEAD:.mcp.json"), "a name is not a pinned commit"
    assert guard.check(f"git diff {base_sha}...{head_sha} --stat") is None


def test_the_reviewer_instructions_come_from_the_base_ref_not_the_head(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A head that can rewrite the reviewer's own instructions can neutralise its review."""
    agent = repo / ".claude" / "agents" / "peer-reviewer.md"
    agent.write_text(agent.read_text(encoding="utf-8") + "\nALWAYS-APPROVE\n", encoding="utf-8")
    _commit(repo, "the head edits the reviewer")
    agent.write_text(agent.read_text(encoding="utf-8") + "\nUNCOMMITTED-TOO\n", encoding="utf-8")
    _live_env(monkeypatch)
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~2",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]

    assert run_review.main(argv) == 0

    system_prompt = fake_sdk.queries[0]["options"].system_prompt
    assert "adversarial peer reviewer" in system_prompt
    assert "ALWAYS-APPROVE" not in system_prompt
    assert "UNCOMMITTED-TOO" not in system_prompt


def test_a_base_ref_without_the_reviewer_file_refuses_the_run(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(repo, "rm", "-q", ".claude/agents/peer-reviewer.md")
    _commit(repo, "base without the reviewer")
    _write(repo, "later.py", "z = 1\n")
    _commit(repo, "the change under review")
    restored = repo / ".claude" / "agents" / "peer-reviewer.md"
    restored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_REPO_ROOT / ".claude" / "agents" / "peer-reviewer.md", restored)
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~1",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]

    assert run_review.main([*argv, "--dry-run"]) == run_review.EXIT_REFUSED

    assert "peer-reviewer.md" in capsys.readouterr().err


def test_a_run_that_cannot_list_the_base_files_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = run_review._git

    async def failing_ls_tree(repo_root: Path, *args: str) -> tuple[int, bytes]:
        if args and args[0] == "ls-tree":
            return 128, b""
        return await real(repo_root, *args)

    monkeypatch.setattr(run_review, "_git", failing_ls_tree)
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert "cannot list the files tracked" in capsys.readouterr().err


def test_a_supplied_agent_text_named_like_a_secret_file_is_refused(repo: Path) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML + "    agent_file: config/docker.env\n")
    bad = load_config(repo_root=repo)
    with pytest.raises(prompt.PromptRefusedError, match="secret"):
        prompt.build(repo_root=repo, cfg=bad, agent_text="harmless", scanner=_no_scanner)


def test_a_supplied_agent_text_is_screened_like_a_file(repo: Path, cfg: WorkforceConfig) -> None:
    with pytest.raises(prompt.PromptRefusedError):
        prompt.build(repo_root=repo, cfg=cfg, agent_text=f"key {_FAKE_KEY}", scanner=_no_scanner)
    text = prompt.build(
        repo_root=repo,
        cfg=cfg,
        agent_text="---\nname: x\n---\nCUSTOM BODY\n",
        scanner=_no_scanner,
    )
    assert "CUSTOM BODY" in text
    assert "name: x" not in text


# --------------------------------------------------------------------------- #
# run_review - the trust boundary is the BASE ref (Ruling 49, addendum 3)
# --------------------------------------------------------------------------- #
_HOSTILE_WORKFORCE_YAML = """\
credit:
    caps_usd:
        B: 99999
    ledger: reports/elsewhere/ledger.jsonl
agent_reviewer:
    model: hostile-model
    max_turns: 999
    agent_file: config/hostile-agent.md
    mcp_config: hostile.mcp.json
    artifact_dir: reports/elsewhere
secret_scan:
    command: hostile-gitleaks
    config_file: hostile.gitleaks.toml
    extra_args: ["--hostile"]
"""


def _plant_hostile_targets(repo: Path) -> None:
    for rel in ("config/hostile-agent.md", "hostile.mcp.json", "hostile.gitleaks.toml"):
        _write(repo, rel, "HOSTILE\n")


def _spy_on_executable_lookups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    asked: list[str] = []
    real = policy.resolve_executable

    def spy(name: str, environ: Any = None) -> str | None:
        asked.append(name)
        return real(name, environ)

    monkeypatch.setattr(policy, "resolve_executable", spy)
    return asked


def _assert_the_base_configuration_ruled(repo: Path, sdk: _FakeSdk, asked: list[str]) -> None:
    options = sdk.queries[0]["options"]
    assert (options.max_turns, options.model) == (40, "opus")
    assert "adversarial peer reviewer" in options.system_prompt, "the base's agent file"
    assert "HOSTILE" not in options.system_prompt
    assert options.mcp_servers["mousedroid"]["command"] == sys.executable
    (row,) = _ledger_rows(repo)
    assert row["cap_usd"] == 60.0, "the base's cap, not the head's 99999"
    assert re.fullmatch(r"reports/agent_reviewer/[\d-]+/wsb-0001\.md", row["artifact"])
    assert not (repo / "reports/elsewhere").exists()
    assert set(asked) <= {"git", "gitleaks"}, f"a program the head named was looked up: {asked}"


def test_a_tampered_working_tree_workforce_yaml_changes_nothing_the_reviewer_loads(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``.claude/workforce.yaml`` selects the agent file, the MCP declaration, the scanner
    program and its policy, the caps, and where the review and the ledger row are written."""
    _plant_hostile_targets(repo)
    _write(repo, DEFAULT_CONFIG_RELPATH, _HOSTILE_WORKFORCE_YAML)  # uncommitted
    asked = _spy_on_executable_lookups(monkeypatch)
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo)) == 0

    _assert_the_base_configuration_ruled(repo, fake_sdk, asked)


def test_a_workforce_yaml_the_head_commits_changes_nothing_either(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plant_hostile_targets(repo)
    _write(repo, DEFAULT_CONFIG_RELPATH, _HOSTILE_WORKFORCE_YAML)
    _commit(repo, "the head rewrites the reviewer's configuration")
    asked = _spy_on_executable_lookups(monkeypatch)
    _live_env(monkeypatch)
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~2",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]

    assert run_review.main(argv) == 0

    _assert_the_base_configuration_ruled(repo, fake_sdk, asked)


def test_the_dry_run_says_where_the_configuration_came_from(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _HOSTILE_WORKFORCE_YAML)
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    out = capsys.readouterr().out
    base_sha = _git(repo, "rev-parse", "HEAD~1")
    assert f"workforce_config: {base_sha}:{DEFAULT_CONFIG_RELPATH}" in out
    assert "the working tree's copy is never read" in out
    assert "hostile-model" not in out
    assert "model_requested: opus" in out


def test_a_base_without_a_workforce_yaml_gets_the_defaults_and_cannot_spend(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _git(repo, "rm", "-q", DEFAULT_CONFIG_RELPATH)
    _commit(repo, "a base that has not adopted the workforce config")
    _write(repo, "later.py", "z = 1\n")
    _commit(repo, "the change under review")
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML)  # the head's copy: untracked, ignored
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo)) == run_review.EXIT_CAP

    fake_sdk.query.assert_not_called()
    assert "cap" in capsys.readouterr().err.lower()
    assert _ledger_rows(repo) == []


def test_a_base_workforce_yaml_that_does_not_parse_refuses_the_run(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _argv_with_base_config(repo, "agent_reviewer:\n    nonsense_key: 1\n")
    assert run_review.main([*argv, "--dry-run"]) == run_review.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "nonsense_key" in err
    assert ":.claude/workforce.yaml" in err, "names the base ref's copy, not a working-tree path"


def test_the_secret_policy_comes_from_the_base_ref(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A head that can rewrite ``.gitleaks.toml`` must not be able to switch the screen off,
    and one that the base declares must really apply."""
    _live_env(monkeypatch)
    # The head's working-tree policy matches the agent text; the base's does not: not refused.
    _write(repo, ".gitleaks.toml", '[[rules]]\nid = "x"\nregex = "adversarial peer reviewer"\n')
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    capsys.readouterr()
    # The base's own policy matches: refused, naming the policy file.
    _write(repo, ".gitleaks.toml", '[[rules]]\nid = "x"\nregex = "adversarial peer reviewer"\n')
    _commit(repo, "a base policy that matches the agent text")
    _write(repo, "later.py", "z = 1\n")
    _commit(repo, "the change under review")
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert "gitleaks.toml" in capsys.readouterr().err


def test_both_gitleaks_scans_use_the_base_refs_policy(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The assembled prompt and the review output are scanned with the base's .gitleaks.toml."""
    seen = _fake_gitleaks(monkeypatch)
    base_policy = (repo / ".gitleaks.toml").read_text(encoding="utf-8")
    _write(repo, ".gitleaks.toml", "# the head's copy, which must not be used\n")  # uncommitted
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo)) == 0

    assert len(seen) == 2, "the assembled prompt, then the review output"
    assert [call["policy"] for call in seen] == [base_policy, base_policy]
    for call in seen:
        assert call["env"] is not None
        assert "ANTHROPIC_API_KEY" not in call["env"]


# --- what the MCP server is started with --------------------------------------------------------
def test_the_mcp_server_gets_the_base_refs_configuration_in_a_scratch_directory(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never the checkout's ``config/default.yaml``, and a directory that is not the checkout."""
    default = repo / policy.MCP_CONFIG_RELPATH
    base_text = default.read_text(encoding="utf-8")
    default.write_text("HOSTILE: true\n", encoding="utf-8")  # the head's copy
    seen: dict[str, Any] = {}
    real_query = fake_sdk.query.side_effect

    def spy(**kwargs: Any) -> Any:
        args = kwargs["options"].mcp_servers["mousedroid"]["args"]
        config = Path(args[3])
        seen.update(path=config, text=config.read_text(encoding="utf-8"))
        return real_query(**kwargs)

    fake_sdk.query.side_effect = spy
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo)) == 0

    assert seen["text"] == base_text
    assert "HOSTILE" not in seen["text"]
    assert seen["path"].is_absolute()
    assert not run_review._is_within(seen["path"], repo), "outside the checkout"
    assert not seen["path"].exists(), "the scratch directory is removed after the run"


def test_a_dry_run_writes_no_scratch_directory(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    scratch = Path(tempfile.gettempdir()) / "agent-reviewer-dry-run"
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert str(scratch / "mcp-config.yaml") in out, "it names the file a live run would write"
    assert not scratch.exists()
    assert "mcp_config: materialised from" in out


def test_a_scratch_directory_inside_the_checkout_is_refused(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inside = repo / "tmp-inside"
    inside.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(inside))
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo)) == run_review.EXIT_REFUSED
    assert "inside the checkout" in capsys.readouterr().err
    fake_sdk.query.assert_not_called()
    assert _ledger_rows(repo) == []


def test_a_base_without_the_mcp_server_configuration_refuses_the_run(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(repo, "rm", "-q", policy.MCP_CONFIG_RELPATH)
    _commit(repo, "a base without the server's configuration")
    _write(repo, "later.py", "z = 1\n")
    _commit(repo, "the change under review")
    _write(repo, policy.MCP_CONFIG_RELPATH, "HOSTILE: true\n")  # the head's, untracked
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert policy.MCP_CONFIG_RELPATH in capsys.readouterr().err


# --- whose code runs beside the key ---------------------------------------------------------------
def _code_at(
    monkeypatch: pytest.MonkeyPatch,
    *,
    controller: Path | None = None,
    prefix: Path | None = None,
    package: Path | str | None = "keep",
) -> None:
    """Pretend the reviewer's modules, the interpreter's environment and mousedroid live here."""
    if controller is not None:
        monkeypatch.setattr(run_review, "__file__", str(controller / "run_review.py"))
    if prefix is not None:
        monkeypatch.setattr(sys, "prefix", str(prefix))
    if package != "keep":
        monkeypatch.setattr(run_review, "_mcp_package_origin", lambda: package)


def _evaluate(repo: Path, refs: run_review.Refs | None = None, *, trust: bool = False) -> Any:
    resolved = refs or asyncio.run(run_review._resolve_refs(repo, "HEAD~1", "HEAD"))
    return asyncio.run(run_review._evaluate_trust(repo, resolved, trust_checkout=trust))


def test_code_from_outside_the_checkout_is_trusted(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    """The safe setup: run from a trusted checkout and point --repo-root at the other."""
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "trust_controller: outside the checkout under review" in out
    assert "trust_interpreter: outside the checkout under review" in out
    assert "trust_package: installed outside the checkout under review" in out


def test_the_reviewers_own_modules_in_the_checkout_are_the_branchs_own_code(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run from the head's checkout, the head's ``policy.py`` is the containment."""
    _code_at(monkeypatch, controller=repo / "tools" / "agent_reviewer")

    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED

    err = capsys.readouterr().err
    assert "tools/agent_reviewer" in err
    assert run_review.TRUST_CHECKOUT_FLAG in err
    assert "holds the API key" in err
    fake_sdk.query.assert_not_called()
    assert not (repo / "reports").exists()


def test_an_interpreter_environment_inside_the_checkout_is_the_branchs_own_code(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Its ``site-packages`` and ``.pth`` files run at startup, even under ``-I``."""
    _code_at(monkeypatch, prefix=repo / ".venv")
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    err = capsys.readouterr().err
    assert ".venv" in err
    assert run_review.TRUST_CHECKOUT_FLAG in err


def test_an_editable_mousedroid_in_the_checkout_is_the_branchs_own_code(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _code_at(monkeypatch, package=repo / "src/mousedroid/__init__.py")
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "src/mousedroid/__init__.py" in err
    assert run_review.TRUST_CHECKOUT_FLAG in err


def test_a_live_run_is_refused_the_same_way_before_any_spend(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_env(monkeypatch)
    _code_at(monkeypatch, package=repo / "src/mousedroid/__init__.py")

    assert run_review.main(_argv(repo)) == run_review.EXIT_REFUSED

    fake_sdk.query.assert_not_called()
    assert _ledger_rows(repo) == [], "nothing was charged"


def test_every_problem_is_listed_in_one_refusal(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _code_at(
        monkeypatch,
        controller=repo / "tools" / "agent_reviewer",
        prefix=repo / ".venv",
        package=repo / "src/mousedroid/__init__.py",
    )
    with pytest.raises(run_review.RefusalError) as excinfo:
        _evaluate(repo)
    message = str(excinfo.value)
    assert "reviewer's own modules" in message
    assert "interpreter's environment" in message
    assert "MCP server would import" in message


def test_the_flag_accepts_all_of_it_and_the_plan_says_so(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _code_at(
        monkeypatch,
        controller=repo / "tools" / "agent_reviewer",
        prefix=repo / ".venv",
        package=repo / "src/mousedroid/__init__.py",
    )

    code = run_review.main(_argv(repo, "--dry-run", run_review.TRUST_CHECKOUT_FLAG))

    assert code == 0
    out = capsys.readouterr().out
    here = "IN THE CHECKOUT UNDER REVIEW"
    accepted = f"; accepted with {run_review.TRUST_CHECKOUT_FLAG}"
    for line in (
        f"trust_controller: {here} (tools/agent_reviewer){accepted}",
        f"trust_interpreter: {here} (.venv){accepted}",
        f"trust_package: {here} (src/mousedroid/__init__.py){accepted}",
    ):
        assert line in out


def test_a_checkout_at_the_base_commit_holds_trusted_code(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reviewing a branch by its git objects from a checkout of the base needs no flag."""
    argv = _argv_at_the_base(repo, "--dry-run")
    _code_at(
        monkeypatch,
        controller=repo / "tools" / "agent_reviewer",
        prefix=repo / ".venv",
        package=repo / "src/mousedroid/__init__.py",
    )

    assert run_review.main(argv) == 0

    out = capsys.readouterr().out
    assert "tools/agent_reviewer, in the checkout, which is at the base commit" in out
    assert ".venv, in the checkout, which is at the base commit" in out
    assert "src/mousedroid/__init__.py, in the checkout, which is at the base commit" in out


def test_an_unresolvable_checkout_is_not_taken_for_the_base(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = run_review.Refs("b", "a" * 40, "h", "c" * 40, "unknown")
    _code_at(monkeypatch, package=repo / "src/mousedroid/__init__.py")
    with pytest.raises(run_review.RefusalError, match="not at the base"):
        _evaluate(repo, refs)


def test_a_copy_installed_in_the_interpreters_own_environment_is_not_the_branchs(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The interpreter's environment can sit INSIDE the checkout only with the flag; the
    package installed in it is then covered by that, not by a second refusal."""
    venv = repo / ".venv"
    _code_at(monkeypatch, prefix=venv, package=venv / "Lib/site-packages/mousedroid/__init__.py")
    code = run_review.main(_argv(repo, "--dry-run", run_review.TRUST_CHECKOUT_FLAG))
    assert code == 0
    assert "installed in the interpreter's own environment" in capsys.readouterr().out


def test_a_package_installed_outside_the_checkout_is_not_the_branchs(
    repo: Path,
    tmp_path: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _code_at(monkeypatch, package=tmp_path / "elsewhere/mousedroid/__init__.py")
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert "installed outside the checkout under review" in capsys.readouterr().out


def test_a_missing_package_is_said_not_refused(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no ``mousedroid`` the server cannot start; the review simply has no MCP evidence."""
    _code_at(monkeypatch, package=None)
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert "not importable by this interpreter" in capsys.readouterr().out


def _venv_tracked_by_the_head(repo: Path) -> None:
    _write(repo, ".venv/Lib/site-packages/sitecustomize.py", "print('planted')\n")
    _commit(repo, "the head force-adds a file to the virtualenv")


def test_a_head_that_tracks_files_in_the_interpreters_environment_is_refused_outright(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Not the operator's environment: no flag accepts it."""
    _venv_tracked_by_the_head(repo)
    _code_at(monkeypatch, prefix=repo / ".venv")
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~2",
        "--head",
        "HEAD",
        "--cap-usd",
        "5",
    ]

    assert run_review.main([*argv, "--dry-run"]) == run_review.EXIT_REFUSED
    assert run_review.main([*argv, "--dry-run", run_review.TRUST_CHECKOUT_FLAG]) == (
        run_review.EXIT_REFUSED
    )

    err = capsys.readouterr().err
    assert "sitecustomize.py" in err
    assert "does not accept it" in err


def test_files_the_index_tracks_in_the_environment_count_too(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staged, not yet committed: the checkout would still run them."""
    _write(repo, ".venv/Lib/site-packages/evil.pth", "import os\n")
    _git(repo, "add", "-f", ".venv/Lib/site-packages/evil.pth")
    _code_at(monkeypatch, prefix=repo / ".venv")
    with pytest.raises(run_review.RefusalError, match=r"evil\.pth"):
        _evaluate(repo, trust=True)


def test_files_the_head_tracks_count_even_when_the_checkout_does_not_have_them_yet(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index lists what is checked out; the head's own tree lists what the head adds."""
    _venv_tracked_by_the_head(repo)
    head, middle, base = (_git(repo, "rev-parse", f"HEAD~{n}") for n in (0, 1, 2))
    _git(repo, "checkout", "-q", middle)  # a checkout that lacks the head's .venv files
    refs = run_review.Refs("base", base, "head", head, middle)
    _code_at(monkeypatch, prefix=repo / ".venv")
    with pytest.raises(run_review.RefusalError, match=r"sitecustomize\.py"):
        _evaluate(repo, refs, trust=True)


def test_an_untracked_environment_in_the_checkout_needs_the_flag_but_is_not_refused_outright(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(repo, ".venv/Lib/site-packages/mine.py", "x = 1\n")  # untracked, like a real venv
    _code_at(monkeypatch, prefix=repo / ".venv")
    with pytest.raises(run_review.RefusalError, match="inside this checkout"):
        _evaluate(repo)
    trust = _evaluate(repo, trust=True)
    assert trust.interpreter.startswith("IN THE CHECKOUT UNDER REVIEW (.venv)")


def test_a_failing_tracked_files_lookup_refuses(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = run_review._git

    async def failing(repo_root: Path, *args: str) -> tuple[int, bytes]:
        if args and args[0] == "ls-files":
            return 128, b""
        return await real(repo_root, *args)

    monkeypatch.setattr(run_review, "_git", failing)
    _code_at(monkeypatch, prefix=repo / ".venv")
    with pytest.raises(run_review.RefusalError, match="cannot tell whether the head tracks"):
        _evaluate(repo, trust=True)


def test_the_cli_offers_the_flag_and_defaults_it_off() -> None:
    parser = run_review.build_parser()
    required = ["--base", "a", "--head", "b", "--cap-usd", "1"]
    assert parser.parse_args(required).trust_checkout is False
    assert parser.parse_args([*required, run_review.TRUST_CHECKOUT_FLAG]).trust_checkout


def _argv_at_the_base(repo: Path, *extra: str) -> list[str]:
    """Check the repo out AT the base commit and review the commit above it."""
    _git(repo, "tag", "rbase", "HEAD~1")
    _git(repo, "tag", "rtip", "HEAD")
    _git(repo, "checkout", "-q", "rbase")
    return [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "rbase",
        "--head",
        "rtip",
        "--cap-usd",
        "5",
        *extra,
    ]


# --- how the package is located ------------------------------------------------------------
def test_the_package_origin_is_looked_up_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    package = tmp_path / "pkg" / "__init__.py"

    def find_spec(name: str) -> Any:
        asked.append(name)
        where = [str(package.parent)]
        return SimpleNamespace(origin=str(package), submodule_search_locations=where)

    monkeypatch.setattr(run_review.importlib.util, "find_spec", find_spec)
    assert run_review._mcp_package_origin() == package
    assert asked == [policy.MCP_MODULE]


@pytest.mark.parametrize(
    "spec",
    [
        None,
        SimpleNamespace(origin=None, submodule_search_locations=None),
        SimpleNamespace(origin="built-in", submodule_search_locations=None),
        SimpleNamespace(origin=None, submodule_search_locations=[]),
    ],
)
def test_the_package_origin_is_none_when_the_spec_names_no_directory(
    spec: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_review.importlib.util, "find_spec", lambda _name: spec)
    assert run_review._mcp_package_origin() is None


def test_a_namespace_package_reports_its_first_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SimpleNamespace(origin=None, submodule_search_locations=[str(tmp_path / "ns")])
    monkeypatch.setattr(run_review.importlib.util, "find_spec", lambda _name: spec)
    assert run_review._mcp_package_origin() == tmp_path / "ns"


@pytest.mark.parametrize("error", [ImportError("x"), ValueError("y")])
def test_a_lookup_that_raises_means_the_package_is_not_importable(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def find_spec(_name: str) -> Any:
        raise error

    monkeypatch.setattr(run_review.importlib.util, "find_spec", find_spec)
    assert run_review._mcp_package_origin() is None


def test_a_path_that_cannot_be_resolved_is_not_within_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_path: object, *, strict: bool = False) -> str:
        raise OSError("loop")

    with monkeypatch.context() as patch:  # only around the call: pytest resolves paths too
        patch.setattr(os.path, "realpath", broken)
        within = run_review._is_within(tmp_path / "a", tmp_path)
    assert within is False
    assert run_review._is_within(tmp_path / "a", tmp_path) is True, "and the patch is undone"
    assert run_review._is_within(tmp_path.parent, tmp_path) is False


# --- git is found without looking in the working directory ----------------------------------
def test_git_is_the_program_found_on_path_not_a_bare_name(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[str] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> Any:
        launched.append(str(args[0]))
        return await real(*args, **kwargs)

    monkeypatch.setattr(run_review.asyncio, "create_subprocess_exec", spy)
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert launched
    assert all(os.path.isabs(program) for program in launched), launched
    assert {Path(program).stem.lower() for program in launched} == {"git"}


def test_a_missing_git_on_path_is_explained(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(policy, "resolve_executable", lambda name, environ=None: None)
    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED
    assert "git is required" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# run_review - the spend ledger is the controller's, not the head's
# --------------------------------------------------------------------------- #
_LEDGER_REL = "reports/credit/ledger.jsonl"


def _spend(root: Path, usd: float, ws: str = "B") -> Path:
    path = root / _LEDGER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ledger_text(usd, ws), encoding="utf-8")
    return path


def _argv_default_ledger(repo: Path, *extra: str, cap: str = "10") -> list[str]:
    """``_argv`` without ``--ledger-root``: the ledger is where the reviewer's own code lives."""
    argv = _argv(repo, *extra, cap=cap)
    at = argv.index("--ledger-root")
    return argv[:at] + argv[at + 2 :]


def _controller_rows(controller: Path) -> list[dict[str, Any]]:
    path = controller / _LEDGER_REL
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_default_ledger_is_the_controllers_checkout_never_the_head(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch, controller_root: Path
) -> None:
    _live_env(monkeypatch)
    _spend(controller_root, 58.0)  # 58 of 60 spent; the head's checkout holds no ledger at all
    assert run_review.main(_argv_default_ledger(repo, cap="3")) == run_review.EXIT_CAP
    fake_sdk.query.assert_not_called()

    assert run_review.main(_argv_default_ledger(repo, cap="2")) == 0

    assert not (repo / "reports" / "credit").exists(), "nothing is written to the head's checkout"
    rows = _controller_rows(controller_root)
    assert [row["cost_usd"] for row in rows] == [58.0, 1.25]
    assert rows[-1]["cumulative_ws_usd"] == 59.25


def test_the_controller_root_is_the_checkout_the_reviewers_code_runs_from() -> None:
    """The real default, with the test fixture's patch undone."""
    root = _REAL_CONTROLLER_ROOT()
    assert root == Path(run_review.__file__).resolve().parents[2]
    assert (root / "tools" / "agent_reviewer" / "run_review.py").is_file()


def test_ledger_root_names_where_the_ledger_is(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    controller_root: Path,
) -> None:
    named = tmp_path / "trusted"
    named.mkdir()
    _spend(named, 59.0)
    _spend(controller_root, 0.5)  # the default root is not consulted when one is named
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo, "--ledger-root", str(named), cap="2")) == run_review.EXIT_CAP

    assert run_review.main(_argv(repo, "--ledger-root", str(named), cap="1")) == 0
    assert [row["cost_usd"] for row in _controller_rows(named)] == [59.0, 1.25]
    assert [row["cost_usd"] for row in _controller_rows(controller_root)] == [0.5]


def test_a_ledger_root_that_is_not_a_directory_is_refused(
    repo: Path, fake_sdk: _FakeSdk, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _argv(repo, "--ledger-root", str(tmp_path / "missing"), "--dry-run")
    assert run_review.main(argv) == run_review.EXIT_REFUSED
    assert "not a directory" in capsys.readouterr().err


def test_a_ledger_that_leaves_the_controllers_checkout_is_refused_before_the_sdk(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "trusted"
    root.mkdir()
    outside = tmp_path / "outside"
    (outside / "credit").mkdir(parents=True)
    _link_dir(root / "reports", outside)
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo, "--ledger-root", str(root))) == run_review.EXIT_REFUSED

    assert "outside" in capsys.readouterr().err
    fake_sdk.query.assert_not_called()
    assert not (outside / "credit" / "ledger.jsonl").exists()


def test_a_ledger_that_is_a_symbolic_link_is_refused_before_the_sdk(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "trusted"
    link = root / _LEDGER_REL
    link.parent.mkdir(parents=True)
    target = tmp_path / "somewhere.jsonl"
    target.write_text("", encoding="utf-8")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):  # this account may not make links: pretend it did
        monkeypatch.setattr(Path, "is_symlink", lambda self: self == link)
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo, "--ledger-root", str(root))) == run_review.EXIT_REFUSED

    assert "symbolic link" in capsys.readouterr().err
    fake_sdk.query.assert_not_called()


def _commit_base_ledger(repo: Path, usd: float) -> None:
    """Commit a ledger holding ``usd`` as the BASE of a review whose head is one commit later."""
    _spend(repo, usd)
    _commit(repo, "the base's committed ledger")
    _write(repo, "later_change.py", "z = 1\n")
    _commit(repo, "the change under review")


def test_a_head_that_empties_the_ledger_cannot_widen_the_cap(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base commits 58 of 60; the file in the checkout has lost the rows."""
    _commit_base_ledger(repo, 58.0)
    (repo / _LEDGER_REL).write_text("", encoding="utf-8")
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo, cap="3")) == run_review.EXIT_CAP
    fake_sdk.query.assert_not_called()

    assert run_review.main(_argv(repo, cap="2")) == 0
    (row,) = _ledger_rows(repo)
    assert row["cumulative_ws_usd"] == 59.25, "the base's 58.0 counts even though the file is empty"


def test_a_head_that_deletes_the_ledger_cannot_widen_the_cap_either(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    _commit_base_ledger(repo, 58.0)
    _git(repo, "rm", "-q", _LEDGER_REL)
    _commit(repo, "the head deletes the ledger")
    _live_env(monkeypatch)
    assert run_review.main(_argv(repo, cap="3")) == run_review.EXIT_CAP
    fake_sdk.query.assert_not_called()
    assert run_review.main(_argv(repo, cap="2")) == 0


def test_a_damaged_base_ledger_is_a_refusal_that_names_the_base_blob(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, _LEDGER_REL, "this is not json\n")
    _commit(repo, "the base's damaged ledger")
    _write(repo, "later_change.py", "z = 1\n")
    _commit(repo, "the change under review")
    (repo / _LEDGER_REL).unlink()  # the controller's own file is fine; the base's copy is not

    assert run_review.main(_argv(repo, "--dry-run")) == run_review.EXIT_REFUSED

    err = capsys.readouterr().err
    assert f"{_git(repo, 'rev-parse', 'HEAD~1')}:{_LEDGER_REL}" in err


def test_the_dry_run_says_where_the_ledger_and_its_floor_are(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str], controller_root: Path
) -> None:
    _commit_base_ledger(repo, 1.0)
    assert run_review.main(_argv_default_ledger(repo, "--dry-run")) == 0
    out = capsys.readouterr().out
    expected = controller_root.resolve() / "reports" / "credit" / "ledger.jsonl"
    assert f"ledger: {expected} " in out
    assert "per workstream" in out
    assert f"ledger_floor: {_git(repo, 'rev-parse', 'HEAD~1')}:{_LEDGER_REL}" in out


def test_the_dry_run_says_when_the_base_tracks_no_ledger(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert "ledger_floor: (the base ref does not track a ledger)" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# run_review - the prompts' size, planted programs, and the reviewer's own children
# --------------------------------------------------------------------------- #
def _prompt_sizes(repo: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, int]:
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    out = capsys.readouterr().out
    system = re.search(r"system_prompt_chars: (\d+)", out)
    user = re.search(r"user_prompt_chars: (\d+)", out)
    assert system is not None
    assert user is not None
    return int(system.group(1)), int(user.group(1))


def test_the_dry_run_states_the_prompt_limit(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert "max_prompt_chars: 200000" in capsys.readouterr().out


def test_prompts_longer_than_the_bases_limit_are_refused_before_any_spend(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    system, user = _prompt_sizes(repo, capsys)
    size = system + user
    assert size > 1_000, "the schema's lower bound has to be below a real prompt"
    _live_env(monkeypatch)

    at_the_limit = _argv_with_base_config(repo, _WORKFORCE_YAML + f"    max_prompt_chars: {size}\n")
    assert run_review.main(at_the_limit) == 0, "a prompt exactly at the limit is accepted"
    capsys.readouterr()
    assert len(fake_sdk.queries) == 1

    over = _argv_with_base_config(repo, _WORKFORCE_YAML + f"    max_prompt_chars: {size - 1}\n")
    assert run_review.main(over) == run_review.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "max_prompt_chars" in err
    assert str(size - 1) in err
    assert str(size) in err
    assert len(fake_sdk.queries) == 1, "no second query: the refusal came before the SDK"
    assert len(_ledger_rows(repo)) == 1, "and before any ledger row"


def test_the_limit_comes_from_the_base_and_not_the_working_tree(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML + "    max_prompt_chars: 1000\n")
    assert run_review.main(_argv(repo, "--dry-run")) == 0, "an uncommitted, tiny limit is ignored"
    assert "max_prompt_chars: 200000" in capsys.readouterr().out


def test_a_head_that_tracks_a_program_named_file_is_refused_before_any_spend(
    repo: Path,
    fake_sdk: _FakeSdk,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(repo, "git.exe", "MZ\n")
    _commit(repo, "the head plants a program")
    _live_env(monkeypatch)

    assert run_review.main(_argv(repo)) == run_review.EXIT_REFUSED

    assert "'git.exe'" in capsys.readouterr().err
    fake_sdk.query.assert_not_called()
    assert _ledger_rows(repo) == []


def test_a_planted_program_in_the_checkout_counts_even_when_the_head_is_clean(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI runs in the CHECKOUT: what it tracks at its own HEAD is what Windows would find."""
    _write(repo, "bash.cmd", "@echo off\n")
    _commit(repo, "a later commit in the checkout, past the head under review")
    argv = ["--repo-root", str(repo), "--ledger-root", str(repo)]
    argv += ["--base", "HEAD~2", "--head", "HEAD~1", "--cap-usd", "5", "--dry-run"]

    assert run_review.main(argv) == run_review.EXIT_REFUSED

    assert "'bash.cmd'" in capsys.readouterr().err


def test_a_planted_program_in_the_head_counts_even_when_the_checkout_is_clean(
    repo: Path, fake_sdk: _FakeSdk, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail closed: the head may be the next thing checked out."""
    _write(repo, "node.cmd", "@echo off\n")
    _commit(repo, "the head plants a program")
    _git(repo, "rm", "-q", "node.cmd")
    _commit(repo, "a later commit in the checkout removes it again")
    argv = ["--repo-root", str(repo), "--ledger-root", str(repo)]
    argv += ["--base", "HEAD~2", "--head", "HEAD~1", "--cap-usd", "5", "--dry-run"]

    assert run_review.main(argv) == run_review.EXIT_REFUSED

    assert "'node.cmd'" in capsys.readouterr().err


def test_a_directory_named_like_a_program_is_not_a_planted_program(
    repo: Path, fake_sdk: _FakeSdk
) -> None:
    """Windows cannot run a directory: only files at the root count."""
    _write(repo, "python/readme.txt", "vendored\n")
    _write(repo, "node/readme.txt", "vendored\n")
    _commit(repo, "vendored directories with program names")
    assert run_review.main(_argv(repo, "--dry-run")) == 0


def test_listing_the_root_of_a_commit_git_does_not_have_is_a_refusal(repo: Path) -> None:
    with pytest.raises(run_review.RefusalError, match="cannot list the files"):
        asyncio.run(run_review._root_files(repo, "f" * 40))


def test_a_checkout_whose_head_cannot_be_resolved_has_no_root_to_check(repo: Path) -> None:
    """``worktree_sha`` is ``unknown`` then; it is skipped rather than handed to git."""
    head = _git(repo, "rev-parse", "HEAD")
    refs = run_review.Refs("base", "a" * 40, "head", head, "unknown")
    asyncio.run(run_review._refuse_planted_programs(repo, refs))


def test_a_program_named_file_below_the_root_is_not_a_planted_program(
    repo: Path, fake_sdk: _FakeSdk
) -> None:
    _write(repo, "tools/git.exe", "MZ\n")
    _write(repo, "docs/python", "text\n")
    _commit(repo, "files with program names, but not where Windows looks")
    assert run_review.main(_argv(repo, "--dry-run")) == 0


def test_the_reviewers_own_git_runs_without_any_credential(
    repo: Path, fake_sdk: _FakeSdk, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "MOUSEDROID_LLM__API_KEY", "GH_TOKEN")
    for name in held:
        monkeypatch.setenv(name, "must-not-reach-git")
    seen: list[dict[str, str]] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(kwargs.get("env") or {}))
        return await real(*args, **kwargs)

    monkeypatch.setattr(run_review.asyncio, "create_subprocess_exec", spy)
    assert run_review.main(_argv(repo, "--dry-run")) == 0
    assert len(seen) >= 4
    for env in seen:
        assert not set(held) & set(env), "a credential reached the reviewer's own git"
        assert env.get("GIT_LITERAL_PATHSPECS") == "1"
        assert "PATH" in env, "it still has what a git needs to run"


def test_the_scanner_child_runs_without_any_credential(
    repo: Path, cfg: WorkforceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_gitleaks(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("MOUSEDROID_LLM__API_KEY", _FAKE_KEY)
    assert prompt.build(repo_root=repo, cfg=cfg, gitleaks_toml=None)
    (call,) = seen
    env = call["env"]
    assert env is not None, "an explicit environment, not the inherited one"
    assert "ANTHROPIC_API_KEY" not in env
    assert "MOUSEDROID_LLM__API_KEY" not in env
    assert "PATH" in env


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
    matcher, read_matcher = options.hooks["PreToolUse"]
    assert matcher.matcher == "Bash"
    (guard,) = matcher.hooks
    assert isinstance(guard, policy.BashGuard)
    assert read_matcher.matcher == "Read|Grep|Glob"
    (read_guard,) = read_matcher.hooks
    assert isinstance(read_guard, policy.ReadGuard)
    assert read_guard.root == str(repo.resolve())
    server = options.mcp_servers["mousedroid"]
    assert server["command"] == sys.executable
    assert server["args"][:3] == ["-I", "-c", policy.MCP_BOOTSTRAP]
    pinned = {_git(repo, "rev-parse", "HEAD~1"), _git(repo, "rev-parse", "HEAD")}
    assert guard.revisions == pinned
    assert options.env["GIT_LITERAL_PATHSPECS"] == "1"
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
    argv = _argv_with_base_config(repo, _WORKFORCE_YAML + "    run_timeout_s: 0.5\n")
    sdk = _FakeSdk()

    def hang(**_kwargs: Any) -> Any:
        async def stream() -> Any:
            yield sdk.AssistantMessage([sdk.TextBlock("started")])
            await asyncio.sleep(30)

        return stream()

    sdk.query = MagicMock(side_effect=hang)
    _install_sdk(monkeypatch, sdk)
    _live_env(monkeypatch)
    assert run_review.main(argv) == run_review.EXIT_FAILED
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
        content: str,
        config: WorkforceConfig,
        *,
        repo_root: Path,
        target: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ScanOutcome:
        calls.append(content)
        # The prompt is scanned first and is clean; the review output is what gets flagged.
        return ScanOutcome(clean=len(calls) == 1, available=True, detail="1 leak (redacted)")

    monkeypatch.setattr("tools.agent_reviewer.prompt.secret_scan.scan_content", scan)
    real_lookup = policy.resolve_executable

    def lookup(name: str, environ: Any = None) -> str | None:
        return sys.executable if name == "gitleaks" else real_lookup(name, environ)

    monkeypatch.setattr(policy, "resolve_executable", lookup)
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
    argv = [
        "--repo-root",
        str(repo),
        "--ledger-root",
        str(repo),
        "--base",
        "HEAD~2",
        "--head",
        "HEAD~1",
        "--cap-usd",
        "5",
    ]
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
    """``max_turns``, the model alias, the caps and the ledger path all come from the YAML
    that is committed at the BASE ref."""
    # (1) max_turns and the model alias follow workforce.yaml.
    argv = _argv_with_base_config(
        repo,
        "credit:\n    caps_usd:\n        B: 4\n    ledger: out/spend.jsonl\n"
        "agent_reviewer:\n    model: sonnet\n    max_turns: 7\n",
        cap="3",
    )
    _live_env(monkeypatch)
    assert run_review.main(argv) == 0
    options = fake_sdk.queries[0]["options"]
    assert options.max_turns == 7
    assert options.model == "sonnet"
    # (2) the cap and the ledger path are the YAML's, too.
    lines = (repo / "out/spend.jsonl").read_text(encoding="utf-8").splitlines()
    (row,) = [json.loads(line) for line in lines]
    assert row["cap_usd"] == 4.0
    assert not (repo / "reports/credit/ledger.jsonl").exists()
    assert run_review.main(argv) == run_review.EXIT_CAP, "1.25 + 3 > 4"


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


def _dotted(node: ast.expr) -> str:
    """``a.b.c`` for a name or attribute chain, else an empty string."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_the_package_never_loads_the_working_trees_workforce_config() -> None:
    """The reviewer's configuration is parsed from the base ref's blob, never ``load_config``."""
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called = {_dotted(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        imported = {
            a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
        }
        assert "load_config" not in {name.split(".")[-1] for name in called | imported}, path.name


def test_the_package_starts_no_program_by_a_bare_name() -> None:
    """``git.exe`` or ``gitleaks.exe`` in the head would run first on Windows: every program the
    package starts is an absolute path from ``policy.resolve_executable``."""
    starters = {"create_subprocess_exec", "run", "Popen", "check_output", "check_call", "call"}
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            assert name != "shutil.which", (path.name, node.lineno)
            if name.split(".")[-1] not in starters or not node.args:
                continue
            first = node.args[0]
            literal = isinstance(first, ast.Constant) and isinstance(first.value, str)
            listed = (
                isinstance(first, ast.List)
                and first.elts
                and isinstance(first.elts[0], ast.Constant)
            )
            assert not literal, (path.name, node.lineno, "a program named by a string literal")
            assert not listed, (path.name, node.lineno, "a program named by a list literal")


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


def test_the_readme_worktree_recipe_cannot_collide_with_a_checked_out_branch() -> None:
    """`git worktree add <dir> <branch>` fails when that branch is checked out in another worktree,
    which the head branch usually is: the recipe detaches at a commit id instead."""
    readme = (_PACKAGE_DIR / "README.md").read_text(encoding="utf-8")
    recipes = [line for line in readme.splitlines() if line.lstrip().startswith("git worktree add")]
    assert len(recipes) == 2
    for line in recipes:
        assert "--detach" in line, line
    assert "<head-branch>" not in "".join(recipes)
