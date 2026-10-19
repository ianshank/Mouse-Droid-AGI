"""The read-only containment of the Agent-SDK peer reviewer (F-059 WS-B / WS-D).

A cloud agent that reads this repository and spends real money is acceptable only
because of what it *cannot* do. Everything that limits it lives here, as plain
data and pure functions, so tests pin it without the SDK, a network or a key. The
layers, in the order the SDK applies them:

1. ``BASE_TOOLS`` (``tools=``): the model is only shown ``Read``, ``Grep``,
   ``Glob`` and ``Bash``. ``Write``, ``Edit``, the web tools and the subagent tool
   do not exist for it.
2. :func:`deny_non_readonly_bash` (a PreToolUse hook): hooks run FIRST. It is a
   whitelist - ``git diff|log|show`` and ``python -m pytest ... --collect-only``,
   nothing else - and it also refuses the options that make those commands write
   a file, run a configured program or read outside the repo. It returns
   nothing when it has no objection: an explicit "allow" would approve the call
   on its own, skipping the allowlist match below, so the hook can only
   tighten, never grant.
3. :func:`disallowed_tools` (deny rules): evaluated even if the hook were to fail.
   Writes, web access, subagents, every actuation tool by name, and reads of
   secret files (``*.env``, ``docker.env*``) and credential directories.
4. ``permission_mode="dontAsk"``: anything not pre-approved is denied instead of
   prompting - a headless run has nobody to ask.
5. ``ALLOWED`` (``allowed_tools=``): the allowlist itself.
6. :func:`build_mcp_servers`: the only MCP server is ``mousedroid``, built from
   ``.mcp.json`` with ``MOUSEDROID_MOCK_HARDWARE=true`` forced and
   ``strict_mcp_config`` set, so the CLI loads no other server (the ``github``
   server in ``.mcp.json`` carries a token and can write; it never gets here).
7. ``setting_sources=[]``: no user, project or local settings file can widen any
   of the above.

Why there is no ``RegexInjectionFilter`` here
---------------------------------------------
The filter guards natural language that leaves the *rover* toward an
actuation-bearing translation (CHARTER section 3). The reviewer holds no
actuation tool and no write tool, runs on the workstation against a mock-hardware
MCP server, and a human reads its output before ``/code-review`` and merge. A
prompt injection planted in repository text can at worst produce a wrong
paragraph in an advisory review. The allowlist above is the control, not a text
filter.

Known limits, stated so nobody has to discover them
----------------------------------------------------
* ``Read`` is not confined to the repository: a bare ``Read`` allow rule reads any
  file the user can. The deny rules cover env files and the usual credential
  directories; they are a screen, not a sandbox. Run the review from an account
  that holds nothing it should not read.
* The Bash guard is a whitelist over a small character set, not a shell parser.
  It fails closed (an unusual-but-harmless command is refused) because the
  alternative is parsing shell.
* ``python -m pytest --collect-only`` imports the repository's test modules, so it
  runs the repository's own import-time code. That is first-party code under
  review; the guard keeps pytest from loading plugins or writing files.

Nothing here imports ``claude_agent_sdk`` or ``mousedroid``.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
import shlex
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from tools.claude_hooks.config import AgentReviewerConfig

#: The one MCP server the reviewer may see (a key of ``.mcp.json``'s ``mcpServers``).
MCP_SERVER: Final = "mousedroid"

#: PreToolUse matcher the Bash guard is registered under.
BASH_HOOK_MATCHER: Final = "Bash"

#: ``dontAsk``: tools not pre-approved are denied instead of prompting.
PERMISSION_MODE: Final = "dontAsk"

#: Built-in tools the session may see at all (``ClaudeAgentOptions.tools``).
BASE_TOOLS: Final[tuple[str, ...]] = ("Read", "Grep", "Glob", "Bash")

#: Tool rules auto-approved (``ClaudeAgentOptions.allowed_tools``). Pinned literally
#: by ``test_allowlist_is_the_planned_set``: widening this is a reviewed edit.
ALLOWED: Final[tuple[str, ...]] = (
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

#: Built-in tools denied outright (``ClaudeAgentOptions.disallowed_tools``).
DENIED: Final[tuple[str, ...]] = (
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
)

#: The subagent tool (named ``Task`` or ``Agent`` depending on the CLI version) is a
#: second tool surface; the reviewer works alone.
SUBAGENT_TOOLS: Final[tuple[str, ...]] = ("Task", "Agent")

#: Every actuation or side-effecting MCP tool, by name. A superset of
#: ``MCPConfig.actuation_tools`` (pinned by a test, so a tool added there reaches
#: this list) plus ``emergency_stop`` and ``translate_nl_mission``, which the MCP
#: default list leaves out for operational reasons that do not apply to a reviewer.
ACTUATION_TOOLS: Final[tuple[str, ...]] = (
    "set_velocity",
    "emergency_stop",
    "translate_nl_mission",
    "export_experience",
    "tensorrt_compile",
    "calibrate_ultrasonic",
)

#: File names (matched on the base name, case-insensitively) that may hold secrets.
#: The prompt builder refuses to include them, the Bash guard refuses to name them
#: and ``Read`` is denied on them.
SECRET_PATH_GLOBS: Final[tuple[str, ...]] = ("*.env", "docker.env*", ".env", ".env.*")

#: Credential locations ``Read`` is denied on: the home-relative ones (modelled on the
#: deny list in Claude Code's own example settings, plus this project's
#: ``~/.mousedroid``), and ``/proc``, where on Linux a process's environment - which
#: holds the API key - can be read. Claude Code also applies ``Read()`` deny rules to
#: the file operands of Bash commands (its changelog says so) and, best effort, to
#: Grep and Glob, so this is a screen across tools - not a sandbox.
CREDENTIAL_READ_DENY: Final[tuple[str, ...]] = (
    "Read(~/.mousedroid/**)",
    "Read(~/.ssh/**)",
    "Read(~/.aws/**)",
    "Read(~/.claude/**)",
    "Read(~/.claude.json)",
    "Read(~/.config/gh/**)",
    "Read(~/.config/git/credentials)",
    "Read(~/.git-credentials)",
    "Read(~/.npmrc)",
    "Read(~/.netrc)",
    "Read(~/_netrc)",
    "Read(//proc/**)",
)

#: Environment forced into the mock MCP server, overriding ``.mcp.json``.
FORCED_MCP_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {
        "MOUSEDROID_MOCK_HARDWARE": "true",
        # Defence in depth: even if an actuation tool name slipped past the rules above,
        # the server refuses to expose or run it.
        "MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS": "false",
    }
)

#: Only ``.mcp.json`` env keys with one of these prefixes reach the MCP server; any
#: rover host, serial port, token or other setting in the file is dropped.
MCP_ENV_PASSTHROUGH_PREFIXES: Final[tuple[str, ...]] = ("MOUSEDROID_MCP__",)

#: Environment forced into the CLI process (and so into its Bash children). The
#: pytest cache and ``__pycache__`` writes are the only side effects a
#: ``--collect-only`` run would otherwise have.
FORCED_PROCESS_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {
        "MOUSEDROID_MOCK_HARDWARE": "true",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
)

# --- Bash guard vocabulary -------------------------------------------------- #
_MAX_COMMAND_CHARS: Final = 4096

# The whole command must be made of these characters. No ; & | < > ` $ ( ) \ ! # or
# whitespace other than a space: chaining, redirection, substitution, variables,
# escapes and comments are all out. Quotes stay (so `--format="%h %s"` works) but
# cannot hide anything, because the dangerous characters are refused inside them too.
_COMMAND_CHARSET: Final = re.compile(r"[\w@%+=:,./^~*?\[\]{}\"' -]+")

# The shell expands these when they are NOT quoted, and an expansion can turn a
# filename into an option (`git diff *` with a file called `--output=x`) or a brace
# list into one (`{--output=f,x}`). The guard sees the text before the shell does, so
# an unquoted one is refused; quoted (`-- '*.py'`) they are literal and fine.
_EXPANSION_CHARS: Final = frozenset("*?[]{}")

_GIT_SUBCOMMANDS: Final = frozenset({"diff", "log", "show"})

# Long options that make a read-only git command write a file (--output), read
# outside the repo (--no-index) or run a program named in git config (the other two).
# git accepts unambiguous abbreviations, so a token that is a prefix of one of these
# is refused as well.
_GIT_DENIED_LONG: Final[tuple[str, ...]] = ("--output", "--no-index", "--ext-diff", "--textconv")

# -O<orderfile> reads an arbitrary file.
_GIT_DENIED_SHORT_PREFIXES: Final[tuple[str, ...]] = ("-O",)

_PYTEST_COLLECT_FLAGS: Final = frozenset({"--collect-only", "--co"})
_PYTEST_PLAIN_FLAGS: Final = frozenset(
    {"-q", "-qq", "--quiet", "-v", "-vv", "--no-header", "--no-summary"}
)
_PYTEST_EXPRESSION_FLAGS: Final = frozenset({"-k", "-m"})
# A -k / -m expression: words, spaces, dots and hyphens, not starting with a hyphen
# (which would make it look like the next option).
_PYTEST_EXPRESSION: Final = re.compile(r"[\w.][\w .-]*")

_DRIVE_PATH: Final = re.compile(r"[A-Za-z]:[\\/]")
_TOKEN_SPLIT: Final = re.compile(r"[:=]")

# --- Credential shapes ------------------------------------------------------- #
# What a real key looks like, as opposed to the documented placeholders
# (`sk-ant-...`, `sk-ant-xyz`): long enough that no placeholder qualifies.
_CREDENTIAL_SHAPES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("an Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("a Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("an AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "a GitHub token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}|\bgithub_pat_[A-Za-z0-9_]{50,}"),
    ),
    ("a private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


class PolicyViolationError(RuntimeError):
    """Raised when the reviewer's inputs would breach its read-only policy."""


# --------------------------------------------------------------------------- #
# Secret screens (shared with the prompt builder and the Bash guard)
# --------------------------------------------------------------------------- #
def secret_path_reason(path: str | Path) -> str | None:
    """Return why ``path`` looks like a secret-bearing file, or ``None``.

    Matches the base name only, case-insensitively and on every platform
    (backslashes are separators here), against :data:`SECRET_PATH_GLOBS`. The
    reason is a predicate ("matches the secret-file pattern ...") for the caller to
    attach to the path it already has.
    """
    name = PurePosixPath(str(path).replace("\\", "/")).name.lower()
    for glob in SECRET_PATH_GLOBS:
        if fnmatch.fnmatchcase(name, glob.lower()):
            return f"matches the secret-file pattern {glob!r}"
    return None


def credential_shape_reason(text: str) -> str | None:
    """Return a description of the credential ``text`` appears to contain, or ``None``.

    The description names the kind of credential, never its value.
    """
    for label, pattern in _CREDENTIAL_SHAPES:
        if pattern.search(text):
            return f"text shaped like {label}"
    return None


# --------------------------------------------------------------------------- #
# The tool lists
# --------------------------------------------------------------------------- #
def disallowed_tools() -> list[str]:
    """Return the deny rules (``ClaudeAgentOptions.disallowed_tools``).

    Deny rules are evaluated even if the PreToolUse hook were to fail, so they repeat
    what the allowlist already excludes: they are the second lock, not the first.
    """
    return [
        *DENIED,
        *SUBAGENT_TOOLS,
        *(f"mcp__{MCP_SERVER}__{name}" for name in ACTUATION_TOOLS),
        *(f"Read(**/{glob})" for glob in SECRET_PATH_GLOBS),
        *CREDENTIAL_READ_DENY,
    ]


# --------------------------------------------------------------------------- #
# The Bash guard
# --------------------------------------------------------------------------- #
_BASH_ALLOWED_SUMMARY: Final = (
    "Bash is limited to `git diff`, `git log`, `git show` and "
    "`python -m pytest ... --collect-only`, as one plain command: no chaining, "
    "pipes, redirection, substitution or variables."
)


def _path_token_reason(word: str) -> str | None:
    """Refuse an absolute, home-relative or repo-escaping path."""
    if word.startswith(("/", "~")) or _DRIVE_PATH.match(word):
        return f"absolute and home-relative paths are not allowed ({word!r})."
    if ".." in word.replace("\\", "/").split("/"):
        return f"paths may not climb out of the repository ({word!r})."
    return None


def _secret_token_reason(word: str) -> str | None:
    """Refuse a word that names a secret-bearing file (``rev:path``, ``--opt=path``, ...)."""
    for piece in _TOKEN_SPLIT.split(word):
        if piece and secret_path_reason(piece):
            return f"{piece!r} names a file that may hold secrets; the reviewer never reads those."
    return None


def _git_option_reason(word: str) -> str | None:
    name = word.split("=", 1)[0]
    if name.startswith("--"):
        if any(denied.startswith(name) for denied in _GIT_DENIED_LONG):
            return (
                f"git option {name} can write a file, read outside the repository or run a "
                "program, so it is not allowed."
            )
        return None
    if word.startswith(_GIT_DENIED_SHORT_PREFIXES):
        return f"git option {word!r} reads an arbitrary file, so it is not allowed."
    return None


def _git_token_reason(word: str) -> str | None:
    if word == "--":
        return None
    if word.startswith("-"):
        return _git_option_reason(word) or _secret_token_reason(word)
    return _path_token_reason(word) or _secret_token_reason(word)


def _check_git(argv: list[str]) -> str | None:
    subcommand = argv[1]
    if subcommand not in _GIT_SUBCOMMANDS:
        return f"`git {subcommand}` is not read-only inspection. {_BASH_ALLOWED_SUMMARY}"
    for word in argv[2:]:
        reason = _git_token_reason(word)
        if reason:
            return reason
    return None


def _pytest_expression_reason(flag: str, rest: list[str], index: int) -> str | None:
    if index >= len(rest) or not _PYTEST_EXPRESSION.fullmatch(rest[index]):
        return f"pytest {flag} needs a plain expression (letters, digits, spaces, `_`, `.`, `-`)."
    return None


def _check_pytest(rest: list[str]) -> str | None:
    collect_only = False
    index = 0
    while index < len(rest):
        word = rest[index]
        if word in _PYTEST_COLLECT_FLAGS:
            collect_only = True
        elif word in _PYTEST_EXPRESSION_FLAGS:
            index += 1
            reason = _pytest_expression_reason(word, rest, index)
            if reason:
                return reason
        elif word in _PYTEST_PLAIN_FLAGS:
            pass
        elif word.startswith("-"):
            return (
                f"pytest option {word!r} is not on the read-only list "
                f"({', '.join(sorted(_PYTEST_COLLECT_FLAGS | _PYTEST_PLAIN_FLAGS))}, -k, -m)."
            )
        else:
            reason = _path_token_reason(word) or _secret_token_reason(word)
            if reason:
                return reason
        index += 1
    if not collect_only:
        return (
            "python -m pytest is only allowed with --collect-only: the reviewer does not run tests."
        )
    return None


def _unquoted_expansion(command: str) -> str | None:
    """Return the first glob or brace character that sits outside quotes, if any."""
    quote = ""
    for char in command:
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char in _EXPANSION_CHARS:
            return char
    return None


def _check_argv(argv: list[str]) -> str | None:
    if len(argv) >= 2 and argv[0] == "git":
        return _check_git(argv)
    if argv[:3] == ["python", "-m", "pytest"]:
        return _check_pytest(argv[3:])
    return _BASH_ALLOWED_SUMMARY


def check_bash_command(command: object) -> str | None:
    """Return why ``command`` is refused, or ``None`` if the reviewer may run it.

    Args:
        command: The ``command`` field of a Bash tool call (anything, so a malformed
            payload is refused rather than raising).

    Returns:
        ``None`` for ``git diff|log|show ...`` and ``python -m pytest ... --collect-only``
        as a single plain command; otherwise a sentence the model can act on.
    """
    if not isinstance(command, str) or not command.strip():
        return "Bash needs a non-empty command string."
    if len(command) > _MAX_COMMAND_CHARS:
        return f"The command is longer than {_MAX_COMMAND_CHARS} characters."
    stripped = command.strip()
    if not _COMMAND_CHARSET.fullmatch(stripped):
        return f"The command contains shell syntax that is not allowed. {_BASH_ALLOWED_SUMMARY}"
    expansion = _unquoted_expansion(stripped)
    if expansion is not None:
        return (
            f"An unquoted {expansion!r} would be expanded by the shell before git sees it; "
            "quote it (for example `-- '*.py'`)."
        )
    try:
        argv = shlex.split(stripped, posix=True)
    except ValueError as exc:
        return f"The command could not be parsed ({exc})."
    return _check_argv(argv)


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


async def deny_non_readonly_bash(
    input_data: Mapping[str, Any] | None,
    tool_use_id: str | None,
    context: Any,
) -> dict[str, Any]:
    """PreToolUse hook: refuse every Bash call that is not read-only git or pytest collection.

    Returns ``{}`` (no opinion) when the command is acceptable, so the permission
    rules still apply - an explicit allow would bypass them. Fails closed: a
    malformed payload or an internal error is a deny, never a pass.

    Args:
        input_data: The SDK's ``PreToolUseHookInput``.
        tool_use_id: The tool call id (unused).
        context: The SDK's hook context (unused).
    """
    try:
        if not isinstance(input_data, Mapping) or "tool_name" not in input_data:
            return _deny("The Bash request was malformed.")
        if input_data["tool_name"] != BASH_HOOK_MATCHER:
            return {}
        tool_input = input_data.get("tool_input")
        command = tool_input.get("command") if isinstance(tool_input, Mapping) else None
        reason = check_bash_command(command)
    except Exception as exc:  # fail closed: a guard that crashes must not wave a call through
        return _deny(f"The read-only Bash guard failed ({type(exc).__name__}); the call is denied.")
    return {} if reason is None else _deny(reason)


# --------------------------------------------------------------------------- #
# The mock-only MCP server
# --------------------------------------------------------------------------- #
def _read_server_entry(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PolicyViolationError(f"cannot read {path.name}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise PolicyViolationError(f"{path.name} is not valid JSON: {exc}") from exc
    servers = raw.get("mcpServers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict):
        raise PolicyViolationError(f"{path.name} has no `mcpServers` object")
    entry = servers.get(MCP_SERVER)
    if not isinstance(entry, dict):
        raise PolicyViolationError(f"{path.name} declares no `{MCP_SERVER}` server")
    return entry


def _string_list(value: object, *, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PolicyViolationError(f"{where} must be a list of strings")
    return list(value)


def _string_map(value: object, *, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise PolicyViolationError(f"{where} must be an object of strings")
    return dict(value)


def build_mcp_servers(repo_root: Path, cfg: AgentReviewerConfig) -> dict[str, dict[str, Any]]:
    """Build the reviewer's ``mcp_servers`` from ``.mcp.json``, mock hardware forced.

    Only the ``mousedroid`` server is taken. Of its environment only
    ``MOUSEDROID_MCP__*`` keys survive, then :data:`FORCED_MCP_ENV` is laid over
    them, so no rover host, serial port or token in the file can reach the server
    and ``MOUSEDROID_MOCK_HARDWARE`` is ``true`` whatever the file says.

    Args:
        repo_root: Repository root.
        cfg: The reviewer's workforce settings (names the ``.mcp.json`` path).

    Returns:
        ``{"mousedroid": {"type": "stdio", "command": ..., "args": [...], "env": {...}}}``.

    Raises:
        PolicyViolationError: The file is missing, malformed, lacks the server, or a
            value would be expanded from the parent environment (``${...}``).
    """
    entry = _read_server_entry(repo_root / cfg.mcp_config)
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        raise PolicyViolationError(f"{cfg.mcp_config}: server `{MCP_SERVER}` has no command")
    args = _string_list(entry.get("args", []), where=f"{cfg.mcp_config}: args")
    env = _string_map(entry.get("env", {}), where=f"{cfg.mcp_config}: env")
    kept = {
        key: value
        for key, value in env.items()
        if key.startswith(MCP_ENV_PASSTHROUGH_PREFIXES) and key not in FORCED_MCP_ENV
    }
    for text in (command, *args, *kept.values()):
        if "${" in text:
            raise PolicyViolationError(
                f"{cfg.mcp_config}: `{text}` would be expanded from the parent environment, "
                "which holds the API key; the reviewer's MCP config takes literal values only"
            )
    return {
        MCP_SERVER: {
            "type": "stdio",
            "command": command,
            "args": args,
            "env": {**kept, **FORCED_MCP_ENV},
        }
    }


# --------------------------------------------------------------------------- #
# The SDK options
# --------------------------------------------------------------------------- #
def reviewer_options(
    *,
    cfg: AgentReviewerConfig,
    repo_root: Path,
    model: str,
    system_prompt: str,
    cap_usd: float,
) -> dict[str, Any]:
    """Return the keyword arguments for ``ClaudeAgentOptions`` (everything but ``hooks``).

    The set of keys is closed and pinned by a test: a new option is a deliberate,
    reviewed addition. ``hooks`` is registered separately (it needs the SDK's
    ``HookMatcher``): ``{"PreToolUse": [HookMatcher("Bash", [deny_non_readonly_bash])]}``.

    Args:
        cfg: The reviewer's workforce settings (``max_turns`` comes from here).
        repo_root: Repository root, used as the session's working directory.
        model: The model alias or id to run.
        system_prompt: The assembled system prompt.
        cap_usd: This run's budget; passed to the SDK as ``max_budget_usd`` so the CLI
            itself stops at it.

    Raises:
        PolicyViolationError: ``cap_usd`` is not a finite number greater than zero (the
            CLI rejects ``0``, and an unbounded budget is the thing being prevented),
            or the MCP config cannot be built.
    """
    if (
        isinstance(cap_usd, bool)
        or not isinstance(cap_usd, (int, float))
        or not math.isfinite(cap_usd)
        or cap_usd <= 0
    ):
        raise PolicyViolationError(f"the run budget must be a finite number > 0, got {cap_usd!r}")
    return {
        "model": model,
        "system_prompt": system_prompt,
        "tools": list(BASE_TOOLS),
        "allowed_tools": list(ALLOWED),
        "disallowed_tools": disallowed_tools(),
        "permission_mode": PERMISSION_MODE,
        "max_turns": cfg.max_turns,
        "max_budget_usd": float(cap_usd),
        "cwd": str(repo_root),
        "mcp_servers": build_mcp_servers(repo_root, cfg),
        "strict_mcp_config": True,
        "setting_sources": [],
        "env": dict(FORCED_PROCESS_ENV),
        # Prompts are delivered exactly as written: no `@/path` file-mention expansion.
        "verbatim_prompts": True,
    }
