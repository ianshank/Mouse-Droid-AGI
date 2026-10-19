"""The read-only containment of the Agent-SDK peer reviewer (F-059 WS-B / WS-D).

Threat model
------------
The reviewer runs on the operator's workstation with an API key in its environment,
against a change set that may be HOSTILE: the head commit controls every file in the
working tree. So nothing the reviewed tree controls may reach an exec sink, an
environment that holds the key, or a path the reviewer should not read. Everything
that enforces that lives here, as plain data and pure functions, so tests pin it
without the SDK, a network or a key. The layers, in the order the SDK applies them:

1. ``BASE_TOOLS`` (``tools=``): the model is only shown ``Read``, ``Grep``,
   ``Glob`` and ``Bash``. ``Write``, ``Edit``, the web tools and the subagent tool
   do not exist for it.
2. :class:`BashGuard` (a PreToolUse hook): hooks run FIRST. A whitelist -
   ``git diff|log|show``, nothing else, as one plain command. It refuses the git
   options that write a file, run a configured program, read outside the repo or
   widen the search to every ref (``--all``, ``--reflog``, ...); pathspecs after
   ``--`` must be literal (no globs, no ``:`` magic, nothing starting with ``-``);
   and ``git show <rev>:<path>`` is accepted only for a non-secret file TRACKED AT
   THE BASE REF, because a path the literal-token screen never sees (a history-wide
   form, a path the head renamed) is a path it cannot vouch for. It returns nothing
   when it has no objection: an explicit "allow" would approve the call on its own,
   skipping the allowlist match below, so the hook can only tighten, never grant.
   There is no pytest and no interpreter on the list: ``--collect-only`` still
   imports the repository's ``conftest.py`` and test modules, which is the head
   commit's code running in a process that holds the key.
3. :func:`disallowed_tools` (deny rules): evaluated even if the hook were to fail.
   Writes, web access, subagents, every actuation tool by name, and reads of
   secret files (``*.env``, ``docker.env*``) and credential directories.
4. ``permission_mode="dontAsk"``: anything not pre-approved is denied instead of
   prompting - a headless run has nobody to ask.
5. ``ALLOWED`` (``allowed_tools=``): the allowlist itself.
6. :func:`build_mcp_servers`: the only MCP server is ``mousedroid``, and its entry
   is BUILT IN CODE from constants - the interpreter is ``sys.executable``, the
   arguments are :data:`MCP_ARGS`, the environment is :data:`MCP_ENV` - never passed
   through from any ``.mcp.json``. The file is read from the trusted BASE ref (not
   the working tree, which the head controls) and only to CONFIRM that it still
   declares the pinned invocation; a mismatch refuses the run. ``strict_mcp_config``
   keeps the CLI from loading any other server (the ``github`` entry of
   ``.mcp.json`` carries a token and can write).
7. ``setting_sources=[]``: no user, project or local settings file can widen any
   of the above - and the head's ``.claude/settings.json`` hooks never run.
8. :func:`process_env`: the environment the reviewer adds for the CLI process is an
   allowlist (``PATH``, ``HOME``, ``LANG``) plus forced constants (literal git
   pathspecs, mock hardware); no credential is copied into it. NOTE: the SDK merges
   it over the inherited environment, so it cannot REMOVE what the CLI inherits -
   the key the CLI needs stays in the CLI's environment. What keeps the key out of
   the processes the CLI starts is that none of them runs tree-controlled code
   (layers 2 and 6), that the MCP server entry blanks the credential variables, and,
   as an opt-in, the CLI's own ``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`` (see README).

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
* ``git log -p`` over the named revisions shows every file those commits ever
  touched, including a secret file committed in the past and still reachable.
  Env files are gitignored and the CI secret scan covers history; the guard does not
  re-scan git's output.

Nothing here imports ``claude_agent_sdk`` or ``mousedroid``.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import shlex
import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass
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
#: by ``test_allowlist_is_the_planned_set``: widening this is a reviewed edit. There
#: is deliberately no pytest rule (see the module docstring, layer 2).
ALLOWED: Final[tuple[str, ...]] = (
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

# --- The MCP server, pinned ---------------------------------------------------- #
#: The module the mock MCP server runs (``python -m <module>``): where the interpreter
#: finds it is what decides whose code the server executes (see ``run_review``).
MCP_MODULE: Final = "mousedroid"

#: The module invocation the mock MCP server is started with, copied from the
#: committed ``.mcp.json``. The entry handed to the SDK is built from THIS, never from
#: the file: the file only has to confirm it (a mismatch refuses the run).
MCP_ARGS: Final[tuple[str, ...]] = ("-m", MCP_MODULE, "--config", "config/default.yaml")

#: What the committed ``.mcp.json`` may name as the interpreter. The entry built for the
#: SDK always uses ``sys.executable`` - an absolute path, so no search of the working
#: directory (which the head controls, and which Windows searches before ``PATH``).
MCP_DECLARED_COMMANDS: Final = frozenset({"python", "python3"})

#: The whole environment the MCP server entry carries (the CLI lays it over its own).
MCP_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {
        # Mock hardware: no rover, no serial port, no host.
        "MOUSEDROID_MOCK_HARDWARE": "true",
        "MOUSEDROID_MCP__ENABLED": "true",
        "MOUSEDROID_MCP__BIND_TRANSPORT": "true",
        "MOUSEDROID_MCP__TRANSPORT": "stdio",
        # Defence in depth: even if an actuation tool name slipped past the rules above,
        # the server refuses to expose or run it.
        "MOUSEDROID_MCP__EXPOSE_ACTUATION_TOOLS": "false",
        # `python -m` puts the working directory first on sys.path; a planted json.py or
        # mousedroid/ in the head would run. 3.11+ honours this; older Pythons ignore it.
        "PYTHONSAFEPATH": "1",
        # The CLI's environment holds the key and may be laid under this entry's: blank
        # every credential variable so the server never sees a value.
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_AUTH_TOKEN": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "MOUSEDROID_LLM__API_KEY": "",
    }
)

# --- The CLI process environment ------------------------------------------------- #
#: Environment for every git the reviewer (or the model) runs: pathspecs are literal -
#: never globs or ``:(magic)`` - and git does not rewrite the index to refresh it.
GIT_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {"GIT_LITERAL_PATHSPECS": "1", "GIT_OPTIONAL_LOCKS": "0"}
)

#: Variables copied from the parent when set; nothing else is copied.
PROCESS_ENV_PASSTHROUGH: Final[tuple[str, ...]] = ("PATH", "HOME", "LANG")

#: Constants forced into the CLI process (and so into its Bash children).
FORCED_PROCESS_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {**GIT_ENV, "MOUSEDROID_MOCK_HARDWARE": "true"}
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
# an unquoted one is refused; quoted they reach git literally.
_EXPANSION_CHARS: Final = frozenset("*?[]{}")

_GIT_SUBCOMMANDS: Final = frozenset({"diff", "log", "show"})

# Long options refused. git accepts unambiguous abbreviations, so a word that is a
# prefix of one of these is refused as well.
_GIT_DENIED_LONG: Final[tuple[str, ...]] = (
    # write a file / read outside the repo / run a program named in git config
    "--output",
    "--no-index",
    "--ext-diff",
    "--textconv",
    # widen the search from the named revisions to every ref, reflog entry or glob
    "--all",
    "--branches",
    "--remotes",
    "--tags",
    "--glob",
    "--reflog",
    "--walk-reflogs",
    "--alternate-refs",
    "--indexed-objects",
)

# -O<orderfile> reads an arbitrary file; -g is --walk-reflogs.
_GIT_DENIED_SHORT_PREFIXES: Final[tuple[str, ...]] = ("-O", "-g")

# A plain revision, range or repo-relative path before `--` (no glob, no `:`).
_REVISION: Final = re.compile(r"[A-Za-z0-9._/@^~+-]+")
# A literal file path after `--`: must not start with `-`, no `* ? [ :`, no `~`.
_LITERAL_PATH: Final = re.compile(r"[A-Za-z0-9._][A-Za-z0-9._/ +@%=,-]*")

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
# The tool lists and the process environment
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


def process_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment the reviewer adds for the CLI process.

    An allowlist: :data:`PROCESS_ENV_PASSTHROUGH` copied from ``environ`` when set,
    then :data:`FORCED_PROCESS_ENV`. No credential is ever copied into it - not
    ``ANTHROPIC_API_KEY``, not ``MOUSEDROID_LLM__API_KEY``, nothing else - however the
    parent's environment is populated.

    The SDK lays this over the CLI's inherited environment, so it cannot remove a
    variable the CLI inherits; see the module docstring (layer 8).

    Args:
        environ: The parent's environment (default: ``os.environ``).
    """
    source = os.environ if environ is None else environ
    env = {name: source[name] for name in PROCESS_ENV_PASSTHROUGH if source.get(name)}
    env.update(FORCED_PROCESS_ENV)
    return env


# --------------------------------------------------------------------------- #
# The Bash guard
# --------------------------------------------------------------------------- #
_BASH_ALLOWED_SUMMARY: Final = (
    "Bash is limited to `git diff`, `git log` and `git show`, as one plain command: no "
    "chaining, pipes, redirection, substitution or variables. Pathspecs go after `--` and "
    "are literal paths (no globs); `git show <rev>:<path>` names only a file tracked at "
    "the base ref - read a file the change adds from `git diff <base>...<head> -- <path>`."
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
                f"git option {name} can write a file, read outside the repository, run a "
                "program or search every ref, so it is not allowed."
            )
        return None
    if word.startswith(_GIT_DENIED_SHORT_PREFIXES):
        return f"git option {word!r} is not allowed (it reads an arbitrary file or walks reflogs)."
    return None


def _pathspec_reason(word: str) -> str | None:
    """Check a word after ``--``: a literal file path, nothing git could interpret."""
    if word.startswith("-"):
        return f"{word!r} looks like an option; after `--` only literal file paths are allowed."
    if not _LITERAL_PATH.fullmatch(word):
        return (
            f"{word!r} is not a literal path: globs, pathspec magic (`:`) and unusual characters "
            "are not allowed after `--`."
        )
    return _path_token_reason(word) or _secret_token_reason(word)


def _rev_path_reason(word: str, base_paths: Collection[str]) -> str | None:
    """Check ``<rev>:<path>``: only a non-secret file tracked at the trusted base ref."""
    rev, _, path = word.partition(":")
    if not rev or not _REVISION.fullmatch(rev):
        return f"{word!r}: before the `:` there must be a plain revision."
    if not path or ":" in path:
        return f"{word!r} is not a <rev>:<path> naming one file."
    if secret_path_reason(path):
        return f"{path!r} names a file that may hold secrets; the reviewer never reads those."
    if path not in base_paths:
        return (
            f"{path!r} is not a file tracked at the base ref, so `git show <rev>:<path>` will "
            "not name it; read a file the change adds from `git diff <base>...<head> -- <path>`."
        )
    return None


def _git_word_reason(word: str, base_paths: Collection[str]) -> str | None:
    """Check a word before ``--``: an option, a ``<rev>:<path>``, or a revision / path."""
    if word.startswith("-"):
        return _git_option_reason(word) or _secret_token_reason(word)
    if ":" in word:
        return _rev_path_reason(word, base_paths)
    if not _REVISION.fullmatch(word):
        return (
            f"{word!r} is not a plain revision or path (letters, digits and . _ / @ ^ ~ + -); "
            "put paths after `--`, and keep them literal."
        )
    return _path_token_reason(word) or _secret_token_reason(word)


def _check_git(argv: list[str], base_paths: Collection[str]) -> str | None:
    subcommand = argv[1]
    if subcommand not in _GIT_SUBCOMMANDS:
        return f"`git {subcommand}` is not read-only inspection. {_BASH_ALLOWED_SUMMARY}"
    after_separator = False
    for word in argv[2:]:
        if word == "--" and not after_separator:
            after_separator = True
            continue
        reason = _pathspec_reason(word) if after_separator else _git_word_reason(word, base_paths)
        if reason:
            return reason
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


def check_bash_command(command: object, *, base_paths: Collection[str] = frozenset()) -> str | None:
    """Return why ``command`` is refused, or ``None`` if the reviewer may run it.

    Args:
        command: The ``command`` field of a Bash tool call (anything, so a malformed
            payload is refused rather than raising).
        base_paths: The files tracked at the trusted base ref - the only files
            ``git show <rev>:<path>`` may name.

    Returns:
        ``None`` for ``git diff|log|show ...`` as a single plain command; otherwise a
        sentence the model can act on.
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
            "quote it - and note that pathspecs are literal here, so a glob matches nothing."
        )
    try:
        argv = shlex.split(stripped, posix=True)
    except ValueError as exc:
        return f"The command could not be parsed ({exc})."
    if len(argv) >= 2 and argv[0] == "git":
        return _check_git(argv, base_paths)
    return _BASH_ALLOWED_SUMMARY


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
    *,
    base_paths: Collection[str] = frozenset(),
) -> dict[str, Any]:
    """PreToolUse hook: refuse every Bash call that is not read-only git inspection.

    Returns ``{}`` (no opinion) when the command is acceptable, so the permission
    rules still apply - an explicit allow would approve the call on its own. Fails
    closed: a malformed payload or an internal error is a deny, never a pass.

    Args:
        input_data: The SDK's ``PreToolUseHookInput``.
        tool_use_id: The tool call id (unused).
        context: The SDK's hook context (unused).
        base_paths: The files tracked at the trusted base ref (see :class:`BashGuard`).
    """
    try:
        if not isinstance(input_data, Mapping) or "tool_name" not in input_data:
            return _deny("The Bash request was malformed.")
        if input_data["tool_name"] != BASH_HOOK_MATCHER:
            return {}
        tool_input = input_data.get("tool_input")
        command = tool_input.get("command") if isinstance(tool_input, Mapping) else None
        reason = check_bash_command(command, base_paths=base_paths)
    except Exception as exc:  # fail closed: a guard that crashes must not wave a call through
        return _deny(f"The read-only Bash guard failed ({type(exc).__name__}); the call is denied.")
    return {} if reason is None else _deny(reason)


@dataclass(frozen=True)
class BashGuard:
    """The PreToolUse hook the reviewer registers: a callable bound to the base ref's files.

    Attributes:
        base_paths: Every file tracked at the trusted base ref. ``git show <rev>:<path>``
            is accepted only when ``<path>`` is one of them (and is not a secret file).
    """

    base_paths: frozenset[str] = frozenset()

    def check(self, command: object) -> str | None:
        """Return why ``command`` is refused, or ``None``."""
        return check_bash_command(command, base_paths=self.base_paths)

    async def __call__(
        self, input_data: Mapping[str, Any] | None, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """Hook entry point: ``{}`` to pass, or a deny payload."""
        return await deny_non_readonly_bash(
            input_data, tool_use_id, context, base_paths=self.base_paths
        )


# --------------------------------------------------------------------------- #
# The mock-only MCP server
# --------------------------------------------------------------------------- #
def _declared_entry(mcp_json: str, source: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(mcp_json)
    except ValueError as exc:
        raise PolicyViolationError(f"{source} is not valid JSON: {exc}") from exc
    servers = raw.get("mcpServers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict):
        raise PolicyViolationError(f"{source} has no `mcpServers` object")
    entry = servers.get(MCP_SERVER)
    if not isinstance(entry, dict):
        raise PolicyViolationError(f"{source} declares no `{MCP_SERVER}` server")
    return entry


def confirm_mcp_declaration(mcp_json: str, source: str) -> None:
    """Confirm that ``mcp_json`` declares the pinned ``mousedroid`` invocation.

    The text is read from the trusted base ref and is NOT used to build anything: it is
    only checked against :data:`MCP_DECLARED_COMMANDS` and :data:`MCP_ARGS`, so that a
    change to how trunk starts the server is noticed (and the pin updated on purpose)
    instead of silently followed.

    Raises:
        PolicyViolationError: The text is not JSON, lacks the server, declares a command
            or arguments other than the pinned ones, or a non-stdio transport.
    """
    entry = _declared_entry(mcp_json, source)
    command = entry.get("command")
    if not isinstance(command, str) or (
        command not in MCP_DECLARED_COMMANDS and command != sys.executable
    ):
        raise PolicyViolationError(
            f"{source}: `{MCP_SERVER}` declares command {command!r}, not one of "
            f"{sorted(MCP_DECLARED_COMMANDS)}; refusing to start a server the trusted base "
            "ref does not describe"
        )
    if entry.get("args") != list(MCP_ARGS):
        raise PolicyViolationError(
            f"{source}: `{MCP_SERVER}` declares args {entry.get('args')!r}, not the pinned "
            f"{list(MCP_ARGS)!r}; update policy.MCP_ARGS (and its test) if trunk changed on purpose"
        )
    if entry.get("type", "stdio") != "stdio" or "url" in entry:
        raise PolicyViolationError(f"{source}: `{MCP_SERVER}` is not a stdio server")


def build_mcp_servers(mcp_json: str, source: str = ".mcp.json") -> dict[str, dict[str, Any]]:
    """Build the reviewer's ``mcp_servers`` from constants, confirmed by the base ref's file.

    The entry is ``{"type": "stdio", "command": sys.executable, "args": MCP_ARGS, "env":
    MCP_ENV}`` - nothing is passed through from ``mcp_json``, which (read from the trusted
    base ref, never the working tree) only has to confirm the pinned invocation.

    Args:
        mcp_json: The text of ``.mcp.json`` at the trusted base ref.
        source: What to call it in error messages (for example ``<sha>:.mcp.json``).

    Raises:
        PolicyViolationError: The file does not confirm the pinned invocation, or there is
            no interpreter to run it with.
    """
    confirm_mcp_declaration(mcp_json, source)
    if not sys.executable:
        raise PolicyViolationError("cannot locate the running interpreter for the MCP server")
    return {
        MCP_SERVER: {
            "type": "stdio",
            "command": sys.executable,
            "args": list(MCP_ARGS),
            "env": dict(MCP_ENV),
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
    mcp_json: str,
    mcp_source: str = ".mcp.json",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the keyword arguments for ``ClaudeAgentOptions`` (everything but ``hooks``).

    The set of keys is closed and pinned by a test: a new option is a deliberate,
    reviewed addition. ``hooks`` is registered separately (it needs the SDK's
    ``HookMatcher``): ``{"PreToolUse": [HookMatcher("Bash", [BashGuard(base_paths)])]}``.

    Args:
        cfg: The reviewer's workforce settings (``max_turns`` comes from here).
        repo_root: Repository root, used as the session's working directory.
        model: The model alias or id to run.
        system_prompt: The assembled system prompt.
        cap_usd: This run's budget; passed to the SDK as ``max_budget_usd`` so the CLI
            itself stops at it.
        mcp_json: ``.mcp.json`` as it is at the TRUSTED BASE REF (not the working tree).
        mcp_source: What to call it in error messages.
        environ: The parent environment :func:`process_env` allowlists from.

    Raises:
        PolicyViolationError: ``cap_usd`` is not a finite number greater than zero (the
            CLI rejects ``0``, and an unbounded budget is the thing being prevented), or
            ``mcp_json`` does not confirm the pinned MCP invocation.
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
        "mcp_servers": build_mcp_servers(mcp_json, mcp_source),
        "strict_mcp_config": True,
        "setting_sources": [],
        "env": process_env(environ),
        # Prompts are delivered exactly as written: no `@/path` file-mention expansion.
        "verbatim_prompts": True,
    }
