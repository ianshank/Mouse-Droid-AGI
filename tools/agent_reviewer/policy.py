"""The read-only containment of the Agent-SDK peer reviewer (F-059 WS-B / WS-D).

Threat model
------------
The reviewer runs on the operator's workstation with an API key in its environment,
against a change set that may be HOSTILE: the head commit controls every file in the
working tree. So the trust boundary is the BASE ref - nothing the reviewed tree controls
may reach an exec sink, an environment that holds the key, a path the reviewer should
not read, or the reviewer's own configuration and instructions. Everything that enforces
that lives here, as plain data and pure functions, so tests pin it without the SDK, a
network or a key. The layers, in the order the SDK applies them:

1. ``BASE_TOOLS`` (``tools=``): the model is only shown ``Read``, ``Grep``,
   ``Glob`` and ``Bash``. ``Write``, ``Edit``, the web tools and the subagent tool
   do not exist for it.
2. :class:`BashGuard` (a PreToolUse hook): hooks run FIRST. A whitelist -
   ``git diff|log|show``, nothing else, as one plain command. It refuses the git
   options that write a file, run a configured program, read outside the repo (also
   ``-O<file>`` bundled behind other short options: ``-pO<file>``), widen the search to
   every ref (``--all``, ``--reflog``, ...) or print blob ids (``--raw``, ``--abbrev``).
   Pathspecs after ``--`` must be literal (no globs, no ``:`` magic, nothing starting with
   ``-``). A revision must be one of the PINNED commits
   (the resolved base and head ids) with ancestry suffixes (``^``, ``~N``): a bare blob
   or tree id, an abbreviation or a ref name could name an object whatever its path, so
   none is accepted. And ``git show <rev>:<path>`` is accepted only for a non-secret
   file TRACKED AT THE BASE REF. The hook returns nothing when it has no objection: an
   explicit "allow" would approve the call on its own, skipping the allowlist match
   below, so it can only tighten, never grant. There is no pytest and no interpreter
   on the list: ``--collect-only`` still imports the repository's ``conftest.py`` and
   test modules, which is the head commit's code running in a process that holds the key.
2b. :class:`ReadGuard` (the second PreToolUse hook, on ``Read|Grep|Glob``): the CLI's deny
   rules are cwd-relative screens, so the hook ``realpath``s every path the tool is given
   (``file_path`` and ``path``: the CLI reads a Grep ``file_path`` as its ``path``), requires
   it to be INSIDE the checkout under review (that also resolves 8.3 short names,
   ``::$DATA`` streams and symlink aliases), and refuses ``.git`` internals, secret-file and
   private-key names, Windows device names (``NUL``, ``COM1``, ...) and glob characters in a
   path. A GLOB is not judged by rewriting it into text - the CLI splits a Grep glob on
   whitespace and commas and hands the pieces to ripgrep, which reads classes, braces, ``?``,
   ``!`` and escapes - but by a strict grammar: letters, digits and ``. _ @ + -``, ``/`` and
   ``*`` only; directory parts are literal names, ``*`` or ``**``; the file part is a literal
   name or ``*.<extension>``. The file part is then matched (``fnmatch``) against concrete
   names made from the secret patterns (``.env``, ``prod.env``, ``docker.env.local``,
   ``.env.production``, ``id_rsa``, ...), and the literal start of the glob is resolved
   against the search directory and must stay inside the checkout. The guard never rewrites
   a call (the CLI already turns the ``Read()`` deny rules into ripgrep exclusions); it
   refuses, answers with a reason and does not raise. The run itself is refused when the
   head or the checkout tracks a symbolic link that can leave the checkout
   (:func:`symlink_escape_reason`). Fails closed.
3. :func:`disallowed_tools` (deny rules): evaluated even if the hook were to fail.
   Writes, web access, subagents, every actuation tool by name, and reads of
   secret files (``*.env``, ``docker.env*``) and credential directories.
4. ``permission_mode="dontAsk"``: anything not pre-approved is denied instead of
   prompting - a headless run has nobody to ask.
5. ``ALLOWED`` (``allowed_tools=``): the allowlist itself.
6. :func:`build_mcp_servers`: the only MCP server is ``mousedroid``, and its entry is
   BUILT IN CODE from constants - never passed through from any ``.mcp.json``. The
   interpreter is ``sys.executable`` run ISOLATED (``-I``: neither the working
   directory nor any ``PYTHON*`` variable reaches the import path), through a fixed
   bootstrap that first changes to a scratch directory outside the checkout (the
   CLI's stdio schema has no ``cwd``) and then runs ``mousedroid`` with a
   configuration file that the runner materialised there from the BASE ref. The ``.mcp.json``
   is read from the base ref too and only CONFIRMS that it still declares the pinned
   invocation; a mismatch refuses the run. ``strict_mcp_config`` keeps the CLI from
   loading any other server (the ``github`` entry of ``.mcp.json`` carries a token and
   can write).
7. ``setting_sources=[]``: no user, project or local settings file can widen any
   of the above - and the head's ``.claude/settings.json`` hooks never run.
8. :func:`process_env`: the environment the reviewer adds for the CLI process is an
   allowlist (``PATH``, ``HOME``, ``LANG``) plus forced constants (literal git
   pathspecs, mock hardware); no credential is copied into it. NOTE: the SDK merges
   it over the inherited environment, so it cannot REMOVE what the CLI inherits -
   the key the CLI needs stays in the CLI's environment. What keeps the key out of
   the processes the CLI starts is that none of them runs tree-controlled code
   (layers 2 and 6), that the MCP server entry blanks the credential variables, and,
   as an opt-in, the CLI's own ``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`` (see README). The
   children the REVIEWER starts itself (its ``git``, its ``gitleaks``) get its environment
   through :func:`scrubbed_environ`, without any variable that can hold a credential: a
   name with KEY, TOKEN, SECRET, PASSWORD or CREDENTIAL(S) as one of its words, and a few
   named ones.
9. :func:`resolve_executable`: ``git`` and ``gitleaks`` are looked up in the absolute
   directories of ``PATH`` only. ``shutil.which`` and Windows' ``CreateProcess`` both
   look in the working directory first, so a ``git.exe`` committed to the head would
   run with the key in its environment. The CLI looks up more programs by bare name than
   can be wrapped, so :func:`planted_program_reason` also refuses a head (or checkout)
   that tracks a root file named like one of them, before any spend.

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
* ``Read``, ``Grep`` and ``Glob`` are confined to the checkout by :class:`ReadGuard`, but
  INSIDE it a ``Grep`` over a directory can still print lines of a secret file the CLI's
  own deny rules do not catch (an env file under another name). Run the review from a
  checkout that holds no real secrets.
* The Bash guard is a whitelist over a small character set, not a shell parser.
  It fails closed (an unusual-but-harmless command is refused) because the
  alternative is parsing shell.
* ``git log -p`` over the pinned revisions shows every file those commits ever
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
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from tools.claude_hooks.config import AgentReviewerConfig

#: The one MCP server the reviewer may see (a key of ``.mcp.json``'s ``mcpServers``).
MCP_SERVER: Final = "mousedroid"

#: PreToolUse matcher the Bash guard is registered under.
BASH_HOOK_MATCHER: Final = "Bash"

#: PreToolUse matcher the read-confinement guard (:class:`ReadGuard`) is registered under.
READ_HOOK_MATCHER: Final = "Read|Grep|Glob"

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

#: The repo-relative configuration file the server is started with. The runner reads it
#: from the BASE ref and hands the server a copy in a temporary directory; the checkout's
#: own copy (which the head controls) is never given to it.
MCP_CONFIG_RELPATH: Final = "config/default.yaml"

#: The invocation the committed ``.mcp.json`` must still declare. It is only a pin to
#: CONFIRM against: the entry handed to the SDK is built from constants, never from the
#: file (see :func:`mcp_launch_args`), and a mismatch refuses the run.
MCP_DECLARED_ARGS: Final[tuple[str, ...]] = ("-m", MCP_MODULE, "--config", MCP_CONFIG_RELPATH)

#: What the committed ``.mcp.json`` may name as the interpreter. The entry built for the
#: SDK always uses ``sys.executable`` - an absolute path, so no search of the working
#: directory (which the head controls, and which Windows searches before ``PATH``).
MCP_DECLARED_COMMANDS: Final = frozenset({"python", "python3"})

#: What the server is started through, as ``python -I -c <this> <absolute config path>``.
#: One line and single quotes only, so no command-line quoting can change it. It changes
#: to the scratch directory that holds the configuration - the CLI's stdio schema has no
#: ``cwd`` - and then does what ``python -m mousedroid --config <file>`` does.
#: ``-I`` keeps the working directory and every ``PYTHON*`` variable off the import path.
MCP_BOOTSTRAP: Final = (
    "import os,runpy,sys;"
    "c=sys.argv[1];"
    "os.chdir(os.path.dirname(c));"
    f"sys.argv=[{MCP_MODULE!r},'--config',c];"
    f"runpy.run_module({MCP_MODULE!r},run_name='__main__',alter_sys=True)"
)

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

#: Constants forced into the CLI process (and so into its Bash children). On Windows the
#: programs the CLI starts are not looked for in the working directory - the checkout under
#: review - before ``PATH``.
FORCED_PROCESS_ENV: Final[Mapping[str, str]] = MappingProxyType(
    {
        **GIT_ENV,
        "MOUSEDROID_MOCK_HARDWARE": "true",
        "NoDefaultCurrentDirectoryInExePath": "1",
    }
)

#: Variables that can hold a credential. The CLI needs the API key; nothing ELSE the
#: reviewer starts - its own git, its own gitleaks - does, so they run without these. The
#: names are the ones this project is known to use, the AWS secret key, and the password
#: variables the word rule below cannot see; the rule catches the rest.
CREDENTIAL_ENV_NAMES: Final = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "MOUSEDROID_LLM__API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "PGPASSWORD",
        "MYSQL_PWD",
    }
)

#: A name made of words (split on ``_``, or any other non-alphanumeric), one of which is KEY,
#: TOKEN, SECRET, PASSWORD or CREDENTIAL(S): ``API_KEY``, a bare ``TOKEN``, ``AWS_ACCESS_KEY_ID``,
#: ``GITHUB_TOKEN``, ``GOOGLE_APPLICATION_CREDENTIALS``. ``KEYBOARD`` is not.
_CREDENTIAL_WORD: Final = re.compile(
    r"(?:^|[^A-Z0-9])(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)(?:[^A-Z0-9]|$)"
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
    # print blob ids, which a later call could try to use as a revision
    "--raw",
    "--abbrev",
)

# -O<orderfile> reads an arbitrary file, alone (`-O/etc/passwd`) or bundled after other
# short options (`-pO/etc/passwd`: git reads that as `-p -O/etc/passwd`); -g is
# --walk-reflogs. The bundled form is refused on any capital O, so a pickaxe string with
# one (`-SOrange`) is refused too: the model attaches it with `--pickaxe...` or spells it out.
_BUNDLED_ORDERFILE: Final = re.compile(r"-[A-Za-z]*O.*")
_GIT_DENIED_SHORT_PREFIXES: Final[tuple[str, ...]] = ("-g",)

# A pinned commit id (SHA-1 or SHA-256, lowercase) plus ancestry suffixes: `^`, `^2`, `~`,
# `~3`. Nothing else names an object the guard lets through: a bare blob or tree id, an
# abbreviation, a ref name or `<rev>^{tree}` could print an object whatever its path.
_COMMIT_ISH: Final = re.compile(r"([0-9a-f]{40}|[0-9a-f]{64})((?:\^[0-9]*|~[0-9]*)*)")
# `A..B` and `A...B`.
_RANGE_SEPARATOR: Final = re.compile(r"\.\.\.?")
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


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def resolve_executable(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the absolute path of the program ``name`` found on ``PATH``, or ``None``.

    Only ABSOLUTE directories of ``PATH`` are searched - never the working directory, and
    not an empty or relative entry. ``shutil.which`` and Windows' ``CreateProcess`` both
    look in the working directory first, so a ``git.exe`` or ``gitleaks.exe`` committed to
    the head under review would otherwise run in the reviewer's own process, with the API
    key in its environment. A ``name`` that already has a directory part is returned only
    when it is absolute and executable.

    Args:
        name: The program, with or without its extension.
        environ: The environment whose ``PATH`` (and, on Windows, ``PATHEXT``) is searched
            (default: ``os.environ``).
    """
    if os.path.dirname(name):
        return name if os.path.isabs(name) and _is_executable(name) else None
    source = os.environ if environ is None else environ
    names = [name]
    if os.name == "nt":
        extensions = [ext for ext in source.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if ext]
        if os.path.splitext(name)[1].lower() not in {ext.lower() for ext in extensions}:
            names = [name + ext for ext in extensions]
    for directory in source.get("PATH", "").split(os.pathsep):
        directory = directory.strip('"')
        if not directory or not os.path.isabs(directory):
            continue
        for candidate in names:
            path = os.path.join(directory, candidate)
            if _is_executable(path):
                return path
    return None


def _is_credential_name(name: str) -> bool:
    upper = name.upper()
    return upper in CREDENTIAL_ENV_NAMES or _CREDENTIAL_WORD.search(upper) is not None


def scrubbed_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return ``environ`` without any variable that can hold a credential.

    For the children the REVIEWER starts itself (its own ``git``, its own ``gitleaks``):
    they read repository objects and scan text, they have no use for the API key, and a
    program that is not the CLI must not be handed it.

    Args:
        environ: The environment to copy (default: ``os.environ``).
    """
    source = os.environ if environ is None else environ
    return {name: value for name, value in source.items() if not _is_credential_name(name)}


# A name the reviewer or the CLI starts by bare name; on Windows a file called like this in
# the working directory is run BEFORE the one on PATH.
_PLANTED_PROGRAM: Final = re.compile(
    r"(?:git|bash|sh|claude|rg|node|npm|npx|gitleaks|cmd|powershell|pwsh"
    r"|python[0-9.]*|pythonw[0-9.]*)"
    r"(?:\.(?:exe|com|bat|cmd|ps1|vbs|vbe|js|jse|wsf|wsh|msc|lnk|scr))?"
    r"[. ]*",  # Windows drops trailing dots and spaces: `git.exe.` is checked out as `git.exe`
    re.IGNORECASE,
)


def planted_program_reason(names: Iterable[str]) -> str | None:
    """Return why a root-level file name looks like a planted program, or ``None``.

    The head is checked out into the directory the CLI runs in. A tracked ``git.exe`` or
    ``bash.cmd`` there is, on Windows, found BEFORE the real program, so the review is
    refused before anything is spent.
    """
    for name in names:
        if _PLANTED_PROGRAM.fullmatch(name):
            return (
                f"the head tracks {name!r} in the repository root, named like a program the "
                "reviewer or the CLI starts; on Windows the working directory is searched first"
            )
    return None


def symlink_escape_reason(link: str, target: str) -> str | None:
    """Return why the tracked symbolic link ``link`` may lead out of the repository, or ``None``.

    ``link`` is the repository-relative path of the link (``docs/latest``) and ``target`` is
    what git stores for it. Only the plainest targets pass: names, with ``..`` in front and
    nowhere else, climbing no higher than the repository root. An absolute or drive-rooted
    target, a ``..`` after a name (which only a walk of the real file system can resolve once
    other links are involved) and a malformed target are all judged to leave: a read tool
    that follows such a link would be reading outside the checkout.
    """
    stored = target.replace("\\", "/")
    if not stored or any(ord(char) < 32 for char in stored):
        return "an empty or malformed target"
    if stored.startswith("/") or re.match(r"[A-Za-z]:", stored):
        return "an absolute target"
    ups = 0
    named = False
    for part in stored.split("/"):
        if part in ("", "."):
            continue
        if part != "..":
            named = True
        elif named:
            return "a '..' after a name, which cannot be followed without the file system"
        else:
            ups += 1
    depth = len([part for part in link.replace("\\", "/").split("/") if part]) - 1
    if ups > depth:
        return "a target above the repository root"
    return None


# --------------------------------------------------------------------------- #
# The Bash guard
# --------------------------------------------------------------------------- #
_BASH_ALLOWED_SUMMARY: Final = (
    "Bash is limited to `git diff`, `git log` and `git show`, as one plain command: no "
    "chaining, pipes, redirection, substitution or variables. Name commits only by the full "
    "ids pinned in the task (optionally followed by ^, ^N or ~N), attach option values to "
    "the option (-n5, --grep=text), put pathspecs after `--` as literal paths (no globs), and "
    "use `git show <commit>:<path>` only for a file tracked at the base - read a file the "
    "change adds from `git diff <base>...<head> -- <path>`."
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
    if _BUNDLED_ORDERFILE.fullmatch(word) or word.startswith(_GIT_DENIED_SHORT_PREFIXES):
        return (
            f"git option {word!r} is not allowed: a capital O anywhere in a bundle of short "
            "options (-O<file>, -pO<file>) makes git read an arbitrary file, and -g walks reflogs."
        )
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


def _revision_part_reason(part: str, revisions: Collection[str], *, word: str) -> str | None:
    """Check one revision: a pinned commit id, optionally with ancestry suffixes."""
    match = _COMMIT_ISH.fullmatch(part)
    if match is not None and match.group(1) in revisions:
        return None
    pinned = " and ".join(sorted(revisions)) or "none (this guard has no pinned commit)"
    return (
        f"{word!r} is not a revision this review allows: name commits only by the full ids "
        f"pinned in the task ({pinned}), optionally followed by ^, ^N or ~N. Put paths after "
        "`--`, and attach option values to the option (-n5, --grep=text): a separate word is "
        "read as a revision."
    )


def _revision_reason(word: str, revisions: Collection[str]) -> str | None:
    """Check a revision expression: a pinned commit alone, or as ``A..B`` / ``A...B``.

    A bare blob or tree id, an abbreviation, a ref name or ``<rev>^{tree}`` could name an
    object whatever its path, so none of them is accepted; neither is a commit id the
    task did not pin. Either end of a range may be empty (``A..``) and a leading ``^``
    excludes a commit.
    """
    parts = _RANGE_SEPARATOR.split(word)
    if len(parts) > 2:
        return f"{word!r}: a range has two ends (A..B or A...B)."
    for part in parts:
        if not part and len(parts) == 2:
            continue
        reason = _revision_part_reason(
            part[1:] if part.startswith("^") else part, revisions, word=word
        )
        if reason:
            return reason
    return None


def _rev_path_reason(
    word: str, base_paths: Collection[str], revisions: Collection[str]
) -> str | None:
    """Check ``<rev>:<path>``: a pinned commit and a non-secret file tracked at the base ref."""
    rev, _, path = word.partition(":")
    reason = _revision_part_reason(rev, revisions, word=word)
    if reason:
        return reason
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


def _git_word_reason(
    word: str, base_paths: Collection[str], revisions: Collection[str]
) -> str | None:
    """Check a word before ``--``: an option, a ``<rev>:<path>``, or a revision expression."""
    if word.startswith("-"):
        return _git_option_reason(word) or _secret_token_reason(word)
    if ":" in word:
        return _rev_path_reason(word, base_paths, revisions)
    return _revision_reason(word, revisions)


def _check_git(
    argv: list[str], base_paths: Collection[str], revisions: Collection[str]
) -> str | None:
    subcommand = argv[1]
    if subcommand not in _GIT_SUBCOMMANDS:
        return f"`git {subcommand}` is not read-only inspection. {_BASH_ALLOWED_SUMMARY}"
    after_separator = False
    for word in argv[2:]:
        if word == "--" and not after_separator:
            after_separator = True
            continue
        reason = (
            _pathspec_reason(word)
            if after_separator
            else _git_word_reason(word, base_paths, revisions)
        )
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


def check_bash_command(
    command: object,
    *,
    base_paths: Collection[str] = frozenset(),
    revisions: Collection[str] = frozenset(),
) -> str | None:
    """Return why ``command`` is refused, or ``None`` if the reviewer may run it.

    Args:
        command: The ``command`` field of a Bash tool call (anything, so a malformed
            payload is refused rather than raising).
        base_paths: The files tracked at the trusted base ref - the only files
            ``git show <rev>:<path>`` may name.
        revisions: The pinned commits (the resolved base and head ids) - the only
            commits a revision word may start from. Empty means none is accepted.

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
        return _check_git(argv, base_paths, revisions)
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
    revisions: Collection[str] = frozenset(),
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
        revisions: The pinned commits (see :class:`BashGuard`).
    """
    try:
        if not isinstance(input_data, Mapping) or "tool_name" not in input_data:
            return _deny("The Bash request was malformed.")
        if input_data["tool_name"] != BASH_HOOK_MATCHER:
            return {}
        tool_input = input_data.get("tool_input")
        command = tool_input.get("command") if isinstance(tool_input, Mapping) else None
        reason = check_bash_command(command, base_paths=base_paths, revisions=revisions)
    except Exception as exc:  # fail closed: a guard that crashes must not wave a call through
        return _deny(f"The read-only Bash guard failed ({type(exc).__name__}); the call is denied.")
    return {} if reason is None else _deny(reason)


@dataclass(frozen=True)
class BashGuard:
    """The PreToolUse hook the reviewer registers: a callable bound to the base ref's facts.

    Attributes:
        base_paths: Every file tracked at the trusted base ref. ``git show <rev>:<path>``
            is accepted only when ``<path>`` is one of them (and is not a secret file).
        revisions: The pinned commits - the resolved base and head ids. A revision word is
            accepted only when it is one of them, optionally with ``^`` / ``~N`` suffixes.
    """

    base_paths: frozenset[str] = frozenset()
    revisions: frozenset[str] = frozenset()

    def check(self, command: object) -> str | None:
        """Return why ``command`` is refused, or ``None``."""
        return check_bash_command(command, base_paths=self.base_paths, revisions=self.revisions)

    async def __call__(
        self, input_data: Mapping[str, Any] | None, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """Hook entry point: ``{}`` to pass, or a deny payload."""
        return await deny_non_readonly_bash(
            input_data,
            tool_use_id,
            context,
            base_paths=self.base_paths,
            revisions=self.revisions,
        )


# --------------------------------------------------------------------------- #
# The Read / Grep / Glob confinement
# --------------------------------------------------------------------------- #
#: The tools :class:`ReadGuard` governs.
_READ_TOOLS: Final = frozenset({"Read", "Grep", "Glob"})

#: Both spellings of "where to look" are judged on every tool, whichever the tool documents:
#: the CLI reads a Grep ``file_path`` as its ``path`` (and a Write ``path`` as ``file_path``),
#: and a hook may be handed the input before or after that rewrite.
_READ_PATH_KEYS: Final = ("file_path", "path")

#: The field that carries a glob: Grep's file filter, Glob's own pattern.
_READ_GLOB_FIELD: Final = MappingProxyType({"Grep": "glob", "Glob": "pattern"})

_MAX_PATH_CHARS: Final = 4096

#: A path field takes a plain path; these make it a pattern (the CLI hands it to ripgrep).
_PATH_PATTERN_CHARS: Final = frozenset("*?[]{}")

#: A Windows device name - ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1-9``, ``LPT1-9`` (also
#: the superscript digits) - with or without an extension, a trailing dot or space or a
#: stream: the OS opens the DEVICE whatever directory the name is in, so such a path is
#: not a file inside the checkout, and resolving it can even raise.
_DEVICE_NAME: Final = re.compile(
    r"(?:con|prn|aux|nul|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3]|conin\$|conout\$)"
    r"(?:[. :].*)?",
    re.IGNORECASE | re.DOTALL,
)

#: Private-key file names the read guard refuses besides :data:`SECRET_PATH_GLOBS`: the usual
#: ``ssh-keygen`` outputs (their ``.pub`` halves are public and stay readable).
PRIVATE_KEY_NAMES: Final[tuple[str, ...]] = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")

#: What ``*`` stands for when a secret-file pattern is turned into concrete names.
_PROBE_FILLERS: Final[tuple[str, ...]] = (
    "",
    "x",
    "prod",
    "local",
    "production",
    ".local",
    ".production",
)

#: A glob is written in this alphabet and no other. Everything a glob engine reads as syntax
#: (``? [ ] { } ! \``), and everything the CLI splits a Grep glob on (whitespace, commas),
#: is outside it, so no glob can mean to the engine what the screen did not see.
_GLOB_ALPHABET: Final = re.compile(r"[A-Za-z0-9_.@+*/-]+")
_GLOB_FILE_PART: Final = re.compile(r"\*\.[A-Za-z0-9_+-]+")


def path_is_inside(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` or below it (both already real paths)."""
    try:
        common = os.path.commonpath([os.path.normcase(path), os.path.normcase(root)])
    except ValueError:  # another drive
        return False
    return common == os.path.normcase(root)


def read_secret_reason(path: str | Path) -> str | None:
    """Return why ``path`` names a file the read guard never opens, or ``None``.

    The project's secret-file patterns (:data:`SECRET_PATH_GLOBS`) plus
    :data:`PRIVATE_KEY_NAMES`, matched on the base name, case-insensitively.
    """
    reason = secret_path_reason(path)
    if reason:
        return reason
    if PurePosixPath(str(path).replace("\\", "/")).name.lower() in PRIVATE_KEY_NAMES:
        return "is a private-key file name"
    return None


def secret_probe_names(patterns: Iterable[str]) -> tuple[str, ...]:
    """Turn secret-file patterns into concrete lower-case names a glob can be tested against.

    ``*.env`` becomes ``.env``, ``prod.env``, ...; ``docker.env*`` becomes ``docker.env``,
    ``docker.env.local``, ...; ``.env.*`` becomes ``.env.local``, ``.env.production``, ...;
    a pattern without ``*`` (``id_rsa``) is its own probe. A glob is judged by asking
    whether it matches one of these names, never by rewriting the glob into text.
    """
    names: list[str] = []
    for pattern in patterns:
        if "*" not in pattern:
            names.append(pattern.lower())
            continue
        names.extend(pattern.replace("*", filler).lower() for filler in _PROBE_FILLERS)
    return tuple(dict.fromkeys(name for name in names if not name.endswith(".")))


#: The names a glob's file part must not be able to match.
_SECRET_PROBES: Final = secret_probe_names((*SECRET_PATH_GLOBS, *PRIVATE_KEY_NAMES))


def _device_name_in(path: str) -> str | None:
    """Return the first component of ``path`` that is a Windows device name, or ``None``."""
    return next((part for part in re.split(r"[\\/]", path) if _DEVICE_NAME.fullmatch(part)), None)


def _real_of(raw: str, root: str) -> str:
    """The real path a tool would open for ``raw`` (links and short names followed)."""
    return os.path.realpath(raw if os.path.isabs(raw) else os.path.join(root, raw))


def _git_internals_reason(real: str, root_real: str, what: str) -> str | None:
    parts = os.path.relpath(real, root_real).replace("\\", "/").split("/")
    if any(part.lower() == ".git" for part in parts):
        return f"{what} is inside .git; use `git show` for repository content."
    return None


def _confined_path_reason(raw: object, root: str) -> str | None:
    """Check one path the tool would open: a plain path that resolves inside ``root``."""
    if not isinstance(raw, str) or not raw.strip():
        return "the tool call names no path."
    if len(raw) > _MAX_PATH_CHARS or any(ord(char) < 32 for char in raw):
        return "the path is too long or contains control characters."
    device = _device_name_in(raw)
    if device is not None:
        return f"{raw!r} names a Windows device ({device!r}); the reviewer reads files only."
    if raw.startswith("~") or "$" in raw or "%" in raw:
        return (
            f"{raw!r} would be expanded (~, $VAR, %VAR%) before it is opened; "
            "give a plain path inside the checkout."
        )
    if _PATH_PATTERN_CHARS.intersection(raw):
        return (
            f"{raw!r} contains glob characters (* ? [ ] {{ }}); a path field takes a plain "
            "path, and patterns go in the glob field."
        )
    if ":" in os.path.splitdrive(raw)[1]:
        return f"{raw!r} contains ':' (an NTFS data stream or a pathspec); give a plain path."
    real = _real_of(raw, root)
    root_real = os.path.realpath(root)
    if not path_is_inside(real, root_real):
        return (
            f"{raw!r} resolves outside the checkout under review (symlinks and short names are "
            "resolved first); the reviewer reads only files inside it."
        )
    reason = _git_internals_reason(real, root_real, repr(raw))
    if reason:
        return reason
    reason = read_secret_reason(real)
    if reason:
        return f"{raw!r} {reason}; the reviewer never reads those."
    return None


def _is_plain_name(segment: str) -> bool:
    """A literal file or directory name of a glob already known to be in the plain alphabet:
    no wildcard, not ``.``, ``..`` or ``...``, not ending in a dot (Windows drops it)."""
    return (
        segment != ""
        and "*" not in segment
        and not segment.startswith("..")
        and not segment.endswith(".")
    )


def _glob_directories_reason(pattern: str, directories: list[str]) -> str | None:
    """Check the directory parts of a glob: each is a literal name, ``*`` or ``**``."""
    for segment in directories:
        if segment == "":
            return f"the glob {pattern!r} has an empty part (a doubled /)."
        if segment.lower() == ".git":
            return f"the glob {pattern!r} names .git; use `git show` for repository content."
        if segment in ("*", "**") or _is_plain_name(segment):
            continue
        if segment.startswith(".."):
            return f"the glob {pattern!r} may not climb out of the checkout."
        return (
            f"the glob {pattern!r} has the directory part {segment!r}; "
            "a directory part is a literal name, * or **."
        )
    return None


def _glob_secret_reason(pattern: str, file_part: str) -> str | None:
    """Return why a glob's file part can reach a secret-file name, or ``None``.

    A literal is judged by name. Any file part is also matched (``fnmatch``) against the
    concrete secret names from :func:`secret_probe_names`, so ``*.env``, ``*.local`` and a
    bare ``*`` are refused while ``*.py`` is not.
    """
    reason = read_secret_reason(file_part)
    if reason is None:
        lowered = file_part.lower()
        probe = next((p for p in _SECRET_PROBES if fnmatch.fnmatchcase(p, lowered)), None)
        if probe is None:
            return None
        reason = f"could match the secret-file name {probe!r}"
    else:
        reason = f"could match a file that {reason}"
    return (
        f"the glob {pattern!r} {reason}; the reviewer never reads those. "
        "Name an extension that is not a secret one, for example *.py."
    )


def _glob_file_part_reason(pattern: str, file_part: str) -> str | None:
    """Check the file part of a glob: a literal name or ``*.<extension>`` (``*`` and ``**``
    alone are let through here and stopped by :func:`_glob_secret_reason`)."""
    if file_part == "":
        return f"the glob {pattern!r} ends with / and names no file; end it with a name or *.py."
    if file_part.lower() == ".git":
        return f"the glob {pattern!r} names .git; use `git show` for repository content."
    if (
        file_part in ("*", "**")
        or _GLOB_FILE_PART.fullmatch(file_part)
        or _is_plain_name(file_part)
    ):
        return None
    if file_part.startswith(".."):
        return f"the glob {pattern!r} may not climb out of the checkout."
    return (
        f"the glob {pattern!r} has the file part {file_part!r}; "
        "the file part is a literal name or *.<extension> (for example *.py)."
    )


def _glob_prefix_reason(pattern: str, base: str, root_real: str) -> str | None:
    """Resolve the literal (wildcard-free) leading part of a glob and keep it inside the root.

    The CLI resolves a glob against the search directory, so ``link/*.py`` reads ``link``
    first. If ``link`` is a symbolic link out of the checkout, that is a way out the glob's
    spelling does not show. A glob with no wildcard at all names one file, which is then
    also judged like a ``Read`` path.
    """
    segments = pattern.split("/")
    literal: list[str] = []
    for segment in segments:
        if "*" in segment:
            break
        literal.append(segment)
    if not literal:
        return None
    real = os.path.realpath(os.path.join(base, *literal))
    if not path_is_inside(real, root_real):
        return (
            f"the glob {pattern!r} starts at {'/'.join(literal)!r}, which resolves outside the "
            "checkout under review (links and short names are followed first)."
        )
    reason = _git_internals_reason(real, root_real, f"the glob {pattern!r}")
    if reason:
        return reason
    if len(literal) == len(segments):  # no wildcard: it names one file
        reason = read_secret_reason(real)
        if reason:
            return f"the glob {pattern!r} {reason}; the reviewer never reads those."
    return None


def _glob_reason(pattern: object, *, base: str, root_real: str) -> str | None:
    """Check a glob: written in the plain alphabet, unable to match a secret-file name, and
    starting somewhere inside the checkout."""
    if not isinstance(pattern, str) or not pattern:
        return "a glob must be a non-empty string."
    if len(pattern) > _MAX_PATH_CHARS or any(ord(char) < 32 for char in pattern):
        return "the glob is too long or contains control characters."
    if pattern.startswith(("/", "~")) or re.match(r"[A-Za-z]:", pattern):
        return (
            f"the glob {pattern!r} is absolute or home-relative; use a pattern inside the checkout."
        )
    if not _GLOB_ALPHABET.fullmatch(pattern):
        return (
            f"the glob {pattern!r} uses a character a glob engine reads as syntax, or the CLI "
            "splits a glob on (a class [..], a brace {..}, ?, !, a backslash, a comma, a space); "
            "write it with letters, digits, . _ @ + - and / and *."
        )
    device = _device_name_in(pattern)
    if device is not None:
        return f"the glob {pattern!r} names a Windows device ({device!r}); files only."
    *directories, file_part = pattern.split("/")
    return (
        _glob_directories_reason(pattern, directories)
        or _glob_secret_reason(pattern, file_part)
        or _glob_file_part_reason(pattern, file_part)
        or _glob_prefix_reason(pattern, base, root_real)
    )


def check_read_access(tool_name: object, tool_input: object, *, root: str) -> str | None:
    """Return why a ``Read`` / ``Grep`` / ``Glob`` call is refused, or ``None``.

    It answers and does not raise: a path the checks cannot resolve (an OS error, a name
    the platform refuses to normalise) is not a path to read, so that too is a reason.

    Args:
        tool_name: The tool (``Read``, ``Grep`` or ``Glob``); anything else is refused.
        tool_input: The call's input (anything, so a malformed payload is refused).
        root: The checkout under review; every path must resolve inside it.
    """
    try:
        return _check_read_access(tool_name, tool_input, root=root)
    except Exception as exc:  # fail closed: what cannot be checked is refused
        return f"the read could not be checked safely ({type(exc).__name__}); it is refused."


def _check_read_access(tool_name: object, tool_input: object, *, root: str) -> str | None:
    if not isinstance(tool_name, str) or tool_name not in _READ_TOOLS:
        return f"{tool_name!r} is not a read tool."
    if not isinstance(tool_input, Mapping):
        return "The read request was malformed."
    root_real = os.path.realpath(root)
    base = root_real  # a glob is resolved against the search directory, else the checkout
    given = [tool_input[key] for key in _READ_PATH_KEYS if tool_input.get(key) is not None]
    if not given and tool_name == "Read":
        return "Read needs a file_path."
    for raw in given:
        reason = _confined_path_reason(raw, root)
        if reason:
            return reason
    if given:
        base = _real_of(given[-1], root)  # `path` wins over `file_path`, as in the CLI
    glob_field = _READ_GLOB_FIELD.get(tool_name)
    glob = None if glob_field is None else tool_input.get(glob_field)
    if glob is None:
        return None
    return _glob_reason(glob, base=base, root_real=root_real)


async def deny_unconfined_reads(
    input_data: Mapping[str, Any] | None,
    tool_use_id: str | None,
    context: Any,
    *,
    root: str,
) -> dict[str, Any]:
    """PreToolUse hook: refuse every Read / Grep / Glob that leaves the checkout.

    Returns ``{}`` (no opinion) when the call is acceptable, so the permission rules still
    apply - an explicit allow would approve the call on its own. Fails closed: a malformed
    payload or an internal error is a deny, never a pass.

    Args:
        input_data: The SDK's ``PreToolUseHookInput``.
        tool_use_id: The tool call id (unused).
        context: The SDK's hook context (unused).
        root: The checkout under review.
    """
    try:
        if not isinstance(input_data, Mapping) or "tool_name" not in input_data:
            return _deny("The read request was malformed.")
        name = input_data["tool_name"]
        if name not in _READ_TOOLS:
            return {}
        reason = check_read_access(name, input_data.get("tool_input"), root=root)
    except Exception as exc:  # fail closed: a guard that crashes must not wave a call through
        return _deny(f"The read guard failed ({type(exc).__name__}); the call is denied.")
    return {} if reason is None else _deny(reason)


@dataclass(frozen=True)
class ReadGuard:
    """The second PreToolUse hook the reviewer registers: reads stay inside the checkout.

    Attributes:
        root: The checkout under review. Every path ``Read``, ``Grep`` and ``Glob`` would
            open is resolved with ``realpath`` and must be inside it.
    """

    root: str

    def check(self, tool_name: object, tool_input: object) -> str | None:
        """Return why the call is refused, or ``None``."""
        return check_read_access(tool_name, tool_input, root=self.root)

    async def __call__(
        self, input_data: Mapping[str, Any] | None, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """Hook entry point: ``{}`` to pass, or a deny payload."""
        return await deny_unconfined_reads(input_data, tool_use_id, context, root=self.root)


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
    only checked against :data:`MCP_DECLARED_COMMANDS` and :data:`MCP_DECLARED_ARGS`, so
    that a change to how trunk starts the server is noticed (and the pin updated on
    purpose) instead of silently followed.

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
    if entry.get("args") != list(MCP_DECLARED_ARGS):
        raise PolicyViolationError(
            f"{source}: `{MCP_SERVER}` declares args {entry.get('args')!r}, not the pinned "
            f"{list(MCP_DECLARED_ARGS)!r}; update policy.MCP_DECLARED_ARGS (and its test) if "
            "trunk changed on purpose"
        )
    if entry.get("type", "stdio") != "stdio" or "url" in entry:
        raise PolicyViolationError(f"{source}: `{MCP_SERVER}` is not a stdio server")


def mcp_launch_args(config_path: Path | str) -> list[str]:
    """Return the interpreter arguments the MCP server is started with.

    ``-I -c <bootstrap> <config>``: isolated mode (neither the working directory nor any
    ``PYTHON*`` variable reaches the import path, on every supported Python), and the
    bootstrap runs ``mousedroid`` from the configuration file's own directory.

    Args:
        config_path: Absolute path of the configuration file the runner materialised from
            the base ref, inside a directory outside the checkout.

    Raises:
        PolicyViolationError: ``config_path`` is not absolute.
    """
    path = str(config_path)
    if not os.path.isabs(path):
        raise PolicyViolationError(f"the MCP configuration path must be absolute, got {path!r}")
    return ["-I", "-c", MCP_BOOTSTRAP, path]


def build_mcp_servers(
    mcp_json: str, source: str = ".mcp.json", *, config_path: Path | str
) -> dict[str, dict[str, Any]]:
    """Build the reviewer's ``mcp_servers`` from constants, confirmed by the base ref's file.

    The entry is ``{"type": "stdio", "command": sys.executable, "args":
    mcp_launch_args(config_path), "env": MCP_ENV}`` - nothing is passed through from
    ``mcp_json``, which (read from the trusted base ref, never the working tree) only has
    to confirm the pinned invocation.

    Args:
        mcp_json: The text of ``.mcp.json`` at the trusted base ref.
        source: What to call it in error messages (for example ``<sha>:.mcp.json``).
        config_path: Absolute path of the server's configuration file (a copy of the base
            ref's, in a temporary directory outside the checkout).

    Raises:
        PolicyViolationError: The file does not confirm the pinned invocation, the path is
            not absolute, or there is no interpreter to run the server with.
    """
    confirm_mcp_declaration(mcp_json, source)
    if not sys.executable:
        raise PolicyViolationError("cannot locate the running interpreter for the MCP server")
    return {
        MCP_SERVER: {
            "type": "stdio",
            "command": sys.executable,
            "args": mcp_launch_args(config_path),
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
    mcp_config_path: Path | str,
    mcp_source: str = ".mcp.json",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the keyword arguments for ``ClaudeAgentOptions`` (everything but ``hooks``).

    The set of keys is closed and pinned by a test: a new option is a deliberate,
    reviewed addition. ``hooks`` is registered separately (it needs the SDK's
    ``HookMatcher``): ``{"PreToolUse": [HookMatcher("Bash", [BashGuard(...)])]}``.

    Args:
        cfg: The reviewer's workforce settings (``max_turns`` comes from here).
        repo_root: Repository root, used as the session's working directory.
        model: The model alias or id to run.
        system_prompt: The assembled system prompt.
        cap_usd: This run's budget; passed to the SDK as ``max_budget_usd`` so the CLI
            itself stops at it.
        mcp_json: ``.mcp.json`` as it is at the TRUSTED BASE REF (not the working tree).
        mcp_config_path: Absolute path of the MCP server's configuration file (see
            :func:`build_mcp_servers`).
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
        "mcp_servers": build_mcp_servers(mcp_json, mcp_source, config_path=mcp_config_path),
        "strict_mcp_config": True,
        "setting_sources": [],
        "env": process_env(environ),
        # Prompts are delivered exactly as written: no `@/path` file-mention expansion.
        "verbatim_prompts": True,
    }
