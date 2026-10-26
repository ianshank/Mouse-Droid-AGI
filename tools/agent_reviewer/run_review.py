"""Run one read-only Agent-SDK peer review of ``--base...--head`` (F-059 WS-B / WS-D).

    python -m tools.agent_reviewer.run_review \\
        --base origin/<trunk> --head <branch> --cap-usd 10 [--opening R3-1,R3-2] \\
        [--out reports/agent_reviewer/<date>/wsb-NNNN.md] [--dry-run] [--repo-root DIR] \\
        [--ledger-root DIR] [--trust-checkout]

``--dry-run`` builds everything a real run would - prompts, policy, SDK options, the
artifact path, the ledger pre-check - and prints it, WITHOUT constructing a client,
resolving a model, calling the network or writing anything. The first real run happens
only after the operator has rotated the API key (``NEXT_STEPS.md`` item 1).

The key is read from ``ANTHROPIC_API_KEY`` in the environment and from nowhere else; there
is no option that takes one, and it is never printed, logged, put in the options or written
to the artifact or the ledger (the SDK's CLI inherits the environment itself).

The head under review may be hostile, so the trust boundary is the BASE ref. The reviewer's
own configuration (``.claude/workforce.yaml``), its agent prompt, the MCP declaration, the MCP
server's configuration and the secret policy (``.gitleaks.toml``) are all read from the base
commit with ``git cat-file`` - the working tree is never opened for them - and the files
``git show <rev>:<path>`` may name are the ones tracked there. The code that would run beside
the CLI that holds the key is checked too: the reviewer's own modules, the interpreter's
environment and the ``mousedroid`` package the MCP server imports. When any of them lives inside
the checkout under review and the checkout is not at the base commit, the run is refused before
any spend unless ``--trust-checkout`` accepts it; an interpreter environment inside the
checkout that the head TRACKS files in is refused outright, and so is a head (or checkout)
that tracks a symbolic link which can leave the checkout. ``Read``, ``Grep`` and ``Glob`` are
confined to the checkout by a second PreToolUse hook (``policy.ReadGuard``).

The spend ledger is the CONTROLLER's, not the head's: it is resolved against ``--ledger-root``
(default: the checkout this reviewer's own code runs from), never a symbolic link, and the
spend it reports is floored by what the base ref's committed copy records, so a head cannot
delete rows to widen the cap. The cap is per workstream, not per head worktree. A ledger root
inside a checkout that is not at the base is a file the head can rewrite, so it needs
``--trust-checkout`` like the code does.

A real run, in order:

1. validate the input and build every prompt and the policy (any refusal stops here);
2. ``tools.credit_ledger.charge`` - raises ``CapExceeded`` BEFORE any client is built;
3. resolve the model alias to an id (``client.models.retrieve``; the configured alias with
   ``resolved_via: config`` if that is unavailable) - the only call before the review;
4. drive ``claude_agent_sdk.query`` until its ``ResultMessage``, under a wall-clock limit;
5. ALWAYS, however step 4 ended (error result, crash, timeout, Ctrl-C): write the artifact
   under ``reports/agent_reviewer/<date>/`` and append one ledger row.

Exit codes: 0 reviewed and recorded; 1 the review did not succeed (or its ledger row could
not be written - the row is printed so it can be appended by hand); 2 refused before any
spend; 3 the workstream cap would be exceeded; 130 interrupted. The ledger ``status`` is
``ok``, ``error_max_turns``, ``error_max_budget_usd``, ``error_during_execution``,
``error_api``, ``error``, ``error:<ExceptionType>``, ``timeout``, ``interrupted`` or
``output_withheld`` (the review matched the credential screen and was not saved).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from tools import credit_ledger
from tools.agent_reviewer import ledger, policy, prompt
from tools.claude_hooks.config import (
    DEFAULT_CONFIG_RELPATH,
    MODEL_ID_PATTERN,
    ConfigError,
    WorkforceConfig,
    parse_config_text,
)
from tools.claude_hooks.logging_setup import get_logger
from tools.claude_hooks.paths import resolve_repo_root

_log = get_logger(__name__)

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
EXIT_REFUSED: Final = 2
EXIT_CAP: Final = 3
EXIT_INTERRUPTED: Final = 130

#: ``resolved_via`` vocabulary (shared with ``tools/llm_latency_probe.py``).
RESOLVED_VIA_RETRIEVE: Final = "models.retrieve"
RESOLVED_VIA_CONFIG: Final = "config"

#: The variable the SDK's CLI reads. A name defined by the SDK, not a tunable.
API_KEY_ENV: Final = "ANTHROPIC_API_KEY"

#: The ledger's ``artifact`` value when the review file could not be written.
ARTIFACT_NOT_WRITTEN: Final = "(not written)"

#: The option that accepts running code that lives inside the checkout under review.
TRUST_CHECKOUT_FLAG: Final = "--trust-checkout"

#: The file the MCP server is given (a copy of the base ref's configuration) and, for a dry
#: run that writes nothing, the directory it would live in.
_MCP_CONFIG_FILENAME: Final = "mcp-config.yaml"
_DRY_RUN_SCRATCH: Final = "agent-reviewer-dry-run"

_GIT_TIMEOUT_S: Final = 30.0
_SYMLINK_MODE: Final = "120000"  # the mode git records for a symbolic link
_MAX_TRACKED_SYMLINKS: Final = 200  # more than this is not a source tree we will follow links in
_DEFAULT_RUN_PREFIX: Final = "wsb"
_DATE_FORMAT: Final = "%Y-%m-%d"
_TIMESTAMP_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"

# A ref starts with an alphanumeric (never `-`, which git would read as an option) and
# holds only characters a ref or a revision expression needs.
_REF_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@^~+-]*")
_SHA_RE: Final = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_OPENING_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MAX_OPENING_IDS: Final = 50
_MODEL_ID_RE: Final = re.compile(MODEL_ID_PATTERN)


class RefusalError(RuntimeError):
    """The run was refused before any spend (bad input or a missing prerequisite)."""


class SdkRejectedError(RuntimeError):
    """The installed ``claude-agent-sdk`` does not accept the reviewer's options."""


_REFUSALS: Final = (
    RefusalError,
    prompt.PromptRefusedError,
    policy.PolicyViolationError,
    ConfigError,
    credit_ledger.LedgerError,
)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _positive_finite(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return value


def _ref(text: str) -> str:
    if not _REF_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a plain git ref (letters, digits and . _ / @ ^ ~ + -, "
            "not starting with `-`)"
        )
    return text


def _opening_ids(text: str) -> tuple[str, ...]:
    ids: list[str] = []
    for raw in text.split(","):
        item = raw.strip()
        if not _OPENING_ID_RE.fullmatch(item):
            raise argparse.ArgumentTypeError(f"{item!r} is not a finding id such as R3-1")
        if item not in ids:
            ids.append(item)
    if len(ids) > _MAX_OPENING_IDS:
        raise argparse.ArgumentTypeError(f"at most {_MAX_OPENING_IDS} opening ids")
    return tuple(ids)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser. It has no option that takes a credential."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.agent_reviewer.run_review",
        description="Run one read-only Agent-SDK peer review of BASE...HEAD under a spend cap.",
    )
    parser.add_argument("--base", required=True, type=_ref, metavar="REF", help="Base ref.")
    parser.add_argument("--head", required=True, type=_ref, metavar="REF", help="Head ref.")
    parser.add_argument(
        "--cap-usd",
        required=True,
        type=_positive_finite,
        metavar="USD",
        help="Budget for THIS run: checked against the workstream cap first, and passed to "
        "the SDK as its hard stop.",
    )
    parser.add_argument(
        "--opening",
        type=_opening_ids,
        default=(),
        metavar="IDS",
        help="Comma-separated finding ids from earlier rounds to report on (e.g. R3-1,R3-2).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Artifact path inside the repository (default: <artifact_dir>/<date>/<run_id>.md).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan; construct no client, resolve no model, write nothing.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Repository root (default: discovered from the workforce tooling).",
    )
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Checkout that holds the spend ledger (default: the checkout this reviewer's own "
        "code runs from, never the head under review). The cap is per workstream: every "
        "review charges this one ledger.",
    )
    parser.add_argument(
        TRUST_CHECKOUT_FLAG,
        action="store_true",
        help="Accept running code that lives inside the checkout under review (the reviewer's "
        "own modules, the interpreter's virtualenv, an editable `mousedroid`) although the "
        "checkout is not at the base commit: it runs beside the CLI that holds the API key. "
        "Off by default; a virtualenv the head tracks files in is refused regardless.",
    )
    return parser


# --------------------------------------------------------------------------- #
# The plan - everything decided before any spend
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Refs:
    """The commits under review, resolved."""

    base_ref: str
    base_sha: str
    head_ref: str
    head_sha: str
    worktree_sha: str


@dataclass(frozen=True)
class CodeTrust:
    """Where the code that runs beside the API key, and the spend ledger, come from."""

    controller: str
    interpreter: str
    package: str
    ledger: str


@dataclass(frozen=True)
class Plan:
    """Everything a run needs, built (and refused) before any client exists."""

    repo_root: Path
    cfg: WorkforceConfig
    refs: Refs
    opening: tuple[str, ...]
    cap_usd: float
    run_id: str
    artifact: Path
    system_prompt: str
    user_prompt: str
    options: dict[str, Any]
    base_paths: frozenset[str]
    config_source: str
    gitleaks_toml: str | None
    trust: CodeTrust
    ledger_root: Path
    ledger_floor: str | None

    @property
    def read_guard(self) -> policy.ReadGuard:
        """The hook that keeps Read, Grep and Glob inside the checkout under review."""
        return policy.ReadGuard(
            root=str(self.repo_root),
            glob_extensions=tuple(self.cfg.agent_reviewer.glob_extensions),
        )

    @property
    def ledger_floor_source(self) -> str:
        """What the base ref's copy of the ledger is called in messages."""
        return f"{self.refs.base_sha}:{self.cfg.credit.ledger}"

    @property
    def guard(self) -> policy.BashGuard:
        """The Bash guard bound to this run's base-ref facts: its files and its two commits."""
        return policy.BashGuard(
            base_paths=self.base_paths,
            revisions=frozenset({self.refs.base_sha, self.refs.head_sha}),
        )

    @property
    def target(self) -> str:
        """The ledger ``target`` column."""
        return f"diff {self.refs.base_ref}...{self.refs.head_ref} @{self.refs.head_sha[:7]}"

    @property
    def artifact_rel(self) -> str:
        """The artifact path, repo-relative with forward slashes."""
        return self.artifact.relative_to(self.repo_root).as_posix()


async def _git(repo_root: Path, *args: str) -> tuple[int, bytes]:
    """Run ``git -C repo_root <args>``; return ``(returncode, stdout)``.

    Every git the reviewer runs has literal pathspecs (``GIT_LITERAL_PATHSPECS=1``), the
    same environment the model's own git gets - except that it runs without any variable
    that can hold a credential: it reads repository objects, and has no use for the key. A
    git that does not finish in time is killed and reported as a failure (``-1``).

    ``git`` is found in the absolute directories of ``PATH`` only: Windows tries the
    working directory first, and a ``git.exe`` committed to the head would run here, in the
    process that holds the API key.

    Raises:
        RefusalError: ``git`` cannot be found or started.
    """
    git = policy.resolve_executable("git")
    if git is None:
        raise RefusalError("git is required to resolve refs (it was not found on PATH)")
    try:
        proc = await asyncio.create_subprocess_exec(
            git,
            "-C",
            str(repo_root),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={**policy.scrubbed_environ(), **policy.GIT_ENV},
        )
    except OSError as exc:
        raise RefusalError(f"git is required to resolve refs ({exc.strerror or exc})") from exc
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, b""
    return (-1 if proc.returncode is None else proc.returncode), out


async def _rev_parse(repo_root: Path, revision: str) -> str | None:
    """Return the commit ``revision`` names in ``repo_root``, or ``None``."""
    code, out = await _git(
        repo_root, "rev-parse", "--verify", "--quiet", "--end-of-options", f"{revision}^{{commit}}"
    )
    sha = out.decode("ascii", "replace").strip()
    return sha if code == 0 and _SHA_RE.fullmatch(sha) else None


async def _tracked_paths(repo_root: Path, commit: str) -> frozenset[str]:
    """Return every file tracked at ``commit`` (what ``git show <rev>:<path>`` may name)."""
    code, out = await _git(repo_root, "ls-tree", "-r", "-z", "--name-only", commit)
    if code != 0:
        raise RefusalError(f"cannot list the files tracked at the base ref {commit}")
    return frozenset(name for name in out.decode("utf-8", "replace").split(chr(0)) if name)


async def _base_blob(repo_root: Path, commit: str, relpath: str) -> str:
    """Return ``relpath`` as it is at ``commit``, never from the head-controlled working tree."""
    code, out = await _git(repo_root, "cat-file", "blob", f"{commit}:{relpath}")
    if code != 0:
        raise RefusalError(f"the base ref {commit} has no {relpath}; refusing to read it elsewhere")
    return out.decode("utf-8", "replace")


async def _resolve_refs(repo_root: Path, base: str, head: str) -> Refs:
    base_sha = await _rev_parse(repo_root, base)
    if base_sha is None:
        raise RefusalError(f"--base {base!r} does not name a commit in {repo_root}")
    head_sha = await _rev_parse(repo_root, head)
    if head_sha is None:
        raise RefusalError(f"--head {head!r} does not name a commit in {repo_root}")
    if base_sha == head_sha:
        raise RefusalError("--base and --head are the same commit; there is nothing to review")
    worktree_sha = await _rev_parse(repo_root, "HEAD") or "unknown"
    return Refs(base, base_sha, head, head_sha, worktree_sha)


@dataclass(frozen=True)
class BaseMaterial:
    """What the run takes from the TRUSTED base ref instead of from the head.

    The reviewer's own configuration is on this list: ``.claude/workforce.yaml`` selects the
    agent file, the MCP declaration, the secret-scan program and its policy file, the caps and
    where the review and the ledger row are written, so a working-tree copy would let a hostile
    head redirect every one of them.
    """

    refs: Refs
    cfg: WorkforceConfig
    config_source: str
    paths: frozenset[str]
    mcp_json: str
    mcp_config: str
    agent_text: str
    gitleaks_toml: str | None
    ledger_floor: str | None


async def _base_blob_if_tracked(
    repo_root: Path, refs: Refs, paths: frozenset[str], relpath: str
) -> str | None:
    """Return ``relpath`` as it is at the base, or ``None`` when the base does not track it."""
    if relpath not in paths:
        return None
    return await _base_blob(repo_root, refs.base_sha, relpath)


async def _load_base_material(repo_root: Path, base: str, head: str) -> BaseMaterial:
    refs = await _resolve_refs(repo_root, base, head)
    paths = await _tracked_paths(repo_root, refs.base_sha)
    config_source = f"{refs.base_sha}:{DEFAULT_CONFIG_RELPATH}"
    config_text = await _base_blob_if_tracked(repo_root, refs, paths, DEFAULT_CONFIG_RELPATH)
    # A base without the file gets the schema defaults, as load_config gives a repository that
    # has not adopted it - and an empty credit block cannot spend.
    cfg = (
        WorkforceConfig()
        if config_text is None
        else parse_config_text(config_text, source=config_source)
    )
    mcp_json = await _base_blob(repo_root, refs.base_sha, cfg.agent_reviewer.mcp_config)
    agent_text = await _base_blob(repo_root, refs.base_sha, cfg.agent_reviewer.agent_file)
    mcp_config = await _base_blob(repo_root, refs.base_sha, policy.MCP_CONFIG_RELPATH)
    gitleaks_toml = await _base_blob_if_tracked(repo_root, refs, paths, cfg.secret_scan.config_file)
    ledger_floor = await _base_blob_if_tracked(repo_root, refs, paths, cfg.credit.ledger)
    return BaseMaterial(
        refs=refs,
        cfg=cfg,
        config_source=config_source,
        paths=paths,
        mcp_json=mcp_json,
        mcp_config=mcp_config,
        agent_text=agent_text,
        gitleaks_toml=gitleaks_toml,
        ledger_floor=ledger_floor,
    )


def _mcp_package_origin() -> Path | None:
    """Return where this interpreter would import the MCP server's package from, or ``None``.

    The server is ``sys.executable -I -c <bootstrap>`` running ``mousedroid``
    (:data:`policy.MCP_BOOTSTRAP`), so what matters is where THIS interpreter finds
    ``mousedroid``, not where the repository keeps it. ``find_spec`` locates a top-level
    package without importing it.
    """
    try:
        spec = importlib.util.find_spec(policy.MCP_MODULE)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and os.path.isabs(spec.origin):
        return Path(spec.origin)
    locations = list(spec.submodule_search_locations or ())
    return Path(locations[0]) if locations else None


def _is_within(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or below it, once links and short names are resolved."""
    try:
        return policy.path_is_inside(os.path.realpath(path), os.path.realpath(root))
    except OSError:
        return False


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _controller_root() -> Path:
    """The checkout this reviewer's own code runs from (``tools/agent_reviewer/..``)."""
    return Path(__file__).resolve().parents[2]


async def _root_files(repo_root: Path, commit: str) -> list[str]:
    """Return the names of the files (not directories) at the root of ``commit``."""
    code, out = await _git(repo_root, "ls-tree", "-z", commit)
    if code != 0:
        raise RefusalError(f"cannot list the files at the root of {commit}")
    names: list[str] = []
    for entry in out.decode("utf-8", "replace").split(chr(0)):
        meta, _, name = entry.partition("\t")
        if name and meta.split(" ")[1:2] == ["blob"]:
            names.append(name)
    return names


async def _refuse_planted_programs(repo_root: Path, refs: Refs) -> None:
    """Refuse a head (or a checkout) whose root holds a file named like a program we start."""
    for commit in dict.fromkeys((refs.head_sha, refs.worktree_sha)):
        if not _SHA_RE.fullmatch(commit):
            continue
        reason = policy.planted_program_reason(await _root_files(repo_root, commit))
        if reason:
            raise RefusalError(reason)


async def _tracked_symlinks(repo_root: Path, commit: str) -> list[tuple[str, str]]:
    """Return ``(path, blob id)`` for every symbolic link tracked at ``commit``."""
    code, out = await _git(repo_root, "ls-tree", "-r", "-z", commit)
    if code != 0:
        raise RefusalError(f"cannot list the tracked files of {commit}")
    links: list[tuple[str, str]] = []
    for entry in out.decode("utf-8", "replace").split(chr(0)):
        meta, _, name = entry.partition("\t")
        fields = meta.split(" ")
        if name and len(fields) == 3 and fields[0] == _SYMLINK_MODE:
            links.append((name, fields[2]))
    return links


async def _staged_symlinks(repo_root: Path) -> list[tuple[str, str]]:
    """Return ``(path, blob id)`` for every symbolic link the checkout's index holds.

    The index is what the next commit would hold and what a read can already find on disk, so a
    link that is staged and not committed (``git add docs/latest``) shows in no ``ls-tree``.
    ``git ls-files -s`` prints ``<mode> <blob> <stage>``, a tab and the path.
    """
    code, out = await _git(repo_root, "ls-files", "-s", "-z")
    if code != 0:
        raise RefusalError("cannot list the files the index of the checkout holds")
    links: list[tuple[str, str]] = []
    for entry in out.decode("utf-8", "replace").split(chr(0)):
        meta, _, name = entry.partition("\t")
        fields = meta.split(" ")
        if name and len(fields) == 3 and fields[0] == _SYMLINK_MODE:
            links.append((name, fields[1]))
    return links


async def _refuse_links_that_leave(
    repo_root: Path, links: list[tuple[str, str]], where: str
) -> None:
    """Refuse the run when one of ``links`` (path, blob id) held at ``where`` can leave."""
    if len(links) > _MAX_TRACKED_SYMLINKS:
        raise RefusalError(
            f"{where} tracks {len(links)} symbolic links (more than "
            f"{_MAX_TRACKED_SYMLINKS}); the reviewer will not follow that many"
        )
    for link, blob in links:
        code, out = await _git(repo_root, "cat-file", "blob", blob)
        if code != 0:
            raise RefusalError(f"cannot read the symbolic link {link!r} at {where}")
        target = out.decode("utf-8", "replace")
        reason = policy.symlink_escape_reason(link, target)
        if reason:
            raise RefusalError(
                f"{where} tracks the symbolic link {link!r} -> {target!r} ({reason}); "
                "Read, Grep and Glob would follow it out of the checkout under review"
            )


async def _refuse_escaping_symlinks(repo_root: Path, refs: Refs) -> None:
    """Refuse a head, checkout or index that tracks a symbolic link which can leave the checkout.

    Read, Grep and Glob resolve what they are given against the checkout. A link the head
    commits - ``docs/latest -> /etc`` - would be a way out that no pattern shows, so the whole
    run is refused before any spend (see :func:`policy.symlink_escape_reason`). The commits are
    read with ``ls-tree``; the index is read too, because a link that is staged and not yet
    committed is already on disk. A link that is only untracked is not seen (README, Known
    limits).
    """
    for commit in dict.fromkeys((refs.head_sha, refs.worktree_sha)):
        if not _SHA_RE.fullmatch(commit):
            continue
        links = await _tracked_symlinks(repo_root, commit)
        await _refuse_links_that_leave(repo_root, links, commit[:12])
    await _refuse_links_that_leave(repo_root, await _staged_symlinks(repo_root), "the index")


async def _tracked_under(repo_root: Path, refs: Refs, directory: Path) -> list[str]:
    """Return the files the head (or the checkout's index) tracks under ``directory``."""
    where = _relative(directory, repo_root)
    names: set[str] = set()
    for args in (
        ("ls-tree", "-r", "-z", "--name-only", refs.head_sha, "--", where),
        ("ls-files", "-z", "--", where),
    ):
        code, out = await _git(repo_root, *args)
        if code != 0:
            raise RefusalError(f"cannot tell whether the head tracks files under {where}")
        names.update(name for name in out.decode("utf-8", "replace").split(chr(0)) if name)
    return sorted(names)


async def _evaluate_trust(
    repo_root: Path, refs: Refs, *, trust_checkout: bool, ledger_path: Path
) -> CodeTrust:
    """Say where the code that runs beside the API key comes from; refuse the branch's own.

    Three pieces of code run in, or beside, the process that holds the key: the reviewer's
    own modules (this package), the interpreter's environment (its ``site-packages`` and
    ``.pth`` files run at startup, even under ``-I``) and the ``mousedroid`` package the MCP
    server imports. Each is the head's whenever it lives inside the checkout under review and
    the checkout is not at the base commit: the head controls every tracked file, and
    ``mousedroid`` is the checkout's own source under an editable install. A checkout that is
    at the base commit holds trusted code, and so does anything outside the checkout.

    The spend ledger is a fourth piece of trust: a ledger file inside a checkout that is not
    at the base is a file the head can rewrite, so ``--ledger-root`` pointing there is
    refused the same way (the base's committed copy still floors the total, but the operator
    has to say they accept the rest).

    An interpreter environment inside the checkout in which the head TRACKS files (a
    force-added ``.venv/...``) is refused whatever the operator says: it is demonstrably not
    the operator's.

    Args:
        repo_root: The resolved checkout the review runs in.
        refs: The resolved commits; ``worktree_sha`` is what the checkout holds.
        trust_checkout: The operator accepted running code from inside the checkout.
        ledger_path: Where the spend ledger file is (or would be created).

    Returns:
        The four lines the plan prints.

    Raises:
        RefusalError: Any piece is the branch's own and ``trust_checkout`` is false, or the
            head tracks files under the interpreter's environment.
    """
    at_base = refs.worktree_sha == refs.base_sha
    problems: list[str] = []

    controller = Path(__file__).resolve().parent
    if not _is_within(controller, repo_root):
        controller_note = "outside the checkout under review"
    else:
        where = _relative(controller, repo_root)
        if at_base:
            controller_note = f"{where}, in the checkout, which is at the base commit"
        else:
            controller_note = f"IN THE CHECKOUT UNDER REVIEW ({where})"
            problems.append(f"the reviewer's own modules ({where}) are in this checkout")

    prefix = Path(sys.prefix)
    if not _is_within(prefix, repo_root):
        interpreter_note = "outside the checkout under review"
    else:
        where = _relative(prefix, repo_root)
        if at_base:
            interpreter_note = f"{where}, in the checkout, which is at the base commit"
        else:
            tracked = await _tracked_under(repo_root, refs, prefix)
            if tracked:
                raise RefusalError(
                    f"the head tracks {len(tracked)} file(s) under the interpreter's own "
                    f"environment {where} (for example {tracked[0]}): that environment belongs "
                    f"to the head, not to you, and {TRUST_CHECKOUT_FLAG} does not accept it. "
                    "Create the virtualenv outside the checkout."
                )
            interpreter_note = f"IN THE CHECKOUT UNDER REVIEW ({where})"
            problems.append(f"the interpreter's environment ({where}) is inside this checkout")

    origin = _mcp_package_origin()
    if origin is None:
        package_note = (
            f"{policy.MCP_MODULE} is not importable by this interpreter "
            "(the MCP server will not start; the review runs without it)"
        )
    elif _is_within(origin, prefix):
        package_note = "installed in the interpreter's own environment"
    elif not _is_within(origin, repo_root):
        package_note = "installed outside the checkout under review"
    else:
        where = _relative(origin, repo_root)
        if at_base:
            package_note = f"{where}, in the checkout, which is at the base commit"
        else:
            package_note = f"IN THE CHECKOUT UNDER REVIEW ({where})"
            problems.append(
                f"the MCP server would import `{policy.MCP_MODULE}` from {where}, "
                "inside this checkout"
            )

    code_problems = len(problems)
    if not _is_within(ledger_path, repo_root):
        ledger_note = "outside the checkout under review"
    else:
        where = _relative(ledger_path, repo_root)
        if at_base:
            ledger_note = f"{where}, in the checkout, which is at the base commit"
        else:
            ledger_note = f"IN THE CHECKOUT UNDER REVIEW ({where})"
            problems.append(
                f"the spend ledger ({where}) is inside this checkout, which the head controls"
            )

    notes = [controller_note, interpreter_note, package_note, ledger_note]
    if problems:
        if not trust_checkout:
            lead = (
                "code inside this checkout would run beside the CLI that holds the API key"
                if code_problems
                else "the spend ledger would live inside a checkout the head controls"
            )
            raise RefusalError(
                f"{lead}, and the checkout is at {refs.worktree_sha[:12]}, not at the base "
                f"{refs.base_sha[:12]}: {'; '.join(problems)}. Run the reviewer from a trusted "
                "checkout and point --repo-root at the one under review (virtualenv outside it, "
                f"`{policy.MCP_MODULE}` installed from the base ref without -e, --ledger-root "
                f"outside it), or pass {TRUST_CHECKOUT_FLAG} if you would run this branch's code "
                "and trust its files anyway."
            )
        notes = [
            f"{note}; accepted with {TRUST_CHECKOUT_FLAG}" if note.startswith("IN THE") else note
            for note in notes
        ]
    return CodeTrust(controller=notes[0], interpreter=notes[1], package=notes[2], ledger=notes[3])


def next_run_id(artifact_root: Path, *, prefix: str = _DEFAULT_RUN_PREFIX) -> str:
    """Return the next ``<prefix>-NNNN`` after the artifacts already under ``artifact_root``."""
    pattern = re.compile(rf"{re.escape(prefix)}-(\d+)\.md")
    highest = 0
    if artifact_root.is_dir():
        for path in artifact_root.glob(f"*/{prefix}-*.md"):
            match = pattern.fullmatch(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"{prefix}-{highest + 1:04d}"


def _artifact_path(
    out: Path | None, repo_root: Path, cfg: WorkforceConfig, run_id: str, now: datetime
) -> Path:
    if out is not None:
        candidate = out if out.is_absolute() else repo_root / out
    else:
        directory = repo_root / cfg.agent_reviewer.artifact_dir / now.strftime(_DATE_FORMAT)
        candidate = directory / f"{run_id}.md"
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise RefusalError(f"the artifact path {candidate} is outside the repository") from exc
    if resolved.exists():
        shown = resolved.relative_to(repo_root).as_posix()
        raise RefusalError(f"the artifact {shown} already exists; not overwriting it")
    return resolved


def _mcp_config_path(text: str, work_dir: Path | None, repo_root: Path) -> Path:
    """Return the file the MCP server reads: the base ref's copy, outside the checkout.

    A dry run writes nothing, so it only names the file a live run would write.

    Raises:
        RefusalError: The scratch directory is inside the checkout.
    """
    if work_dir is None:
        return Path(tempfile.gettempdir()) / _DRY_RUN_SCRATCH / _MCP_CONFIG_FILENAME
    if _is_within(work_dir, repo_root):
        raise RefusalError(
            f"the scratch directory {work_dir} is inside the checkout; point TMPDIR (TEMP on "
            "Windows) outside it"
        )
    path = work_dir / _MCP_CONFIG_FILENAME
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


async def _plan(args: argparse.Namespace, work_dir: Path | None) -> Plan:
    repo_root = (resolve_repo_root() if args.repo_root is None else args.repo_root).resolve()
    material = await _load_base_material(repo_root, args.base, args.head)
    cfg = material.cfg
    reviewer = cfg.agent_reviewer
    refs = material.refs
    ledger_root = (args.ledger_root or _controller_root()).resolve()
    trust = await _evaluate_trust(
        repo_root,
        refs,
        trust_checkout=args.trust_checkout,
        ledger_path=ledger_root / cfg.credit.ledger,
    )
    await _refuse_planted_programs(repo_root, refs)
    await _refuse_escaping_symlinks(repo_root, refs)
    system_prompt = prompt.build(
        repo_root=repo_root,
        cfg=cfg,
        agent_text=material.agent_text,
        gitleaks_toml=material.gitleaks_toml,
    )
    user_prompt = prompt.build_user_prompt(
        base_ref=refs.base_ref,
        base_sha=refs.base_sha,
        head_ref=refs.head_ref,
        head_sha=refs.head_sha,
        worktree_sha=refs.worktree_sha,
        opening=args.opening,
    )
    size = len(system_prompt) + len(user_prompt)
    if size > reviewer.max_prompt_chars:
        raise RefusalError(
            f"the prompts are {size} characters, above agent_reviewer.max_prompt_chars "
            f"({reviewer.max_prompt_chars}) in the base's workforce.yaml; nothing was sent"
        )
    options = policy.reviewer_options(
        cfg=reviewer,
        repo_root=repo_root,
        model=reviewer.model,
        system_prompt=system_prompt,
        cap_usd=args.cap_usd,
        mcp_json=material.mcp_json,
        mcp_config_path=_mcp_config_path(material.mcp_config, work_dir, repo_root),
        mcp_source=f"{refs.base_sha}:{reviewer.mcp_config}",
    )
    run_id = next_run_id(repo_root / reviewer.artifact_dir, prefix=f"ws{reviewer.ws.lower()}")
    artifact = _artifact_path(args.out, repo_root, cfg, run_id, datetime.now(timezone.utc))
    return Plan(
        repo_root=repo_root,
        cfg=cfg,
        refs=refs,
        opening=tuple(args.opening),
        cap_usd=args.cap_usd,
        run_id=run_id,
        artifact=artifact,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        options=options,
        base_paths=material.paths,
        config_source=material.config_source,
        gitleaks_toml=material.gitleaks_toml,
        trust=trust,
        ledger_root=ledger_root,
        ledger_floor=material.ledger_floor,
    )


def _build_plan(args: argparse.Namespace, work_dir: Path | None = None) -> Plan:
    """Build the whole plan, or raise a refusal. ``work_dir`` is None for a dry run."""
    return asyncio.run(_plan(args, work_dir))


# --------------------------------------------------------------------------- #
# The SDK seam (the only place claude_agent_sdk is touched)
# --------------------------------------------------------------------------- #
def _import_sdk() -> Any | None:
    """Return the ``claude_agent_sdk`` module, or ``None`` if the extra is not installed."""
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    return claude_agent_sdk


def _sdk_version(sdk: Any) -> str:
    return str(getattr(sdk, "__version__", "unknown"))


def _sdk_options(sdk: Any, kwargs: Mapping[str, Any], plan: Plan) -> Any:
    """Build ``ClaudeAgentOptions`` from the policy kwargs plus the two PreToolUse hooks.

    The Bash guard (git only, pinned revisions) and the read guard (``Read|Grep|Glob``
    confined to the checkout) are separate matchers; both fail closed.

    Raises:
        SdkRejectedError: The installed SDK does not know one of the option names.
    """
    matchers = [
        sdk.HookMatcher(matcher=policy.BASH_HOOK_MATCHER, hooks=[plan.guard]),
        sdk.HookMatcher(matcher=policy.READ_HOOK_MATCHER, hooks=[plan.read_guard]),
    ]
    try:
        return sdk.ClaudeAgentOptions(**kwargs, hooks={"PreToolUse": matchers})
    except TypeError as exc:
        raise SdkRejectedError(
            f"claude-agent-sdk {_sdk_version(sdk)} rejected the reviewer options: {exc}"
        ) from exc


async def _aclose(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    try:
        outcome = close()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:  # a failed close must not mask the lookup result
        _log.debug("agent_reviewer_client_close_failed")


async def resolve_model(alias: str, *, timeout_s: float) -> tuple[str, str]:
    """Resolve the model ``alias`` to an id with ``client.models.retrieve``.

    The client is built without arguments, so the Anthropic SDK reads the key from the
    environment itself; nothing here ever holds it.

    Returns:
        ``(model, resolved_via)``: the id and ``"models.retrieve"``, or - when the
        ``anthropic`` package is missing, the lookup fails or times out, or the answer
        has no usable id - the configured alias and ``"config"``. The review never fails
        on the lookup: the bundled CLI resolves an alias such as ``opus`` itself.
    """
    try:
        import anthropic
    except ImportError:
        return alias, RESOLVED_VIA_CONFIG
    client: Any = None
    try:
        client = anthropic.AsyncAnthropic()
        info = await asyncio.wait_for(client.models.retrieve(alias), timeout=timeout_s)
    except Exception as exc:
        _log.warning(
            "agent_reviewer_model_resolve_failed", alias=alias, error_type=type(exc).__name__
        )
        return alias, RESOLVED_VIA_CONFIG
    finally:
        if client is not None:
            await _aclose(client)
    resolved = getattr(info, "id", None)
    # The id becomes `--model <id>` on the CLI's command line: use it only if it is as
    # plain as a configured alias.
    if isinstance(resolved, str) and _MODEL_ID_RE.fullmatch(resolved.strip()):
        return resolved.strip(), RESOLVED_VIA_RETRIEVE
    return alias, RESOLVED_VIA_CONFIG


@dataclass
class _RunState:
    """What the SDK run produced so far; filled in as messages arrive."""

    model: str
    resolved_via: str = RESOLVED_VIA_CONFIG
    sdk_version: str = "unknown"
    started: bool = False
    result: Any = None
    texts: list[str] = field(default_factory=list)
    served: list[str] = field(default_factory=list)
    error: BaseException | None = None

    def observe(self, sdk: Any, message: Any) -> None:
        """Record one SDK message."""
        self.started = True
        if isinstance(message, sdk.ResultMessage):
            self.result = message
        elif isinstance(message, sdk.AssistantMessage):
            text = "".join(
                block.text for block in message.content if isinstance(block, sdk.TextBlock)
            )
            if text.strip():
                self.texts.append(text)
            served = getattr(message, "model", None)
            if isinstance(served, str) and served not in self.served:
                self.served.append(served)

    def served_models(self) -> list[str]:
        """The model ids the API actually served, as reported by the SDK."""
        models = list(self.served)
        usage = getattr(self.result, "model_usage", None)
        if isinstance(usage, Mapping):
            models += [name for name in usage if isinstance(name, str) and name not in models]
        return models

    def review_text(self) -> str:
        """The review: the final result text, else whatever the assistant said before it ended."""
        final = getattr(self.result, "result", None)
        if isinstance(final, str) and final.strip():
            return final
        return "\n\n".join(self.texts)


async def _consume(sdk: Any, options: Any, user_prompt: str, state: _RunState) -> None:
    # aclosing: when the run is cancelled (timeout, Ctrl-C) the generator is closed here,
    # which is what makes the SDK stop its CLI subprocess before the books are settled.
    async with aclosing(sdk.query(prompt=user_prompt, options=options)) as stream:
        async for message in stream:
            state.observe(sdk, message)


async def _drive(plan: Plan, sdk: Any, state: _RunState) -> None:
    """Resolve the model and run the review, leaving the outcome in ``state``.

    Never raises for an ordinary failure: the caller settles the books from ``state``
    whatever happened. A cancellation is recorded and re-raised.
    """
    reviewer = plan.cfg.agent_reviewer
    try:
        state.model, state.resolved_via = await resolve_model(
            reviewer.model, timeout_s=reviewer.resolve_timeout_s
        )
        _log.info("agent_reviewer_model_resolved", via=state.resolved_via)
        options = _sdk_options(sdk, {**plan.options, "model": state.model}, plan)
        await asyncio.wait_for(
            _consume(sdk, options, plan.user_prompt, state), timeout=reviewer.run_timeout_s
        )
    except asyncio.CancelledError as exc:
        state.error = exc
        raise
    except Exception as exc:  # SDK errors, ResultError after an error result, a timeout
        state.error = exc


# --------------------------------------------------------------------------- #
# Settling the books - always runs after a spend
# --------------------------------------------------------------------------- #
def _output_refusal(text: str, plan: Plan) -> str | None:
    """Return why the review text must not be saved, or ``None``."""
    reason = policy.credential_shape_reason(text)
    if reason:
        return reason
    try:
        prompt.scan_with_gitleaks(
            text, cfg=plan.cfg, label="the review output", gitleaks_toml=plan.gitleaks_toml
        )
    except prompt.PromptRefusedError as exc:
        return str(exc)
    return None


def _render_artifact(
    plan: Plan, state: _RunState, accounting: ledger.RunAccounting, body: str
) -> str:
    reviewer = plan.cfg.agent_reviewer
    turns = getattr(state.result, "num_turns", None)
    served = ", ".join(f"`{name}`" for name in state.served_models()) or "unknown"
    lines = [
        f"# Agent-SDK peer review {plan.run_id}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| run_id | {plan.run_id} |",
        f"| status | {accounting.status} |",
        f"| workstream | {reviewer.ws} |",
        f"| base | `{plan.refs.base_ref}` @ `{plan.refs.base_sha}` |",
        f"| head | `{plan.refs.head_ref}` @ `{plan.refs.head_sha}` |",
        f"| working tree | `{plan.refs.worktree_sha}` |",
        f"| opening items | {', '.join(plan.opening) or 'none'} |",
        f"| model requested | `{reviewer.model}` |",
        f"| model resolved | `{state.model}` ({state.resolved_via}) |",
        f"| models served | {served} |",
        f"| turns | {turns if isinstance(turns, int) else 'unknown'} (max {reviewer.max_turns}) |",
        f"| cost | ${accounting.cost_usd:.4f} ({accounting.cost_source}) |",
        f"| run cap | ${plan.cap_usd:.2f} |",
        f"| generated | {datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)} |",
        f"| tool | tools.agent_reviewer, claude-agent-sdk {state.sdk_version} |",
        "",
        "---",
        "",
        body.replace("\r\n", "\n").strip(),
        "",
    ]
    return "\n".join(lines)


def _write_artifact(plan: Plan, text: str) -> str:
    """Write the artifact (never over an existing file); return its ledger path value."""
    try:
        plan.artifact.parent.mkdir(parents=True, exist_ok=True)
        with plan.artifact.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except OSError as exc:
        print(
            f"WARNING: the artifact could not be written ({exc.strerror or exc}); the review "
            "text is lost, the ledger row is still recorded.",
            file=sys.stderr,
        )
        return ARTIFACT_NOT_WRITTEN
    return plan.artifact_rel


def _record(plan: Plan, state: _RunState, accounting: ledger.RunAccounting, artifact: str) -> bool:
    try:
        row = ledger.record_run(
            plan.cfg,
            accounting=accounting,
            run_id=plan.run_id,
            target=plan.target,
            model_requested=plan.cfg.agent_reviewer.model,
            model_resolved=state.model,
            resolved_via=state.resolved_via,
            artifact=artifact,
            ledger_root=plan.ledger_root,
            floor_text=plan.ledger_floor,
            floor_source=plan.ledger_floor_source,
        )
    except ledger.LedgerWriteError as exc:
        print(
            "LEDGER ROW NOT WRITTEN - the money is spent. Fix the ledger, then append this row "
            "(plus ts, cap_usd, cumulative_ws_usd) by hand:",
            file=sys.stderr,
        )
        print(json.dumps(exc.entry, separators=(",", ":")), file=sys.stderr)
        print(f"  cause: {exc}", file=sys.stderr)
        return False
    _log.info(
        "agent_reviewer_recorded",
        run_id=plan.run_id,
        status=accounting.status,
        cost_usd=accounting.cost_usd,
        cost_source=accounting.cost_source,
        cumulative_ws_usd=row["cumulative_ws_usd"],
    )
    return True


def _settle(plan: Plan, state: _RunState, *, interrupted: bool) -> int:
    """Account for the run, write the artifact and the ledger row; return the exit code."""
    accounting = ledger.account_run(
        state.result, started=state.started, cap_usd=plan.cap_usd, error=state.error
    )
    body = state.review_text() or "*(the run ended before the reviewer produced any text)*"
    try:
        withheld = _output_refusal(body, plan)
    except Exception:  # fail closed: if the screen itself breaks, the text is not saved
        withheld = "the credential screen could not run"
    if withheld is not None:
        accounting = dataclasses.replace(accounting, status=ledger.STATUS_OUTPUT_WITHHELD)
        body = (
            f"*(review output withheld: it matched the credential screen - {withheld}. The run "
            "was charged; check what the reviewer could read before running it again.)*"
        )
        print(f"WARNING: review output withheld ({withheld}).", file=sys.stderr)
    try:
        artifact = _write_artifact(plan, _render_artifact(plan, state, accounting, body))
    except Exception as exc:  # the ledger row matters more than the file: never skip it
        print(
            f"WARNING: the artifact step failed ({type(exc).__name__}); recording the run anyway.",
            file=sys.stderr,
        )
        artifact = ARTIFACT_NOT_WRITTEN
    recorded = _record(plan, state, accounting, artifact)
    print(f"review: {artifact}")
    print(f"run_id: {plan.run_id}")
    print(f"status: {accounting.status}")
    print(f"cost_usd: {accounting.cost_usd} ({accounting.cost_source})")
    if interrupted:
        return EXIT_INTERRUPTED
    succeeded = (
        recorded and accounting.status == ledger.STATUS_OK and artifact != ARTIFACT_NOT_WRITTEN
    )
    return EXIT_OK if succeeded else EXIT_FAILED


# --------------------------------------------------------------------------- #
# The two modes
# --------------------------------------------------------------------------- #
def _print_plan(plan: Plan, headroom: credit_ledger.Headroom, sdk_line: str) -> None:
    reviewer = plan.cfg.agent_reviewer
    options = plan.options
    server = options["mcp_servers"][policy.MCP_SERVER]
    key_state = "set" if os.environ.get(API_KEY_ENV, "").strip() else "not set"
    print("agent_reviewer dry-run: no client constructed, no model lookup, no network, no spend")
    print(f"ws: {reviewer.ws}")
    print(f"run_id: {plan.run_id}")
    print(f"artifact: {plan.artifact_rel} (would be written)")
    print(f"base: {plan.refs.base_ref} = {plan.refs.base_sha}")
    print(f"head: {plan.refs.head_ref} = {plan.refs.head_sha}")
    print(f"working_tree: {plan.refs.worktree_sha}")
    print(f"workforce_config: {plan.config_source} (the working tree's copy is never read)")
    print(f"base_tracked_files: {len(plan.base_paths)} (what `git show <rev>:<path>` may name)")
    print(f"opening: {', '.join(plan.opening) or '(none)'}")
    print(f"model_requested: {reviewer.model}")
    print(f"model_resolved: {options['model']}")
    print(f"resolved_via: {RESOLVED_VIA_CONFIG} (a real run asks models.retrieve first)")
    print(f"max_turns: {options['max_turns']}")
    print(f"cap_usd: {plan.cap_usd}")
    print(
        f"ledger: {plan.ledger_root / plan.cfg.credit.ledger} "
        "(the controller's checkout; the cap is per workstream, not per head worktree)"
    )
    floor_note = (
        f"{plan.ledger_floor_source} (rows the base already holds count even if the file lost them)"
        if plan.ledger_floor is not None
        else "(the base ref does not track a ledger)"
    )
    print(f"ledger_floor: {floor_note}")
    print(f"credit_cap_usd: {headroom.cap_usd}")
    print(f"credit_spent_usd: {headroom.spent_usd}")
    print(f"credit_remaining_after_this_run_usd: {headroom.remaining_usd}")
    print(f"permission_mode: {options['permission_mode']}")
    print(f"tools: {', '.join(options['tools'])}")
    for title, rules in (
        ("allowed_tools", options["allowed_tools"]),
        ("disallowed_tools", options["disallowed_tools"]),
    ):
        print(f"{title}:")
        for rule in rules:
            print(f"  - {rule}")
    print(f"mcp_servers: {policy.MCP_SERVER} ({server['command']} {' '.join(server['args'])})")
    print(
        f"mcp_config: materialised from {plan.refs.base_sha}:{policy.MCP_CONFIG_RELPATH} into a "
        "scratch directory outside the checkout; the server runs isolated (-I) from there"
    )
    print(f"trust_controller: {plan.trust.controller}")
    print(f"trust_interpreter: {plan.trust.interpreter}")
    print(f"trust_package: {plan.trust.package}")
    print(f"trust_ledger: {plan.trust.ledger}")
    print(
        f"mcp_json: {plan.refs.base_sha}:{reviewer.mcp_config} confirms the pinned invocation "
        "(built from constants; the working tree's copy is never read)"
    )
    print("mcp_env:")
    for name, value in server["env"].items():
        print(f"  {name}={value}")
    print("process_env:")
    for name, value in options["env"].items():
        shown = "(from the parent)" if name in policy.PROCESS_ENV_PASSTHROUGH else value
        print(f"  {name}={shown}")
    print(f"read_guard: Read, Grep and Glob confined to {plan.repo_root}")
    extensions = " ".join(reviewer.glob_extensions) or "(none: no wildcard file part)"
    print(f"glob_extensions: {extensions} (the base's agent_reviewer.glob_extensions)")
    print(f"system_prompt_chars: {len(plan.system_prompt)}")
    print(f"user_prompt_chars: {len(plan.user_prompt)}")
    print(f"max_prompt_chars: {reviewer.max_prompt_chars} (both prompts together)")
    print(f"sdk: {sdk_line}")
    print(f"{API_KEY_ENV}: {key_state}")


def _precheck(plan: Plan) -> credit_ledger.Headroom:
    """Authorise the run against the controller's ledger, floored by the base's copy."""
    return ledger.precheck(
        plan.cfg,
        plan.cap_usd,
        ledger_root=plan.ledger_root,
        floor_text=plan.ledger_floor,
        floor_source=plan.ledger_floor_source,
    )


def _dry_run(plan: Plan) -> int:
    try:
        headroom = _precheck(plan)
        sdk = _import_sdk()
        if sdk is None:
            sdk_line = (
                "not installed (options not validated against it; install the agent_reviewer extra)"
            )
        else:
            _sdk_options(sdk, plan.options, plan)
            sdk_line = f"claude-agent-sdk {_sdk_version(sdk)} (options accepted)"
    except credit_ledger.CapExceeded as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_CAP
    except (SdkRejectedError, *_REFUSALS) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    _print_plan(plan, headroom, sdk_line)
    return EXIT_OK


def _live(plan: Plan) -> int:
    if not os.environ.get(API_KEY_ENV, "").strip():
        print(
            f"refused: {API_KEY_ENV} is not set. Export it in the environment (it is never "
            "taken from the command line) after rotating the key; see NEXT_STEPS.md item 1.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    sdk = _import_sdk()
    if sdk is None:
        print(
            "refused: claude-agent-sdk is not installed. Install the extra: "
            'pip install -e ".[agent_reviewer,anthropic]"',
            file=sys.stderr,
        )
        return EXIT_REFUSED
    try:
        _sdk_options(sdk, plan.options, plan)  # the names are real, before any spend
        headroom = _precheck(plan)
    except credit_ledger.CapExceeded as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_CAP
    except (SdkRejectedError, *_REFUSALS) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    _log.info(
        "agent_reviewer_charged",
        ws=plan.cfg.agent_reviewer.ws,
        run_id=plan.run_id,
        cap_usd=headroom.cap_usd,
        spent_usd=headroom.spent_usd,
        run_cap_usd=plan.cap_usd,
    )
    state = _RunState(model=plan.cfg.agent_reviewer.model, sdk_version=_sdk_version(sdk))
    interrupted = False
    try:
        asyncio.run(_drive(plan, sdk, state))
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        if state.error is None:
            state.error = KeyboardInterrupt()
    return _settle(plan, state, interrupted=interrupted)


def _execute(args: argparse.Namespace, work_dir: Path | None) -> int:
    try:
        plan = _build_plan(args, work_dir)
    except _REFUSALS as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return _dry_run(plan) if args.dry_run else _live(plan)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return _execute(args, None)
    # The MCP server's configuration lives in a scratch directory outside the checkout for the
    # length of the run. The CLI may outlive the run on Windows and keep it as a working
    # directory, so a failed cleanup is not an error.
    scratch = tempfile.TemporaryDirectory(prefix="agent-reviewer-", ignore_cleanup_errors=True)
    with scratch as directory:
        return _execute(args, Path(directory).resolve())


if __name__ == "__main__":
    sys.exit(main())
