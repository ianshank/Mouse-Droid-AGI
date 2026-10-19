"""The reviewer's prompts, and the refusal to put a secret into any prompt.

``build()`` assembles the system prompt: the body of ``.claude/agents/peer-reviewer.md``
(its YAML front-matter stripped), a short statement of the run's constraints, the
rule for when an MCP diagnostic counts as evidence, and the house output shape.
``build_user_prompt()`` is the one-message task: which commits to review.

Everything that is read from the repository on the way in goes through
:func:`read_includable`, which REFUSES - raises :class:`PromptRefusedError`, never
quietly skips - a file when:

* its name matches ``*.env``, ``docker.env*`` (or the dotenv variants):
  checked on the NAME before the file is opened;
* its text contains something shaped like a credential (an Anthropic key, a Hugging
  Face token, an AWS key id, a GitHub token, a private-key block); or
* its text matches a regex declared in ``.gitleaks.toml`` (read at run time, so the
  repo's own secret policy is the single source - the placeholders it allowlists are
  refused here too: a cloud prompt has no use for a fake key);

and the assembled prompt is then scanned by the workforce's own ``gitleaks`` wrapper
(the default ruleset ``.gitleaks.toml`` extends), so a pattern the Python screens do
not know still has a chance to stop it. When ``gitleaks`` is not installed the scan
is skipped with a warning, unless ``secret_scan.strict`` is true.

A refusal never repeats the text that triggered it.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from tools.agent_reviewer import policy
from tools.claude_hooks import secret_scan
from tools.claude_hooks.config import WorkforceConfig
from tools.claude_hooks.logging_setup import get_logger
from tools.claude_hooks.secret_scan import ScanOutcome

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 CI leg
    import tomli as tomllib

_log = get_logger(__name__)

#: A scanner takes the prompt text and reports whether it is clean (tests inject one).
Scanner = Callable[[str], ScanOutcome]

_FRONT_MATTER_FENCE: Final = "---"
_SCAN_TARGET: Final = "reviewer-prompt.md"
_DETAIL_CHARS: Final = 300

OPERATING_CONSTRAINTS: Final = """\
## How this run works

This is a non-interactive run: nobody can answer a question, and your final message is
saved verbatim as the review. If the context you need is missing, say what is missing and
end with NEEDS_CONTEXT instead of guessing.

You can read files and run `git diff`, `git log` and `git show`; you cannot write, edit, run
tests or any other program, or reach the network. Pathspecs go after `--` and are literal (a
glob matches nothing), `--all` and the other whole-history options are refused, and
`git show <rev>:<path>` works only for a file that exists at the base - read a file the change
adds from `git diff <base>...<head> -- <path>`. A refused tool call is the harness doing its
job, not a defect in the code under review: do not report it, and do not retry it in another
form."""

DIAGNOSTICS_EVIDENCE_RULE: Final = """\
## When an MCP diagnostic is evidence

The `mousedroid` MCP server on this machine runs against mock hardware. A diagnostics
result (`system_info`, `lidar_diagnostics`, `esp32_diagnostics`) is evidence about the
rover only when its `source` is `live_driver` or `probe` AND its status is not `mock` or
`unconfigured`. A result with no `source` field - which is all the `esp32_diagnostics` stub
returns - is not evidence: never cite it for a claim about the hardware."""

HOUSE_OUTPUT_SHAPE: Final = """\
## House output shape

Write the review as Markdown with exactly these sections, in this order, and nothing
before the first heading or after the last line.

### Verdict table
One row per claim you checked: `| Claim | Verdict | Evidence |`. Verdict is CONFIRMED,
REFUTED or CLEAN. Evidence names what you read - a symbol (`module.py::NAME`), a command and
its result, or a test.

### Defects (severity-ordered)
`| ID | Sev | Finding |`, most severe first (Critical, High, Med, Low), leading with the
finding the author least wants to hear. Every finding carries one of [Certain] (verified
against the tree), [Likely] (strong inference) or [Guessing] (gap-filling), and cites by
symbol, never by line number.

### Survives
What is correct and stays unchanged, so the review is calibrated rather than a list of faults.

### Load-bearing pins
A numbered list of the tests, gates and invariants that must keep holding for the
conclusions above to stay true.

### Verdict:
Exactly one of APPROVE, REQUEST_CHANGES or NEEDS_CONTEXT, then one sentence of reason."""


class PromptRefusedError(ValueError):
    """Raised when a prompt (or a file going into one) must not be built."""


def strip_front_matter(text: str) -> str:
    """Return ``text`` without a leading ``---`` ... ``---`` YAML block.

    Text that does not open with a fence is returned unchanged.

    Raises:
        PromptRefusedError: The text opens a front-matter block that never closes.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != _FRONT_MATTER_FENCE:
        return text
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONT_MATTER_FENCE:
            return "".join(lines[index + 1 :]).lstrip("\r\n")
    raise PromptRefusedError("the agent file opens a YAML front-matter block that never closes")


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _regex_sources(data: Mapping[str, Any]) -> list[str]:
    """Collect the regexes a gitleaks config declares (allowlists and custom rules)."""
    found: list[str] = []
    allowlist = data.get("allowlist")
    if isinstance(allowlist, dict):
        found += _strings(allowlist.get("regexes"))
    allowlists = data.get("allowlists")
    if isinstance(allowlists, list):
        for item in allowlists:
            if isinstance(item, dict):
                found += _strings(item.get("regexes"))
    rules = data.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict) and isinstance(rule.get("regex"), str):
                found.append(rule["regex"])
    return found


def gitleaks_regexes(repo_root: Path, cfg: WorkforceConfig) -> list[re.Pattern[str]]:
    """Compile the regexes declared in the repository's ``.gitleaks.toml``.

    A missing file is an empty list (the repo has no custom secret policy). A file
    that exists but cannot be read or compiled is a refusal: a prompt must not be
    built while the policy that guards it is unreadable.

    Raises:
        PromptRefusedError: The file is not valid TOML, or a regex in it does not compile.
    """
    relpath = cfg.secret_scan.config_file
    path = repo_root / relpath
    if not path.is_file():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PromptRefusedError(
            f"{relpath} cannot be read as TOML ({type(exc).__name__}); not building a prompt "
            "without the secret policy"
        ) from exc
    patterns: list[re.Pattern[str]] = []
    for source in _regex_sources(data):
        try:
            patterns.append(re.compile(source))
        except re.error as exc:
            raise PromptRefusedError(f"a regex in {relpath} does not compile ({exc})") from exc
    return patterns


def _screen_content(text: str, *, label: str, patterns: Sequence[re.Pattern[str]]) -> None:
    reason = policy.credential_shape_reason(text)
    if reason:
        raise PromptRefusedError(f"refusing to include {label}: {reason}")
    for pattern in patterns:
        if pattern.search(text):
            raise PromptRefusedError(
                f"refusing to include {label}: it matches a regex declared in .gitleaks.toml"
            )


def read_includable(path: str | Path, *, repo_root: Path, cfg: WorkforceConfig) -> str:
    """Read a repository file for inclusion in a prompt, or refuse.

    Args:
        path: Repo-relative (or absolute, inside the repo) path.
        repo_root: Repository root.
        cfg: Workforce config (names the ``.gitleaks.toml`` to read).

    Returns:
        The file's text.

    Raises:
        PromptRefusedError: The path is outside the repository, its name is a secret-file
            pattern, it cannot be read as UTF-8, or its text looks like a credential
            or matches a ``.gitleaks.toml`` regex.
    """
    root = repo_root.resolve()
    candidate = Path(path)
    full = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        rel = full.relative_to(root).as_posix()
    except ValueError as exc:
        raise PromptRefusedError(
            f"refusing to include {path}: it is outside the repository"
        ) from exc
    reason = policy.secret_path_reason(rel)
    if reason:
        raise PromptRefusedError(f"refusing to include {rel}: {reason}")
    try:
        text = full.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PromptRefusedError(f"cannot read {rel} as UTF-8 text ({type(exc).__name__})") from exc
    _screen_content(text, label=rel, patterns=gitleaks_regexes(repo_root, cfg))
    return text


def screen_included_text(text: str, *, label: str, repo_root: Path, cfg: WorkforceConfig) -> None:
    """Refuse ``text`` (supplied by the caller, not read from the working tree) as a prompt part.

    The same screens :func:`read_includable` applies to a file: the name ``label`` is checked
    against the secret-file patterns, the text against the credential shapes and the
    ``.gitleaks.toml`` regexes.

    Raises:
        PromptRefusedError: The name or the text is refused.
    """
    reason = policy.secret_path_reason(label)
    if reason:
        raise PromptRefusedError(f"refusing to include {label}: {reason}")
    _screen_content(text, label=label, patterns=gitleaks_regexes(repo_root, cfg))


def scan_with_gitleaks(
    text: str,
    *,
    repo_root: Path,
    cfg: WorkforceConfig,
    label: str,
    scanner: Scanner | None = None,
) -> None:
    """Run the workforce's ``gitleaks`` wrapper over ``text``; raise on a finding.

    Args:
        text: The text about to leave the machine (a prompt) or be committed (a review).
        repo_root: Repository root (locates ``.gitleaks.toml``).
        cfg: Workforce config (the scanner command and ``secret_scan.strict``).
        label: What is being scanned, for the message.
        scanner: Replaces the real scan (tests).

    Raises:
        PromptRefusedError: ``gitleaks`` reported a finding, or it is unavailable while
            ``secret_scan.strict`` is true. When it is unavailable and not strict the
            scan is skipped with a warning.
    """
    if scanner is not None:
        outcome = scanner(text)
    else:
        outcome = secret_scan.scan_content(text, cfg, repo_root=repo_root, target=_SCAN_TARGET)
    if outcome.available and not outcome.clean:
        raise PromptRefusedError(
            f"gitleaks reported a finding in {label}: {outcome.detail[:_DETAIL_CHARS]}"
        )
    if outcome.available:
        return
    if cfg.secret_scan.strict:
        raise PromptRefusedError(
            f"gitleaks is unavailable ({outcome.detail}) and secret_scan.strict is true; "
            f"refusing {label}"
        )
    _log.warning("agent_reviewer_gitleaks_unavailable", label=label, detail=outcome.detail)


def build(
    *,
    repo_root: Path,
    cfg: WorkforceConfig,
    extra_files: Sequence[str | Path] = (),
    agent_text: str | None = None,
    scanner: Scanner | None = None,
) -> str:
    """Assemble the reviewer's system prompt.

    Args:
        repo_root: Repository root.
        cfg: Workforce config (names the agent file).
        extra_files: Further repository files to append as context. Each goes through
            :func:`read_includable`.
        agent_text: The agent file's text. The CLI passes it as read from the TRUSTED base
            ref: the working tree belongs to the head under review, and a head that can
            rewrite the reviewer's own instructions can neutralise its review. When omitted
            the file is read from the working tree (callers that trust it).
        scanner: Replaces the ``gitleaks`` scan of the assembled prompt (tests).

    Returns:
        The prompt: agent body (front-matter stripped), run constraints, the MCP evidence
        rule, any context files, then the house output shape.

    Raises:
        PromptRefusedError: Any included file, or the assembled prompt, was refused.
    """
    if agent_text is None:
        agent = read_includable(cfg.agent_reviewer.agent_file, repo_root=repo_root, cfg=cfg)
    else:
        screen_included_text(
            agent_text, label=cfg.agent_reviewer.agent_file, repo_root=repo_root, cfg=cfg
        )
        agent = agent_text
    sections = [strip_front_matter(agent), OPERATING_CONSTRAINTS, DIAGNOSTICS_EVIDENCE_RULE]
    for extra in extra_files:
        text = read_includable(extra, repo_root=repo_root, cfg=cfg)
        sections.append(f"## Context file: {Path(extra).as_posix()}\n\n{text}")
    sections.append(HOUSE_OUTPUT_SHAPE)
    prompt = "\n\n".join(section.strip() for section in sections) + "\n"
    scan_with_gitleaks(
        prompt, repo_root=repo_root, cfg=cfg, label="the assembled prompt", scanner=scanner
    )
    return prompt


def build_user_prompt(
    *,
    base_ref: str,
    base_sha: str,
    head_ref: str,
    head_sha: str,
    worktree_sha: str,
    opening: Sequence[str],
) -> str:
    """Return the one-message task: review ``base...head``.

    Args:
        base_ref: The base as the operator named it.
        base_sha: The base commit.
        head_ref: The head as the operator named it.
        head_sha: The head commit.
        worktree_sha: The commit the working tree is at. When it is not the head, the
            reviewer is told not to trust ``Read`` for the head's files, and to use
            ``git show`` (for a file that exists at the base) or the diff instead.
        opening: Finding ids from earlier review rounds to report on (may be empty).
    """
    lines = [
        "Review the change set below as the adversarial peer reviewer.",
        "",
        f"- base: {base_ref} = {base_sha}",
        f"- head: {head_ref} = {head_sha}",
        f"- start with `git diff --stat {base_sha}...{head_sha}`, then read each changed "
        f"file's diff with `git diff {base_sha}...{head_sha} -- <path>` (a literal path).",
    ]
    if worktree_sha == head_sha:
        lines.append(
            "- the working tree is checked out at the head, so Read, Grep and Glob see the "
            "changed files."
        )
    else:
        lines.append(
            f"- the working tree is at {worktree_sha}, NOT the head: do not use Read for the "
            f"head's files. For a file that exists at the base use "
            f"`git show {head_sha}:<path>`; for a file the change adds, read it from the diff."
        )
    if opening:
        lines += [
            "",
            "Opening items - finding ids from earlier review rounds recorded in this repository "
            "(search `openspec/changes/**/peer-review.md`). Report each as closed or still "
            "open at the head:",
            f"  {', '.join(opening)}",
        ]
    lines += [
        "",
        "Verify claims against the tree, not against the change's own description. Your final "
        "message is the review, in the house output shape.",
    ]
    return "\n".join(lines) + "\n"
