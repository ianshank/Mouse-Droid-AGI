"""Run one read-only Agent-SDK peer review of ``--base...--head`` (F-059 WS-B / WS-D).

    python -m tools.agent_reviewer.run_review \\
        --base origin/<trunk> --head <branch> --cap-usd 10 [--opening R3-1,R3-2] \\
        [--out reports/agent_reviewer/<date>/wsb-NNNN.md] [--dry-run] [--repo-root DIR]

``--dry-run`` builds everything a real run would - prompts, policy, SDK options, the
artifact path, the ledger pre-check - and prints it, WITHOUT constructing a client,
resolving a model, calling the network or writing anything. The first real run happens
only after the operator has rotated the API key (``NEXT_STEPS.md`` item 1).

The key is read from ``ANTHROPIC_API_KEY`` in the environment and from nowhere else; there
is no option that takes one, and it is never printed, logged, put in the options or written
to the artifact or the ledger (the SDK's CLI inherits the environment itself).

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
import inspect
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from tools import credit_ledger
from tools.agent_reviewer import ledger, policy, prompt
from tools.claude_hooks.config import MODEL_ID_PATTERN, ConfigError, WorkforceConfig, load_config
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

_GIT_TIMEOUT_S: Final = 30.0
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

    @property
    def target(self) -> str:
        """The ledger ``target`` column."""
        return f"diff {self.refs.base_ref}...{self.refs.head_ref} @{self.refs.head_sha[:7]}"

    @property
    def artifact_rel(self) -> str:
        """The artifact path, repo-relative with forward slashes."""
        return self.artifact.relative_to(self.repo_root).as_posix()


async def _rev_parse(repo_root: Path, revision: str) -> str | None:
    """Return the commit ``revision`` names in ``repo_root``, or ``None``."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{revision}^{{commit}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RefusalError(f"git is required to resolve refs ({exc.strerror or exc})") from exc
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    sha = out.decode("ascii", "replace").strip()
    return sha if proc.returncode == 0 and _SHA_RE.fullmatch(sha) else None


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


def _build_plan(args: argparse.Namespace) -> Plan:
    repo_root = (resolve_repo_root() if args.repo_root is None else args.repo_root).resolve()
    cfg = load_config(repo_root=repo_root)
    reviewer = cfg.agent_reviewer
    refs = asyncio.run(_resolve_refs(repo_root, args.base, args.head))
    system_prompt = prompt.build(repo_root=repo_root, cfg=cfg)
    user_prompt = prompt.build_user_prompt(
        base_ref=refs.base_ref,
        base_sha=refs.base_sha,
        head_ref=refs.head_ref,
        head_sha=refs.head_sha,
        worktree_sha=refs.worktree_sha,
        opening=args.opening,
    )
    options = policy.reviewer_options(
        cfg=reviewer,
        repo_root=repo_root,
        model=reviewer.model,
        system_prompt=system_prompt,
        cap_usd=args.cap_usd,
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
    )


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


def _sdk_options(sdk: Any, kwargs: Mapping[str, Any]) -> Any:
    """Build ``ClaudeAgentOptions`` from the policy kwargs plus the Bash-guard hook.

    Raises:
        SdkRejectedError: The installed SDK does not know one of the option names.
    """
    matcher = sdk.HookMatcher(
        matcher=policy.BASH_HOOK_MATCHER, hooks=[policy.deny_non_readonly_bash]
    )
    try:
        return sdk.ClaudeAgentOptions(**kwargs, hooks={"PreToolUse": [matcher]})
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
        options = _sdk_options(sdk, {**plan.options, "model": state.model})
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
            text, repo_root=plan.repo_root, cfg=plan.cfg, label="the review output"
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
            repo_root=plan.repo_root,
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
    print(f"opening: {', '.join(plan.opening) or '(none)'}")
    print(f"model_requested: {reviewer.model}")
    print(f"model_resolved: {options['model']}")
    print(f"resolved_via: {RESOLVED_VIA_CONFIG} (a real run asks models.retrieve first)")
    print(f"max_turns: {options['max_turns']}")
    print(f"cap_usd: {plan.cap_usd}")
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
    print("mcp_env:")
    for name, value in server["env"].items():
        print(f"  {name}={value}")
    print(f"system_prompt_chars: {len(plan.system_prompt)}")
    print(f"user_prompt_chars: {len(plan.user_prompt)}")
    print(f"sdk: {sdk_line}")
    print(f"{API_KEY_ENV}: {key_state}")


def _dry_run(plan: Plan) -> int:
    try:
        headroom = ledger.precheck(plan.cfg, plan.cap_usd, repo_root=plan.repo_root)
        sdk = _import_sdk()
        if sdk is None:
            sdk_line = (
                "not installed (options not validated against it; install the agent_reviewer extra)"
            )
        else:
            _sdk_options(sdk, plan.options)
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
        _sdk_options(sdk, plan.options)  # the option names are real, before any spend
        headroom = ledger.precheck(plan.cfg, plan.cap_usd, repo_root=plan.repo_root)
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        plan = _build_plan(args)
    except _REFUSALS as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return _dry_run(plan) if args.dry_run else _live(plan)


if __name__ == "__main__":
    sys.exit(main())
