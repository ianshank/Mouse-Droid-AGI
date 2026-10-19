"""Pydantic schema for ``.claude/workforce.yaml`` — the workforce config source.

Workforce invariant I-2 ("no hardcoded values"): every threshold, gate key, path
glob and budget consumed by a hook, the AQA regression test or a workforce skill
is declared here and read from YAML. Numeric and string literals in this module
are schema *defaults*; no other workforce module may inline them.

Backwards compatibility (invariant I-1): every field carries a default, so a
repository with no ``workforce.yaml`` — or one written before a field existed —
loads successfully with the documented defaults. ``extra="forbid"`` is applied
at every level so a typo (``frozen_path`` for ``frozen_paths``) fails loudly at
load time instead of silently disabling a gate.
"""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from tools.claude_hooks.paths import resolve_repo_root

#: Repo-relative location of the workforce config file.
DEFAULT_CONFIG_RELPATH = ".claude/workforce.yaml"

_STRICT = ConfigDict(extra="forbid")


class ConfigError(RuntimeError):
    """Raised when the workforce configuration cannot be loaded or validated."""


class FreezeConfig(BaseModel):
    """Capability freeze-gate settings.

    The gate denies edits to :attr:`frozen_paths` until the feature identified by
    :attr:`feature_key` reaches :attr:`done_status` in :attr:`features_file`,
    mechanising the plan rule "hardware readiness preempts all in-flight software
    streams". When the feature completes, the gate self-disables with no code
    change.
    """

    model_config = _STRICT

    enabled: bool = True
    feature_key: str = "F-008"
    features_file: str = "features.yaml"
    frozen_paths: list[str] = Field(default_factory=list)
    override_env: str = "MOUSEDROID_WORKFORCE_ALLOW_FROZEN"
    done_status: str = "done"

    @field_validator("feature_key", "features_file", "override_env", "done_status")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject whitespace-only identifiers that would silently break the gate."""
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("features_file")
    @classmethod
    def _reject_absolute(cls, value: str) -> str:
        """Keep the catalog path repo-relative (invariant I-3, portability)."""
        p = Path(value)
        if p.is_absolute() or PurePosixPath(value).is_absolute() or ".." in p.parts:
            raise ValueError("must be a repo-relative path without '..' traversal")
        return value


class SecretScanConfig(BaseModel):
    """Edit-time secret-scan settings.

    Wraps the repository's existing scanner and allowlist rather than
    introducing a second policy: the same ``gitleaks`` binary and
    ``.gitleaks.toml`` regex-only allowlist the CI gate uses.
    """

    model_config = _STRICT

    enabled: bool = True
    command: str = "gitleaks"
    config_file: str = ".gitleaks.toml"
    timeout_s: float = Field(default=20.0, gt=0.0, le=600.0)
    #: When the scanner binary is missing (or times out), deny instead of
    #: warning. ``False`` mirrors the advisory posture of the CI job.
    strict: bool = False
    extra_args: list[str] = Field(default_factory=list)
    #: Skip scanning content larger than this; a multi-megabyte generated file
    #: would blow the hook's latency budget without adding signal.
    max_bytes: int = Field(default=1_000_000, gt=0)

    @field_validator("config_file")
    @classmethod
    def _reject_absolute(cls, value: str) -> str:
        """Keep the allowlist path repo-relative.

        It is joined onto the repo root before being handed to the scanner, so
        an absolute or ``..``-traversing value would point the scan's config at
        an arbitrary file. Mirrors the guard on
        :attr:`FreezeConfig.features_file`.
        """
        p = Path(value)
        if p.is_absolute() or PurePosixPath(value).is_absolute() or ".." in p.parts:
            raise ValueError("must be a repo-relative path without '..' traversal")
        return value


class MypyProfile(BaseModel):
    """Per-path mypy invocation profile for the advisory post-edit check.

    mypy derives a *module name* from a file's path, so a single argument set
    cannot serve two trees that sit differently relative to their package base.
    ``src/mousedroid/**`` resolves correctly under mypy's defaults, but
    ``tools/claude_hooks/**`` — a package under a plain directory that imports
    itself as ``tools.claude_hooks.*`` — needs ``--explicit-package-bases`` with
    ``MYPYPATH`` at the repository root. Without them mypy finds the file twice
    ("Source file found twice under different module names") and **checks
    nothing**, reporting that collision as the only finding.

    This is the same split CI already runs as two separate mypy invocations
    (``.github/workflows/ci.yml``: ``mypy src/`` in ``typecheck``, and
    ``MYPYPATH=. mypy tools/claude_hooks/ --explicit-package-bases`` in
    ``local-gates``). A profile list is how the edit-time check reproduces that
    split without hardcoding a tree name in the hook.
    """

    model_config = _STRICT

    paths: list[str] = Field(
        min_length=1,
        description=(
            "Repo-relative glob patterns, evaluated in order. The first profile "
            "matching the edited file supplies that file's mypy invocation. At "
            "least one pattern is required: a profile with no paths can never "
            "match, so it would be silently inert -- and this schema's whole "
            "posture (extra='forbid') is that a configuration mistake fails loudly."
        ),
    )
    args: list[str] = Field(
        default_factory=list,
        description=(
            "Full mypy argument list for matching files, replacing "
            "post_edit.mypy_args rather than extending it — a profile exists "
            "because the default set is wrong for the tree, so appending to it "
            "would keep the wrong flags."
        ),
    )
    mypy_path: str | None = Field(
        default=None,
        description=(
            "Repo-relative value exported as MYPYPATH for matching files, or "
            "null to inherit the ambient environment. '.' means the repository "
            "root, which is what a package under a plain directory needs."
        ),
    )

    @field_validator("mypy_path")
    @classmethod
    def _reject_absolute_mypy_path(cls, value: str | None) -> str | None:
        """Keep MYPYPATH repo-relative and inside the checkout (invariant I-3).

        An absolute path here pins the config to one machine, which is what
        ``portability.find_absolute_paths`` exists to stop -- and since the hook
        joins this onto the repo root, a value that escapes the checkout sends
        MYPYPATH somewhere arbitrary.

        ``PurePosixPath`` alone is not enough, and an earlier version of this
        validator that used only it was strictly weaker than the authority its
        docstring cited: ``PurePosixPath("C:\\Users\\me").is_absolute()`` is
        ``False``, so Windows drive-letter paths sailed through, even though
        ``portability`` ships a dedicated ``_WINDOWS_ABSOLUTE_RE`` for exactly
        that shape. ``~`` and ``..`` are rejected for the same reason: both
        resolve outside the checkout.
        """
        if value is None:
            return value
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("must be repo-relative, not absolute")
        if value.startswith("~"):
            raise ValueError("must be repo-relative; '~' expands outside the checkout")
        if ".." in PurePosixPath(value).parts:
            raise ValueError("must stay inside the checkout; '..' escapes the repo root")
        return value


class PostEditConfig(BaseModel):
    """Advisory post-edit check settings (report-only by platform contract)."""

    model_config = _STRICT

    enabled: bool = True
    #: Checker names to run, in order. Unknown names are skipped with a warning,
    #: so adding a checker later is a config edit, not a code change.
    checks: list[str] = Field(default_factory=lambda: ["ruff", "mypy"])
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    #: Only these suffixes are checked; anything else is skipped.
    suffixes: list[str] = Field(default_factory=lambda: [".py"])
    ruff_args: list[str] = Field(default_factory=lambda: ["check", "--no-fix"])
    mypy_args: list[str] = Field(
        default_factory=lambda: ["--ignore-missing-imports", "--follow-imports=silent"]
    )
    mypy_profiles: list[MypyProfile] = Field(
        default_factory=lambda: [
            MypyProfile(
                paths=["tools/**"],
                args=[
                    "--ignore-missing-imports",
                    "--follow-imports=silent",
                    "--explicit-package-bases",
                ],
                mypy_path=".",
            )
        ],
        description=(
            "Per-path mypy invocation overrides, first match wins. Files matching "
            "no profile use mypy_args unchanged, so this field is additive: a "
            "config written before it existed keeps its exact behaviour for "
            "src/ and gains the correct invocation for tools/."
        ),
    )


class CoverageConfig(BaseModel):
    """Coverage budgets for the workforce tooling itself.

    The repository-wide gate measures ``src/mousedroid`` only and cannot see
    ``tools/``; :attr:`tools_line_min` drives a dedicated, additive invocation.
    """

    model_config = _STRICT

    tools_line_min: int = Field(default=85, ge=0, le=100)
    #: Reported, never blocking, until a baseline exists (see the truthful-claims
    #: requirement in the dev-governance spec).
    tools_branch_min_advisory: int = Field(default=0, ge=0, le=100)
    tools_source: str = "tools/claude_hooks"


class DocsConfig(BaseModel):
    """Documentation budgets used by the docs-consolidation phase."""

    model_config = _STRICT

    core_max_lines: int = Field(default=250, gt=0)
    surfaces_dir: str = "docs/claude/surfaces"


class WorktreeConfig(BaseModel):
    """Worktree-per-change settings."""

    model_config = _STRICT

    prefix: str = "mdcw-"


class EvidenceConfig(BaseModel):
    """Evidence-audit policy.

    ``local_only_declared`` records report families the repository deliberately
    gitignores; a claim backed by one of those plus a CHANGELOG/plan reference is
    a declared local-only evidence chain, not a finding.
    """

    model_config = _STRICT

    tracked_roots: list[str] = Field(default_factory=lambda: ["reports", "smoke-reports"])
    stale_after_days: int = Field(default=90, gt=0)
    local_only_declared: list[str] = Field(default_factory=list)


class AgentsConfig(BaseModel):
    """Subagent asset contract enforced by the AQA regression test."""

    model_config = _STRICT

    directory: str = ".claude/agents"
    max_lines: int = Field(default=60, gt=0)
    required_frontmatter_keys: list[str] = Field(
        default_factory=lambda: ["name", "description", "tools"]
    )
    #: Frontmatter keys Claude Code understands for a subagent. Anything else is
    #: silently ignored by the platform, so the AQA test flags it as drift.
    allowed_frontmatter_keys: list[str] = Field(
        default_factory=lambda: ["name", "description", "tools", "model"]
    )
    #: Permission-pattern punctuation. The platform accepts bare tool names only
    #: in agent frontmatter, so these characters indicate an unsupported
    #: ``Bash(git diff*)``-style declaration.
    forbidden_tool_chars: list[str] = Field(default_factory=lambda: ["(", ")", "*"])


class RatchetBudgetItem(BaseModel):
    """A single ratchet-down-only numeric budget tracked for early warning.

    ``marker`` is the literal substring counted per source line — mirrors the
    inline ``line.count(marker)`` logic the hard-fail regression tests already
    use (named ``marker``, not ``token``: a keyword argument literally spelled
    ``token=`` trips ruff's S106 hardcoded-credential heuristic on every
    construction site, including this class's own defaults below). ``ceiling``
    is that same hard-fail threshold; ``warn_threshold``, when set, fires a
    WARN once the count crosses it, *before* the budget goes red, so drift is
    visible at edit time instead of only at PR time.
    """

    model_config = _STRICT

    name: str
    marker: str
    scope_glob: str = "src/mousedroid/**/*.py"
    ceiling: int = Field(ge=0)
    warn_threshold: int | None = Field(default=None, ge=0)

    @field_validator("name", "marker")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def _warn_threshold_within_ceiling(self) -> RatchetBudgetItem:
        """A warn threshold above the ceiling could never fire before the
        hard-fail gate does, defeating the point of an early warning."""
        if self.warn_threshold is not None and self.warn_threshold > self.ceiling:
            raise ValueError(
                f"warn_threshold ({self.warn_threshold}) must not exceed ceiling ({self.ceiling})"
            )
        return self


class RatchetBudgetsConfig(BaseModel):
    """Ratchet-down-only numeric budgets tracked for early-warning drift.

    Defaults reproduce today's hard-fail-only regression budgets
    (``tests/regression/test_suppression_budget.py``,
    ``test_hardcoded_value_marker_budget.py``) exactly — wiring this config in
    is behavior-neutral until a consumer reads from it. Explicitly out of
    scope: ``_ALLOWED_SRC_PER_FILE_IGNORES``, a set-membership pin with no
    "approaching the ceiling" state to warn about in advance.
    """

    model_config = _STRICT

    enabled: bool = True
    items: list[RatchetBudgetItem] = Field(
        default_factory=lambda: [
            RatchetBudgetItem(name="noqa", marker="noqa", ceiling=19, warn_threshold=17),
            RatchetBudgetItem(
                name="type_ignore", marker="type: ignore", ceiling=8, warn_threshold=7
            ),
            RatchetBudgetItem(
                name="hardcoded_ok", marker="# hardcoded-ok", ceiling=24, warn_threshold=22
            ),
        ]
    )


def _require_repo_relative(value: str) -> str:
    """Return ``value`` if it is a non-blank repo-relative path without ``..``.

    Shared by the credit sub-sections that join a configured path onto the repo
    root: an absolute or ``..``-traversing value would point a read or a write
    outside the checkout. Both path flavours are checked because the checkout may
    sit on either OS (``PurePosixPath`` alone calls ``C:/x`` relative).
    """
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        not value.strip()
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ValueError("must be a non-empty repo-relative path without '..' traversal")
    return value


class ModelPrice(BaseModel):
    """Standard (non-batch) token prices of one model, in USD per million tokens.

    Both directions are required: a half-specified price cannot bound a cost, and
    a zero price would make every estimate zero and so bypass the cap.
    """

    model_config = _STRICT

    input_usd_per_mtok: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="USD per million input tokens at the standard (non-batch) rate.",
    )
    output_usd_per_mtok: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description="USD per million output tokens at the standard (non-batch) rate.",
    )


class CreditPrices(BaseModel):
    """The token prices the credit tools use to price an estimate and a recorded cost.

    Fails closed like :class:`CreditConfig`: ``models`` defaults to empty, so a
    model nobody priced cannot be estimated and therefore cannot be submitted.
    """

    model_config = _STRICT

    batch_multiplier: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Fraction of the standard token price the Message Batches API bills "
            "(0.5 = half price). Defaults to 1.0, no discount assumed, so an estimate "
            "made without this key can only over-state the cost."
        ),
    )
    models: dict[str, ModelPrice] = Field(
        default_factory=dict,
        description=(
            "Standard prices per model id. A model absent from this map cannot be "
            "estimated, so a tool that prices its spend before the call refuses it."
        ),
    )

    @field_validator("models")
    @classmethod
    def _reject_blank_model_names(cls, value: dict[str, ModelPrice]) -> dict[str, ModelPrice]:
        """Reject a blank model id, which could never match a request."""
        for name in value:
            if not name.strip():
                raise ValueError("model ids must not be empty or whitespace-only")
        return value


class BatchTriageConfig(BaseModel):
    """Settings of the Batch-API vulture triage (F-059 WS-C, ``tools/dead_code_triage_batch.py``).

    Every knob the tool would otherwise inline: which model answers and how it is
    asked, how much source it is shown, how big a request may get, how the pre-call
    estimate is made, how long to wait for the batch, and where the artifacts live.
    The defaults reproduce the plan's figures, so a workforce.yaml written before
    this section existed behaves identically.
    """

    model_config = _STRICT

    workstream: str = Field(
        default="C",
        description="Key of credit.caps_usd that pays for the triage run (F-059 WS-C).",
    )
    model: str = Field(
        default="claude-sonnet-5-5",
        description=(
            "Model id the triage requests ask for. The tool resolves it through "
            "client.models.retrieve before submitting and records both ids."
        ),
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description=(
            "max_tokens of each triage request. Thinking tokens count against it, so "
            "it also bounds how much a model that thinks may spend before it answers."
        ),
    )
    effort: Literal["low", "medium", "high"] | None = Field(
        default="low",
        description=(
            "output_config.effort sent with each request; null sends none (the model's "
            "default, which thinks longest). Capped at high: xhigh and max are rejected "
            "together with a disabled thinking mode."
        ),
    )
    thinking: Literal["adaptive", "between_tools"] | None = Field(
        default=None,
        description=(
            "Explicit thinking mode; null sends no thinking field (the model default). "
            "Claude Sonnet 5.5 answers HTTP 400 to type disabled; between_tools is its "
            "way to switch thinking off, accepted at effort high or below."
        ),
    )
    source_root: str = Field(
        default="src/mousedroid",
        description=(
            "Repo-relative tree the audit scans. Findings are grouped by the first path "
            "segment below it, and the part of a recorded absolute path before it is dropped."
        ),
    )
    context_lines: int = Field(
        default=20,
        ge=0,
        description="Lines of source shown on each side of a finding (the excerpt window).",
    )
    max_findings_per_request: int = Field(
        default=20,
        gt=0,
        description=(
            "Findings sent in one request; a larger package is split. Bounds the answer "
            "(about 75 tokens per finding) so it fits max_tokens beside any thinking."
        ),
    )
    max_hint_names: int = Field(
        default=300,
        ge=0,
        description=(
            "Cap on the Protocol member and validator names listed in one prompt; the "
            "rest are summarised as a count."
        ),
    )
    chars_per_token: float = Field(
        default=3.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Characters per token assumed when sizing a prompt for the estimate. Code runs "
            "near 3, so 3.0 over-states the input and keeps the estimate an upper bound."
        ),
    )
    assumed_input_tokens_per_request: int = Field(
        default=8000,
        ge=0,
        description=(
            "Floor for one request's input tokens in the estimate (the plan's 8K per "
            "chunk); a measured prompt above it is priced as measured."
        ),
    )
    poll_interval_s: float = Field(
        default=60.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Seconds between processing_status checks while collecting a batch.",
    )
    poll_timeout_s: float = Field(
        default=90000.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Seconds collect waits before giving up (the batch keeps running). A batch "
            "expires 24 hours after creation, so the default is 25 hours."
        ),
    )
    batches_dir: str = Field(
        default="reports/credit/batches",
        description="Repo-relative directory of the per-submit batch records (<UTC date>.json).",
    )
    analysis_dir: str = Field(
        default="docs/analysis",
        description="Repo-relative directory the default triage document is written to.",
    )
    refuse_globs: list[str] = Field(
        default_factory=lambda: ["*.env", "docker.env*", ".env*"],
        description=(
            "File-name patterns (fnmatch, case-insensitive) the prompt builder refuses to "
            "read, so an environment file can never become part of a request."
        ),
    )

    @field_validator("workstream", "model")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject whitespace-only identifiers, which could never match anything."""
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("source_root", "batches_dir", "analysis_dir")
    @classmethod
    def _reject_unsafe_path(cls, value: str) -> str:
        """Keep every configured path inside the checkout (it is joined onto the repo root)."""
        return _require_repo_relative(value)

    @field_validator("refuse_globs")
    @classmethod
    def _reject_blank_globs(cls, value: list[str]) -> list[str]:
        """Reject a blank pattern, which would silently match nothing."""
        if any(not pattern.strip() for pattern in value):
            raise ValueError("patterns must not be empty or whitespace-only")
        return value


class CreditConfig(BaseModel):
    """Spend caps and ledger location for the F-059 credit workstreams.

    Consumed by ``tools/credit_ledger.py``: ``charge()`` refuses an API call
    BEFORE it is made when the workstream's recorded spend plus the estimate
    would exceed its cap, and ``record()`` appends one JSON object per run to
    :attr:`ledger`. The numbers live in ``.claude/workforce.yaml``, never in code.

    :attr:`prices` and :attr:`batch_triage` serve ``tools/dead_code_triage_batch.py``:
    the price table its pre-call estimate is made from, and the triage run's own
    settings. Both default so that a workforce.yaml written before them still loads.

    Fails closed: :attr:`caps_usd` defaults to empty, so a repository that has
    not adopted the block — or an unknown workstream name — cannot spend. This
    is the one section whose "missing means permissive" default would be wrong:
    every other section defaults to today's behaviour; this one guards money.
    """

    model_config = _STRICT

    caps_usd: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-workstream spend cap in USD, keyed by workstream name (A/B/C in "
            "the F-059 plan). A workstream absent from this map has no cap and "
            "may not spend; a cap of 0 disables a workstream explicitly."
        ),
    )
    ledger: str = Field(
        default="reports/credit/ledger.jsonl",
        description=(
            "Repo-relative path of the append-only JSON-lines spend ledger "
            "(tracked: it is the evidence that ledgers stayed under their caps)."
        ),
    )
    prices: CreditPrices = Field(
        default_factory=CreditPrices,
        description=(
            "Token prices used to price a pre-call estimate and a recorded cost. Empty by "
            "default: a model with no price cannot be estimated, so it cannot be submitted."
        ),
    )
    batch_triage: BatchTriageConfig = Field(
        default_factory=BatchTriageConfig,
        description="Settings of the Batch-API vulture triage (tools/dead_code_triage_batch.py).",
    )

    @field_validator("caps_usd")
    @classmethod
    def _validate_caps(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject blank workstream names and caps that cannot bound spend.

        ``nan`` fails every comparison (a ``nan`` cap would never be exceeded),
        ``inf`` is no cap at all, and a negative cap is a typo for a number.
        """
        for workstream, cap in value.items():
            if not workstream.strip():
                raise ValueError("workstream names must not be empty or whitespace-only")
            if not math.isfinite(cap) or cap < 0:
                raise ValueError(f"cap for workstream {workstream!r} must be a finite number >= 0")
        return value

    @field_validator("ledger")
    @classmethod
    def _reject_unsafe_ledger_path(cls, value: str) -> str:
        """Keep the ledger inside the checkout (it is joined onto the repo root)."""
        p = Path(value)
        if (
            not value.strip()
            or p.is_absolute()
            or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or ".." in p.parts
        ):
            raise ValueError("must be a non-empty repo-relative path without '..' traversal")
        return value


class WorkforceConfig(BaseModel):
    """Root workforce configuration."""

    model_config = _STRICT

    freeze: FreezeConfig = Field(default_factory=FreezeConfig)
    secret_scan: SecretScanConfig = Field(default_factory=SecretScanConfig)
    post_edit: PostEditConfig = Field(default_factory=PostEditConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    docs: DocsConfig = Field(default_factory=DocsConfig)
    worktree: WorktreeConfig = Field(default_factory=WorktreeConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    ratchet_budgets: RatchetBudgetsConfig = Field(default_factory=RatchetBudgetsConfig)
    credit: CreditConfig = Field(default_factory=CreditConfig)


def load_config(
    path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> WorkforceConfig:
    """Load and validate the workforce configuration.

    Args:
        path: Explicit config file path. Defaults to
            ``<repo_root>/.claude/workforce.yaml``.
        repo_root: Repository root. Resolved via
            :func:`~tools.claude_hooks.paths.resolve_repo_root` when omitted.

    Returns:
        The validated configuration. A missing file yields schema defaults, so
        the tooling stays functional in a repository that has not adopted the
        config yet (invariant I-1).

    Raises:
        ConfigError: The file exists but is unreadable, is not a YAML mapping,
            or fails schema validation (including unknown keys).
    """
    root = resolve_repo_root() if repo_root is None else repo_root
    config_path = (root / DEFAULT_CONFIG_RELPATH) if path is None else Path(path)

    if not config_path.is_file():
        return WorkforceConfig()

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read workforce config {config_path}: {exc}") from exc

    try:
        parsed: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in workforce config {config_path}: {exc}") from exc

    if parsed is None:
        return WorkforceConfig()
    if not isinstance(parsed, dict):
        raise ConfigError(
            f"workforce config {config_path} must contain a YAML mapping, "
            f"got {type(parsed).__name__}"
        )

    try:
        return WorkforceConfig.model_validate(parsed)
    except ValidationError as exc:
        raise ConfigError(f"invalid workforce config {config_path}: {exc}") from exc
