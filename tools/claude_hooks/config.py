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

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
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
        if Path(value).is_absolute() or ".." in Path(value).parts:
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
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("must be a repo-relative path without '..' traversal")
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
