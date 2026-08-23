# tests/regression/test_claude_workforce_aqa.py
"""AQA: Claude Code workforce asset contracts.

The PR gate for everything under ``.claude/`` plus the hook package. Implemented
as a regression test rather than a new GitHub Actions job, matching the pattern
``test_skill_commands_aqa.py`` established: the regression tier already runs
across the Python matrix, so a new workflow would buy nothing and add a startup
surface to keep green.

Reuses the host/IP rule from :mod:`tools.validate_skill_commands` (the
``test_foundry_plan_doc.py`` precedent) so the policy lives in exactly one place.

Contracts pinned here:

* every workforce threshold lives in ``.claude/workforce.yaml`` and validates;
* workforce assets stay portable (no absolute paths, no host/IP literals);
* subagent frontmatter uses only platform-supported keys, with **bare** tool
  names — permission patterns like ``Bash(git diff*)`` are silently ignored by
  the platform, so they must fail here instead;
* wired hook commands point at files that exist;
* the legacy ``.claude/commands/`` layout stays deleted;
* pre-existing settings survive the hooks block being added.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.claude_hooks.config import DEFAULT_CONFIG_RELPATH, WorkforceConfig, load_config
from tools.claude_hooks.portability import find_absolute_paths
from tools.validate_skill_commands import find_hardcoded_hosts

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_DIR = _REPO_ROOT / ".claude"
_SETTINGS = _CLAUDE_DIR / "settings.json"
_LEGACY_COMMANDS = _CLAUDE_DIR / "commands"
_HOOK_PACKAGE = _REPO_ROOT / "tools" / "claude_hooks"

# All files that contain text which must stay portable. The tuple structure is
# enumerated, so a new asset is covered the moment it lands.
_TEXT_SUFFIXES = frozenset({".md", ".yaml", ".yml", ".json"})


def _config() -> WorkforceConfig:
    return load_config(repo_root=_REPO_ROOT)


def _claude_text_assets() -> list[Path]:
    if not _CLAUDE_DIR.is_dir():
        return []

    def _is_valid_asset(path: Path) -> bool:
        return path.suffix in _TEXT_SUFFIXES and ".local." not in path.name

    assets: list[Path] = []
    for root, dirs, files in os.walk(_CLAUDE_DIR):
        if "worktrees" in dirs:
            dirs.remove("worktrees")
        assets.extend(Path(root) / f for f in files if _is_valid_asset(Path(root) / f))
    return sorted(assets)


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    """Return the YAML front-matter mapping of ``path`` (empty when absent)."""
    text = path.read_text(encoding="utf-8-sig")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    parsed = yaml.safe_load(parts[1])
    return parsed if isinstance(parsed, dict) else {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_workforce_config_is_present_and_valid() -> None:
    """The checked-in config must satisfy its own schema."""
    assert (_REPO_ROOT / DEFAULT_CONFIG_RELPATH).is_file(), (
        f"{DEFAULT_CONFIG_RELPATH} is missing — workforce thresholds must have a home"
    )
    cfg = _config()
    assert cfg.freeze.frozen_paths, "freeze.frozen_paths is empty — the gate would never fire"


def _git_tracked(*pathspecs: str) -> list[str]:
    """Return the git-tracked files matching ``pathspecs``."""
    result = subprocess.run(
        ["git", "ls-files", "--", *pathspecs],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.parametrize(
    "relpath",
    [DEFAULT_CONFIG_RELPATH, ".claude/settings.json"],
)
def test_shared_claude_assets_are_git_tracked(relpath: str) -> None:
    """Shared `.claude/` assets must actually ship, not just exist locally.

    `.gitignore` excludes `.claude/*` for session state, so a new shared asset
    is untracked by default: it works on the author's machine and is simply
    absent everywhere else. That is not hypothetical — the workforce config hit
    exactly this and was caught only by CI. The negation entries in `.gitignore`
    are what make these files shippable, and this test is what keeps them so.
    """
    assert _git_tracked(relpath), (
        f"{relpath} is not tracked by git — it exists locally but will be absent "
        f"in CI and in every clone. Add a '!{relpath}' negation to .gitignore."
    )


def test_freeze_gate_targets_a_real_feature_catalog() -> None:
    """The configured catalog exists and actually declares the gate feature."""
    cfg = _config()
    catalog = _REPO_ROOT / cfg.freeze.features_file
    assert catalog.is_file(), f"freeze.features_file {cfg.freeze.features_file} does not exist"
    parsed = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    features = parsed.get("features") if isinstance(parsed, dict) else parsed
    ids = {str(entry.get("id", "")) for entry in features if isinstance(entry, dict)}
    assert cfg.freeze.feature_key in ids, (
        f"freeze.feature_key {cfg.freeze.feature_key} is not in {cfg.freeze.features_file}"
    )


def test_coverage_source_directory_exists() -> None:
    """The dedicated coverage invocation must point at a real package."""
    cfg = _config()
    assert (_REPO_ROOT / cfg.coverage.tools_source).is_dir()


# ---------------------------------------------------------------------------
# Portability (invariant I-3)
# ---------------------------------------------------------------------------


def test_claude_assets_carry_no_absolute_paths() -> None:
    offenders: list[str] = []
    for asset in _claude_text_assets():
        found = find_absolute_paths(asset.read_text(encoding="utf-8"))
        offenders.extend(f"{asset.relative_to(_REPO_ROOT)}: {hit}" for hit in found)
    assert not offenders, f"absolute paths under .claude/ break portability: {offenders}"


def test_claude_assets_carry_no_hardcoded_hosts() -> None:
    offenders: list[str] = []
    for asset in _claude_text_assets():
        found = find_hardcoded_hosts(asset.read_text(encoding="utf-8"))
        offenders.extend(f"{asset.relative_to(_REPO_ROOT)}: {hit}" for hit in found)
    assert not offenders, f"hardcoded host/IP under .claude/: {offenders}"


# ---------------------------------------------------------------------------
# Subagent frontmatter contract
# ---------------------------------------------------------------------------


def _agent_files() -> list[Path]:
    agents_dir = _REPO_ROOT / _config().agents.directory
    return sorted(agents_dir.glob("*.md")) if agents_dir.is_dir() else []


def test_agent_files_declare_required_frontmatter() -> None:
    cfg = _config()
    for agent in _agent_files():
        meta = _parse_frontmatter(agent)
        missing = [key for key in cfg.agents.required_frontmatter_keys if not meta.get(key)]
        assert not missing, f"{agent.name} is missing frontmatter keys {missing}"


def test_agent_files_use_only_supported_frontmatter_keys() -> None:
    cfg = _config()
    allowed = set(cfg.agents.allowed_frontmatter_keys)
    for agent in _agent_files():
        unknown = sorted(set(_parse_frontmatter(agent)) - allowed)
        assert not unknown, (
            f"{agent.name} declares frontmatter keys the platform ignores: {unknown}"
        )


def test_agent_tools_are_bare_names_not_permission_patterns() -> None:
    """Agent frontmatter accepts bare tool names only.

    A ``Bash(git diff*)``-style entry looks like it restricts the agent but is
    not honoured in agent frontmatter, so it must fail loudly here.
    """
    cfg = _config()
    for agent in _agent_files():
        tools = _parse_frontmatter(agent).get("tools", "")
        rendered = ", ".join(tools) if isinstance(tools, list) else str(tools)
        offenders = [char for char in cfg.agents.forbidden_tool_chars if char in rendered]
        assert not offenders, (
            f"{agent.name} uses permission-pattern syntax {offenders} in 'tools'; "
            "agent frontmatter supports bare tool names only"
        )


def test_agent_files_stay_within_line_budget() -> None:
    cfg = _config()
    for agent in _agent_files():
        lines = len(agent.read_text(encoding="utf-8").splitlines())
        assert lines <= cfg.agents.max_lines, (
            f"{agent.name} is {lines} lines (budget {cfg.agents.max_lines})"
        )


# The complete roster per D-4 in the workforce design.
_EXPECTED_AGENTS = frozenset(
    {
        "peer-reviewer",
        "security-scanner",
        "config-guardian",
        "openspec-author",
        "test-engineer",
        "doc-reconciler",
        "hw-evidence-auditor",
    }
)


def test_complete_agent_roster_is_present() -> None:
    """All seven D-4 agents must exist — a missing agent is a gap in governance."""
    present = {agent.stem for agent in _agent_files()}
    missing = _EXPECTED_AGENTS - present
    assert not missing, f"agent roster is incomplete — missing: {sorted(missing)}"


def test_agent_name_matches_filename() -> None:
    """The frontmatter name must match the file stem for discoverability."""
    for agent in _agent_files():
        meta = _parse_frontmatter(agent)
        name = meta.get("name", "")
        assert name == agent.stem, (
            f"{agent.name}: frontmatter name '{name}' does not match stem '{agent.stem}'"
        )


def test_agent_files_are_git_tracked() -> None:
    """Agents must ship — untracked files are invisible in CI and clones."""
    for agent in _agent_files():
        relpath = str(agent.relative_to(_REPO_ROOT)).replace("\\", "/")
        assert _git_tracked(relpath), f"{relpath} is not tracked by git — add a .gitignore negation"


# ---------------------------------------------------------------------------
# SKILLS.md index accuracy — F-030
# ---------------------------------------------------------------------------
#
# SKILLS.md documents two shapes of entry: a plain "###" procedure (files to
# read, commands to run, no dedicated skill directory required — deliberate,
# not drift), and a real ".claude/skills/<name>/" directory invocable as
# "/<name>". The class of bug this session kept re-finding is a real,
# invocable capability that the index never mentions — 12 of 17 skill
# directories were absent from SKILLS.md entirely before this fix, and the
# "Subagent skills" table named seven subagents that do not exist in this
# repository. Both are now pinned by DISCOVERY (glob the real directories,
# not a hardcoded roster) so the next skill or agent added without an index
# entry fails loudly rather than rotting invisibly the same way.

_SKILLS_DIR = _REPO_ROOT / ".claude" / "skills"
_SKILLS_MD = _REPO_ROOT / "SKILLS.md"


def _skill_directories() -> list[str]:
    """Every real, invocable skill under .claude/skills/, by directory name."""
    if not _SKILLS_DIR.is_dir():
        return []
    return sorted(d.name for d in _SKILLS_DIR.iterdir() if d.is_dir())


def test_every_skill_directory_is_mentioned_in_the_index() -> None:
    """A real skill absent from SKILLS.md is invisible to any agent using the

    index as documented ("Discovery" section: "grep this file"). This does not
    require a full ### entry — appearing anywhere (e.g. the Workforce skills
    table) is sufficient; only total absence is the failure mode this pins.
    """
    text = _SKILLS_MD.read_text(encoding="utf-8")
    missing = sorted(name for name in _skill_directories() if name not in text)
    assert not missing, (
        f"skill director{'y is' if len(missing) == 1 else 'ies are'} not mentioned "
        f"anywhere in SKILLS.md: {missing} — a real, invocable skill the index "
        "does not know about"
    )


def test_every_agent_is_listed_in_the_subagent_skills_table() -> None:
    """The Subagent skills table must name real agents, not a stale roster.

    Verified defect this corrects: the table previously listed seven
    plugin-namespaced subagent names (``security-auditor``,
    ``feature-dev:code-reviewer``, ...) matching none of the seven real
    ``.claude/agents/`` files.
    """
    text = _SKILLS_MD.read_text(encoding="utf-8")
    table_start = text.find("## Subagent skills")
    assert table_start != -1, "SKILLS.md lost its Subagent skills section"
    table_text = text[table_start:]
    missing = sorted(agent.stem for agent in _agent_files() if f"`{agent.stem}`" not in table_text)
    assert not missing, f"agents missing from the Subagent skills table: {missing}"


# ---------------------------------------------------------------------------
# settings.json wiring
# ---------------------------------------------------------------------------


def test_settings_json_is_valid() -> None:
    assert _SETTINGS.is_file()
    data = json.loads(_SETTINGS.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_settings_permissions_are_preserved() -> None:
    """Adding hooks must not disturb the pre-existing permission allowlist."""
    data = json.loads(_SETTINGS.read_text(encoding="utf-8"))
    allow = data.get("permissions", {}).get("allow", [])
    assert isinstance(allow, list)
    assert allow, "permissions.allow was emptied — the hooks block must be additive"


def _hook_commands() -> list[str]:
    data = json.loads(_SETTINGS.read_text(encoding="utf-8"))
    commands: list[str] = []
    hooks = data.get("hooks", {})
    if not isinstance(hooks, dict):
        return commands
    for entries in hooks.values():
        for entry in entries if isinstance(entries, list) else []:
            for handler in entry.get("hooks", []) if isinstance(entry, dict) else []:
                command = handler.get("command") if isinstance(handler, dict) else None
                if isinstance(command, str):
                    commands.append(command)
    return commands


def _module_target(command: str) -> Path | None:
    """Return the file a ``python -m pkg.mod`` hook command resolves to."""
    tokens = command.split()
    if "-m" not in tokens:
        return None
    spec = tokens[tokens.index("-m") + 1]
    return _REPO_ROOT / Path(*spec.split(".")).with_suffix(".py")


def test_wired_hook_commands_reference_existing_modules() -> None:
    """Every wired hook must resolve to a real file.

    A hook command pointing at a missing module fails on *every* edit, so this
    is the highest-value assertion in the file.
    """
    for command in _hook_commands():
        target = _module_target(command)
        assert target is not None, f"hook command is not a 'python -m' invocation: {command}"
        assert target.is_file(), f"hook module not found: {target.relative_to(_REPO_ROOT)}"


def test_wired_hook_commands_run_from_the_project_directory() -> None:
    """Hook commands stay portable and importable.

    ``python -m tools.claude_hooks.<mod>`` needs the repository root on
    ``sys.path``; running the module file by path does not provide that, so the
    command must ``cd`` to ``$CLAUDE_PROJECT_DIR`` first.
    """
    for command in _hook_commands():
        assert "$CLAUDE_PROJECT_DIR" in command, (
            f"hook command must resolve via $CLAUDE_PROJECT_DIR, got: {command}"
        )
        assert " -m " in command, (
            "hook must use 'python -m package.module' so the repo root is importable; "
            f"got: {command}"
        )


# ---------------------------------------------------------------------------
# Layout invariants
# ---------------------------------------------------------------------------


def test_legacy_commands_dir_stays_deleted() -> None:
    """The migrated-away layout must not return (foundry plan WS-F7a)."""
    assert not _LEGACY_COMMANDS.exists(), (
        ".claude/commands/ has been resurrected — add skills under "
        ".claude/skills/<name>/SKILL.md instead"
    )


@pytest.mark.parametrize(
    "module",
    [
        "config",
        "docs_trimmer",
        "freeze_gate",
        "hookio",
        "logging_setup",
        "paths",
        "portability",
        "secret_scan",
    ],
)
def test_hook_package_modules_are_present(module: str) -> None:
    assert (_HOOK_PACKAGE / f"{module}.py").is_file()


def test_hook_package_has_no_runtime_package_import() -> None:
    """Hooks must never import the robot runtime.

    A hook runs on every Write/Edit; importing ``mousedroid`` would drag torch,
    faiss and lmdb into that path and make edits crawl.
    """
    offenders: list[str] = []
    for module in sorted(_HOOK_PACKAGE.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        if "import mousedroid" in text or "from mousedroid" in text:
            offenders.append(module.name)
    assert not offenders, f"hook modules import the runtime package: {offenders}"


# ---------------------------------------------------------------------------
# Phase 5: MCP Configuration & Worktree Runbooks (F-024)
# ---------------------------------------------------------------------------

_MCP_JSON = _REPO_ROOT / ".mcp.json"
_WORKTREES_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "worktrees.md"
_MCP_EVALUATION_DOC = _REPO_ROOT / "docs" / "claude" / "surfaces" / "mcp-evaluation.md"
_MCP_NEXT_STEPS = _REPO_ROOT / "docs" / "MCP_NEXT_STEPS.md"


def test_mcp_json_is_valid_json() -> None:
    """The checked-in .mcp.json must parse as valid JSON."""
    assert _MCP_JSON.is_file(), ".mcp.json is missing from repository root"
    data = json.loads(_MCP_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "mcpServers" in data
    assert isinstance(data["mcpServers"], dict)


def test_mcp_json_is_secretless() -> None:
    """.mcp.json must not contain hardcoded secret literals (only ${VAR} expansions)."""
    text = _MCP_JSON.read_text(encoding="utf-8")
    assert "ghp_" not in text, "Hardcoded GitHub token found in .mcp.json"
    assert "github_pat_" not in text, "Hardcoded GitHub PAT found in .mcp.json"
    data = json.loads(text)
    for server_name, server_cfg in data.get("mcpServers", {}).items():
        env = server_cfg.get("env", {})
        for k, v in env.items():
            if "TOKEN" in k or "SECRET" in k or "KEY" in k:
                val_str = str(v)
                assert val_str.startswith("${"), (
                    f"Credential key {k} in server {server_name} is not an env var expansion: {v}"
                )
                assert val_str.endswith("}"), (
                    f"Credential key {k} in server {server_name} is not an env var expansion: {v}"
                )


def test_mcp_json_mousedroid_server_matches_operator_guide() -> None:
    """The mousedroid server stanza in .mcp.json must match docs/MCP_OPERATOR_GUIDE.md."""
    data = json.loads(_MCP_JSON.read_text(encoding="utf-8"))
    mousedroid = data.get("mcpServers", {}).get("mousedroid")
    assert mousedroid is not None, "mousedroid server is missing from .mcp.json"
    assert mousedroid.get("command") == "python"
    assert mousedroid.get("args") == ["-m", "mousedroid", "--config", "config/default.yaml"]
    env = mousedroid.get("env", {})
    assert env.get("MOUSEDROID_MCP__ENABLED") == "true"
    assert env.get("MOUSEDROID_MCP__BIND_TRANSPORT") == "true"
    assert env.get("MOUSEDROID_MCP__TRANSPORT") == "stdio"
    assert "MOUSEDROID_MOCK_HARDWARE" in env


def test_mcp_json_github_server_configured() -> None:
    """The github server stanza in .mcp.json must be present and secretless."""
    data = json.loads(_MCP_JSON.read_text(encoding="utf-8"))
    github = data.get("mcpServers", {}).get("github")
    assert github is not None, "github server is missing from .mcp.json"
    assert github.get("command") == "npx"
    assert "@modelcontextprotocol/server-github" in github.get("args", [])


def test_worktree_runbook_is_present_and_structured() -> None:
    """docs/runbooks/worktrees.md must exist and document lifecycle & guardrails."""
    assert _WORKTREES_RUNBOOK.is_file(), "docs/runbooks/worktrees.md is missing"
    content = _WORKTREES_RUNBOOK.read_text(encoding="utf-8")
    assert "git worktree list" in content
    assert "git worktree add" in content
    assert "git worktree remove" in content
    assert "mdcw-" in content


def test_mcp_evaluation_surface_doc_is_present() -> None:
    """docs/claude/surfaces/mcp-evaluation.md must exist and record evaluate-first decisions."""
    assert _MCP_EVALUATION_DOC.is_file(), "docs/claude/surfaces/mcp-evaluation.md is missing"
    content = _MCP_EVALUATION_DOC.read_text(encoding="utf-8")
    assert "grafana" in content.lower()
    assert "huggingface" in content.lower()
    assert "mousedroid" in content.lower()
    assert "github" in content.lower()


def test_mcp_next_steps_checkbox_ticked() -> None:
    """docs/MCP_NEXT_STEPS.md must have the Claude Code .mcp.json checkbox ticked."""
    content = _MCP_NEXT_STEPS.read_text(encoding="utf-8")
    assert "- [x] Same for **Claude Code** (`.mcp.json` template)." in content
