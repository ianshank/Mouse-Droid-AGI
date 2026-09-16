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
* wired hook commands resolve an interpreter that can actually import the hook
  package — a bare ``python3`` silently disabled every gate (see
  :func:`test_wired_hook_commands_do_not_invoke_a_bare_interpreter`);
* the legacy ``.claude/commands/`` layout stays deleted;
* pre-existing settings survive the hooks block being added.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.claude_hooks.config import DEFAULT_CONFIG_RELPATH, WorkforceConfig, load_config
from tools.claude_hooks.paths import path_matches_any
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
    directories = _skill_directories()
    # A positive control: if .claude/skills/ ever went missing or empty, the
    # assertion below would pass vacuously over zero skills -- which is not
    # the same claim as "every skill is indexed", and is exactly the failure
    # mode a pin exists to rule out, not fall into.
    assert directories, (
        f"{_SKILLS_DIR} is missing or has no skill directories -- the "
        "membership check below cannot mean anything over an empty set"
    )
    text = _SKILLS_MD.read_text(encoding="utf-8")
    missing = sorted(name for name in directories if name not in text)
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
    agents = _agent_files()
    # Same positive control as the skills index above: an empty agent roster
    # would make the membership check below pass over nothing.
    assert agents, (
        f"{_REPO_ROOT / _config().agents.directory} is missing or has no agent "
        "files -- the membership check below cannot mean anything over an "
        "empty set"
    )
    text = _SKILLS_MD.read_text(encoding="utf-8")
    table_start = text.find("## Subagent skills")
    assert table_start != -1, "SKILLS.md lost its Subagent skills section"
    # Bounded to the NEXT level-two heading, not end-of-file -- otherwise an
    # agent name mentioned anywhere later in SKILLS.md (in a different
    # section) would satisfy this check even if the table itself omits it.
    table_end = text.find("\n## ", table_start + len("## Subagent skills"))
    table_text = text[table_start : table_end if table_end != -1 else None]
    missing = sorted(agent.stem for agent in agents if f"`{agent.stem}`" not in table_text)
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
    ``sys.path``; running the module file by path does not provide that. Claude
    Code does not guarantee the hook's working directory, so the command must
    locate the project through ``$CLAUDE_PROJECT_DIR`` — today by handing that
    path to :data:`_INTERPRETER_WRAPPER`, which chdirs before exec.
    """
    for command in _hook_commands():
        assert "$CLAUDE_PROJECT_DIR" in command, (
            f"hook command must resolve via $CLAUDE_PROJECT_DIR, got: {command}"
        )
        assert " -m " in command, (
            "hook must use 'python -m package.module' so the repo root is importable; "
            f"got: {command}"
        )


#: Wrapper that resolves a capable interpreter for every wired hook.
_INTERPRETER_WRAPPER_RELPATH = "tools/claude_hooks/run_hook.sh"
_INTERPRETER_WRAPPER = _REPO_ROOT / _INTERPRETER_WRAPPER_RELPATH

#: Skip for every test that shells out to the wrapper.
#:
#: ``run_hook.sh`` is a POSIX shell script, and on the ``test-windows`` runner
#: ``bash`` resolves to WSL's ``bash.exe`` with **no distribution installed** — it
#: exits 1 printing "Windows Subsystem for Linux has no installed distributions"
#: as UTF-16, so every one of these tests fails for a reason that has nothing to
#: do with the wrapper. That is not a hypothetical: three of them went red on the
#: ``test-windows (3.11)`` leg exactly this way.
#:
#: The hooks themselves are unaffected — Claude Code runs the command it is given,
#: and a Windows host needs a different wrapper invocation anyway (which is why
#: ``_resolve_interpreter`` carries the ``.venv/Scripts/python.exe`` candidate).
#: The two-part form is the house guard, taken byte-for-byte from
#: ``tests/unit/scripts/test_repin_tags.py`` (itself copied from
#: ``test_archive_stale_branches.py``), whose comment documents this exact
#: failure: ``os.name == "nt"`` is NOT redundant with ``shutil.which("bash")``,
#: because on the Windows runner ``which()`` *finds* the WSL shim. This is now the
#: third file in this repo to learn that the hard way.
_REQUIRES_POSIX_SHELL = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="run_hook.sh is a POSIX script; the Windows runner's bash is a WSL shim with no distro",
)

#: Interpreter spellings that resolve through ``PATH`` rather than by capability.
_BARE_INTERPRETERS = ("python", "python3", "python3.10", "python3.11", "python3.12", "py")


def test_wired_hook_commands_do_not_invoke_a_bare_interpreter() -> None:
    """The highest-value pin in this file: a bare ``python3`` disables every gate.

    This is not hypothetical. The hooks shipped as ``cd "$CLAUDE_PROJECT_DIR" &&
    python3 -m tools.claude_hooks.<mod>``. ``python3`` is whatever is first on the
    hook process's ``PATH`` — the *system* interpreter in this project's
    containers, which has no pydantic — so :mod:`tools.claude_hooks.config` failed
    to import, the hook exited 1, and Claude Code treats a non-2 exit from a
    ``PreToolUse`` hook as a non-blocking hook *error*. The F-008 freeze gate and
    the edit-time secret scan both failed open, and nothing was visible: a gate
    that never blocks looks exactly like a gate with nothing to block.

    The command must therefore go through the wrapper, which picks an interpreter
    that can actually import the hook package.
    """
    for command in _hook_commands():
        tokens = command.split()
        if "-m" not in tokens:
            # Without this, tokens.index below raises ValueError and the failure
            # reads as a broken test rather than a broken hook command.
            pytest.fail(f"hook command is not a 'python -m' invocation: {command}")
        offenders = [
            token
            for token in tokens[: tokens.index("-m")]
            if Path(token.strip('"')).name in _BARE_INTERPRETERS
        ]
        assert not offenders, (
            f"hook command invokes {offenders} from PATH, which is not guaranteed to "
            f"have the hook package's dependencies — route it through "
            f"{_INTERPRETER_WRAPPER_RELPATH} instead. Command: {command}"
        )
        assert _INTERPRETER_WRAPPER_RELPATH in command, (
            f"hook command must resolve its interpreter through "
            f"{_INTERPRETER_WRAPPER_RELPATH}, got: {command}"
        )


def test_interpreter_wrapper_is_tracked_and_executable() -> None:
    """The wrapper is only useful if it ships and can run.

    ``.gitignore`` does not cover ``tools/``, so tracking is the weaker risk
    here; a wrapper that lost its executable bit would still work under the
    wired ``bash <path>`` form, so this pins the stronger contract of both.
    """
    assert _INTERPRETER_WRAPPER.is_file()
    assert _git_tracked(_INTERPRETER_WRAPPER_RELPATH), (
        f"{_INTERPRETER_WRAPPER_RELPATH} is untracked — every hook would fail in CI "
        "and in every fresh clone."
    )
    assert os.access(_INTERPRETER_WRAPPER, os.X_OK), (
        f"{_INTERPRETER_WRAPPER_RELPATH} is not executable"
    )


def _decoy_interpreter_dir(tmp_path: Path, name: str) -> Path:
    """A directory holding an executable ``name`` that always fails.

    Stands in for the *incapable* interpreter the wrapper must refuse. Without a
    decoy, a capability test is environment-dependent theatre: on a CI runner
    ``pip install -e .`` goes into ``setup-python``'s interpreter with no
    virtualenv, so ``python3`` on PATH is already capable and a wrapper degraded
    to ``exec python3 "$@"`` would pass. The decoy makes PATH hostile everywhere,
    which is what makes the pin fire on the runner as well as locally.
    """
    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    decoy = decoy_dir / name
    decoy.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    decoy.chmod(0o755)
    return decoy_dir


@_REQUIRES_POSIX_SHELL
@pytest.mark.parametrize("decoy_name", ["python3", "python"])
def test_interpreter_wrapper_resolves_an_interpreter_that_can_import_the_hooks(
    tmp_path: Path, decoy_name: str
) -> None:
    """An executable pin: the wrapper's *output* must be a capable interpreter.

    Asserting on the wrapper's text would pass for a wrapper that resolves the
    wrong interpreter. Running it is what proves the gates can execute.

    PATH is replaced by a directory whose only ``python3``/``python`` exits 9, so
    the wrapper can only succeed by preferring the project virtualenv over PATH.
    An earlier version of this test left the ambient PATH in place; peer review
    showed it passed on CI regardless of the wrapper's behaviour, because CI has
    no virtualenv and its PATH interpreter is already capable.
    """
    decoy_dir = _decoy_interpreter_dir(tmp_path, decoy_name)
    completed = subprocess.run(
        [
            "bash",
            str(_INTERPRETER_WRAPPER),
            "-c",
            "import tools.claude_hooks.config; print('ok')",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "CLAUDE_PROJECT_DIR": str(_REPO_ROOT),
            # Prepended, not replacing: the decoy must shadow python3/python
            # while `bash`, `cd` and `command -v` stay resolvable. Replacing PATH
            # outright made this test die with FileNotFoundError on `bash`.
            "PATH": os.pathsep.join([str(decoy_dir), os.environ.get("PATH", "")]),
        },
    )
    if completed.returncode == 9:
        pytest.fail(
            f"the wrapper exec'd the incapable {decoy_name} decoy from PATH instead of "
            "probing for one that can import the hook package — every gate fails open"
        )
    assert completed.returncode == 0, (
        f"the wrapper resolved an interpreter that cannot import the hook package — "
        f"every gate would fail open.\nstdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    assert "ok" in completed.stdout


@_REQUIRES_POSIX_SHELL
def test_interpreter_wrapper_fails_fast_with_no_arguments() -> None:
    """An argument-less wiring must error, not hang.

    Without a guard the wrapper would ``exec`` a bare interpreter, which reads
    the hook payload on stdin as a REPL script and blocks until Claude Code's
    hook timeout. A hang mid-edit is worse than a visible error, and the timeout
    is 15-90s depending on the hook.
    """
    completed = subprocess.run(
        ["bash", str(_INTERPRETER_WRAPPER)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(_REPO_ROOT)},
        stdin=subprocess.DEVNULL,
    )
    assert completed.returncode != 0
    assert "no arguments" in completed.stderr


def test_post_edit_mypy_profile_covers_the_hook_package_itself() -> None:
    """The shipped config must cover the tree whose module resolution collides.

    ``tools/claude_hooks`` is a package under a plain directory that imports
    itself as ``tools.claude_hooks.*``. Under mypy's defaults every file there
    resolves to two module names, so mypy reports that collision and checks
    nothing — the post-edit hook then answers every edit to the workforce tooling
    with a finding that is not about the edit. A wrong advisory finding is worse
    than none: it trains contributors to ignore the hook, which is how the
    fail-open bug above survived unnoticed.
    """
    profiles = _config().post_edit.mypy_profiles
    matched = [
        profile
        for profile in profiles
        if path_matches_any("tools/claude_hooks/config.py", profile.paths) is not None
    ]
    assert matched, (
        "no post_edit.mypy_profiles entry matches tools/claude_hooks/**; the "
        "edit-time mypy check reports a module-name collision instead of real findings"
    )
    profile = matched[0]
    assert "--explicit-package-bases" in profile.args, (
        f"profile {profile.paths} matches the hook package but omits "
        "--explicit-package-bases, so mypy still resolves each file twice"
    )
    assert profile.mypy_path is not None, (
        f"profile {profile.paths} needs MYPYPATH at the repo root; "
        "--explicit-package-bases alone leaves the collision in place"
    )


@_REQUIRES_POSIX_SHELL
def test_interpreter_wrapper_honours_the_explicit_override(tmp_path: Path) -> None:
    """``MOUSEDROID_PYTHON`` wins outright, so an operator can pin an interpreter.

    Pointed at a *sentinel* rather than at ``sys.executable``. An earlier version
    set ``MOUSEDROID_PYTHON=sys.executable`` and asserted the wrapper printed
    ``sys.executable`` — which in a ``.venv`` checkout is exactly what
    auto-resolution produces anyway, so the assertion could not distinguish
    "override honoured" from "override ignored". Peer review caught it by deleting
    the override branch and watching the test still pass. A sentinel that no
    resolution path could ever pick is what makes this discriminate.
    """
    sentinel = tmp_path / "sentinel-python"
    sentinel.write_text("#!/bin/sh\necho OVERRIDE_HONOURED\n", encoding="utf-8")
    sentinel.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(_INTERPRETER_WRAPPER), "-c", "import sys; print(sys.executable)"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "CLAUDE_PROJECT_DIR": str(_REPO_ROOT),
            "MOUSEDROID_PYTHON": str(sentinel),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "OVERRIDE_HONOURED", (
        "MOUSEDROID_PYTHON was ignored — the wrapper resolved its own interpreter "
        f"instead of the operator's. stdout: {completed.stdout!r}"
    )


@_REQUIRES_POSIX_SHELL
def test_interpreter_wrapper_rejects_a_non_executable_override(tmp_path: Path) -> None:
    """A typo'd override must name itself, not die as a bare shell error.

    Used verbatim without an executability check, ``MOUSEDROID_PYTHON=/no/such/py``
    reaches ``exec`` and exits 127 with only bash's "No such file or directory" —
    a message that never mentions hooks or gates, so an operator reads it as
    unrelated noise while every gate is silently bypassed.
    """
    completed = subprocess.run(
        ["bash", str(_INTERPRETER_WRAPPER), "-c", "pass"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env={
            **os.environ,
            "CLAUDE_PROJECT_DIR": str(_REPO_ROOT),
            "MOUSEDROID_PYTHON": str(tmp_path / "does-not-exist"),
        },
    )
    assert completed.returncode != 0
    assert "MOUSEDROID_PYTHON" in completed.stderr
    assert "gates cannot run" in completed.stderr


@_REQUIRES_POSIX_SHELL
@pytest.mark.parametrize("exit_code", [0, 1, 2, 42])
def test_interpreter_wrapper_preserves_the_exit_code(exit_code: int) -> None:
    """Exit-code fidelity is the whole contract Claude Code reads.

    A PreToolUse deny is exit 0 plus a JSON payload; exit 2 blocks; anything else
    non-zero is a non-blocking hook *error* — which is precisely how the original
    bug hid. A future edit replacing ``exec`` with a call plus post-processing
    would silently convert every deny into an error, so the pass-through is
    pinned rather than assumed.
    """
    completed = subprocess.run(
        ["bash", str(_INTERPRETER_WRAPPER), "-c", f"import sys; sys.exit({exit_code})"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(_REPO_ROOT)},
    )
    assert completed.returncode == exit_code, (
        f"the wrapper rewrote exit {exit_code} as {completed.returncode}; a deny or a "
        "block would be delivered as the wrong decision"
    )


@_REQUIRES_POSIX_SHELL
def test_interpreter_wrapper_passes_stdin_through_unread() -> None:
    """The hook payload arrives on stdin and must reach the hook intact.

    The capability probe runs a real interpreter, so anything it read from stdin
    would be consumed before the hook saw it — the payload would vanish and every
    hook would no-op on an empty document. ``</dev/null`` on the probe is what
    prevents that, and this is what proves it.
    """
    payload = '{"tool_name":"Edit","tool_input":{"file_path":"README.md"}}'
    completed = subprocess.run(
        ["bash", str(_INTERPRETER_WRAPPER), "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        cwd=_REPO_ROOT,
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(_REPO_ROOT)},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == payload, (
        f"stdin did not survive the wrapper: {completed.stdout!r} != {payload!r}"
    )


@_REQUIRES_POSIX_SHELL
def test_interpreter_wrapper_falls_back_to_its_own_location() -> None:
    """With ``CLAUDE_PROJECT_DIR`` unset the wrapper must still find the repo.

    Its own header advertises this as "what makes this runnable by hand and from
    the test suite", but every other test here sets the variable, so the branch
    had no coverage at all. Run from a directory that is not the repo root, so a
    wrapper relying on the ambient cwd fails.
    """
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PROJECT_DIR"}
    completed = subprocess.run(
        ["bash", str(_INTERPRETER_WRAPPER), "-c", "import tools.claude_hooks.config; print('ok')"],
        cwd=_REPO_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    assert completed.returncode == 0, (
        f"the script-relative fallback did not resolve the project root.\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )
    assert "ok" in completed.stdout


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


def _launches_the_wrapper(node: ast.FunctionDef) -> bool:
    """Whether ``node`` contains a ``subprocess.run(["bash", <wrapper>, ...])`` call.

    Matched structurally rather than by substring. A substring check on ``"bash"``
    and ``_INTERPRETER_WRAPPER`` is what the first version of this sweep used, and
    it matched the sweep functions *themselves* — they mention both names in order
    to look for them. Self-matching made every run red regardless of the code
    under test, which is the same class of mistake as a vacuous pin: the assertion
    stopped describing the tree.
    """
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "run"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "subprocess"
        ):
            continue
        if not call.args or not isinstance(call.args[0], ast.List):
            continue
        argv = call.args[0].elts
        starts_with_bash = (
            bool(argv) and isinstance(argv[0], ast.Constant) and argv[0].value == "bash"
        )
        names_wrapper = any(
            isinstance(inner, ast.Name) and inner.id == "_INTERPRETER_WRAPPER"
            for element in argv
            for inner in ast.walk(element)
        )
        if starts_with_bash and names_wrapper:
            return True
    return False


def _wrapper_subprocess_tests() -> list[ast.FunctionDef]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test_")
        and _launches_the_wrapper(node)
    ]


def test_every_wrapper_subprocess_test_is_marked_posix_only() -> None:
    """The durable half of the Windows fix: no such test may forget the marker.

    Three of these went red on the ``test-windows (3.11)`` leg because they shell
    out to ``bash``, which on that runner is WSL with no distribution installed.
    Adding the marker one test at a time is how the *fourth* one gets forgotten,
    so this discovers them from the source instead.

    Asserted against the AST rather than by running anything, so it names the
    offending test on every platform — including the Linux legs, where a missing
    marker is otherwise invisible until Windows goes red.
    """
    offenders = [
        node.name
        for node in _wrapper_subprocess_tests()
        if not any(
            isinstance(dec, ast.Name) and dec.id == "_REQUIRES_POSIX_SHELL"
            for dec in node.decorator_list
        )
    ]
    assert not offenders, (
        f"these tests shell out to run_hook.sh without @_REQUIRES_POSIX_SHELL and "
        f"will fail on the test-windows leg: {offenders}"
    )


def test_the_wrapper_subprocess_sweep_actually_finds_tests() -> None:
    """Anti-vacuity: a sweep that matches nothing passes unconditionally."""
    found = [node.name for node in _wrapper_subprocess_tests()]
    assert len(found) >= 5, f"the wrapper-subprocess sweep found only {found}"
