# tests/unit/tools/claude_hooks/test_config.py
"""Unit tests for the workforce configuration schema.

Two contracts matter most here: a missing or empty config must still yield a
usable object (backwards compatibility), and an unknown key must fail loudly
rather than silently disabling a gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.claude_hooks.config import (
    DEFAULT_CONFIG_RELPATH,
    ConfigError,
    WorkforceConfig,
    load_config,
)


def _write_config(root: Path, body: str) -> Path:
    path = root / DEFAULT_CONFIG_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Defaults + backwards compatibility
# ---------------------------------------------------------------------------


def test_missing_file_yields_defaults(tmp_path: Path) -> None:
    cfg = load_config(repo_root=tmp_path)
    assert isinstance(cfg, WorkforceConfig)
    assert cfg.freeze.feature_key == "F-008"
    assert cfg.coverage.tools_line_min == 85
    assert cfg.secret_scan.strict is False
    assert cfg.ratchet_budgets.enabled is True
    assert {item.name for item in cfg.ratchet_budgets.items} == {
        "noqa",
        "type_ignore",
        "hardcoded_ok",
    }


def test_empty_file_yields_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "")
    assert load_config(repo_root=tmp_path).freeze.enabled is True


def test_comment_only_file_yields_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "# nothing but a comment\n")
    assert load_config(repo_root=tmp_path).docs.core_max_lines == 250


def test_partial_config_keeps_defaults_for_absent_sections(tmp_path: Path) -> None:
    _write_config(tmp_path, "freeze:\n    feature_key: F-999\n")
    cfg = load_config(repo_root=tmp_path)
    assert cfg.freeze.feature_key == "F-999"
    # Untouched sections keep their defaults — a config written before a section
    # existed still loads.
    assert cfg.coverage.tools_line_min == 85
    assert cfg.secret_scan.command == "gitleaks"


def test_explicit_path_overrides_repo_root(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text("docs:\n    core_max_lines: 42\n", encoding="utf-8")
    assert load_config(custom, repo_root=tmp_path).docs.core_max_lines == 42


def test_repo_config_file_is_valid() -> None:
    """The checked-in .claude/workforce.yaml must satisfy its own schema."""
    repo_root = Path(__file__).resolve().parents[4]
    cfg = load_config(repo_root=repo_root)
    assert cfg.freeze.frozen_paths, "repo config should declare at least one frozen path"
    assert cfg.freeze.feature_key.startswith("F-")


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "unknown_section:\n    a: 1\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(repo_root=tmp_path)
    assert "unknown_section" in str(excinfo.value)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    # The exact typo this guard exists for: frozen_path vs frozen_paths.
    _write_config(tmp_path, "freeze:\n    frozen_path:\n        - src/**\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(repo_root=tmp_path)
    assert "frozen_path" in str(excinfo.value)


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "freeze: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(repo_root=tmp_path)


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(repo_root=tmp_path)


def test_unreadable_file_is_rejected(tmp_path: Path) -> None:
    # A directory where a file is expected: read_text raises OSError.
    path = tmp_path / DEFAULT_CONFIG_RELPATH
    path.mkdir(parents=True)
    # is_file() is False for a directory, so this falls through to defaults.
    assert load_config(repo_root=tmp_path).freeze.enabled is True


@pytest.mark.parametrize(
    "body",
    [
        "coverage:\n    tools_line_min: 101\n",
        "coverage:\n    tools_line_min: -1\n",
        "secret_scan:\n    timeout_s: 0\n",
        "secret_scan:\n    timeout_s: -5\n",
        "secret_scan:\n    max_bytes: 0\n",
        "docs:\n    core_max_lines: 0\n",
        "evidence:\n    stale_after_days: 0\n",
        "agents:\n    max_lines: 0\n",
        "post_edit:\n    timeout_s: 100000\n",
    ],
)
def test_out_of_range_values_are_rejected(tmp_path: Path, body: str) -> None:
    _write_config(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "freeze:\n    feature_key: '   '\n",
        "freeze:\n    features_file: ''\n",
        "freeze:\n    override_env: '  '\n",
        "freeze:\n    done_status: ''\n",
    ],
)
def test_blank_freeze_identifiers_are_rejected(tmp_path: Path, body: str) -> None:
    _write_config(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize(
    "features_file",
    ["/etc/features.yaml", "../features.yaml", "a/../../features.yaml"],
)
def test_non_relative_features_file_is_rejected(tmp_path: Path, features_file: str) -> None:
    _write_config(tmp_path, f"freeze:\n    features_file: '{features_file}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "freeze:\n    frozen_paths: not-a-list\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize(
    "config_file",
    ["/etc/passwd", "../../etc/passwd", "a/../../outside.toml"],
)
def test_non_relative_secret_scan_config_is_rejected(tmp_path: Path, config_file: str) -> None:
    """The allowlist path is joined onto the repo root before the scan runs.

    An absolute or traversing value would point the scanner's config at an
    arbitrary file, so it gets the same guard as freeze.features_file.
    """
    _write_config(tmp_path, f"secret_scan:\n    config_file: '{config_file}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_relative_secret_scan_config_is_accepted(tmp_path: Path) -> None:
    _write_config(tmp_path, "secret_scan:\n    config_file: custom/.gitleaks.toml\n")
    assert load_config(repo_root=tmp_path).secret_scan.config_file == "custom/.gitleaks.toml"


# ---------------------------------------------------------------------------
# post_edit.mypy_profiles
# ---------------------------------------------------------------------------


def test_mypy_profiles_default_covers_the_tools_tree(tmp_path: Path) -> None:
    """The shipped default must fix the tree that needs fixing.

    ``tools/claude_hooks`` is a package under a plain directory that imports
    itself as ``tools.claude_hooks.*``. Without ``--explicit-package-bases`` and
    ``MYPYPATH`` at the repo root, mypy finds each file twice and checks nothing
    — the "Source file found twice under different module names" error is then
    the *only* finding the post-edit hook reports for a real defect.
    """
    profiles = load_config(repo_root=tmp_path).post_edit.mypy_profiles
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.paths == ["tools/**"]
    assert "--explicit-package-bases" in profile.args
    assert profile.mypy_path == "."


def test_legacy_config_without_mypy_profiles_keeps_src_behaviour(tmp_path: Path) -> None:
    """A config written before the field existed still loads, unchanged.

    Invariant I-1: the field is additive. A pre-existing ``post_edit`` block that
    sets only ``mypy_args`` keeps that value, and picks up the default profile.
    """
    _write_config(
        tmp_path,
        "post_edit:\n    mypy_args:\n        - --ignore-missing-imports\n",
    )
    post_edit = load_config(repo_root=tmp_path).post_edit
    assert post_edit.mypy_args == ["--ignore-missing-imports"]
    assert [profile.paths for profile in post_edit.mypy_profiles] == [["tools/**"]]


def test_mypy_profiles_round_trip_from_yaml(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "post_edit:\n"
        "    mypy_profiles:\n"
        "        - paths:\n"
        "              - other/**\n"
        "          args:\n"
        "              - --strict\n"
        "          mypy_path: stubs\n",
    )
    profiles = load_config(repo_root=tmp_path).post_edit.mypy_profiles
    assert len(profiles) == 1
    assert profiles[0].paths == ["other/**"]
    assert profiles[0].args == ["--strict"]
    assert profiles[0].mypy_path == "stubs"


def test_empty_mypy_profiles_list_is_accepted(tmp_path: Path) -> None:
    """Opting out is legal: every file then uses ``mypy_args`` unchanged."""
    _write_config(tmp_path, "post_edit:\n    mypy_profiles: []\n")
    assert load_config(repo_root=tmp_path).post_edit.mypy_profiles == []


def test_mypy_profile_without_mypy_path_inherits_the_environment(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "post_edit:\n    mypy_profiles:\n        - paths:\n              - x/**\n",
    )
    assert load_config(repo_root=tmp_path).post_edit.mypy_profiles[0].mypy_path is None


@pytest.mark.parametrize(
    "mypy_path",
    [
        "/home/someone/repo",
        "/etc",
        # Windows drive-letter forms. PurePosixPath(...).is_absolute() is False
        # for both, so a POSIX-only validator accepted them -- and on Windows
        # `repo_root / "C:\\Users\\me"` IS "C:\\Users\\me", so MYPYPATH landed
        # off-checkout. Caught by peer review; these two are the pin.
        "C:\\Users\\me\\repo",
        "C:/Users/me/repo",
        # Both escape the checkout the value is joined onto.
        "~/elsewhere",
        "../../..",
    ],
)
def test_absolute_mypy_path_is_rejected(tmp_path: Path, mypy_path: str) -> None:
    """Invariant I-3: an absolute MYPYPATH pins the config to one checkout."""
    _write_config(
        tmp_path,
        f"post_edit:\n    mypy_profiles:\n        - paths:\n              - x/**\n"
        f"          mypy_path: '{mypy_path}'\n",
    )
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_mypy_profile_without_paths_is_rejected(tmp_path: Path) -> None:
    """A profile with no paths can never match, so it must not load quietly.

    ``extra="forbid"`` already makes a mistyped *key* loud; an omitted ``paths``
    was silent, which is the same class of mistake with the opposite outcome.
    """
    _write_config(tmp_path, "post_edit:\n    mypy_profiles:\n        - args: [--strict]\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_mypy_profile_with_empty_paths_list_is_rejected(tmp_path: Path) -> None:
    """Explicitly empty is the same inert profile, spelled differently."""
    _write_config(
        tmp_path,
        "post_edit:\n    mypy_profiles:\n        - paths: []\n          args: [--strict]\n",
    )
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_mypy_profile_unknown_key_is_rejected(tmp_path: Path) -> None:
    """``extra='forbid'``: a typo must not silently disable the profile."""
    _write_config(
        tmp_path,
        "post_edit:\n    mypy_profiles:\n        - paths:\n              - x/**\n"
        "          mypypath: .\n",
    )
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


# ---------------------------------------------------------------------------
# ratchet_budgets
# ---------------------------------------------------------------------------


def test_partial_ratchet_budgets_override_keeps_other_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "ratchet_budgets:\n    enabled: false\n")
    cfg = load_config(repo_root=tmp_path)
    assert cfg.ratchet_budgets.enabled is False
    # Untouched: the default three-item budget list survives.
    assert len(cfg.ratchet_budgets.items) == 3


def test_ratchet_budgets_items_round_trip(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "ratchet_budgets:\n"
        "    items:\n"
        "        - name: custom\n"
        "          marker: '# custom-marker'\n"
        "          ceiling: 3\n",
    )
    cfg = load_config(repo_root=tmp_path)
    assert len(cfg.ratchet_budgets.items) == 1
    item = cfg.ratchet_budgets.items[0]
    assert item.name == "custom"
    assert item.ceiling == 3
    assert item.warn_threshold is None
    # scope_glob keeps its own default when the override omits it.
    assert item.scope_glob == "src/mousedroid/**/*.py"


def test_ratchet_budget_item_warn_threshold_above_ceiling_is_rejected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "ratchet_budgets:\n"
        "    items:\n"
        "        - name: custom\n"
        "          marker: x\n"
        "          ceiling: 3\n"
        "          warn_threshold: 5\n",
    )
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "ratchet_budgets:\n"
        "    items:\n"
        "        - name: ''\n"
        "          marker: x\n"
        "          ceiling: 1\n",
        "ratchet_budgets:\n"
        "    items:\n"
        "        - name: x\n"
        "          marker: '  '\n"
        "          ceiling: 1\n",
    ],
)
def test_ratchet_budget_item_blank_identifiers_are_rejected(tmp_path: Path, body: str) -> None:
    _write_config(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_ratchet_budget_item_negative_ceiling_is_rejected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "ratchet_budgets:\n"
        "    items:\n"
        "        - name: x\n"
        "          marker: x\n"
        "          ceiling: -1\n",
    )
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_ratchet_budgets_unknown_key_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "ratchet_budgets:\n    max_items: 5\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(repo_root=tmp_path)
    assert "max_items" in str(excinfo.value)


# ---------------------------------------------------------------------------
# credit (F-059: spend caps + ledger location for tools/credit_ledger.py)
# ---------------------------------------------------------------------------


def test_credit_defaults_fail_closed_with_no_caps(tmp_path: Path) -> None:
    """No ``credit:`` block => no workstream has a cap => nothing may spend."""
    cfg = load_config(repo_root=tmp_path)
    assert cfg.credit.caps_usd == {}
    assert cfg.credit.ledger == "reports/credit/ledger.jsonl"


def test_credit_block_round_trips(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "credit:\n"
        "    caps_usd:\n"
        "        A: 10\n"
        "        B: 60.5\n"
        "    ledger: reports/x/spend.jsonl\n",
    )
    cfg = load_config(repo_root=tmp_path)
    assert cfg.credit.caps_usd == {"A": 10.0, "B": 60.5}
    assert cfg.credit.ledger == "reports/x/spend.jsonl"


def test_credit_partial_block_keeps_the_default_ledger_path(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    caps_usd:\n        A: 1\n")
    assert load_config(repo_root=tmp_path).credit.ledger == "reports/credit/ledger.jsonl"


def test_credit_does_not_disturb_other_sections(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    caps_usd:\n        A: 1\n")
    cfg = load_config(repo_root=tmp_path)
    assert cfg.freeze.feature_key == "F-008"
    assert cfg.coverage.tools_line_min == 85


def test_credit_unknown_key_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    cap_usd:\n        A: 1\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(repo_root=tmp_path)
    assert "cap_usd" in str(excinfo.value)


@pytest.mark.parametrize("cap", ["-1", ".nan", ".inf", "-.inf"])
def test_credit_cap_must_be_a_finite_non_negative_number(tmp_path: Path, cap: str) -> None:
    _write_config(tmp_path, f"credit:\n    caps_usd:\n        A: {cap}\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_credit_zero_cap_is_allowed_and_disables_a_workstream(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    caps_usd:\n        A: 0\n")
    assert load_config(repo_root=tmp_path).credit.caps_usd == {"A": 0.0}


def test_credit_blank_workstream_name_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    caps_usd:\n        ' ': 5\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize("ledger", ["/etc/ledger.jsonl", "C:/ledger.jsonl", "../outside.jsonl"])
def test_credit_ledger_path_must_stay_inside_the_repo(tmp_path: Path, ledger: str) -> None:
    _write_config(tmp_path, f"credit:\n    ledger: '{ledger}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_credit_blank_ledger_path_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    ledger: ' '\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_repo_config_declares_the_credit_block() -> None:
    """The checked-in workforce.yaml carries caps, so the ledger can refuse overspend."""
    repo_root = Path(__file__).resolve().parents[4]
    cfg = load_config(repo_root=repo_root)
    assert cfg.credit.caps_usd, "credit.caps_usd must be declared in .claude/workforce.yaml"
    assert all(cap > 0 for cap in cfg.credit.caps_usd.values())


# ---------------------------------------------------------------------------
# agent_reviewer (F-059 WS-B: the read-only Agent-SDK peer reviewer)
# ---------------------------------------------------------------------------


def test_agent_reviewer_defaults(tmp_path: Path) -> None:
    """No ``agent_reviewer:`` block => the planned reviewer settings."""
    reviewer = load_config(repo_root=tmp_path).agent_reviewer
    assert reviewer.ws == "B"
    assert reviewer.model == "opus"
    assert reviewer.max_turns == 40
    assert reviewer.agent_file == ".claude/agents/peer-reviewer.md"
    assert reviewer.mcp_config == ".mcp.json"
    assert reviewer.artifact_dir == "reports/agent_reviewer"
    assert reviewer.resolve_timeout_s > 0
    assert reviewer.run_timeout_s > 0


def test_agent_reviewer_block_round_trips(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "agent_reviewer:\n"
        "    ws: C\n"
        "    model: sonnet\n"
        "    max_turns: 7\n"
        "    agent_file: agents/x.md\n"
        "    mcp_config: cfg/mcp.json\n"
        "    artifact_dir: out/reviews\n"
        "    resolve_timeout_s: 3.5\n"
        "    run_timeout_s: 90\n",
    )
    reviewer = load_config(repo_root=tmp_path).agent_reviewer
    assert (reviewer.ws, reviewer.model, reviewer.max_turns) == ("C", "sonnet", 7)
    assert reviewer.agent_file == "agents/x.md"
    assert reviewer.mcp_config == "cfg/mcp.json"
    assert reviewer.artifact_dir == "out/reviews"
    assert reviewer.resolve_timeout_s == 3.5
    assert reviewer.run_timeout_s == 90.0


def test_agent_reviewer_partial_block_keeps_the_other_defaults(tmp_path: Path) -> None:
    _write_config(tmp_path, "agent_reviewer:\n    max_turns: 12\n")
    reviewer = load_config(repo_root=tmp_path).agent_reviewer
    assert reviewer.max_turns == 12
    assert reviewer.model == "opus"
    assert reviewer.artifact_dir == "reports/agent_reviewer"


def test_agent_reviewer_does_not_disturb_other_sections(tmp_path: Path) -> None:
    _write_config(tmp_path, "agent_reviewer:\n    max_turns: 12\n")
    cfg = load_config(repo_root=tmp_path)
    assert cfg.freeze.feature_key == "F-008"
    assert cfg.credit.caps_usd == {}


def test_agent_reviewer_unknown_key_is_rejected(tmp_path: Path) -> None:
    _write_config(tmp_path, "agent_reviewer:\n    max_turn: 12\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(repo_root=tmp_path)
    assert "max_turn" in str(excinfo.value)


@pytest.mark.parametrize("turns", ["0", "-3"])
def test_agent_reviewer_max_turns_must_be_positive(tmp_path: Path, turns: str) -> None:
    """The SDK reads ``max_turns=0`` as "no limit", so 0 must never reach it."""
    _write_config(tmp_path, f"agent_reviewer:\n    max_turns: {turns}\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize("field", ["ws", "model"])
def test_agent_reviewer_blank_identifiers_are_rejected(tmp_path: Path, field: str) -> None:
    _write_config(tmp_path, f"agent_reviewer:\n    {field}: ' '\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize("ws", ["../x", "a b", "B/C", "B.md", "x" * 17])
def test_agent_reviewer_ws_is_a_plain_name(tmp_path: Path, ws: str) -> None:
    """The workstream builds run ids and file names, so it may not carry a path."""
    _write_config(tmp_path, f"agent_reviewer:\n    ws: '{ws}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize("model", ["-x", "--dangerously-skip-permissions", "a b", "opus;x"])
def test_agent_reviewer_model_cannot_look_like_a_cli_option(tmp_path: Path, model: str) -> None:
    """The alias is passed as ``--model <value>``; a leading dash would read as an option."""
    _write_config(tmp_path, f"agent_reviewer:\n    model: '{model}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize("model", ["opus", "claude-opus-5-5", "anthropic.claude-x:0"])
def test_agent_reviewer_model_accepts_aliases_and_ids(tmp_path: Path, model: str) -> None:
    _write_config(tmp_path, f"agent_reviewer:\n    model: '{model}'\n")
    assert load_config(repo_root=tmp_path).agent_reviewer.model == model


@pytest.mark.parametrize("field", ["agent_file", "mcp_config", "artifact_dir"])
@pytest.mark.parametrize("path", ["/etc/x", "C:/x", "../x", " "])
def test_agent_reviewer_paths_must_stay_inside_the_repo(
    tmp_path: Path, field: str, path: str
) -> None:
    _write_config(tmp_path, f"agent_reviewer:\n    {field}: '{path}'\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


@pytest.mark.parametrize("field", ["resolve_timeout_s", "run_timeout_s"])
@pytest.mark.parametrize("value", ["0", "-1", ".nan", ".inf"])
def test_agent_reviewer_timeouts_must_be_positive_and_finite(
    tmp_path: Path, field: str, value: str
) -> None:
    _write_config(tmp_path, f"agent_reviewer:\n    {field}: {value}\n")
    with pytest.raises(ConfigError):
        load_config(repo_root=tmp_path)


def test_repo_config_declares_the_agent_reviewer_block() -> None:
    """The checked-in workforce.yaml states the reviewer's turn cap and model alias."""
    repo_root = Path(__file__).resolve().parents[4]
    raw = (repo_root / DEFAULT_CONFIG_RELPATH).read_text(encoding="utf-8")
    assert "\nagent_reviewer:" in raw, "agent_reviewer must be declared in .claude/workforce.yaml"
    reviewer = load_config(repo_root=repo_root).agent_reviewer
    assert reviewer.max_turns == 40
    assert reviewer.model == "opus"
