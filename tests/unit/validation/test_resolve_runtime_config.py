"""Source-reporting runtime config resolution (F-059 task 4.7).

``resolve_runtime_config_paths`` answers "which overlays?". The re-arm CLI also has to
say *where that answer came from*, because a silent fall-through to ``default.yaml`` is
exactly the no-op task 4.7 closes. ``resolve_runtime_config`` is the same precedence with
the provenance kept, and ``resolve_runtime_config_paths`` is defined as its ``.paths`` so
the two can never disagree about which overlays a process loads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.validation.runtime import (
    RUNTIME_CONFIG_ENV_VARS,
    RuntimeConfigResolution,
    resolve_runtime_config,
    resolve_runtime_config_paths,
)

_CSV_VARS = ("MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS")
_SINGLE_VARS = ("MOUSEDROID_CONFIG", "MOUSEDROID_JETSON_CONFIG")


@pytest.fixture(autouse=True)
def _no_ambient_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's shell must not decide what these tests resolve."""
    for var in RUNTIME_CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_the_env_var_names_are_the_four_the_runtime_documents() -> None:
    """Precedence order, CSV lists first: this tuple is what the CLI's hint prints."""
    assert (*_CSV_VARS, *_SINGLE_VARS) == RUNTIME_CONFIG_ENV_VARS


def test_nothing_set_resolves_to_the_default() -> None:
    assert resolve_runtime_config() == RuntimeConfigResolution(
        paths=(), source="default", env_var=None
    )


def test_explicit_paths_win_and_report_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", "a.yaml,b.yaml")
    monkeypatch.setenv("MOUSEDROID_CONFIG", "c.yaml")

    resolved = resolve_runtime_config(["x.yaml", Path("y.yaml")])

    assert resolved == RuntimeConfigResolution(
        paths=(Path("x.yaml"), Path("y.yaml")), source="cli", env_var=None
    )


@pytest.mark.parametrize("var", _CSV_VARS)
def test_a_csv_list_var_reports_env_csv_and_which_var(
    monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    monkeypatch.setenv(var, "a.yaml, b.yaml ,,c.yaml")

    resolved = resolve_runtime_config()

    assert resolved.source == "env_csv"
    assert resolved.env_var == var
    assert resolved.paths == (Path("a.yaml"), Path("b.yaml"), Path("c.yaml"))


@pytest.mark.parametrize("var", _SINGLE_VARS)
def test_a_single_path_var_reports_env_single_and_which_var(
    monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    monkeypatch.setenv(var, "  /etc/mousedroid/jetson_production.yaml  ")

    resolved = resolve_runtime_config()

    assert resolved.source == "env_single"
    assert resolved.env_var == var
    assert resolved.paths == (Path("/etc/mousedroid/jetson_production.yaml"),)


def test_a_csv_list_outranks_a_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_CONFIG", "single.yaml")
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", "list.yaml")

    resolved = resolve_runtime_config()

    assert (resolved.source, resolved.paths) == ("env_csv", (Path("list.yaml"),))


def test_the_two_csv_vars_and_the_two_single_vars_rank_in_documented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", "jetson-list.yaml")
    monkeypatch.setenv("MOUSEDROID_CONFIGS", "list.yaml")
    assert resolve_runtime_config().env_var == "MOUSEDROID_CONFIGS"

    monkeypatch.delenv("MOUSEDROID_CONFIGS")
    monkeypatch.delenv("MOUSEDROID_JETSON_CONFIGS")
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIG", "jetson-single.yaml")
    monkeypatch.setenv("MOUSEDROID_CONFIG", "single.yaml")
    assert resolve_runtime_config().env_var == "MOUSEDROID_CONFIG"


def test_blank_values_fall_through_rather_than_resolving_to_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set-but-blank var is not an overlay; the next tier still gets its turn."""
    monkeypatch.setenv("MOUSEDROID_CONFIGS", " , ,")
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", "   ")
    monkeypatch.setenv("MOUSEDROID_CONFIG", "single.yaml")

    resolved = resolve_runtime_config(["", "  "])

    assert (resolved.source, resolved.env_var) == ("env_single", "MOUSEDROID_CONFIG")
    assert resolved.paths == (Path("single.yaml"),)


def test_blank_everything_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_CONFIG", "   ")

    assert resolve_runtime_config(["  "]).source == "default"


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"MOUSEDROID_CONFIG": "a.yaml"},
        {"MOUSEDROID_JETSON_CONFIG": "a.yaml"},
        {"MOUSEDROID_CONFIGS": "a.yaml,b.yaml", "MOUSEDROID_CONFIG": "c.yaml"},
        {"MOUSEDROID_JETSON_CONFIGS": "a.yaml, b.yaml"},
        {"MOUSEDROID_CONFIGS": " ,", "MOUSEDROID_CONFIG": "c.yaml"},
    ],
)
@pytest.mark.parametrize("explicit", [None, [], ["x.yaml"], ["x.yaml", "y.yaml"]])
def test_the_paths_helper_is_exactly_the_resolutions_paths(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    explicit: list[str] | None,
) -> None:
    """One implementation of the precedence: the two entry points cannot drift."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert resolve_runtime_config_paths(explicit) == resolve_runtime_config(explicit).paths
