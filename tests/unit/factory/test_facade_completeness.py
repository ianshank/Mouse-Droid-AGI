"""Facade-completeness regression for the `factory/` package (ADR-017).

`src/mousedroid/factory.py` (5,140 lines) was split into a 19-submodule
`src/mousedroid/factory/` package plus its `__init__.py` facade (20 files
total); `factory/__init__.py` is a pure
re-export facade whose job is to make every builder defined in any
submodule reachable as `mousedroid.factory.<name>`, exactly as it was
reachable on the pre-split flat module. A missed re-export is easy to
introduce silently (add a new `build_*` to a submodule, forget the
`__init__.py` import) and easy to miss in review, because the failure
mode is a downstream `ImportError` in an unrelated caller, not a loud
signal at the point of the missing edit.

This is a *structural*, forward-looking check (not a one-time snapshot
diffed against the deleted flat file): it walks every `factory/*.py`
submodule via `ast`, collects every top-level function/class definition
whose name does not start with `_` (the facade's public contract), and
asserts each one is reachable from `mousedroid.factory` as the identical
object. A second, explicit assertion pins the 12 private names real
tests import directly (`tests/unit/factory/test_build_orchestrator_greeter.py`,
`tests/unit/factory/test_factory.py`, `tests/unit/factory/test_factory_observability.py`,
`tests/unit/factory/test_factory_esp32_discovery.py`,
`tests/regression/test_pr106_backwards_compat.py`,
`tests/unit/factory/test_build_on_device_coordinator.py`,
`tests/unit/factory/test_compose_weight_update_loader.py`,
`tests/integration/test_on_device_sim_soak.py`,
`tests/integration/test_pr134_ws4_gate_integration.py`) so a future
rename or removal of any one of them fails loudly here first.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mousedroid.factory as _factory_module

_FACTORY_DIR = Path(__file__).resolve().parents[3] / "src" / "mousedroid" / "factory"

# The 12 private symbols documented in ADR-017 / the decomposition plan as
# re-exported outside `__all__` because a real test imports them directly by
# name — mirrors config/schema/__init__.py's `_WORLD_MODEL_DEFAULT_REPO_ID`
# pattern. Pinned explicitly because private names are excluded from the
# auto-discovered public-surface check below by design.
_DOCUMENTED_PRIVATE_REEXPORTS = frozenset(
    {
        "_MAX_REPLAY_COUNT_CHUNK",
        "_count_new_replay_records",
        "_count_replay_records",
        "_load_replay_batch",
        "_load_replay_sequence_batch",
        "_build_held_out_sequence_batch",
        "_build_on_device_gate_runner",
        "_compose_weight_update_loader",
        "_build_orchestrator_greeter",
        "_resolve_bdi_weights",
        "_resolve_tracking_uri",
        "_resolve_esp32_serial_via_usbc_discovery",
    }
)


def _public_top_level_defs(path: Path) -> dict[str, ast.AST]:
    """Every module-scope function/class def in `path` not starting with `_`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defs: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and (
            not node.name.startswith("_")
        ):
            defs[node.name] = node
    return defs


def _submodule_files() -> list[Path]:
    return sorted(p for p in _FACTORY_DIR.glob("*.py") if p.name != "__init__.py")


def test_every_public_builder_is_reexported_by_the_facade() -> None:
    """Every public top-level def in a `factory/*.py` submodule is on `mousedroid.factory`
    AND listed in `__all__`.

    Forward-looking: this is what actually catches a future missed
    re-export, unlike a one-time diff against the deleted flat file.
    Checks both halves of the facade's advertised contract, not just
    `hasattr` reachability -- a builder can be imported into
    `factory/__init__.py`'s namespace (so `mousedroid.factory.build_x`
    resolves) while still missing from `__all__` (so
    `from mousedroid.factory import *` silently omits it). Neither this
    test's own reachability check alone, nor
    `test_every_all_entry_actually_resolves_on_the_module` (which only
    checks the opposite direction: every *listed* `__all__` entry
    resolves), would catch that on its own.
    """
    missing: list[str] = []
    for path in _submodule_files():
        for name in _public_top_level_defs(path):
            if not hasattr(_factory_module, name) or name not in _factory_module.__all__:
                missing.append(f"{path.name}::{name}")
    assert not missing, (
        "public symbol(s) defined in a factory/ submodule but not fully "
        "re-exported (either missing the import, or imported but absent "
        f"from __all__) from factory/__init__.py: {missing}"
    )


def test_reexported_builders_are_the_same_object_not_a_shadow_copy() -> None:
    """`mousedroid.factory.<name>` must be an identity re-export, not a redefinition.

    Guards against someone "fixing" a missing re-export by pasting a
    duplicate function body into `__init__.py` instead of importing it --
    that would pass the completeness check above while silently forking
    behavior between the two copies.
    """
    mismatched: list[str] = []
    for path in _submodule_files():
        module_name = f"mousedroid.factory.{path.stem}"
        import importlib

        submodule = importlib.import_module(module_name)
        for name in _public_top_level_defs(path):
            facade_obj = getattr(_factory_module, name, None)
            submodule_obj = getattr(submodule, name, None)
            if facade_obj is not None and facade_obj is not submodule_obj:
                mismatched.append(f"{path.name}::{name}")
    assert not mismatched, f"facade re-export is not identity-equal to source: {mismatched}"


def test_documented_private_reexports_are_all_present() -> None:
    """The 12 private names real tests import directly must stay importable."""
    missing = [
        name for name in sorted(_DOCUMENTED_PRIVATE_REEXPORTS) if not hasattr(_factory_module, name)
    ]
    assert not missing, (
        f"documented private re-export(s) missing from factory/__init__.py: {missing} "
        "-- a real test imports these by name (see module docstring)"
    )


def test_documented_private_reexports_are_excluded_from_all() -> None:
    """The 12 private names are re-exported but never advertised in `__all__`.

    Matches config/schema/__init__.py's established convention: present
    for direct import, excluded from the public `__all__` surface.
    """
    exported_in_all = sorted(
        name for name in _DOCUMENTED_PRIVATE_REEXPORTS if name in _factory_module.__all__
    )
    assert not exported_in_all, (
        f"private re-export(s) unexpectedly listed in __all__: {exported_in_all}"
    )


def test_every_all_entry_actually_resolves_on_the_module() -> None:
    """The inverse of the completeness check above: `__all__` must not overclaim.

    `test_every_public_builder_is_reexported_by_the_facade` only checks that
    every real submodule symbol made it INTO `__all__`; it says nothing about
    whether every NAME LISTED in `__all__` actually resolves. Those are
    different failure modes -- an `__all__` entry can go dangling on its own
    (e.g. from a leftover string literal after the import backing it was
    removed) with no missing-re-export symptom at all. Neither `mypy --strict`
    nor `ruff --select F822` catches this for a plain `__init__.py` module
    (confirmed empirically), and `from mousedroid.factory import *` -- the
    one thing `__all__` actually controls -- raises `AttributeError` at
    import time for anyone who does it. Historically confirmed non-vacuous:
    a `TYPE_CHECKING`/`TypeVar` cleanup once left exactly these two names
    stringly-referenced in `__all__` with nothing behind them.
    """
    dangling = sorted(
        name for name in _factory_module.__all__ if not hasattr(_factory_module, name)
    )
    assert not dangling, (
        f"__all__ entry/entries do not resolve on mousedroid.factory: {dangling} -- "
        "`from mousedroid.factory import *` raises AttributeError for these"
    )


def test_every_submodule_stays_accessible_as_a_facade_attribute() -> None:
    """Every real `factory/*.py` submodule must resolve as `mousedroid.factory.<name>`.

    An earlier version of `factory/__init__.py` explicitly `del`eted each
    submodule name (`arm`, `orchestrator`, ...) from its own namespace right
    after using it, purely to match the pre-split flat module's `dir()`
    output. No test pinned that goal, and it produced a real, version-
    dependent bug: several tests correctly patch
    `mousedroid.factory.orchestrator.build_cognitive_core` rather than the
    facade re-export (see `test_factory.py`, "patch where it's used, not
    where it's defined"). `unittest.mock.patch`'s dotted-path resolver falls
    back to a plain `import mousedroid.factory.orchestrator` when the
    attribute is missing -- but once a submodule is already cached in
    `sys.modules`, that fallback re-import is a no-op and does NOT restore
    the deleted attribute on Python 3.10 (confirmed: it does on 3.11+, a
    real interpreter behavior difference). The patch raised
    `AttributeError: module 'mousedroid.factory' has no attribute
    'orchestrator'` on Python 3.10 CI as a direct result -- reproducible in
    a single fresh process with zero other tests run first, so it was never
    a matter of test order. This test pins the fix (leave submodules
    visible) in a way that fails on every Python version, not just 3.10.
    """
    missing = sorted(
        path.stem for path in _submodule_files() if not hasattr(_factory_module, path.stem)
    )
    assert not missing, (
        f"submodule(s) not accessible as mousedroid.factory.<name>: {missing} -- "
        "a mock.patch('mousedroid.factory.<name>.<symbol>') target for any of these "
        "can silently break depending on Python version (see this test's docstring)"
    )
