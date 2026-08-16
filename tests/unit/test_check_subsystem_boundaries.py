"""Tests for scripts/check_subsystem_boundaries.py.

Loads the script as a module (mirrors ``test_check_no_hardcoded_values.py``)
and exercises its AST/introspection logic. The positive-control test
reproduces the exact shape of the real violation this checker was written
to catch (learning/offline_rl.py importing NoOpExperimentLogger at module
scope, before its fix) to prove it would have been flagged.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pydantic
import pytest


def _load_checker_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parent.parent.parent
    script_path = repo_root / "scripts" / "check_subsystem_boundaries.py"

    spec = importlib.util.spec_from_file_location("check_subsystem_boundaries", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    # The script defines a @dataclass(frozen=True) class; dataclasses'
    # forward-ref resolution looks the module up by name in sys.modules
    # while exec_module runs, so it must be registered first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load_checker_module()


def _tree(source: str) -> ast.Module:
    return ast.parse(source)


class TestModuleLevelImportExtraction:
    def test_finds_top_level_mousedroid_import(self, checker: ModuleType) -> None:
        tree = _tree("from mousedroid.safety.monitor import SafetyMonitor\n")
        found = checker._module_level_mousedroid_imports(tree)
        assert len(found) == 1
        assert found[0].module == "mousedroid.safety.monitor"

    def test_ignores_non_mousedroid_import(self, checker: ModuleType) -> None:
        tree = _tree("from typing import Any\n")
        assert checker._module_level_mousedroid_imports(tree) == []

    def test_ignores_function_scoped_import(self, checker: ModuleType) -> None:
        """Deferred imports are the established, endorsed escape hatch."""
        tree = _tree(
            "def build() -> None:\n"
            "    from mousedroid.telemetry.failure_recorder import NullFailureRecorder\n"
            "    return NullFailureRecorder()\n"
        )
        assert checker._module_level_mousedroid_imports(tree) == []

    def test_ignores_type_checking_block(self, checker: ModuleType) -> None:
        tree = _tree(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from mousedroid.training.observability import ExperimentLoggerProtocol\n"
        )
        assert checker._module_level_mousedroid_imports(tree) == []

    def test_ignores_typing_dot_type_checking_block(self, checker: ModuleType) -> None:
        tree = _tree(
            "import typing\nif typing.TYPE_CHECKING:\n    from mousedroid.foo.bar import Baz\n"
        )
        assert checker._module_level_mousedroid_imports(tree) == []


class TestSharedKernelAndProtocolExemptions:
    def test_config_schema_is_shared_kernel(self, checker: ModuleType) -> None:
        assert checker._is_shared_kernel("mousedroid.config.schema")

    def test_config_loader_is_shared_kernel(self, checker: ModuleType) -> None:
        assert checker._is_shared_kernel("mousedroid.config.loader")

    def test_unrelated_config_submodule_is_not_shared_kernel(self, checker: ModuleType) -> None:
        assert not checker._is_shared_kernel("mousedroid.config.migration_helpers")

    def test_protocol_module_detected(self, checker: ModuleType) -> None:
        assert checker._is_protocol_module("mousedroid.safety.protocol")
        assert checker._is_protocol_module("mousedroid.arm.protocols")
        assert checker._is_protocol_module("mousedroid.harness.approval.protocol")

    def test_non_protocol_module_not_detected(self, checker: ModuleType) -> None:
        assert not checker._is_protocol_module("mousedroid.safety.monitor")


# Fixture types for the structural-exemption checks below — real objects,
# not stubs, so `_is_dto_enum_exception_or_protocol` runs its actual
# isinstance/issubclass logic rather than a mock standing in for it.
@dataclasses.dataclass
class _FixtureDataclass:
    x: int = 0


class _FixtureEnum(enum.Enum):
    A = 1


class _FixtureModel(pydantic.BaseModel):
    x: int = 0


class _FixtureError(Exception):
    pass


class _FixtureProtocol(Protocol):
    def foo(self) -> None: ...


class _FixtureConcreteClass:
    pass


class TestStructuralExemption:
    def test_dataclass_is_exempt(self, checker: ModuleType) -> None:
        assert checker._is_dto_enum_exception_or_protocol(_FixtureDataclass)

    def test_enum_is_exempt(self, checker: ModuleType) -> None:
        assert checker._is_dto_enum_exception_or_protocol(_FixtureEnum)

    def test_pydantic_basemodel_is_exempt(self, checker: ModuleType) -> None:
        assert checker._is_dto_enum_exception_or_protocol(_FixtureModel)

    def test_exception_subclass_is_exempt(self, checker: ModuleType) -> None:
        assert checker._is_dto_enum_exception_or_protocol(_FixtureError)

    def test_protocol_subclass_is_exempt(self, checker: ModuleType) -> None:
        assert checker._is_dto_enum_exception_or_protocol(_FixtureProtocol)

    def test_plain_concrete_class_is_not_exempt(self, checker: ModuleType) -> None:
        assert not checker._is_dto_enum_exception_or_protocol(_FixtureConcreteClass)


class TestNeedsReview:
    def test_real_exception_is_exempt(self, checker: ModuleType) -> None:
        assert not checker._needs_review("builtins", "ValueError")

    def test_real_concrete_class_is_flagged(self, checker: ModuleType) -> None:
        assert checker._needs_review("collections", "OrderedDict")

    def test_plain_function_is_not_a_concrete_type(self, checker: ModuleType) -> None:
        assert not checker._needs_review("os.path", "join")

    def test_unresolvable_import_fails_closed(self, checker: ModuleType) -> None:
        assert checker._needs_review("mousedroid.this_subsystem_does_not_exist", "Whatever")


class TestPositiveControl:
    """Reproduce the exact real violation this checker was added to catch."""

    def test_flags_the_original_offline_rl_violation_shape(
        self, checker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before its fix, learning/offline_rl.py did exactly this.

        A module-level import of the concrete NoOpExperimentLogger class
        from a peer subsystem (training.observability), used as a
        constructor default. Resolution hits the REAL, already-installed
        mousedroid.training.observability.noop_logger.NoOpExperimentLogger
        — a genuine concrete class, correctly not a dataclass, Enum,
        exception, or Protocol.
        """
        fake_learning = tmp_path / "src" / "mousedroid" / "learning"
        fake_learning.mkdir(parents=True)
        fixture_file = fake_learning / "offline_rl_fixture.py"
        fixture_file.write_text(
            "from mousedroid.training.observability.noop_logger import NoOpExperimentLogger\n"
            "\n"
            "\n"
            "class Trainer:\n"
            "    def __init__(self, logger=None):\n"
            "        self._logger = logger or NoOpExperimentLogger()\n"
        )

        monkeypatch.setattr(checker, "_SRC", tmp_path / "src" / "mousedroid")
        monkeypatch.setattr(checker, "_REPO_ROOT", tmp_path)

        violations = checker._check_file(fixture_file, frozenset({"learning", "training"}))

        assert len(violations) == 1
        assert violations[0].names == ("NoOpExperimentLogger",)
        assert violations[0].module == "mousedroid.training.observability.noop_logger"

    def test_deferred_variant_of_the_same_import_is_not_flagged(
        self, checker: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fixed shape (TYPE_CHECKING + function-scoped import) must pass clean."""
        fake_learning = tmp_path / "src" / "mousedroid" / "learning"
        fake_learning.mkdir(parents=True)
        fixture_file = fake_learning / "offline_rl_fixture.py"
        fixture_file.write_text(
            "from __future__ import annotations\n"
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from mousedroid.training.observability import ExperimentLoggerProtocol\n"
            "\n"
            "\n"
            "class Trainer:\n"
            "    def __init__(self, logger: ExperimentLoggerProtocol | None = None) -> None:\n"
            "        if logger is None:\n"
            "            from mousedroid.training.observability import NoOpExperimentLogger\n"
            "\n"
            "            logger = NoOpExperimentLogger()\n"
            "        self._logger = logger\n"
        )

        monkeypatch.setattr(checker, "_SRC", tmp_path / "src" / "mousedroid")
        monkeypatch.setattr(checker, "_REPO_ROOT", tmp_path)

        violations = checker._check_file(fixture_file, frozenset({"learning", "training"}))
        assert violations == []
