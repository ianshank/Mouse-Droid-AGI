"""Unit tests for ``scripts/generate_package_map.py`` against a fixture tree.

Correctness of the F-053 package import map comes from these tests, not from
rendering Mermaid (no renderer exists in the toolchain). The fixture tree is
small enough to assert exact edges, including the ``TYPE_CHECKING`` filter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._script_loader import load_script_module

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "package_map"
_SRC = _FIXTURES / "src" / "mousedroid"
_EPICS = _FIXTURES / "epics.yaml"


@pytest.fixture(scope="module")
def gen():
    return load_script_module("generate_package_map")


class TestDiscoverAndPurpose:
    def test_discovers_fixture_packages(self, gen) -> None:
        assert gen.discover_packages(_SRC) == ("alpha", "beta", "gamma")

    def test_purpose_from_docstring_first_line(self, gen) -> None:
        purpose = gen.purpose_from_init(_SRC / "alpha" / "__init__.py")
        assert purpose == "Alpha package - root of the fixture graph."

    def test_missing_docstring_fails_closed(self, gen) -> None:
        bare = _FIXTURES / "no_docstring" / "src" / "mousedroid" / "bare" / "__init__.py"
        with pytest.raises(ValueError, match=r"no module docstring|empty"):
            gen.purpose_from_init(bare)


class TestImportGraph:
    def test_type_checking_only_file_adds_no_edge(self, gen, tmp_path: Path) -> None:
        pkg = tmp_path / "mousedroid" / "solo"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text('"""Solo package."""\n', encoding="utf-8")
        (pkg / "tc_only.py").write_text(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from mousedroid.other.mod import Thing\n",
            encoding="utf-8",
        )
        other = tmp_path / "mousedroid" / "other"
        other.mkdir()
        (other / "__init__.py").write_text('"""Other package."""\n', encoding="utf-8")
        packages = gen.discover_packages(tmp_path / "mousedroid")
        edges = gen.collect_import_edges(tmp_path / "mousedroid", packages)
        assert edges["solo"] == set()

    def test_runtime_import_is_recorded(self, gen) -> None:
        packages = gen.discover_packages(_SRC)
        edges = gen.collect_import_edges(_SRC, packages)
        assert edges["alpha"] == {"beta", "gamma"}
        assert edges["beta"] == set()
        assert edges["gamma"] == set()

    def test_dependents_are_inverted(self, gen) -> None:
        packages = gen.discover_packages(_SRC)
        edges = gen.collect_import_edges(_SRC, packages)
        infos = gen.build_package_infos(_SRC, packages, edges)
        assert infos["beta"].dependents == ("alpha",)
        assert infos["gamma"].dependents == ("alpha",)
        assert infos["alpha"].dependents == ()


class TestEpicRoster:
    def test_load_epics_partitions_packages(self, gen) -> None:
        packages = gen.discover_packages(_SRC)
        epics = gen.load_epics(_EPICS, packages)
        assert [e.id for e in epics] == ["left", "right"]
        assert epics[0].packages == ("alpha", "beta")
        assert epics[1].packages == ("gamma",)

    def test_missing_package_in_roster_fails(self, gen, tmp_path: Path) -> None:
        bad = tmp_path / "epics.yaml"
        bad.write_text(
            "epics:\n  - id: only\n    title: Only\n    packages: [alpha]\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not assigned"):
            gen.load_epics(bad, ("alpha", "beta"))

    def test_unknown_package_in_roster_fails(self, gen, tmp_path: Path) -> None:
        bad = tmp_path / "epics.yaml"
        bad.write_text(
            "epics:\n  - id: only\n    title: Only\n    packages: [alpha, ghost]\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown"):
            gen.load_epics(bad, ("alpha",))


class TestRenderAndCheck:
    def test_render_labels_import_map(self, gen) -> None:
        packages = gen.discover_packages(_SRC)
        epics = gen.load_epics(_EPICS, packages)
        edges = gen.collect_import_edges(_SRC, packages)
        infos = gen.build_package_infos(_SRC, packages, edges)
        body = gen.render_epic_document(epics[0], infos)
        assert "import map, not a dataflow map" in body
        assert "## `alpha`" in body
        assert "mousedroid.beta" not in body  # package-level names only
        assert "beta" in body

    def test_normalize_for_compare_handles_crlf_and_backslashes(self, gen) -> None:
        raw = "a\\b\r\nc\\d\r"
        assert gen.normalize_for_compare(raw) == "a/b\nc/d\n"

    def test_check_detects_staleness(self, gen, tmp_path: Path) -> None:
        out_dir = tmp_path / "package-map"
        index = tmp_path / "package-map.md"
        outputs = gen.generate(
            src_root=_SRC,
            epics_path=_EPICS,
            out_dir=out_dir,
            index_path=index,
        )
        # Missing files → problems
        assert gen.check_outputs(outputs)
        gen.write_outputs(outputs)
        assert gen.check_outputs(outputs) == []
        # Corrupt one file
        index.write_text(index.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
        problems = gen.check_outputs(outputs)
        assert any("stale" in p for p in problems)

    def test_generate_is_deterministic(self, gen) -> None:
        first = gen.generate(
            src_root=_SRC,
            epics_path=_EPICS,
            out_dir=Path("/tmp/unused-a"),
            index_path=Path("/tmp/unused-index-a.md"),
        )
        second = gen.generate(
            src_root=_SRC,
            epics_path=_EPICS,
            out_dir=Path("/tmp/unused-b"),
            index_path=Path("/tmp/unused-index-b.md"),
        )
        # Compare by relative role (index vs epic id), not absolute path
        first_bodies = sorted(first.values())
        second_bodies = sorted(second.values())
        assert first_bodies == second_bodies
        for body in first_bodies:
            assert "T00:" not in body  # no timestamps
            assert "C:\\" not in body
            assert "/Users/" not in body
            assert "/home/" not in body


class TestCli:
    def test_check_mode_exit_codes(self, gen, tmp_path: Path) -> None:
        out_dir = tmp_path / "package-map"
        index = tmp_path / "package-map.md"
        argv = [
            "--src-root",
            str(_SRC),
            "--epics",
            str(_EPICS),
            "--out-dir",
            str(out_dir),
            "--index",
            str(index),
        ]
        assert gen.main([*argv]) == 0  # write
        assert gen.main([*argv, "--check"]) == 0
        index.write_text("stale\n", encoding="utf-8")
        assert gen.main([*argv, "--check"]) == 1
