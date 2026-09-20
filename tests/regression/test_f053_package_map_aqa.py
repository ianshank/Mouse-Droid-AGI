"""AQA — F-053 Phase 5: the generated package import map stays fresh.

Regenerate-and-diff under normalised comparison (line endings + path
separators) so the gate holds on ``test-windows``. Determinism pins:
sorted output, no timestamps, no absolute paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._script_loader import load_script_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "mousedroid"
_EPICS = _REPO_ROOT / "scripts" / "package_map_epics.yaml"
_OUT_DIR = _REPO_ROOT / "docs" / "architecture" / "package-map"
_INDEX = _REPO_ROOT / "docs" / "architecture" / "package-map.md"


@pytest.fixture(scope="module")
def gen():
    return load_script_module("generate_package_map")


class TestPackageImportMapIsFresh:
    def test_committed_outputs_match_a_fresh_generation(self, gen) -> None:
        outputs = gen.generate(
            src_root=_SRC,
            epics_path=_EPICS,
            out_dir=_OUT_DIR,
            index_path=_INDEX,
        )
        problems = gen.check_outputs(outputs)
        assert problems == [], (
            "package import map is stale or incomplete — run `make package-map` "
            f"and commit the result: {problems}"
        )

    def test_generation_is_deterministic_and_portable(self, gen) -> None:
        first = gen.generate(
            src_root=_SRC,
            epics_path=_EPICS,
            out_dir=_OUT_DIR,
            index_path=_INDEX,
        )
        second = gen.generate(
            src_root=_SRC,
            epics_path=_EPICS,
            out_dir=_OUT_DIR,
            index_path=_INDEX,
        )
        assert first == second
        for path, body in first.items():
            normalised = gen.normalize_for_compare(body)
            assert normalised == body.replace("\r\n", "\n").replace("\\", "/")
            assert "T00:" not in body
            assert str(_REPO_ROOT) not in body
            assert str(_REPO_ROOT).replace("\\", "/") not in body
            # Index path appears only as a repo-relative link, never absolute.
            assert "C:" not in body
            assert path.suffix == ".md"

    def test_every_package_is_covered_exactly_once(self, gen) -> None:
        packages = gen.discover_packages(_SRC)
        epics = gen.load_epics(_EPICS, packages)
        rostered = [name for epic in epics for name in epic.packages]
        assert sorted(rostered) == sorted(packages)
        assert len(rostered) == len(set(rostered))

    def test_header_declares_import_map_not_dataflow(self, gen) -> None:
        text = _INDEX.read_text(encoding="utf-8")
        assert "import map, not a dataflow map" in text
        assert "TYPE_CHECKING" in text

    def test_telemetry_init_has_a_docstring(self) -> None:
        """Phase 5.3: the empty telemetry ``__init__.py`` must carry a purpose line."""
        init = _SRC / "telemetry" / "__init__.py"
        content = init.read_text(encoding="utf-8")
        assert content.strip(), "telemetry/__init__.py must not be empty"
        assert content.lstrip().startswith('"""') or content.lstrip().startswith("'''")
