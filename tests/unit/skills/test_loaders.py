"""Tests for ``mousedroid.skills.loaders``."""

from __future__ import annotations

from pathlib import Path

from mousedroid.skills.loaders import (
    CodeRegisteredLoader,
    MarkdownAgentLoader,
    YAMLManifestLoader,
)
from mousedroid.skills.protocol import SkillSpec

# ---------------------------------------------------------------------------
# CodeRegisteredLoader
# ---------------------------------------------------------------------------


def test_code_loader_yields_specs() -> None:
    specs = (
        SkillSpec(name="a"),
        SkillSpec(name="b"),
    )
    loader = CodeRegisteredLoader(specs)
    out = list(loader.load())
    assert [s.name for s in out] == ["a", "b"]


# ---------------------------------------------------------------------------
# YAMLManifestLoader
# ---------------------------------------------------------------------------


def test_yaml_loader_single_skill_per_file(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "diag.yaml").write_text(
        """
name: diag
description: run diagnostics
tools:
  - health_check
  - mic_diagnostics
system_prompt: You are a diagnostics agent.
""",
        encoding="utf-8",
    )
    loader = YAMLManifestLoader("skills/*.yaml", root=tmp_path)
    out = list(loader.load())
    assert len(out) == 1
    spec = out[0]
    assert spec.name == "diag"
    assert spec.tool_names == frozenset({"health_check", "mic_diagnostics"})
    assert spec.source.startswith("manifest:")


def test_yaml_loader_list_of_skills(tmp_path: Path) -> None:
    (tmp_path / "skills.yaml").write_text(
        """
- name: a
  tools: [health_check]
- name: b
  tools: [esp32_diagnostics]
""",
        encoding="utf-8",
    )
    loader = YAMLManifestLoader("skills.yaml", root=tmp_path)
    out = list(loader.load())
    assert {s.name for s in out} == {"a", "b"}


def test_yaml_loader_invalid_yaml_logged_and_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("::: not valid :::", encoding="utf-8")
    loader = YAMLManifestLoader("*.yaml", root=tmp_path)
    out = list(loader.load())
    assert out == []


def test_yaml_loader_missing_name_skipped(tmp_path: Path) -> None:
    (tmp_path / "noname.yaml").write_text(
        "description: missing name\ntools: []\n", encoding="utf-8"
    )
    loader = YAMLManifestLoader("*.yaml", root=tmp_path)
    out = list(loader.load())
    assert out == []


def test_yaml_loader_tools_must_be_list(tmp_path: Path) -> None:
    (tmp_path / "bad_tools.yaml").write_text("name: x\ntools: not_a_list\n", encoding="utf-8")
    loader = YAMLManifestLoader("*.yaml", root=tmp_path)
    out = list(loader.load())
    assert out == []


def test_yaml_loader_empty_file_skipped(tmp_path: Path) -> None:
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    loader = YAMLManifestLoader("*.yaml", root=tmp_path)
    assert list(loader.load()) == []


# ---------------------------------------------------------------------------
# MarkdownAgentLoader
# ---------------------------------------------------------------------------


def test_markdown_loader_with_front_matter(tmp_path: Path) -> None:
    md = tmp_path / "scout.md"
    md.write_text(
        """---
name: scout
description: explore the area
tools:
  - health_check
---
# Scout
You patrol the perimeter and report anomalies.
""",
        encoding="utf-8",
    )
    loader = MarkdownAgentLoader([tmp_path])
    specs = list(loader.load())
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "scout"
    assert "patrol" in spec.system_prompt
    assert spec.tool_names == frozenset({"health_check"})


def test_markdown_loader_skips_files_without_front_matter(tmp_path: Path) -> None:
    plain = tmp_path / "agent.md"
    plain.write_text("# Just a design doc, no front matter\n", encoding="utf-8")
    loader = MarkdownAgentLoader([tmp_path])
    assert list(loader.load()) == []


def test_markdown_loader_missing_directory_returns_empty(tmp_path: Path) -> None:
    loader = MarkdownAgentLoader([tmp_path / "does_not_exist"])
    assert list(loader.load()) == []


def test_markdown_loader_invalid_yaml_in_front_matter_skipped(
    tmp_path: Path,
) -> None:
    md = tmp_path / "broken.md"
    md.write_text(
        """---
name: x
tools: [unbalanced
---
body
""",
        encoding="utf-8",
    )
    loader = MarkdownAgentLoader([tmp_path])
    assert list(loader.load()) == []


def test_markdown_loader_explicit_system_prompt_in_front_matter(
    tmp_path: Path,
) -> None:
    md = tmp_path / "x.md"
    md.write_text(
        """---
name: explicit
system_prompt: Pre-baked prompt
tools: []
---
# Body would normally become the system prompt
""",
        encoding="utf-8",
    )
    loader = MarkdownAgentLoader([tmp_path])
    specs = list(loader.load())
    assert specs[0].system_prompt == "Pre-baked prompt"
