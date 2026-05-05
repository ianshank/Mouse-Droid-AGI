"""Unit tests for :class:`MarkdownReplayExporter`."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import MemoryConfig
from mousedroid.memory.episodic import EpisodicReplay
from mousedroid.memory.exporter import (
    SCHEMA_VERSION,
    MarkdownReplayExporter,
    MemoryExporterProtocol,
)


def _replay(payloads: list[object]) -> EpisodicReplay:
    cfg = MemoryConfig(episodic_capacity=max(1, len(payloads) or 1))
    r = EpisodicReplay(cfg, seed=0)
    for p in payloads:
        r.push(p, priority=1.0)
    return r


def test_protocol_runtime_check(tmp_path: Path) -> None:
    exporter = MarkdownReplayExporter(tmp_path / "MEMORY.md")
    assert isinstance(exporter, MemoryExporterProtocol)


def test_max_entries_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_entries"):
        MarkdownReplayExporter(tmp_path / "MEMORY.md", max_entries=0)


@pytest.mark.asyncio
async def test_no_op_on_empty_replay(tmp_path: Path) -> None:
    out_path = tmp_path / "MEMORY.md"
    exporter = MarkdownReplayExporter(out_path)
    result = await exporter.export(_replay([]))
    assert result is None
    assert not out_path.exists()


@pytest.mark.asyncio
async def test_writes_markdown_with_front_matter_and_entries(tmp_path: Path) -> None:
    out_path = tmp_path / "MEMORY.md"
    exporter = MarkdownReplayExporter(out_path, max_entries=8)
    payloads = [{"step": i, "action": "forward"} for i in range(5)]
    result = await exporter.export(_replay(payloads))
    assert result == out_path
    body = out_path.read_text(encoding="utf-8")
    assert body.startswith("---\n")
    assert f"schema_version: {SCHEMA_VERSION}" in body
    assert "replay_size: 5" in body
    assert "## Recent experiences" in body


@pytest.mark.asyncio
async def test_atomic_write_uses_tmp_then_rename(tmp_path: Path) -> None:
    out_path = tmp_path / "MEMORY.md"
    out_path.write_text("PREVIOUS", encoding="utf-8")
    exporter = MarkdownReplayExporter(out_path)
    await exporter.export(_replay([{"a": 1}]))
    # Old contents fully replaced; sibling .tmp file does not linger.
    body = out_path.read_text(encoding="utf-8")
    assert "PREVIOUS" not in body
    siblings = list(tmp_path.iterdir())
    assert all(not s.name.endswith(".tmp") for s in siblings)


@pytest.mark.asyncio
async def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "subdir" / "deeper" / "MEMORY.md"
    exporter = MarkdownReplayExporter(nested)
    await exporter.export(_replay([{"k": "v"}]))
    assert nested.exists()


@pytest.mark.asyncio
async def test_export_failure_returns_none_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "MEMORY.md"
    exporter = MarkdownReplayExporter(out_path)

    def boom(_path: Path) -> None:
        raise OSError("disk full")

    # Force write_text to fail; exporter should swallow and return None.
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **k: boom(self))
    result = await exporter.export(_replay([{"any": "thing"}]))
    assert result is None


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    payloads=st.lists(
        st.dictionaries(
            keys=st.text(
                min_size=1,
                max_size=8,
                alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            ),
            values=st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.floats(allow_nan=False, allow_infinity=False, width=32),
                st.text(
                    min_size=0,
                    max_size=20,
                    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
                ),
            ),
            max_size=4,
        ),
        min_size=1,
        max_size=10,
    ),
)
@pytest.mark.asyncio
async def test_property_arbitrary_payloads_serialise_safely(
    payloads: list[dict[str, object]], tmp_path: Path
) -> None:
    out_path = tmp_path / "MEMORY.md"
    exporter = MarkdownReplayExporter(out_path, max_entries=10)
    result = await exporter.export(_replay(payloads))
    assert result == out_path
    body = out_path.read_text(encoding="utf-8")
    assert "## Recent experiences" in body
