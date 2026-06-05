"""Unit tests for Phase-1 situational recognition in CommentaryEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from mousedroid.commentary.engine import CommentaryEngine
from mousedroid.commentary.protocol import CommentaryFacts, GroundedReferentStoreProtocol
from mousedroid.common.time.protocol import MockClock
from mousedroid.config.schema import CommentaryConfig, MemoryConfig, MetricsConfig
from mousedroid.memory.semantic import SemanticIndex
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.voice.protocol import VoiceEngineProtocol

_DIM = 256


def _facts(embedding: np.ndarray | None) -> CommentaryFacts:
    return CommentaryFacts(
        min_clearance_m=5.0,
        forward_distance_m=5.0,
        audio_rms=0.0,
        speed_mps=0.0,
        turn_rate=0.0,
        battery_v=12.0,
        novelty=None,
        is_emergency=False,
        lidar_valid=True,
        audio_valid=True,
        timestamp=0.0,
        embedding=embedding,
    )


def _voice() -> AsyncMock:
    v = AsyncMock(spec=VoiceEngineProtocol)
    v.play_phrase = AsyncMock(return_value=(1, 0.0))
    return v


def _cfg(**over: object) -> CommentaryConfig:
    base: dict[str, object] = {
        "enabled": True,
        "composer": "template",
        "recognition_enabled": True,
        "allow_without_novelty": True,
        "min_interval_s": 0.0,
        "recognition_min_interval_s": 30.0,
        "recognition_distance_threshold": 0.5,
    }
    base.update(over)
    return CommentaryConfig(**base)  # type: ignore[arg-type]


def _engine(
    cfg: CommentaryConfig,
    *,
    voice: AsyncMock | None = None,
    store: GroundedReferentStoreProtocol | None = None,
    embedding_dim: int | None = _DIM,
    reg: MetricsRegistry | None = None,
    clock: MockClock | None = None,
) -> CommentaryEngine:
    from mousedroid.commentary.composers import TemplateCommentaryComposer

    return CommentaryEngine(
        cfg,
        voice_engine=voice if voice is not None else _voice(),
        composer=TemplateCommentaryComposer(cfg),
        metrics=reg,
        clock=clock if clock is not None else MockClock(),
        referent_store=store if store is not None else SemanticIndex(MemoryConfig()),
        embedding_dim=embedding_dim,
    )


def _vec(scale: float = 1.0) -> np.ndarray:
    return np.full(_DIM, scale, dtype=np.float32)


def _reason(reg: MetricsRegistry) -> str | None:
    snap = reg._commentary_suppressed.snapshot()
    return next(iter(snap), None) if snap else None


# --------------------------------------------------------------------------- #
# store-on-first-visit -> recognise-on-revisit loop
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_first_visit_stores_referent_via_novelty_fire() -> None:
    reg = MetricsRegistry(MetricsConfig())
    store = SemanticIndex(MemoryConfig())
    voice = _voice()
    eng = _engine(_cfg(), voice=voice, store=store, reg=reg)
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1  # novelty narration
    assert store.size == 1  # learned this place
    assert reg._commentary_referents_stored.value == 1


@pytest.mark.asyncio
async def test_revisit_narrates_recognition_recalling_phrase() -> None:
    reg = MetricsRegistry(MetricsConfig())
    store = SemanticIndex(MemoryConfig())
    voice = _voice()
    clock = MockClock()
    eng = _engine(_cfg(), voice=voice, store=store, reg=reg, clock=clock)
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()  # first visit (stores)
    clock.advance(40.0)
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()  # revisit -> recognition
    spoken = voice.play_phrase.await_args.args[0]
    assert "last time I said" in spoken
    assert reg._commentary_recognitions.value == 1
    assert store.size == 1  # recognised place NOT re-stored (natural dedup)


@pytest.mark.asyncio
async def test_recognition_cooldown_suppresses_repeat() -> None:
    reg = MetricsRegistry(MetricsConfig())
    store = SemanticIndex(MemoryConfig())
    eng = _engine(_cfg(recognition_min_interval_s=30.0), store=store, reg=reg)
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()  # store
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()  # recognised + narrated (clock at 0, last_recog=-inf)
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()  # immediate -> recognition cooldown
    assert _reason(reg) == "recognition_cooldown"


@pytest.mark.asyncio
async def test_recognised_place_does_not_fire_novelty() -> None:
    store = SemanticIndex(MemoryConfig())
    voice = _voice()
    eng = _engine(_cfg(recognition_min_interval_s=0.0), voice=voice, store=store)
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()  # store + novelty fire (1)
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()  # recognised -> recognition narration (2)
    # Both utterances happened, but the place was stored exactly once.
    assert voice.play_phrase.await_count == 2
    assert store.size == 1


# --------------------------------------------------------------------------- #
# Guards: disabled, dim mismatch, no embedding, store cap, empty-after-transform
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_recognition_inert_when_store_absent() -> None:
    """No referent store -> Phase-0 behaviour (no recognition, no storing)."""
    from mousedroid.commentary.composers import TemplateCommentaryComposer

    cfg = _cfg()
    voice = _voice()
    eng = CommentaryEngine(
        cfg,
        voice_engine=voice,
        composer=TemplateCommentaryComposer(cfg),
        referent_store=None,  # Phase-0: recognition inert
    )
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1  # novelty only; nothing stored/recognised


@pytest.mark.asyncio
async def test_embedding_dim_mismatch_skips_recognition() -> None:
    store = SemanticIndex(MemoryConfig())
    voice = _voice()
    eng = _engine(_cfg(), voice=voice, store=store, embedding_dim=_DIM)
    eng.observe(None, _facts(np.ones(8, dtype=np.float32)))  # wrong width
    await eng._evaluate_and_speak()
    assert voice.play_phrase.await_count == 1  # novelty fires
    assert store.size == 0  # mismatched embedding never stored (no FAISS crash)


@pytest.mark.asyncio
async def test_missing_embedding_skips_recognition() -> None:
    store = SemanticIndex(MemoryConfig())
    eng = _engine(_cfg(), store=store)
    eng.observe(None, _facts(None))
    await eng._evaluate_and_speak()
    assert store.size == 0


@pytest.mark.asyncio
async def test_max_referents_cap_stops_storing() -> None:
    store = SemanticIndex(MemoryConfig())
    eng = _engine(_cfg(recognition_max_referents=1), store=store)
    eng.observe(None, _facts(_vec(1.0)))
    await eng._evaluate_and_speak()  # stores 1st
    eng.observe(None, _facts(_vec(9.0)))  # far enough to be novel, not recognised
    await eng._evaluate_and_speak()  # cap reached -> not stored
    assert store.size == 1


@pytest.mark.asyncio
async def test_recognition_empty_after_transform_suppressed() -> None:
    reg = MetricsRegistry(MetricsConfig())
    store = SemanticIndex(MemoryConfig())
    # All-article recognition template -> rocky_transform yields "".
    cfg = _cfg(recognition_template="the a an {phrase}", recognition_min_interval_s=0.0)
    eng = _engine(cfg, store=store, reg=reg)
    # Seed a referent whose key is itself all-articles so the formatted line is
    # also all-articles -> empty after transform.
    store.store("the", _vec())
    eng.observe(None, _facts(_vec()))
    await eng._evaluate_and_speak()
    assert _reason(reg) == "empty_after_transform"


@pytest.mark.asyncio
async def test_recognition_debug_logs() -> None:
    from structlog.testing import capture_logs

    store = SemanticIndex(MemoryConfig())
    eng = _engine(_cfg(), store=store)
    with capture_logs() as logs:
        eng.observe(None, _facts(_vec()))
        await eng._evaluate_and_speak()  # probe (empty store) + store
    events = {e.get("event") for e in logs}
    assert "commentary_recognition_probe" not in events  # empty store -> no probe
    assert "commentary_referent_stored" in events
