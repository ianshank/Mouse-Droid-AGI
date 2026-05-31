"""Property-based tests for voice engine components."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import SpeakerConfig
from mousedroid.hardware.audio.mock_speaker import MockSpeaker
from mousedroid.voice.phrase_bank import DEFAULT_PHRASES
from mousedroid.voice.rocky import Priority, rocky_transform

_ARTICLES = {"the", "a", "an"}


@given(text=st.text(min_size=0, max_size=200))
@settings(max_examples=50)
def test_rocky_transform_zero_intensity_never_grows(text: str) -> None:
    """At zero intensity, output has fewer or equal words (articles stripped)."""
    result = rocky_transform(text, intensity=0.0)
    assert len(result.split()) <= len(text.split())


@given(text=st.text(alphabet=st.characters(whitelist_categories=("L", "Zs")), min_size=1))
@settings(max_examples=50)
def test_rocky_transform_strips_articles(text: str) -> None:
    """No articles remain after transformation."""
    result = rocky_transform(text, intensity=0.0)
    for word in result.split():
        assert word.lower() not in _ARTICLES


@given(intensity=st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=30)
def test_rocky_transform_intensity_range(intensity: float) -> None:
    """Transform works for any intensity in [0, 1]."""
    result = rocky_transform("path is clear", intensity=intensity)
    assert isinstance(result, str)


@given(data=st.data())
# deadline=None: this test spins up a fresh asyncio loop + structlog handlers per
# example, which Windows cold-starts in 500-900ms on the first iteration before
# warming to <2ms. The hypothesis default 200ms deadline produced spurious
# FlakyFailure reports during PR #106 pre-commit; the assertion (chunk
# independence) does not depend on timing so the deadline buys nothing real.
@settings(max_examples=30, deadline=None)
def test_mock_speaker_chunks_are_copies(data: st.DataObject) -> None:
    """Written chunks are independent copies of the original array."""
    import asyncio

    size = data.draw(st.integers(min_value=1, max_value=2048))
    arr = np.random.default_rng(42).standard_normal(size).astype(np.float32)
    speaker = MockSpeaker(SpeakerConfig(chunk_size=size))
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(speaker.write_chunk(arr))
    finally:
        loop.close()
    arr[:] = 999.0  # Mutate original
    chunks = speaker.get_written_chunks()
    assert not np.allclose(chunks[0], arr)


def test_priority_ordering_transitive() -> None:
    """Priority enum ordering is transitive: EMERGENCY > HIGH > NORMAL."""
    assert Priority.EMERGENCY > Priority.HIGH
    assert Priority.HIGH > Priority.NORMAL
    assert Priority.EMERGENCY > Priority.NORMAL


def test_phrase_bank_all_events_non_empty() -> None:
    """Every phrase bank event has at least one phrase."""
    for event, phrases in DEFAULT_PHRASES.items():
        assert len(phrases) > 0, f"{event!r} has no phrases"
        for phrase in phrases:
            assert isinstance(phrase, str)
            assert len(phrase) > 0
