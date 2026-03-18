"""Tests for LogRingBuffer — ring buffer, structlog processor, subscriptions."""

from __future__ import annotations

import asyncio

from mousedroid.telemetry.log_buffer import LogRingBuffer


def test_log_buffer_initial_state():
    buf = LogRingBuffer(maxlen=100)
    assert buf.size == 0
    assert buf.get_recent() == []


def test_log_buffer_captures_events():
    buf = LogRingBuffer(maxlen=10)
    event = {"event": "test", "level": "info"}
    result = buf(None, "info", event)
    assert buf.size == 1
    assert result is event  # passthrough


def test_log_buffer_passthrough_semantics():
    """Processor must return event_dict unchanged."""
    buf = LogRingBuffer(maxlen=10)
    event = {"event": "hello", "x": 42}
    result = buf(None, "info", event)
    assert result is event
    assert result["event"] == "hello"
    assert result["x"] == 42


def test_log_buffer_ring_behavior():
    buf = LogRingBuffer(maxlen=3)
    for i in range(5):
        buf(None, "info", {"n": i})
    assert buf.size == 3
    recent = buf.get_recent()
    assert [e["n"] for e in recent] == [2, 3, 4]


def test_get_recent_with_n():
    buf = LogRingBuffer(maxlen=10)
    for i in range(10):
        buf(None, "info", {"n": i})
    recent = buf.get_recent(3)
    assert len(recent) == 3
    assert [e["n"] for e in recent] == [7, 8, 9]


def test_get_recent_more_than_available():
    buf = LogRingBuffer(maxlen=10)
    for i in range(3):
        buf(None, "info", {"n": i})
    recent = buf.get_recent(100)
    assert len(recent) == 3


def test_subscribe_receives_events():
    buf = LogRingBuffer(maxlen=10)
    sub = buf.subscribe()
    buf(None, "info", {"event": "test1"})
    assert not sub.empty()
    entry = sub.get_nowait()
    assert entry["event"] == "test1"


def test_unsubscribe_stops_receiving():
    buf = LogRingBuffer(maxlen=10)
    sub = buf.subscribe()
    buf.unsubscribe(sub)
    buf(None, "info", {"event": "test2"})
    assert sub.empty()


def test_unsubscribe_nonexistent_is_safe():
    buf = LogRingBuffer(maxlen=10)
    fake_queue: asyncio.Queue = asyncio.Queue()
    buf.unsubscribe(fake_queue)  # should not raise


def test_multiple_subscribers():
    buf = LogRingBuffer(maxlen=10)
    sub1 = buf.subscribe()
    sub2 = buf.subscribe()
    buf(None, "info", {"event": "shared"})
    assert not sub1.empty()
    assert not sub2.empty()
    assert sub1.get_nowait()["event"] == "shared"
    assert sub2.get_nowait()["event"] == "shared"


def test_subscriber_queue_overflow_drops():
    buf = LogRingBuffer(maxlen=200)
    sub = buf.subscribe()
    # Fill subscriber queue (maxsize=100)
    for i in range(150):
        buf(None, "info", {"n": i})
    # Should not raise, drops silently
    assert sub.qsize() == 100


def test_buffer_copies_events():
    """Ensure buffered events are copies, not references."""
    buf = LogRingBuffer(maxlen=10)
    event = {"event": "original", "mutable": [1, 2]}
    buf(None, "info", event)
    recent = buf.get_recent()
    # The buffered copy should be a different object
    assert recent[0] is not event
