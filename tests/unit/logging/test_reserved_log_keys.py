"""Unit tests for ``safe_log_extra`` — the reserved-log-key guard.

structlog's bound-logger methods have the signature ``meth(event, **kw)``, so
splatting a caller-controlled mapping that happens to carry ``event`` raises
``TypeError``.  Other keys owned by the processor chain (``level``,
``timestamp``, ...) are silently overwritten instead.  ``safe_log_extra``
renames those keys rather than dropping them.
"""

from __future__ import annotations

import pytest

from mousedroid.logging.setup import EXTRA_KEY_PREFIX, RESERVED_LOG_KEYS, safe_log_extra


class TestReservedKeyRenaming:
    """Reserved keys are renamed, never dropped."""

    @pytest.mark.parametrize("key", sorted(RESERVED_LOG_KEYS))
    def test_every_reserved_key_is_renamed(self, key: str) -> None:
        """No reserved key survives under its original name."""
        result = safe_log_extra({key: "value"})

        assert key not in result

    @pytest.mark.parametrize("key", sorted(RESERVED_LOG_KEYS))
    def test_every_reserved_key_value_is_preserved(self, key: str) -> None:
        """The caller's value is carried over to the prefixed name."""
        result = safe_log_extra({key: "value"})

        assert result[f"{EXTRA_KEY_PREFIX}{key}"] == "value"

    def test_non_reserved_keys_pass_through_unchanged(self) -> None:
        """Well-behaved keys keep their original names."""
        result = safe_log_extra({"attempt": 3, "device": "USB Audio"})

        assert result == {"attempt": 3, "device": "USB Audio"}

    def test_empty_mapping_returns_empty_dict(self) -> None:
        """An empty mapping is a no-op."""
        assert safe_log_extra({}) == {}

    def test_returns_a_new_mapping(self) -> None:
        """The caller's mapping is never mutated."""
        original = {"event": "x"}

        safe_log_extra(original)

        assert original == {"event": "x"}


class TestOccupiedKeys:
    """``occupied`` protects keys the call site has already claimed."""

    def test_occupied_key_is_renamed(self) -> None:
        """A key listed in occupied is treated as reserved."""
        result = safe_log_extra({"subsystem": "spoofed"}, occupied=["subsystem"])

        assert result == {f"{EXTRA_KEY_PREFIX}subsystem": "spoofed"}

    def test_occupied_accepts_a_dict(self) -> None:
        """Passing the live payload dict iterates its keys."""
        payload = {"subsystem": "voice", "reason": "timeout"}

        result = safe_log_extra({"reason": "spoofed"}, occupied=payload)

        assert result == {f"{EXTRA_KEY_PREFIX}reason": "spoofed"}

    def test_unoccupied_keys_still_pass_through(self) -> None:
        """occupied narrows nothing beyond the keys it names."""
        result = safe_log_extra({"attempt": 1}, occupied=["subsystem"])

        assert result == {"attempt": 1}


class TestCollisionOfCollision:
    """Renaming must not clobber a caller key that already uses the prefix."""

    def test_both_values_survive(self) -> None:
        """extra={"event": a, "extra_event": b} keeps both values."""
        result = safe_log_extra({"event": "a", "extra_event": "b"})

        assert sorted(result.values()) == ["a", "b"]  # type: ignore[type-var]

    def test_escalates_prefix_until_free(self) -> None:
        """The second collision gains a second prefix."""
        result = safe_log_extra({"event": "a", "extra_event": "b"})

        assert result[f"{EXTRA_KEY_PREFIX}event"] == "a"
        assert result[f"{EXTRA_KEY_PREFIX}{EXTRA_KEY_PREFIX}event"] == "b"

    def test_reverse_insertion_order_also_survives(self) -> None:
        """Dict ordering does not cause a lost value."""
        result = safe_log_extra({"extra_event": "b", "event": "a"})

        assert sorted(result.values()) == ["a", "b"]  # type: ignore[type-var]


class TestOutputIsSafeToSplat:
    """The returned mapping never collides with an owned key."""

    def test_output_never_contains_a_reserved_key(self) -> None:
        """Every reserved key at once still yields a splat-safe mapping."""
        hostile = dict.fromkeys(RESERVED_LOG_KEYS, "v")

        result = safe_log_extra(hostile)

        assert RESERVED_LOG_KEYS.isdisjoint(result)

    def test_no_values_are_lost(self) -> None:
        """Key count is preserved — renaming, not dropping."""
        hostile = {key: f"v_{key}" for key in RESERVED_LOG_KEYS}

        result = safe_log_extra(hostile)

        assert len(result) == len(hostile)
        assert sorted(result.values()) == sorted(hostile.values())  # type: ignore[type-var]
