"""Backwards-compat pin — the reserved-key guard must not rename existing fields.

``safe_log_extra`` renames only keys that collide with something owned. Every
other caller key passes through untouched, which is what keeps existing Grafana
panels and alert rules keyed on ``elapsed_s``, ``tick``, ``h_norm``, ``strategy``
and friends working now that the guard is in place.

The inventory of call sites is **derived from the source tree** by walking the
AST, not hand-transcribed. An earlier revision of this file listed the payloads
by hand and silently missed five of them (three in
``telemetry/server/_lifecycle.py`` and both ``telemetry/server/_ws_handlers.py``
negotiation paths), which is exactly the staleness this discovery step removes: a
new ``record(..., extra=...)`` site is picked up automatically, and one whose keys
collide fails :func:`test_colliding_keys_are_exactly_the_known_set` until someone
decides consciously that the rename is correct.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import structlog.testing

from mousedroid.config.schema import MetricsConfig
from mousedroid.logging.setup import EXTRA_KEY_PREFIX, RESERVED_LOG_KEYS, safe_log_extra
from mousedroid.telemetry.failure_recorder import (
    FailureRecorder,
    NullFailureRecorder,
    PrometheusFailureRecorder,
)
from mousedroid.telemetry.metrics import MetricsRegistry

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mousedroid"

# The field names PrometheusFailureRecorder.record puts in its own payload and
# passes as ``occupied``; pinned by TestRecorderLogShapeUnchanged below.
_RECORDER_OWNED = ("subsystem", "reason", "log_level")

# Sites whose ``extra=`` is a local variable rather than a dict literal, so the
# AST walk cannot read their keys. Transcribed by hand and kept honest by
# test_every_extra_site_is_accounted_for, which fails if a new one appears.
_NON_LITERAL_EXTRAS: dict[str, tuple[str, ...]] = {
    # rocky._record_drop builds ``recorder_extra`` then merges its own ``extra``
    # argument into it: {"event", "priority"} plus the per-call-site fields.
    "voice/rocky.py": ("event", "priority", "cooldown_s", "elapsed_s", "queue_full"),
}

# (module-relative path, key) pairs the guard deliberately renames, because the
# key collides with a structlog-owned or recorder-owned name. Both are *fixes*:
# before the guard, rocky's ``event`` raised TypeError and lost the whole line,
# and _ws_handlers' ``reason`` silently overwrote the recorder's own ``reason``
# field, so the log said the WS close reason instead of "ws_negotiation_failed".
_KNOWN_COLLISIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("voice/rocky.py", "event"),
        ("telemetry/server/_ws_handlers.py", "reason"),
    }
)


def _is_record_call(node: ast.Call) -> bool:
    """Return True if ``node`` is a ``<...>failure_recorder.record(...)`` call."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "record":
        return False
    return ast.unparse(func.value).endswith("failure_recorder")


def _literal_extra_keys(node: ast.Call) -> tuple[bool, tuple[str, ...] | None]:
    """Return ``(has_extra, literal string keys)``; keys is None if not a literal."""
    for kw in node.keywords:
        if kw.arg != "extra":
            continue
        if isinstance(kw.value, ast.Dict):
            return True, tuple(
                k.value
                for k in kw.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            )
        return True, None
    return False, None


def _discover() -> tuple[dict[str, tuple[str, ...]], list[str], int]:
    """Walk ``src/mousedroid`` for ``FailureRecorder.record`` call sites.

    Returns:
        ``(literal_keys_by_path, paths_with_non_literal_extra, total_record_calls)``
        with paths POSIX-style and relative to ``src/mousedroid``.
    """
    literal: dict[str, list[str]] = {}
    non_literal: list[str] = []
    total = 0

    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_SRC_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_record_call(node):
                continue
            total += 1
            has_extra, keys = _literal_extra_keys(node)
            if not has_extra:
                continue
            if keys is None:
                non_literal.append(rel)
            else:
                literal.setdefault(rel, []).extend(keys)

    deduped = {key: tuple(dict.fromkeys(values)) for key, values in literal.items()}
    return deduped, non_literal, total


_LITERAL_EXTRAS, _NON_LITERAL_PATHS, _TOTAL_RECORD_CALLS = _discover()

# Every payload the guard must be checked against: discovered literals plus the
# hand-transcribed non-literal sites.
_ALL_PAYLOADS: list[tuple[str, tuple[str, ...]]] = sorted(
    {**_LITERAL_EXTRAS, **_NON_LITERAL_EXTRAS}.items()
)
_PAYLOAD_IDS = [path for path, _ in _ALL_PAYLOADS]


class TestInventoryIsTrustworthy:
    """The AST discovery must actually find the call sites."""

    def test_record_calls_were_found(self) -> None:
        """A silently-empty walk would make every other test here vacuous."""
        assert _TOTAL_RECORD_CALLS >= 15

    def test_known_payload_modules_were_found(self) -> None:
        """Naming the modules catches a broken walk more precisely than a count."""
        expected = {
            "orchestrator/_action_mixin.py",
            "orchestrator/_world_model_state_mixin.py",
            "telemetry/server/_lifecycle.py",
            "telemetry/server/_ws_handlers.py",
            "voice/rocky.py",
        }

        missing = expected - {path for path, _ in _ALL_PAYLOADS}

        assert not missing, (
            f"the AST walk no longer sees `extra=` payloads in {sorted(missing)}. "
            "Either those modules were renamed (update this set) or _discover() "
            "is broken and every pass-through test here is silently vacuous."
        )

    def test_every_extra_site_is_accounted_for(self) -> None:
        """A new non-literal ``extra=`` site must be transcribed, not skipped."""
        untranscribed = set(_NON_LITERAL_PATHS) - set(_NON_LITERAL_EXTRAS)

        assert not untranscribed, (
            f"{sorted(untranscribed)} pass `extra=` as a variable, so this test "
            "cannot read their keys from the AST. Add them to _NON_LITERAL_EXTRAS "
            "with their field names so the guard stays pinned for those log lines."
        )


@pytest.mark.parametrize(("path", "keys"), _ALL_PAYLOADS, ids=_PAYLOAD_IDS)
def test_non_colliding_keys_pass_through_unchanged(path: str, keys: tuple[str, ...]) -> None:
    """Field names that collide with nothing keep their exact spelling."""
    payload: dict[str, object] = dict.fromkeys(keys, "v")

    result = safe_log_extra(payload, occupied=_RECORDER_OWNED)

    expected = {k for k in keys if k not in RESERVED_LOG_KEYS and k not in _RECORDER_OWNED}
    assert expected <= set(result), f"{path}: guard renamed a non-colliding field"
    for key in expected:
        assert result[key] == "v"


@pytest.mark.parametrize(("path", "keys"), _ALL_PAYLOADS, ids=_PAYLOAD_IDS)
def test_no_value_is_dropped_for_any_call_site(path: str, keys: tuple[str, ...]) -> None:
    """Guarding a real payload never loses a field."""
    payload: dict[str, object] = dict.fromkeys(keys, "v")

    result = safe_log_extra(payload, occupied=_RECORDER_OWNED)

    assert len(result) == len(payload), f"{path}: guard dropped a field"


def test_colliding_keys_are_exactly_the_known_set() -> None:
    """Only the two reviewed collisions get renamed — a new one must be decided on."""
    observed = {
        (path, key)
        for path, keys in _ALL_PAYLOADS
        for key in keys
        if key in RESERVED_LOG_KEYS or key in _RECORDER_OWNED
    }

    assert observed == _KNOWN_COLLISIONS, (
        "the set of call-site keys the guard renames changed.\n"
        f"  new:  {sorted(observed - _KNOWN_COLLISIONS)}\n"
        f"  gone: {sorted(_KNOWN_COLLISIONS - observed)}\n"
        "A new entry means a caller's log field is about to be renamed — confirm "
        "that is intended, then update _KNOWN_COLLISIONS and the CHANGELOG."
    )


def test_known_collisions_are_actually_renamed() -> None:
    """The known collisions really do get the prefix, not silent passthrough."""
    for path, key in sorted(_KNOWN_COLLISIONS):
        result = safe_log_extra({key: "v"}, occupied=_RECORDER_OWNED)

        assert key not in result, f"{path}: {key!r} was not namespaced"
        assert result[f"{EXTRA_KEY_PREFIX}{key}"] == "v"


class TestRecorderLogShapeUnchanged:
    """The emitted log event keeps its established field names."""

    def test_event_name_unchanged(self) -> None:
        """The structlog event name is still subsystem_failure_recorded."""
        rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "piper_timeout")

        assert logs[0]["event"] == "subsystem_failure_recorded"

    def test_owned_field_names_unchanged(self) -> None:
        """subsystem / reason / log_level remain the recorder's field names."""
        rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        with structlog.testing.capture_logs() as logs:
            rec.record("telemetry", "bind_failed", level="error")

        log = logs[0]
        assert log["subsystem"] == "telemetry"
        assert log["reason"] == "bind_failed"
        assert log["log_level"] == "error"

    def test_owned_names_match_the_occupied_set_used_here(self) -> None:
        """``_RECORDER_OWNED`` must track what record() actually emits."""
        rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "piper_timeout")

        emitted = set(logs[0]) - {"event"}
        assert emitted == set(_RECORDER_OWNED)

    def test_extra_none_still_supported(self) -> None:
        """extra=None remains a valid call."""
        rec = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "retry_exhausted", extra=None)

        assert logs[0]["event"] == "subsystem_failure_recorded"


class TestProtocolSignatureUnchanged:
    """The FailureRecorder protocol surface did not shift."""

    def test_record_parameter_names(self) -> None:
        """record() still takes (subsystem, reason, *, level, extra)."""
        params = list(inspect.signature(PrometheusFailureRecorder.record).parameters)

        assert params == ["self", "subsystem", "reason", "level", "extra"]

    def test_level_and_extra_remain_keyword_only(self) -> None:
        """level and extra stay keyword-only with their defaults."""
        sig = inspect.signature(PrometheusFailureRecorder.record)

        assert sig.parameters["level"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["level"].default == "warning"
        assert sig.parameters["extra"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["extra"].default is None

    def test_both_implementations_still_satisfy_the_protocol(self) -> None:
        """Guard did not break structural conformance."""
        prometheus = PrometheusFailureRecorder(MetricsRegistry(MetricsConfig()))

        assert isinstance(prometheus, FailureRecorder)
        assert isinstance(NullFailureRecorder(), FailureRecorder)
