"""Tests for the WebSocket serialization negotiation module."""

from __future__ import annotations

from mousedroid.telemetry.serialization import (
    REASON_INVALID_HELLO,
    REASON_NO_OVERLAP,
    REASON_UNSUPPORTED_VERSION,
    SUPPORTED_SERIALIZATIONS,
    build_default_ack,
    negotiate,
)

_DEFAULT_KW: dict[str, object] = {
    "server_serialization": "json",
    "server_protocol_version": 1,
    "msgpack_client_lib_url": "https://example.test/msgpack.js",
}


class TestBuildDefaultAck:
    """``build_default_ack`` produces the no-hello fallback envelope."""

    def test_payload_shape(self) -> None:
        ack = build_default_ack(serialization="json", protocol_version=1)
        assert ack == {
            "hello_ack": {
                "ok": True,
                "negotiated": False,
                "serialization": "json",
                "protocol_version": 1,
            }
        }


class TestNegotiateHappyPaths:
    """Successful negotiations pick a mutually-acceptable serialization."""

    def test_prefers_client_preferred_when_supported(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 1,
                "supported_serializations": ["json", "msgpack"],
                "preferred_serialization": "msgpack",
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is True
        assert result.serialization == "msgpack"
        assert result.reason is None
        assert result.ack_payload["hello_ack"]["negotiated"] is True

    def test_falls_back_to_server_default_when_overlap_exists(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 1,
                "supported_serializations": ["json", "msgpack"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is True
        assert result.serialization == "json"  # server default

    def test_picks_first_overlap_when_server_default_missing(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 1,
                "supported_serializations": ["msgpack"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is True
        assert result.serialization == "msgpack"


class TestNegotiateFailures:
    """Invalid / mismatched hellos produce structured rejections."""

    def test_missing_hello_key(self) -> None:
        result = negotiate({"hi": "there"}, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.reason == REASON_INVALID_HELLO

    def test_non_dict_payload(self) -> None:
        result = negotiate("hello", **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.reason == REASON_INVALID_HELLO

    def test_supported_serializations_not_a_list(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 1,
                "supported_serializations": "json,msgpack",
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.reason == REASON_INVALID_HELLO

    def test_no_serialization_overlap(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 1,
                "supported_serializations": ["cbor", "yaml"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.reason == REASON_NO_OVERLAP

    def test_unsupported_protocol_version(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 99,
                "supported_serializations": ["json"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.reason == REASON_UNSUPPORTED_VERSION

    def test_zero_protocol_version_rejected(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 0,
                "supported_serializations": ["json"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.reason == REASON_UNSUPPORTED_VERSION

    def test_protocol_version_not_an_int(self) -> None:
        hello = {
            "hello": {
                "protocol_version": "one",
                "supported_serializations": ["json"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.reason == REASON_INVALID_HELLO


class TestAckEnrichment:
    """Successful and failed acks both carry helpful diagnostic fields."""

    def test_success_ack_includes_msgpack_url(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 1,
                "supported_serializations": ["json"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert result.ack_payload["hello_ack"]["msgpack_client_lib_url"] == (
            "https://example.test/msgpack.js"
        )

    def test_failure_ack_lists_supported_serializations(self) -> None:
        hello = {
            "hello": {
                "protocol_version": 1,
                "supported_serializations": ["cbor"],
            }
        }
        result = negotiate(hello, **_DEFAULT_KW)  # type: ignore[arg-type]
        assert sorted(result.ack_payload["hello_ack"]["supported_serializations"]) == sorted(
            SUPPORTED_SERIALIZATIONS
        )
