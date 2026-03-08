"""Experience record protocol — interface for all schema versions."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class ExperienceProtocol(Protocol):
    """Interface for experience records (all schema versions)."""

    @property
    def schema_version(self) -> int:
        """Record schema version (SCHEMA_VERSION constant — NEVER change)."""
        ...

    def serialize(self) -> bytes:
        """Serialize record to msgpack bytes."""
        ...

    @classmethod
    def deserialize(cls, data: bytes) -> Self:
        """Deserialize record from msgpack bytes."""
        ...
