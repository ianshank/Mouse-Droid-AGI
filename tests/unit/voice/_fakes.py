"""Shared voice-test stand-ins.

Holds the canonical ``FakeVoiceEngine`` test-double used by the
greeting + factory test modules. Importable as
``from tests.unit.voice._fakes import FakeVoiceEngine`` because
``pyproject.toml`` configures pytest's ``pythonpath = ['src', '.']``.

Centralising prevents protocol drift: when
:class:`mousedroid.voice.protocol.VoiceEngineProtocol` gains a new
method, only this module needs an update; every test importing
``FakeVoiceEngine`` picks up the change for free.
"""

from __future__ import annotations


class FakeVoiceEngine:
    """In-memory ``VoiceEngineProtocol`` stand-in.

    Records every ``play_phrase`` call into ``utterances`` so callers
    can assert chirp-before-message ordering, runtime-name overrides,
    and lifecycle isolation (the greeter MUST NOT touch ``start`` /
    ``stop`` — those flags expose that contract to tests).

    The recording behaviour is harmless for callers that don't read
    ``utterances`` (e.g. factory wiring tests that only need a
    protocol-conformant engine returned by the test seam), so a single
    capable fixture replaces two duplicated stubs.
    """

    def __init__(self) -> None:
        self.started: bool = False
        self.stopped: bool = False
        self.utterances: list[str] = []

    @property
    def is_ready(self) -> bool:  # pragma: no cover — not exercised by greeter
        return self.started

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def speak(self, event: str, context: dict[str, float] | None = None) -> None:
        # Greeter never calls speak() — the protocol requires it. If a
        # caller ever needs speak() recording, extend here rather than
        # branching the class.
        raise NotImplementedError  # pragma: no cover

    async def play_phrase(self, text: str) -> tuple[int, float]:
        self.utterances.append(text)
        # Return plausible (samples, peak) without driving real audio.
        return (len(text) * 100, 0.5)


__all__ = ["FakeVoiceEngine"]
