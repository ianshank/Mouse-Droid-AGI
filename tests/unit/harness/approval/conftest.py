"""Shared test fixtures for approval gate tests."""

from __future__ import annotations

from mousedroid.harness.approval.protocol import ApprovalDecision, ApprovalRequest
from mousedroid.security.injection_filter import InjectionRejected


class DummyGate:
    """Always-approving gate for composition testing."""

    name = "dummy"

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, decided_by=self.name)


class DummyFilter:
    """Prompt injection filter that rejects text containing 'bad'."""

    def sanitize(self, text: str) -> str:
        if "bad" in text:
            raise InjectionRejected("Found bad word")
        return text
