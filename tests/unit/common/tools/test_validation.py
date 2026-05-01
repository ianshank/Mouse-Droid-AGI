"""Tests for ``mousedroid.common.tools.validation``.

Verifies that ``ValidatedToolSpec`` adds Pydantic input/output validation
without affecting plain ``ToolSpec`` dispatch.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.common.tools.validation import (
    ToolInputValidationError,
    ToolOutputValidationError,
    ValidatedToolSpec,
    validate_input,
    validate_output,
)


class _PingInput(BaseModel):
    """Input schema for the ping tool."""

    message: str = Field(..., min_length=1)
    count: int = Field(1, ge=1, le=10)


class _PingOutput(BaseModel):
    """Output schema for the ping tool."""

    echoed: str
    count: int


async def _ping_handler(message: str, count: int = 1) -> dict[str, Any]:
    return {"echoed": message * count, "count": count}


async def _bad_output_handler(message: str = "x", count: int = 1) -> dict[str, Any]:
    # Return value violates the output schema (missing 'echoed').
    return {"count": count}


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


def test_validated_spec_defaults_are_none() -> None:
    spec = ValidatedToolSpec(
        name="ping",
        description="echo",
        handler=_ping_handler,
    )
    assert spec.input_schema is None
    assert spec.output_schema is None
    assert spec.requires_approval is False


def test_validated_spec_is_a_tool_spec() -> None:
    """``ValidatedToolSpec`` must be a subclass of ``ToolSpec`` so callers
    that type-annotate ``ToolSpec`` keep working without changes."""
    spec = ValidatedToolSpec(
        name="t",
        description="d",
        handler=_ping_handler,
    )
    assert isinstance(spec, ToolSpec)


# ---------------------------------------------------------------------------
# validate_input / validate_output helpers
# ---------------------------------------------------------------------------


def test_validate_input_returns_kwargs_when_no_schema() -> None:
    spec = ValidatedToolSpec(name="t", description="", handler=_ping_handler)
    out = validate_input(spec, {"foo": 1})
    assert out == {"foo": 1}


def test_validate_input_passes_when_valid() -> None:
    spec = ValidatedToolSpec(
        name="ping",
        description="",
        handler=_ping_handler,
        input_schema=_PingInput,
    )
    out = validate_input(spec, {"message": "hi", "count": 2})
    assert out == {"message": "hi", "count": 2}


def test_validate_input_raises_typed_error_on_failure() -> None:
    spec = ValidatedToolSpec(
        name="ping",
        description="",
        handler=_ping_handler,
        input_schema=_PingInput,
    )
    with pytest.raises(ToolInputValidationError):
        validate_input(spec, {"message": "", "count": 999})


def test_validate_output_returns_value_when_no_schema() -> None:
    spec = ValidatedToolSpec(name="t", description="", handler=_ping_handler)
    out = validate_output(spec, {"any": "thing"})
    assert out == {"any": "thing"}


def test_validate_output_raises_typed_error_on_failure() -> None:
    spec = ValidatedToolSpec(
        name="ping",
        description="",
        handler=_ping_handler,
        output_schema=_PingOutput,
    )
    with pytest.raises(ToolOutputValidationError):
        validate_output(spec, {"count": 1})


# ---------------------------------------------------------------------------
# ToolRegistry.dispatch with ValidatedToolSpec
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.mark.asyncio
async def test_dispatch_validates_input_and_output(registry: ToolRegistry) -> None:
    spec = ValidatedToolSpec(
        name="ping",
        description="echo",
        handler=_ping_handler,
        input_schema=_PingInput,
        output_schema=_PingOutput,
    )
    registry.register(spec)
    result = await registry.dispatch("ping", message="hi", count=3)
    assert result == {"echoed": "hihihi", "count": 3}


@pytest.mark.asyncio
async def test_dispatch_rejects_invalid_input(registry: ToolRegistry) -> None:
    spec = ValidatedToolSpec(
        name="ping",
        description="echo",
        handler=_ping_handler,
        input_schema=_PingInput,
    )
    registry.register(spec)
    with pytest.raises(ToolInputValidationError):
        await registry.dispatch("ping", message="", count=1)


@pytest.mark.asyncio
async def test_dispatch_rejects_invalid_output(registry: ToolRegistry) -> None:
    spec = ValidatedToolSpec(
        name="bad",
        description="bad",
        handler=_bad_output_handler,
        output_schema=_PingOutput,
    )
    registry.register(spec)
    with pytest.raises(ToolOutputValidationError):
        await registry.dispatch("bad", message="ok")


@pytest.mark.asyncio
async def test_plain_tool_spec_still_dispatches_unchanged(registry: ToolRegistry) -> None:
    """Backwards compatibility: plain ToolSpec runs the legacy fast path."""

    async def echo(value: str = "x") -> str:
        return value

    spec = ToolSpec(name="echo", description="", handler=echo)
    registry.register(spec)
    result = await registry.dispatch("echo", value="hello")
    assert result == "hello"


@pytest.mark.asyncio
async def test_validated_spec_without_schemas_runs_legacy_path(
    registry: ToolRegistry,
) -> None:
    """A ValidatedToolSpec with both schemas=None must not validate."""
    spec = ValidatedToolSpec(
        name="t",
        description="",
        handler=_ping_handler,
        input_schema=None,
        output_schema=None,
    )
    registry.register(spec)
    result = await registry.dispatch("t", message="hi", count=1)
    assert result["count"] == 1


def test_requires_approval_default_is_false() -> None:
    spec = ValidatedToolSpec(
        name="t",
        description="",
        handler=_ping_handler,
    )
    assert spec.requires_approval is False


def test_requires_approval_can_be_set() -> None:
    spec = ValidatedToolSpec(
        name="t",
        description="",
        handler=_ping_handler,
        requires_approval=True,
    )
    assert spec.requires_approval is True
