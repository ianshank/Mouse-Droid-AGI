"""Tool I/O schema validation — extends ToolSpec with optional Pydantic schemas.

This module is fully backwards compatible. The base ``ToolSpec`` keeps its
zero-overhead dispatch path; validation runs only when a tool is registered
as a ``ValidatedToolSpec``. All schemas are Pydantic v2 ``BaseModel``
subclasses — no ``jsonschema`` dependency is required.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from mousedroid.common.tools.registry import ToolSpec
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class ToolInputValidationError(ValueError):
    """Raised when a tool's input arguments fail Pydantic validation."""


class ToolOutputValidationError(ValueError):
    """Raised when a tool's return value fails Pydantic validation."""


@dataclass(frozen=True)
class ValidatedToolSpec(ToolSpec):
    """Tool specification augmented with optional Pydantic I/O schemas.

    Attributes:
        name: Inherited — unique tool identifier.
        description: Inherited — human-readable description.
        handler: Inherited — async callable that executes the tool.
        input_schema: Optional Pydantic model class validating ``**kwargs``
            passed to ``ToolRegistry.dispatch``. ``None`` disables input
            validation (backwards compatible with plain ``ToolSpec``).
        output_schema: Optional Pydantic model class validating the
            handler's return value. ``None`` disables output validation.
        requires_approval: Hint to the MCP tool bridge that this tool's
            invocation must pass through the configured ``ApprovalGate``.
            Defaults to ``False`` so existing tools dispatch unchanged.
    """

    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    requires_approval: bool = False


def validate_input(spec: ValidatedToolSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate ``kwargs`` against ``spec.input_schema`` (no-op if None).

    Args:
        spec: The validated tool specification.
        kwargs: Keyword arguments supplied to ``dispatch``.

    Returns:
        The validated, normalised kwargs (Pydantic may coerce types).
        When ``input_schema`` is ``None``, ``kwargs`` is returned unchanged.

    Raises:
        ToolInputValidationError: If validation fails.
    """
    schema = spec.input_schema
    if schema is None:
        return kwargs
    try:
        validated = schema.model_validate(kwargs)
    except ValidationError as exc:
        _log.warning(
            "tool_input_validation_failed",
            tool=spec.name,
            error=str(exc),
            keys=sorted(kwargs.keys()),
        )
        msg = f"Invalid input for tool {spec.name!r}: {exc}"
        raise ToolInputValidationError(msg) from exc
    _log.debug("tool_input_validated", tool=spec.name)
    return validated.model_dump()


def validate_output(spec: ValidatedToolSpec, result: Any) -> Any:
    """Validate ``result`` against ``spec.output_schema`` (no-op if None).

    Args:
        spec: The validated tool specification.
        result: The handler's raw return value.

    Returns:
        The validated value (Pydantic may coerce types). When
        ``output_schema`` is ``None``, ``result`` is returned unchanged.

    Raises:
        ToolOutputValidationError: If validation fails.
    """
    schema = spec.output_schema
    if schema is None:
        return result
    try:
        validated = schema.model_validate(result)
    except ValidationError as exc:
        _log.warning(
            "tool_output_validation_failed",
            tool=spec.name,
            error=str(exc),
        )
        msg = f"Invalid output from tool {spec.name!r}: {exc}"
        raise ToolOutputValidationError(msg) from exc
    _log.debug("tool_output_validated", tool=spec.name)
    return validated.model_dump()


__all__ = [
    "ToolInputValidationError",
    "ToolOutputValidationError",
    "ValidatedToolSpec",
    "validate_input",
    "validate_output",
]


# Helper used by the registry; not exported.
_HandlerCallable = Callable[..., Awaitable[Any]]
