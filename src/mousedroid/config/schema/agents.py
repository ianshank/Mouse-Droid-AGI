"""External agent framework configuration models.

Off-loop mission interpretation via Google ADK, long-horizon memory via
Honcho, and cloud actions via Composio. All default-OFF and strictly
outside the 30 Hz control loop.
"""

from __future__ import annotations

from pydantic import Field, SecretStr

from mousedroid.config.schema._primitives import StrictBaseModel, _settings_default_factory

__all__ = [
    "ADKConfig",
    "AgentsConfig",
    "ComposioConfig",
    "HonchoConfig",
]


class ADKConfig(StrictBaseModel):
    """Configuration for Google ADK mission decomposition."""

    enabled: bool = Field(
        default=False,
        description="Whether ADK off-loop mission decomposition is enabled."
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="The Gemini model to use for decomposition."
    )
    timeout_s: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout in seconds for ADK API calls."
    )
    max_sub_tasks: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of sub-tasks to decompose a command into."
    )


class HonchoConfig(StrictBaseModel):
    """Configuration for Honcho long-horizon memory."""

    enabled: bool = Field(
        default=False,
        description="Whether Honcho long-horizon memory synchronization is enabled."
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Honcho."
    )
    app_name: str = Field(
        default="mousedroid",
        description="Application name to use for Honcho."
    )
    sanitize_recalled: bool = Field(
        default=True,
        description="Whether to sanitize recalled text before use."
    )
    max_recall_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return from memory recall."
    )


class ComposioConfig(StrictBaseModel):
    """Configuration for Composio cloud actions."""

    enabled: bool = Field(
        default=False,
        description="Whether Composio cloud action execution is enabled."
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Composio."
    )
    dry_run: bool = Field(
        default=True,
        description="If true, simulates cloud actions without executing them."
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="List of allowed tools that can be executed."
    )


class AgentsConfig(StrictBaseModel):
    """External agent framework configuration models.

    Off-loop mission interpretation via Google ADK, long-horizon memory via
    Honcho, and cloud actions via Composio. All default-OFF and strictly
    outside the 30 Hz control loop.
    """

    adk: ADKConfig = Field(
        default_factory=_settings_default_factory(ADKConfig),
        description="Google ADK configuration."
    )
    honcho: HonchoConfig = Field(
        default_factory=_settings_default_factory(HonchoConfig),
        description="Honcho configuration."
    )
    composio: ComposioConfig = Field(
        default_factory=_settings_default_factory(ComposioConfig),
        description="Composio configuration."
    )
