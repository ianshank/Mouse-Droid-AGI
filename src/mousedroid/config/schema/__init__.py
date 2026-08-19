"""Configuration schema package — single source of truth for all settings.

Public import surface is unchanged: ``from mousedroid.config.schema import Settings``,
``from mousedroid.config.schema import ESP32Config``, etc. all continue to work exactly
as before. The former flat ``schema.py`` module (101 ``BaseModel`` classes plus the root
``Settings``) is now a package split by domain — see the individual submodules
(``hardware``, ``cognitive``, ``world_model``, ``learning``, ``reward_safety``,
``telemetry``, ``llm``, ``voice``, ``sim``, ``gcp_cloud``, ``arm``, ``training``,
``harness_mcp``, ``misc``, ``root``, and the shared ``_primitives``) for the actual model
definitions. This file only re-exports them so every existing caller keeps working
unchanged; ``__all__`` matches the full pre-split ``dir()`` surface exactly.
"""

from __future__ import annotations

import copy
import enum
import sys
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from mousedroid.config.schema._primitives import (
    ESP32CommandSetLiteral,
    PlatformType,
    RangeF,
    ReplayOutcomeLiteral,
    Self,
    StrEnum,
    VLAActiveBackendLiteral,
    VLABackendLiteral,
)
from mousedroid.config.schema.arm import (
    ArmConfig,
    ArmCurriculumConfig,
    ArmPerceptionConfig,
    ArmPlanningConfig,
    ArmSimConfig,
    ArmTaskConfig,
    ArmTrainingConfig,
    PPOConfig,
)
from mousedroid.config.schema.cognitive import (
    CognitiveConfig,
    CuriosityConfig,
    MemoryConfig,
    MetacognitiveConfig,
    SurpriseConfig,
)

# Private (leading-underscore) constant re-exported ONLY because a real
# caller (tests/unit/cloud/test_weight_update_poll_config.py) already does
# ``from mousedroid.config.schema import _WORLD_MODEL_DEFAULT_REPO_ID`` — the
# pre-split flat schema.py exposed it incidentally as a module global. It is
# deliberately NOT in ``__all__``: it stays out of the wildcard-export /
# dir()-enumerated public surface, matching its underscore-privacy intent.
from mousedroid.config.schema.gcp_cloud import (
    _WORLD_MODEL_DEFAULT_REPO_ID as _WORLD_MODEL_DEFAULT_REPO_ID,
)
from mousedroid.config.schema.gcp_cloud import (
    CloudConfig,
    GCPConfig,
    GCPFirestoreConfig,
    GCPLoggingConfig,
    GCPMonitoringConfig,
    GCPPubSubConfig,
    GCPSimulationConfig,
    GCPStorageConfig,
    GCPTrainingConfig,
    WeightUpdatePollConfig,
)
from mousedroid.config.schema.hardware import (
    WAVESHARE_STOCK_BAUD,
    CameraConfig,
    ESP32Config,
    HailoConfig,
    HealthConfig,
    HostEnvConfig,
    JetsonConfig,
    LidarConfig,
    MotorControllerConfig,
    MotorLimitsConfig,
    UltrasonicConfig,
    USBCDiscoveryConfig,
    USBCEndpointSpec,
)
from mousedroid.config.schema.harness_mcp import (
    HarnessApprovalConfig,
    HarnessConfig,
    HarnessHooksConfig,
    HarnessJournalConfig,
    HarnessTrackerConfig,
    MCPConfig,
    MCPResourcesConfig,
    OpenClawConfig,
    OpenClawMemoryConfig,
    OpenClawPolicyConfig,
    SkillsConfig,
)
from mousedroid.config.schema.learning import (
    GrowthConfig,
    LearningConfig,
    OfflineRLConfig,
    OnDeviceLearningConfig,
)
from mousedroid.config.schema.llm import (
    LLMConfig,
    LLMReplannerConfig,
    MissionConfig,
    MissionParserConfig,
    MissionReplannerConfig,
    VLAConfig,
)
from mousedroid.config.schema.misc import (
    BaselinesConfig,
    CircuitBreakerConfig,
    DomainRandomizationConfig,
    ExperienceConfig,
    GreetingConfig,
    LoggingConfig,
    LoopConfig,
    RetryConfig,
    RobotConfig,
)
from mousedroid.config.schema.reward_safety import (
    RewardConfig,
    SafetyConfig,
    SafetyProjectorConfig,
    ThreeLawsConfig,
    VLMProgressConfig,
)
from mousedroid.config.schema.root import (
    Settings,
    apply_aliases,
    migrate_group_sections,
    migrate_section_aliases,
    migrate_section_transforms,
    milliseconds_to_seconds,
    seconds_to_hz,
    seconds_to_milliseconds,
)
from mousedroid.config.schema.sim import (
    MujocoSimConfig,
    RoverActionConfig,
    RoverConfig,
    RoverInertialConfig,
    RoverObservationConfig,
    RoverRewardConfig,
    RoverSimConfig,
    RoverTaskConfig,
)
from mousedroid.config.schema.telemetry import (
    ExperimentLoggerConfig,
    MetricsConfig,
    ObservabilityConfig,
    TelemetryAuthConfig,
    TelemetryConfig,
)
from mousedroid.config.schema.training import (
    DriftTrainingConfig,
    GPUConfig,
    ReplayMixerConfig,
    TrainingAnnotationConfig,
    TrainingConfig,
    TrainingConstitutionalConfig,
    TrainingGenerationConfig,
    TrainingPipelineConfig,
    TrainingReplayConfig,
    TrainingWarmstartConfig,
)
from mousedroid.config.schema.voice import (
    FaceDisplayConfig,
    MicrophoneConfig,
    SpeakerConfig,
    VoiceConfig,
)
from mousedroid.config.schema.world_model import (
    DEFAULT_UCB_CANDIDATES,
    DEFAULT_UCB_TARGET_MS,
    DualStreamTrainingConfig,
    MCTSConfig,
    ModelConfig,
    WorldModelConfig,
    WorldModelMemoryConfig,
)

__all__ = [
    "DEFAULT_UCB_CANDIDATES",
    "DEFAULT_UCB_TARGET_MS",
    "WAVESHARE_STOCK_BAUD",
    "Any",
    "ArmConfig",
    "ArmCurriculumConfig",
    "ArmPerceptionConfig",
    "ArmPlanningConfig",
    "ArmSimConfig",
    "ArmTaskConfig",
    "ArmTrainingConfig",
    "BaseModel",
    "BaseSettings",
    "BaselinesConfig",
    "CameraConfig",
    "CircuitBreakerConfig",
    "CloudConfig",
    "CognitiveConfig",
    "CuriosityConfig",
    "DomainRandomizationConfig",
    "DriftTrainingConfig",
    "DualStreamTrainingConfig",
    "ESP32CommandSetLiteral",
    "ESP32Config",
    "ExperienceConfig",
    "ExperimentLoggerConfig",
    "FaceDisplayConfig",
    "Field",
    "Final",
    "GCPConfig",
    "GCPFirestoreConfig",
    "GCPLoggingConfig",
    "GCPMonitoringConfig",
    "GCPPubSubConfig",
    "GCPSimulationConfig",
    "GCPStorageConfig",
    "GCPTrainingConfig",
    "GPUConfig",
    "GreetingConfig",
    "GrowthConfig",
    "HailoConfig",
    "HarnessApprovalConfig",
    "HarnessConfig",
    "HarnessHooksConfig",
    "HarnessJournalConfig",
    "HarnessTrackerConfig",
    "HealthConfig",
    "HostEnvConfig",
    "JetsonConfig",
    "LLMConfig",
    "LLMReplannerConfig",
    "LearningConfig",
    "LidarConfig",
    "Literal",
    "LoggingConfig",
    "LoopConfig",
    "MCPConfig",
    "MCPResourcesConfig",
    "MCTSConfig",
    "MemoryConfig",
    "MetacognitiveConfig",
    "MetricsConfig",
    "MicrophoneConfig",
    "MissionConfig",
    "MissionParserConfig",
    "MissionReplannerConfig",
    "ModelConfig",
    "MotorControllerConfig",
    "MotorLimitsConfig",
    "MujocoSimConfig",
    "ObservabilityConfig",
    "OfflineRLConfig",
    "OnDeviceLearningConfig",
    "OpenClawConfig",
    "OpenClawMemoryConfig",
    "OpenClawPolicyConfig",
    "PPOConfig",
    "Path",
    "PlatformType",
    "RangeF",
    "ReplayMixerConfig",
    "ReplayOutcomeLiteral",
    "RetryConfig",
    "RewardConfig",
    "RobotConfig",
    "RoverActionConfig",
    "RoverConfig",
    "RoverInertialConfig",
    "RoverObservationConfig",
    "RoverRewardConfig",
    "RoverSimConfig",
    "RoverTaskConfig",
    "SafetyConfig",
    "SafetyProjectorConfig",
    "SecretStr",
    "Self",
    "Settings",
    "SettingsConfigDict",
    "SkillsConfig",
    "SpeakerConfig",
    "StrEnum",
    "SurpriseConfig",
    "TelemetryAuthConfig",
    "TelemetryConfig",
    "ThreeLawsConfig",
    "TrainingAnnotationConfig",
    "TrainingConfig",
    "TrainingConstitutionalConfig",
    "TrainingGenerationConfig",
    "TrainingPipelineConfig",
    "TrainingReplayConfig",
    "TrainingWarmstartConfig",
    "USBCDiscoveryConfig",
    "USBCEndpointSpec",
    "UltrasonicConfig",
    "VLAActiveBackendLiteral",
    "VLABackendLiteral",
    "VLAConfig",
    "VLMProgressConfig",
    "VoiceConfig",
    "WeightUpdatePollConfig",
    "WorldModelConfig",
    "WorldModelMemoryConfig",
    "annotations",
    "apply_aliases",
    "copy",
    "enum",
    "field_validator",
    "migrate_group_sections",
    "migrate_section_aliases",
    "migrate_section_transforms",
    "milliseconds_to_seconds",
    "model_validator",
    "seconds_to_hz",
    "seconds_to_milliseconds",
    "sys",
]
