"""Mission parser — NL commands to structured intents.

Parses natural language commands into structured ``MissionIntent`` objects
that can drive the navigation system without requiring full LLM inference
for common patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Protocol, runtime_checkable

from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class IntentType(Enum):
    """Classified mission intent types."""

    VELOCITY = "velocity"
    NAVIGATION = "navigation"
    STOP = "stop"
    PATROL = "patrol"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MissionIntent:
    """Structured representation of a parsed NL mission command.

    Attributes:
        intent_type: Classification of the command.
        goal_vector: Velocity targets (normalised to [-1, 1]).
        confidence: Parser confidence in [0, 1].
        raw_command: Original NL command.
        parameters: Additional intent-specific parameters.
    """

    intent_type: IntentType = IntentType.UNKNOWN
    goal_vector: GoalVector = field(default_factory=GoalVector)
    confidence: float = 0.0
    raw_command: str = ""
    parameters: dict[str, float | str] = field(default_factory=dict)


@runtime_checkable
class MissionParserProtocol(Protocol):
    """Interface for NL command -> structured intent parsing."""

    def parse(self, nl_command: str) -> MissionIntent:
        """Parse a natural language command into a structured intent.

        Args:
            nl_command: Natural language mission command.

        Returns:
            Parsed ``MissionIntent``.
        """
        ...


class RuleBasedMissionParser:
    """Rule-based mission parser using regex patterns.

    Handles common navigation commands without LLM inference.
    Falls back to ``IntentType.UNKNOWN`` for ambiguous commands
    that require full LLM processing.
    """

    # Compiled patterns for common commands
    _STOP_RE = re.compile(
        r"^(stop|halt|freeze|emergency stop|e[\-\s]?stop|hold|stay)$",
        re.IGNORECASE,
    )
    _FORWARD_RE = re.compile(
        r"^(go|move|drive|travel)\s+(forward|ahead|straight)",
        re.IGNORECASE,
    )
    _BACKWARD_RE = re.compile(
        r"^(go|move|drive|travel)\s+(backward|back|reverse)",
        re.IGNORECASE,
    )
    _LEFT_RE = re.compile(
        r"^(turn|rotate|spin)\s+(left|counter[\-\s]?clockwise)",
        re.IGNORECASE,
    )
    _RIGHT_RE = re.compile(
        r"^(turn|rotate|spin)\s+(right|clockwise)",
        re.IGNORECASE,
    )
    _STRAFE_LEFT_RE = re.compile(
        r"^(strafe|slide|move)\s+left",
        re.IGNORECASE,
    )
    _STRAFE_RIGHT_RE = re.compile(
        r"^(strafe|slide|move)\s+right",
        re.IGNORECASE,
    )
    _PATROL_RE = re.compile(
        r"^patrol\s+(.+)",
        re.IGNORECASE,
    )
    _ANGLE_RE = re.compile(r"(\d+)\s*(degrees?|deg|°)", re.IGNORECASE)
    _SPEED_RE = re.compile(
        r"(slow(?:ly)?|fast|full\s*speed|half\s*speed|quickly)",
        re.IGNORECASE,
    )

    _SPEED_MAP: ClassVar[dict[str, float]] = {
        "slow": 0.3,
        "slowly": 0.3,
        "half speed": 0.5,
        "fast": 0.8,
        "quickly": 0.8,
        "full speed": 1.0,
    }

    def parse(self, nl_command: str) -> MissionIntent:
        """Parse NL command into structured intent.

        Args:
            nl_command: Natural language mission command.

        Returns:
            Parsed ``MissionIntent`` with appropriate type and goal vector.
        """
        cmd = nl_command.strip()
        if not cmd:
            return MissionIntent(
                intent_type=IntentType.UNKNOWN,
                raw_command=cmd,
                confidence=0.0,
            )

        # Stop commands
        if self._STOP_RE.match(cmd):
            return MissionIntent(
                intent_type=IntentType.STOP,
                goal_vector=GoalVector(vx_target=0.0, vy_target=0.0, omega_target=0.0),
                confidence=1.0,
                raw_command=cmd,
            )

        # Extract speed modifier
        speed = self._extract_speed(cmd)

        # Forward motion
        if self._FORWARD_RE.search(cmd):
            return MissionIntent(
                intent_type=IntentType.VELOCITY,
                goal_vector=GoalVector(vx_target=speed, vy_target=0.0, omega_target=0.0),
                confidence=0.9,
                raw_command=cmd,
            )

        # Backward motion
        if self._BACKWARD_RE.search(cmd):
            return MissionIntent(
                intent_type=IntentType.VELOCITY,
                goal_vector=GoalVector(vx_target=-speed, vy_target=0.0, omega_target=0.0),
                confidence=0.9,
                raw_command=cmd,
            )

        # Turn left
        if self._LEFT_RE.search(cmd):
            omega = self._extract_rotation_magnitude(cmd)
            return MissionIntent(
                intent_type=IntentType.VELOCITY,
                goal_vector=GoalVector(vx_target=0.0, vy_target=0.0, omega_target=omega),
                confidence=0.9,
                raw_command=cmd,
                parameters=self._extract_angle_params(cmd),
            )

        # Turn right
        if self._RIGHT_RE.search(cmd):
            omega = self._extract_rotation_magnitude(cmd)
            return MissionIntent(
                intent_type=IntentType.VELOCITY,
                goal_vector=GoalVector(vx_target=0.0, vy_target=0.0, omega_target=-omega),
                confidence=0.9,
                raw_command=cmd,
                parameters=self._extract_angle_params(cmd),
            )

        # Strafe left
        if self._STRAFE_LEFT_RE.search(cmd):
            return MissionIntent(
                intent_type=IntentType.VELOCITY,
                goal_vector=GoalVector(vx_target=0.0, vy_target=-speed, omega_target=0.0),
                confidence=0.9,
                raw_command=cmd,
            )

        # Strafe right
        if self._STRAFE_RIGHT_RE.search(cmd):
            return MissionIntent(
                intent_type=IntentType.VELOCITY,
                goal_vector=GoalVector(vx_target=0.0, vy_target=speed, omega_target=0.0),
                confidence=0.9,
                raw_command=cmd,
            )

        # Patrol
        patrol_match = self._PATROL_RE.match(cmd)
        if patrol_match:
            location = patrol_match.group(1).strip()
            return MissionIntent(
                intent_type=IntentType.PATROL,
                goal_vector=GoalVector(vx_target=0.5, vy_target=0.0, omega_target=0.0),
                confidence=0.8,
                raw_command=cmd,
                parameters={"location": location},
            )

        # Avoid obstacles — navigation intent
        if re.search(r"avoid\s+obstacles?", cmd, re.IGNORECASE):
            return MissionIntent(
                intent_type=IntentType.NAVIGATION,
                goal_vector=GoalVector(vx_target=0.3, vy_target=0.0, omega_target=0.0),
                confidence=0.7,
                raw_command=cmd,
                parameters={"mode": "obstacle_avoidance"},
            )

        # Unknown — needs LLM
        _log.debug("mission_parser_unknown_command", command=cmd)
        return MissionIntent(
            intent_type=IntentType.UNKNOWN,
            raw_command=cmd,
            confidence=0.0,
        )

    def _extract_speed(self, cmd: str) -> float:
        """Extract speed modifier from command text.

        Args:
            cmd: Command text.

        Returns:
            Speed multiplier in (0, 1].
        """
        match = self._SPEED_RE.search(cmd)
        if match:
            key = match.group(1).lower()
            return self._SPEED_MAP.get(key, 0.5)
        return 0.5

    def _extract_rotation_magnitude(self, cmd: str) -> float:
        """Extract rotation magnitude from command text.

        Args:
            cmd: Command text.

        Returns:
            Rotation magnitude in (0, 1].
        """
        angle_match = self._ANGLE_RE.search(cmd)
        if angle_match:
            degrees = float(angle_match.group(1))
            # Normalize: 90 degrees -> 0.5, 180 -> 1.0
            return min(1.0, degrees / 180.0)
        return 0.5

    def _extract_angle_params(self, cmd: str) -> dict[str, float | str]:
        """Extract angle parameters if specified.

        Args:
            cmd: Command text.

        Returns:
            Dict with angle_degrees if found.
        """
        angle_match = self._ANGLE_RE.search(cmd)
        if angle_match:
            return {"angle_degrees": float(angle_match.group(1))}
        return {}
