"""Robot arm platform protocol interfaces.

All arm component interfaces use ``@runtime_checkable`` structural typing,
following the same pattern as ``mousedroid.hardware.protocols``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class DetectedObject:
    """Detected object with pose and classification.

    Attributes:
        object_id: Unique identifier for the detected object.
        class_name: Object class (e.g., "disk_1", "shirt_white").
        confidence: Detection confidence score [0, 1].
        position_m: 3D position in world frame (x, y, z) metres.
        orientation_rad: Orientation as Euler angles (roll, pitch, yaw) radians.
        bbox: 2D bounding box [x_min, y_min, x_max, y_max] pixels.
    """

    __slots__ = ("bbox", "class_name", "confidence", "object_id", "orientation_rad", "position_m")

    def __init__(
        self,
        object_id: str,
        class_name: str,
        confidence: float,
        position_m: NDArray[np.float64],
        orientation_rad: NDArray[np.float64],
        bbox: NDArray[np.float64],
    ) -> None:
        """Initialise detected object.

        Args:
            object_id: Unique object identifier.
            class_name: Object class name.
            confidence: Detection confidence [0, 1].
            position_m: World-frame position (3,).
            orientation_rad: Euler angles (3,).
            bbox: Bounding box (4,).
        """
        self.object_id = object_id
        self.class_name = class_name
        self.confidence = confidence
        self.position_m = position_m
        self.orientation_rad = orientation_rad
        self.bbox = bbox


class SymbolicState:
    """Symbolic state representation for PDDL planning.

    Attributes:
        predicates: Set of active PDDL predicates (e.g., ``on(disk1, peg_A)``).
        objects: Mapping of object names to their types.
    """

    __slots__ = ("objects", "predicates")

    def __init__(
        self,
        predicates: frozenset[str],
        objects: dict[str, str],
    ) -> None:
        """Initialise symbolic state.

        Args:
            predicates: Active PDDL predicates.
            objects: Object name -> type mapping.
        """
        self.predicates = predicates
        self.objects = objects

    def __eq__(self, other: object) -> bool:
        """Check equality by predicates and objects."""
        if not isinstance(other, SymbolicState):
            return NotImplemented
        return self.predicates == other.predicates and self.objects == other.objects

    def __hash__(self) -> int:
        """Hash by predicates."""
        return hash(self.predicates)


class PlanStep:
    """Single step in a symbolic plan.

    Attributes:
        action: Action name (e.g., "move").
        args: Action arguments (e.g., ["disk1", "peg_A", "peg_C"]).
    """

    __slots__ = ("action", "args")

    def __init__(self, action: str, args: list[str]) -> None:
        """Initialise plan step.

        Args:
            action: PDDL action name.
            args: Action argument list.
        """
        self.action = action
        self.args = args

    def __repr__(self) -> str:
        """Readable representation."""
        return f"{self.action}({', '.join(self.args)})"


# ---------------------------------------------------------------------------
# Protocol interfaces
# ---------------------------------------------------------------------------


@runtime_checkable
class ArmDriverProtocol(Protocol):
    """Interface for robot arm hardware drivers (SO-ARM100, mock)."""

    async def get_joint_states(self) -> NDArray[np.float64]:
        """Read current joint angles.

        Returns:
            Joint angles in radians, shape ``(dof,)``.
        """
        ...

    async def send_joint_command(self, target_angles: NDArray[np.float64]) -> None:
        """Command arm to target joint angles.

        Args:
            target_angles: Target joint angles in radians, shape ``(dof,)``.
        """
        ...

    async def close_gripper(self) -> float:
        """Close gripper and return grip force.

        Returns:
            Measured grip force in Newtons.
        """
        ...

    async def open_gripper(self) -> None:
        """Open gripper fully."""
        ...

    async def emergency_stop(self) -> None:
        """Immediately halt all arm motion."""
        ...

    async def home(self) -> None:
        """Move arm to home position."""
        ...

    async def start(self) -> None:
        """Initialise arm driver connection."""
        ...

    async def stop(self) -> None:
        """Shut down arm driver connection."""
        ...

    @property
    def dof(self) -> int:
        """Degrees of freedom."""
        ...


@runtime_checkable
class ArmPerceptionProtocol(Protocol):
    """Interface for robot arm perception stack (Layer 0)."""

    async def detect_objects(self) -> list[DetectedObject]:
        """Detect objects in current camera frame.

        Returns:
            List of detected objects with poses.
        """
        ...

    async def get_symbolic_state(self) -> SymbolicState:
        """Convert current detections to symbolic PDDL state.

        Returns:
            Symbolic state with active predicates.
        """
        ...

    async def start(self) -> None:
        """Start perception pipeline."""
        ...

    async def stop(self) -> None:
        """Stop perception pipeline."""
        ...


@runtime_checkable
class ArmPlannerProtocol(Protocol):
    """Interface for symbolic planning (Layer 1)."""

    def plan(self, initial_state: SymbolicState, goal_state: SymbolicState) -> list[PlanStep]:
        """Generate optimal plan from initial to goal state.

        Args:
            initial_state: Current symbolic state.
            goal_state: Target symbolic state.

        Returns:
            Ordered list of plan steps.

        Raises:
            PlanningError: If no valid plan exists.
        """
        ...

    def replan(
        self, current_state: SymbolicState, goal_state: SymbolicState, error: str
    ) -> list[PlanStep]:
        """Generate recovery plan after execution failure.

        Args:
            current_state: Current (possibly unexpected) symbolic state.
            goal_state: Original target state.
            error: Description of the failure.

        Returns:
            Recovery plan steps.
        """
        ...


@runtime_checkable
class SymbolicPlannerBackend(Protocol):
    """Interface for a single symbolic-planning backend (Layer 1 internal).

    A backend turns a generated PDDL domain/problem pair into an ordered plan.
    Returning ``None`` signals "this backend could not produce a plan" (engine
    unavailable, timeout, internal error, or no solution) so the orchestrating
    :class:`~mousedroid.arm.planning.symbolic_planner.SymbolicPlanner` can defer
    to its fallback backend. A backend that is total (e.g. the recursive
    Tower-of-Hanoi solver) never returns ``None``.
    """

    def search(self, domain_pddl: str, problem_pddl: str) -> list[PlanStep] | None:
        """Attempt to solve the PDDL problem.

        Args:
            domain_pddl: Generated PDDL domain definition.
            problem_pddl: Generated PDDL problem definition.

        Returns:
            An ordered list of plan steps, or ``None`` if this backend could not
            produce a plan (caller should fall back).
        """
        ...


@runtime_checkable
class ArmControllerProtocol(Protocol):
    """Interface for low-level motor control (Layer 3)."""

    async def execute_action(self, action: NDArray[np.float64]) -> dict[str, Any]:
        """Execute continuous action (joint deltas or end-effector target).

        Args:
            action: Action vector from RL policy.

        Returns:
            Execution result with keys: success, achieved_state, info.
        """
        ...

    async def execute_primitive(self, primitive_name: str, target: NDArray[np.float64]) -> bool:
        """Execute pre-trained action primitive.

        Args:
            primitive_name: Primitive name ("grasp", "place", "transit").
            target: Target pose for the primitive.

        Returns:
            True if primitive executed successfully.
        """
        ...


@runtime_checkable
class ArmEnvironmentProtocol(Protocol):
    """Interface for Gymnasium-compatible arm training environments."""

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (observation, info).
        """
        ...

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action to execute.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        ...

    def render(self) -> NDArray[np.uint8] | None:
        """Render current state.

        Returns:
            RGB image array or None if rendering disabled.
        """
        ...

    def close(self) -> None:
        """Clean up environment resources."""
        ...
