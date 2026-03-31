"""Convert object detections to symbolic PDDL state.

Bridges the perception stack (Layer 0) with the symbolic planner
(Layer 1) by mapping detected object poses to PDDL predicates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.arm.protocols import DetectedObject, SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTaskConfig

_log = get_logger(__name__)


class StateExtractor:
    """Extract symbolic PDDL state from object detections.

    Maps detected disk positions to peg assignments and generates
    PDDL predicates like ``on(disk1, peg_A)`` and ``clear(disk1)``.

    Args:
        task_cfg: Task configuration with peg/basket positions.
    """

    def __init__(self, task_cfg: ArmTaskConfig) -> None:
        """Initialise state extractor.

        Args:
            task_cfg: Task config containing peg positions and counts.
        """
        self._task_cfg = task_cfg
        self._peg_positions = np.array(task_cfg.peg_positions, dtype=np.float64)
        self._peg_names = [f"peg_{chr(65 + i)}" for i in range(task_cfg.num_pegs)]
        _log.info(
            "state_extractor_init",
            num_pegs=task_cfg.num_pegs,
            peg_names=self._peg_names,
        )

    def _assign_to_peg(self, position: NDArray[np.float64]) -> str:
        """Assign an object position to the nearest peg.

        Args:
            position: Object XYZ position, shape ``(3,)``.

        Returns:
            Peg name (e.g., "peg_A").
        """
        distances = np.linalg.norm(self._peg_positions - position, axis=1)
        nearest_idx = int(np.argmin(distances))
        return self._peg_names[nearest_idx]

    def extract(self, detections: list[DetectedObject]) -> SymbolicState:
        """Convert detections to symbolic PDDL state.

        Assigns each detected disk to its nearest peg and generates
        stacking predicates based on vertical ordering.

        Args:
            detections: List of detected objects with populated poses.

        Returns:
            Symbolic state with PDDL predicates and object mappings.
        """
        # Filter to disk detections only
        disks = [d for d in detections if d.class_name.startswith("disk")]

        # Group disks by assigned peg
        peg_stacks: dict[str, list[DetectedObject]] = {name: [] for name in self._peg_names}
        for disk in disks:
            peg = self._assign_to_peg(disk.position_m)
            peg_stacks[peg].append(disk)

        # Sort each peg stack by height (z-coordinate, ascending = bottom first)
        for peg_name in peg_stacks:
            peg_stacks[peg_name].sort(key=lambda d: d.position_m[2])

        # Generate PDDL predicates
        predicates: set[str] = set()
        objects: dict[str, str] = {}

        for peg_name, stack in peg_stacks.items():
            objects[peg_name] = "peg"
            if not stack:
                predicates.add(f"clear({peg_name})")
                continue

            # Bottom disk is on the peg
            predicates.add(f"on({stack[0].object_id}, {peg_name})")
            objects[stack[0].object_id] = "disk"

            # Each subsequent disk is on the one below
            for i in range(1, len(stack)):
                predicates.add(f"on({stack[i].object_id}, {stack[i - 1].object_id})")
                objects[stack[i].object_id] = "disk"

            # Top disk is clear
            predicates.add(f"clear({stack[-1].object_id})")

        _log.debug("state_extracted", predicates=len(predicates), objects=len(objects))
        return SymbolicState(predicates=frozenset(predicates), objects=objects)
