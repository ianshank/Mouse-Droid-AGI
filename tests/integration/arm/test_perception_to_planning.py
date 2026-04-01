"""Integration test: perception -> symbolic state -> planner pipeline."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.perception.state_extractor import StateExtractor
from mousedroid.arm.planning.pddl_domain import optimal_move_count
from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner
from mousedroid.arm.protocols import DetectedObject, SymbolicState
from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig


def _make_disk(disk_id: str, peg_x: float, z: float) -> DetectedObject:
    """Create a disk detection at a peg position."""
    return DetectedObject(
        object_id=disk_id,
        class_name=f"disk_{disk_id.split('_')[-1]}",
        confidence=0.95,
        position_m=np.array([peg_x, 0.0, z], dtype=np.float64),
        orientation_rad=np.zeros(3, dtype=np.float64),
        bbox=np.array([0, 0, 50, 50], dtype=np.float64),
    )


class TestPerceptionToPlanningPipeline:
    """Integration: detections -> symbolic state -> optimal plan."""

    def test_3_disk_pipeline_produces_optimal_plan(self) -> None:
        """Full pipeline: 3 disks on peg A -> extract state -> plan -> 7 moves."""
        task_cfg = ArmTaskConfig(num_disks=3, num_pegs=3)
        planning_cfg = ArmPlanningConfig()

        # Simulate 3 disks stacked on peg A (x=0.20)
        detections = [
            _make_disk("disk_1", peg_x=0.20, z=0.3),  # smallest, top
            _make_disk("disk_2", peg_x=0.20, z=0.2),
            _make_disk("disk_3", peg_x=0.20, z=0.1),  # largest, bottom
        ]

        # Step 1: Extract symbolic state
        extractor = StateExtractor(task_cfg)
        state = extractor.extract(detections)

        assert "on(disk_3, peg_A)" in state.predicates
        assert "clear(disk_1)" in state.predicates

        # Step 2: Plan
        planner = SymbolicPlanner(planning_cfg, task_cfg)
        goal = SymbolicState(predicates=frozenset(), objects={})
        steps = planner.plan(state, goal)

        # Step 3: Verify optimality
        expected = optimal_move_count(3)
        assert len(steps) == expected, f"Expected {expected} moves, got {len(steps)}"

    def test_5_disk_pipeline_produces_31_moves(self) -> None:
        """5-disk pipeline produces exactly 31 optimal moves."""
        task_cfg = ArmTaskConfig(
            num_disks=5,
            num_pegs=3,
            peg_positions=[[0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]],
        )
        planning_cfg = ArmPlanningConfig()

        detections = [_make_disk(f"disk_{i + 1}", peg_x=0.20, z=0.1 * (5 - i)) for i in range(5)]

        extractor = StateExtractor(task_cfg)
        state = extractor.extract(detections)

        planner = SymbolicPlanner(planning_cfg, task_cfg)
        goal = SymbolicState(predicates=frozenset(), objects={})
        steps = planner.plan(state, goal)

        assert len(steps) == optimal_move_count(5)

    def test_all_moves_are_valid(self) -> None:
        """Every move in the plan references valid disk and peg names."""
        task_cfg = ArmTaskConfig(num_disks=3, num_pegs=3)
        planning_cfg = ArmPlanningConfig()

        planner = SymbolicPlanner(planning_cfg, task_cfg)
        state = SymbolicState(predicates=frozenset(), objects={})
        steps = planner.plan(state, state)

        valid_disks = {f"disk_{i}" for i in range(1, 4)}
        valid_pegs = {"peg_A", "peg_B", "peg_C"}
        valid_objects = valid_disks | valid_pegs

        for step in steps:
            assert step.action == "move"
            assert len(step.args) == 3
            assert step.args[0] in valid_disks  # disk being moved
            assert step.args[1] in valid_objects  # source
            assert step.args[2] in valid_objects  # target
            assert step.args[1] != step.args[2]  # not same source/target
