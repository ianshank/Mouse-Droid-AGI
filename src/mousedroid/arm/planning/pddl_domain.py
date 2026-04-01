"""PDDL domain and problem generator for Tower of Hanoi.

Generates valid PDDL domain and problem files from the current
symbolic state and task configuration. Guarantees optimal solutions
with exactly 2^n - 1 moves for n disks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.arm.protocols import SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTaskConfig

_log = get_logger(__name__)

HANOI_DOMAIN_PDDL = """\
(define (domain tower-of-hanoi)
  (:requirements :strips)
  (:predicates
    (on ?x ?y)
    (clear ?x)
    (smaller ?x ?y)
    (disk ?x)
    (peg ?x)
  )
  (:action move
    :parameters (?disk ?from ?to)
    :precondition (and
      (disk ?disk)
      (clear ?disk)
      (clear ?to)
      (on ?disk ?from)
      (smaller ?disk ?to)
    )
    :effect (and
      (on ?disk ?to)
      (clear ?from)
      (not (on ?disk ?from))
      (not (clear ?to))
    )
  )
)
"""


def generate_domain() -> str:
    """Generate the Tower of Hanoi PDDL domain definition.

    Returns:
        PDDL domain string.
    """
    return HANOI_DOMAIN_PDDL


def generate_problem(
    task_cfg: ArmTaskConfig,
    initial_state: SymbolicState | None = None,
) -> str:
    """Generate a Tower of Hanoi PDDL problem from task config.

    If no initial state is provided, generates the standard initial
    state with all disks stacked on peg_A in order.

    Args:
        task_cfg: Task config with num_disks and num_pegs.
        initial_state: Optional current symbolic state to plan from.

    Returns:
        PDDL problem string.
    """
    num_disks = task_cfg.num_disks
    num_pegs = task_cfg.num_pegs

    disk_names = [f"disk_{i + 1}" for i in range(num_disks)]
    peg_names = [f"peg_{chr(65 + i)}" for i in range(num_pegs)]

    # Objects
    objects = " ".join(disk_names) + " " + " ".join(peg_names)

    # Initial state predicates
    if initial_state is not None:
        init_preds = "\n    ".join(
            f"({pred})" if not pred.startswith("(") else pred
            for pred in sorted(initial_state.predicates)
        )
    else:
        # Default: all disks on peg_A, smallest on top
        init_lines: list[str] = []
        for name in disk_names:
            init_lines.append(f"(disk {name})")
        for name in peg_names:
            init_lines.append(f"(peg {name})")

        # Smaller-than relationships (disk_1 is smallest)
        all_objects = disk_names + peg_names
        for i, d in enumerate(disk_names):
            for j in range(i + 1, len(all_objects)):
                init_lines.append(f"(smaller {d} {all_objects[j]})")

        # All disks stacked on peg_A
        init_lines.append(f"(on {disk_names[-1]} {peg_names[0]})")
        for i in range(num_disks - 2, -1, -1):
            init_lines.append(f"(on {disk_names[i]} {disk_names[i + 1]})")

        # Clear: top disk and all pegs except A
        init_lines.append(f"(clear {disk_names[0]})")
        for name in peg_names[1:]:
            init_lines.append(f"(clear {name})")

        init_preds = "\n    ".join(init_lines)

    # Goal: all disks on last peg
    goal_lines: list[str] = []
    goal_lines.append(f"(on {disk_names[-1]} {peg_names[-1]})")
    for i in range(num_disks - 2, -1, -1):
        goal_lines.append(f"(on {disk_names[i]} {disk_names[i + 1]})")
    goal_preds = "\n      ".join(goal_lines)

    problem = f"""\
(define (problem hanoi-{num_disks}disk)
  (:domain tower-of-hanoi)
  (:objects {objects})
  (:init
    {init_preds}
  )
  (:goal (and
      {goal_preds}
    )
  )
)
"""
    _log.debug("pddl_problem_generated", num_disks=num_disks, num_pegs=num_pegs)
    return problem


def optimal_move_count(num_disks: int) -> int:
    """Calculate optimal move count for Tower of Hanoi.

    Args:
        num_disks: Number of disks.

    Returns:
        Optimal number of moves: 2^n - 1.
    """
    return (1 << num_disks) - 1
