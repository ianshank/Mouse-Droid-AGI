"""Factory builders — SO-ARM100 manipulator stack and its LLM replanner backend.

Driver, planner, environment, controller, perception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.arm.protocols import (
        ArmControllerProtocol,
        ArmDriverProtocol,
        ArmEnvironmentProtocol,
        ArmPerceptionProtocol,
        ArmPlannerProtocol,
        SymbolicPlannerBackend,
    )
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol

_log = get_logger(__name__)


def build_llm_replanner(cfg: Settings) -> Any:
    """Build the configured arm LLM replanner backend.

    Returns a concrete :class:`LLMReplannerProtocol` implementation. When
    ``cfg.arm_planning.llm_replanner`` is None or the config disables the
    backend, returns :class:`NullLLMReplanner` so the existing
    :class:`Replanner` fall-back to symbolic planning is preserved.

    Args:
        cfg: Root settings.

    Returns:
        A concrete LLM replanner conforming to ``LLMReplannerProtocol``.
    """
    from mousedroid.arm.planning.llm_replanners.null_backend import (
        NullLLMReplanner,
    )

    if cfg.arm_planning is None or cfg.arm_planning.llm_replanner is None:
        return NullLLMReplanner()
    rp_cfg = cfg.arm_planning.llm_replanner
    if not rp_cfg.enabled or rp_cfg.backend == "null":
        return NullLLMReplanner()
    if rp_cfg.backend == "anthropic":
        from mousedroid.arm.planning.llm_replanners.anthropic_backend import (
            AnthropicReplanner,
            AnthropicSDKMissingError,
        )

        try:
            return AnthropicReplanner(rp_cfg)
        except AnthropicSDKMissingError as exc:
            _log.warning(
                "anthropic_replanner_sdk_missing_falling_back",
                error=str(exc),
            )
            return NullLLMReplanner()
    # 'llama' is reserved for a future llm_gateway-backed implementation;
    # defaulting to null until that backend ships keeps factory honest.
    _log.debug("llm_replanner_backend_not_implemented", backend=rp_cfg.backend)
    return NullLLMReplanner()


def build_arm_driver(cfg: Settings) -> ArmDriverProtocol:
    """Build robot arm hardware driver based on config.

    Args:
        cfg: Root settings (must have arm config populated).

    Returns:
        Arm driver conforming to ``ArmDriverProtocol``.

    Raises:
        ValueError: If arm config is not populated.
    """
    if cfg.arm is None:
        msg = "arm config required for robot arm platform"
        raise ValueError(msg)

    if cfg.mock_hardware:
        from mousedroid.arm.hardware.mock_arm_driver import MockArmDriver

        _log.info("arm_driver_mock_built")
        return MockArmDriver(cfg.arm)

    from mousedroid.arm.hardware.so_arm100_driver import SoArm100Driver

    _log.info("arm_driver_real_built", port=cfg.arm.serial_port)
    return SoArm100Driver(cfg.arm)


def build_symbolic_planner_backend(cfg: Settings) -> SymbolicPlannerBackend:
    """Build the primary symbolic-planning backend selected by config.

    Standard-factory entry point for the ``planner_backend`` selection promised
    by the ``SymbolicPlanner`` refactor; delegates to the single-source-of-truth
    :func:`~mousedroid.arm.planning.symbolic_planner.make_primary_backend`.

    Args:
        cfg: Root settings (must have arm_planning and arm_task populated).

    Returns:
        Backend conforming to ``SymbolicPlannerBackend``.

    Raises:
        ValueError: If arm planning or task config is not populated.
    """
    if cfg.arm_planning is None or cfg.arm_task is None:
        msg = "arm_planning and arm_task configs required for arm planner backend"
        raise ValueError(msg)

    from mousedroid.arm.planning.symbolic_planner import make_primary_backend

    return make_primary_backend(cfg.arm_planning, cfg.arm_task)


def build_arm_planner(cfg: Settings) -> ArmPlannerProtocol:
    """Build symbolic planner for arm manipulation tasks.

    Args:
        cfg: Root settings (must have arm_planning and arm_task populated).

    Returns:
        Planner conforming to ``ArmPlannerProtocol``.

    Raises:
        ValueError: If arm planning or task config is not populated.
    """
    if cfg.arm_planning is None or cfg.arm_task is None:
        msg = "arm_planning and arm_task configs required for arm planner"
        raise ValueError(msg)

    from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner

    _log.info("arm_planner_built", backend=cfg.arm_planning.planner_backend)
    return SymbolicPlanner(
        cfg.arm_planning,
        cfg.arm_task,
        primary_backend=build_symbolic_planner_backend(cfg),
    )


def build_arm_environment(cfg: Settings) -> ArmEnvironmentProtocol:
    """Build Gymnasium environment for arm training.

    Args:
        cfg: Root settings (must have arm_task and arm_training populated).

    Returns:
        Environment conforming to ``ArmEnvironmentProtocol``.

    Raises:
        ValueError: If arm task or training config is not populated.
    """
    if cfg.arm_task is None or cfg.arm_training is None:
        msg = "arm_task and arm_training configs required for arm environment"
        raise ValueError(msg)

    dof = cfg.arm.dof if cfg.arm is not None else 6

    if cfg.arm_task.task_type == "laundry_sorting":
        from mousedroid.arm.environments.laundry_sorting import LaundrySortingEnv

        _log.info("arm_env_laundry_built")
        return LaundrySortingEnv(cfg.arm_task, cfg.arm_training, dof=dof)

    from mousedroid.arm.environments.tower_of_hanoi import TowerOfHanoiEnv

    _log.info("arm_env_hanoi_built", num_disks=cfg.arm_task.num_disks)
    return TowerOfHanoiEnv(cfg.arm_task, cfg.arm_training, dof=dof)


def build_arm_controller(cfg: Settings) -> ArmControllerProtocol:
    """Build RL controller for arm manipulation.

    Composes a ``SACAgent`` with ``ActionPrimitives`` inside an
    ``ArmController`` that implements ``ArmControllerProtocol``.

    Args:
        cfg: Root settings (must have arm and arm_training populated).

    Returns:
        Controller conforming to ``ArmControllerProtocol``.

    Raises:
        ValueError: If arm or arm_training config is not populated.
    """
    if cfg.arm_training is None or cfg.arm is None:
        msg = "arm and arm_training configs required for arm controller"
        raise ValueError(msg)

    from mousedroid.arm.control.controller import ArmController
    from mousedroid.arm.control.primitives import ActionPrimitives
    from mousedroid.arm.control.sac_agent import SACAgent

    driver = build_arm_driver(cfg)
    agent = SACAgent(cfg.arm_training)
    primitives = ActionPrimitives(cfg.arm, driver)
    _log.info("arm_controller_built", algorithm=cfg.arm_training.algorithm)
    return ArmController(agent, primitives)


def build_arm_perception(
    cfg: Settings,
    hailo_runtime: HailoRuntimeProtocol | None = None,
) -> ArmPerceptionProtocol:
    """Build perception pipeline for arm manipulation.

    When a Hailo-8 runtime is provided and ``yolo_backend`` is
    ``"hailo"`` or ``"auto"``, YOLO detection is offloaded to the
    accelerator via :class:`HailoYOLODetector`.

    Args:
        cfg: Root settings (must have arm_perception and arm_task populated).
        hailo_runtime: Optional Hailo-8 runtime for accelerated YOLO detection.

    Returns:
        Perception facade conforming to ``ArmPerceptionProtocol``.

    Raises:
        ValueError: If arm_perception or arm_task config is not populated.
    """
    if cfg.arm_perception is None or cfg.arm_task is None:
        msg = "arm_perception and arm_task configs required for perception"
        raise ValueError(msg)

    import numpy as np

    from mousedroid.arm.perception.facade import ArmPerception

    # Default identity intrinsics — overridden by camera driver at runtime
    intrinsics = np.eye(3, dtype=np.float64)
    intrinsics[0, 0] = cfg.arm_perception.default_focal_length  # fx
    intrinsics[1, 1] = cfg.arm_perception.default_focal_length  # fy
    intrinsics[0, 2] = cfg.arm_perception.default_principal_x  # cx
    intrinsics[1, 2] = cfg.arm_perception.default_principal_y  # cy

    # Build Hailo YOLO detector if available
    detector = None
    if hailo_runtime is not None and cfg.arm_perception.yolo_backend in ("hailo", "auto"):
        from mousedroid.arm.perception.hailo_detector import HailoYOLODetector

        detector = HailoYOLODetector(cfg.arm_perception, hailo_runtime)
        _log.info("arm_perception_hailo_yolo_built")

    _log.info("arm_perception_built", camera=cfg.arm_perception.depth_camera_type)
    return ArmPerception(
        cfg.arm_perception,
        cfg.arm_task,
        intrinsics,
        object_detector=detector,
    )


# ---------------------------------------------------------------------------
# 4WD rover sim-to-real factory functions (Phase A scaffold)
# ---------------------------------------------------------------------------
