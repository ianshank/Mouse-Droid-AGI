---
description: Full-cycle robot arm manipulation training — Tower of Hanoi to laundry sorting
status: frozen
unfreeze: "F-008 done on the Jetson + 30-day Phase-3b production soak (see NEXT_STEPS.md)"
---

# Robot Arm Trainer

You are a senior robotics software architect working on MouseDroidAGI's robot arm training platform. Follow all invariants from CLAUDE.md.

## Project Configuration

- **Robot Model**: $ARGUMENTS (default: SO-ARM100)
- **Task**: Tower of Hanoi (primary), Laundry Sorting (generalization)
- **Sim Engine**: MuJoCo
- **RL Algorithm**: SAC + HER
- **Depth Camera**: RealSense D435i
- **Compute**: RTX 5070 (training), Jetson Orin Nano (inference)

## Architecture: Hierarchical 4-Layer System

### Layer 0 — Perception (`arm/perception/`)
- RealSense D435i depth camera driver
- YOLO object detection (disks, garments)
- 6-DoF pose estimation (PnP solver)
- Symbolic state extraction: detections -> PDDL predicates

### Layer 1 — Symbolic Planning (`arm/planning/`)
- PDDL domain/problem generation for Tower of Hanoi
- Pyperplan integration for optimal plan solving
- LLM replanner for failure recovery (Claude API)
- Laundry sorting rule engine

### Layer 2 — World Modeling (reuse `world_model/`)
- RSSM latent dynamics model
- MCTS for subgoal search
- Dreamer-V3 imagination rollouts

### Layer 3 — Motor Control (`arm/control/`)
- SAC + HER goal-conditioned policy (Stable-Baselines3)
- Pre-trained grasp/place action primitives
- Trajectory generation and smoothing

## Milestones

1. **Simulation Environment** — MuJoCo scene with robot + Tower of Hanoi puzzle, Gymnasium wrapper
2. **Perception Stack** — Depth processing, YOLO detection, pose estimation, symbolic state
3. **Symbolic Planning** — PDDL domain, Pyperplan solver, optimality validation
4. **RL Training Pipeline** — SAC+HER agent, reward shaping, curriculum learning (1->3->5 disks)
5. **Sim-to-Real Transfer** — Domain randomization, ONNX export, Jetson deployment
6. **Laundry Generalization** — YOLOv11 laundry training, soft-body grasping, 3-basket routing
7. **Production Hardening** — Observability, fault recovery, CI/CD

## Quality Gates Per Milestone

- Unit tests: 90%+ coverage on new code
- Integration test: full stack smoke test in sim
- Performance: training converges within budget, inference < 50ms
- All public APIs documented (Google docstrings)
- `ruff check` + `mypy --strict` clean

## Key Constraints

- No hardcoded robot parameters — use URDF/YAML configs
- No hardcoded poses — load from `config/robot_arm_default.yaml`
- Version perception models (v1_disk_detector, v2_laundry_yolo)
- GPU-agnostic: auto-detect CUDA, fall back to CPU, support TensorRT
- Reproducible: seed numpy/torch/mujoco, log git SHA
- All arm imports are lazy (only when `platform == robot_arm`)
