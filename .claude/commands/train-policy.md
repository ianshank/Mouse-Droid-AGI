---
description: Train an RL policy for the robot arm platform
---

# Train Policy

Train a reinforcement learning policy for the robot arm. Uses SAC+HER by default.

## Arguments

- `$ARGUMENTS` — config file path or training preset (default: `configs/hanoi_3disk.yaml`)

## Workflow

1. Validate environment setup (MuJoCo imports, CUDA availability, config loads)
2. Create Gymnasium environment from config (tower_of_hanoi or laundry_sorting)
3. Initialize SAC+HER agent with hyperparameters from config
4. Apply curriculum stage settings (num_disks, reward weights)
5. Run training loop with:
   - WandB/TensorBoard logging
   - Checkpoints every N steps (from config)
   - Video rollouts every M steps
   - Early stopping on convergence
6. Evaluate final policy (success rate, episode length)
7. Export checkpoint and log results

## Key Files

- `src/mousedroid/arm/control/sac_agent.py` — SAC+HER agent
- `src/mousedroid/arm/environments/tower_of_hanoi.py` — Gymnasium env
- `src/mousedroid/arm/environments/curriculum.py` — Curriculum manager
- `src/mousedroid/arm/environments/reward_shaping.py` — Reward functions
- `config/robot_arm_training.yaml` — Training hyperparameters

## Expected Output

- Policy checkpoint: `weights/arm/{task}_{stage}_final.pt`
- Training metrics logged to TensorBoard/WandB
- Evaluation summary printed to console
