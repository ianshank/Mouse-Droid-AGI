"""Robot arm training platform — Tower of Hanoi to laundry sorting.

Implements a hierarchical 4-layer architecture:

- Layer 0 (Perception): Depth camera -> YOLO -> 6-DoF pose -> symbolic state
- Layer 1 (Planning): PDDL symbolic planner + LLM replanner
- Layer 2 (World Model): RSSM latent dynamics (reuses ``world_model/``)
- Layer 3 (Control): SAC+HER goal-conditioned policy + grasp/place primitives
"""
