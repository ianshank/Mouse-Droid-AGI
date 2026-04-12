"""GPU pre-training pipeline for MouseDroid (ADR-005).

Provides orchestration, GPU monitoring, and batch tuning for the
4-phase training pipeline: RSSM -> Warm-start -> BDI -> Constitutional RL.
"""

from __future__ import annotations
