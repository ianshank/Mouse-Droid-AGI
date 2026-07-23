"""World model — RSSM, dual-stream RSSM, encoder, and MCTS planner."""

import importlib.util

from mousedroid.world_model.bounded_context import BoundedContextMemory
from mousedroid.world_model.checkpoint_migration import (
    StateDict,
    load_rssm_with_migration,
    migrate_state_dict,
)
from mousedroid.world_model.encoder import MultimodalEncoder
from mousedroid.world_model.mcts import MCTSPlanner
from mousedroid.world_model.protocol import (
    LatentContextProtocol,
    SafetyTraceProtocol,
    WorldModelProtocol,
)
from mousedroid.world_model.rssm import RSSM
from mousedroid.world_model.stream_fusion import StreamFusion

__all__ = [
    "RSSM",
    "BoundedContextMemory",
    "LatentContextProtocol",
    "MCTSPlanner",
    "MultimodalEncoder",
    "SafetyTraceProtocol",
    "StateDict",
    "StreamFusion",
    "WorldModelProtocol",
    "load_rssm_with_migration",
    "migrate_state_dict",
]

# CfC classes are only exported when the ncps package is available.
# The lazy import inside CfCWrapper means a bare `from ... import CfCWrapper`
# would succeed even without ncps; checking here ensures the guard is reliable.
if importlib.util.find_spec("ncps") is not None:
    from mousedroid.world_model.cfc_cell import CfCWrapper
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

    __all__ += ["CfCWrapper", "DualStreamRSSM"]
