"""World model — RSSM, dual-stream RSSM, encoder, and MCTS planner."""

from mousedroid.world_model.encoder import MultimodalEncoder
from mousedroid.world_model.mcts import MCTSPlanner
from mousedroid.world_model.protocol import SafetyTraceProtocol, WorldModelProtocol
from mousedroid.world_model.rssm import RSSM
from mousedroid.world_model.stream_fusion import StreamFusion

__all__ = [
    "RSSM",
    "MCTSPlanner",
    "MultimodalEncoder",
    "SafetyTraceProtocol",
    "StreamFusion",
    "WorldModelProtocol",
]

try:
    from mousedroid.world_model.cfc_cell import CfCWrapper
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

    __all__ += ["CfCWrapper", "DualStreamRSSM"]
except ImportError:  # pragma: no cover
    pass  # ncps not installed; CfC features unavailable
