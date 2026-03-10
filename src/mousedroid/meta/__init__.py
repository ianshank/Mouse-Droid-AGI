"""Meta-learning — in-context learning and MAML."""

from mousedroid.meta.in_context import InContextLearner
from mousedroid.meta.maml import MAMLAdapter
from mousedroid.meta.protocol import MetaLearnerProtocol

__all__ = [
    "InContextLearner",
    "MAMLAdapter",
    "MetaLearnerProtocol",
]
