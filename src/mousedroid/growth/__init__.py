"""Knowledge distillation for model growth."""

from mousedroid.growth.distillation import KnowledgeDistiller
from mousedroid.growth.protocol import GrowthProtocol

__all__ = [
    "GrowthProtocol",
    "KnowledgeDistiller",
]
