"""Knowledge distillation for model growth."""

from mousedroid.growth.coordinator import GrowthDistillationCoordinator, SampleBatchFn
from mousedroid.growth.distillation import DistillObjective, KnowledgeDistiller
from mousedroid.growth.protocol import GrowthProtocol
from mousedroid.growth.slot_store import GrowthSlotStore, StudentSlot
from mousedroid.growth.student import StudentVLAPolicy, VLATeacherModule

__all__ = [
    "DistillObjective",
    "GrowthDistillationCoordinator",
    "GrowthProtocol",
    "GrowthSlotStore",
    "KnowledgeDistiller",
    "SampleBatchFn",
    "StudentSlot",
    "StudentVLAPolicy",
    "VLATeacherModule",
]
