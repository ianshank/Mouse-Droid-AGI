"""Import all protocol modules and verify protocol classes exist.

This gives coverage credit for pure-interface protocol files.
"""

from __future__ import annotations

from typing import Protocol


def test_curiosity_protocol_importable():
    from mousedroid.curiosity.protocol import CuriosityProtocol

    assert issubclass(CuriosityProtocol, Protocol)


def test_efficiency_protocol_importable():
    from mousedroid.efficiency.protocol import EfficiencyProtocol

    assert issubclass(EfficiencyProtocol, Protocol)


def test_growth_protocol_importable():
    from mousedroid.growth.protocol import GrowthProtocol

    assert issubclass(GrowthProtocol, Protocol)


def test_learning_protocol_importable():
    from mousedroid.learning.protocol import ContinualLearnerProtocol

    assert issubclass(ContinualLearnerProtocol, Protocol)


def test_memory_protocol_importable():
    from mousedroid.memory.protocol import MemoryProtocol, ReplayBufferProtocol

    assert issubclass(MemoryProtocol, Protocol)
    assert issubclass(ReplayBufferProtocol, Protocol)


def test_meta_protocol_importable():
    from mousedroid.meta.protocol import MetaLearnerProtocol

    assert issubclass(MetaLearnerProtocol, Protocol)


def test_reward_protocol_importable():
    from mousedroid.reward.protocol import RewardModelProtocol

    assert issubclass(RewardModelProtocol, Protocol)


def test_openclaw_protocol_importable():
    from mousedroid.openclaw.protocol import OpenClawProtocol

    assert issubclass(OpenClawProtocol, Protocol)


def test_scaling_protocol_importable():
    from mousedroid.scaling.protocol import ScalingProtocol

    assert issubclass(ScalingProtocol, Protocol)
