"""Builtin :class:`SkillSpec` packages exposed to OpenClaw.

Each module exports a single ``SPEC: SkillSpec`` describing a portable
skill that an OpenClaw agent on the Mac mini host can invoke either via
the REST mission endpoint (Phase A) or the MCP transport (Phase B).
The factory wires these into the :class:`SkillRegistry` only when
``cfg.openclaw.enabled`` is True so existing deployments still see an
empty registry.

The descriptors here MUST stay aligned with the corresponding
``docs/openclaw_skills/<name>/SKILL.md`` files; the test
``tests/unit/skills/builtin/test_skill_specs_match_docs.py`` enforces
that pairing.
"""

from __future__ import annotations

from mousedroid.skills.builtin.navigate import SPEC as NAVIGATE_SPEC
from mousedroid.skills.builtin.sensor_report import SPEC as SENSOR_REPORT_SPEC
from mousedroid.skills.builtin.voice import SPEC as VOICE_SPEC
from mousedroid.skills.builtin.world_model import SPEC as WORLD_MODEL_SPEC


def all_builtin_specs() -> tuple[object, ...]:
    """Return every builtin :class:`SkillSpec` in stable registration order.

    Stable order matters because :meth:`SkillRegistry.register` logs the
    name of every skill at INFO and operators correlate those lines
    against the publishable docs.
    """
    return (
        NAVIGATE_SPEC,
        SENSOR_REPORT_SPEC,
        VOICE_SPEC,
        WORLD_MODEL_SPEC,
    )


__all__ = [
    "NAVIGATE_SPEC",
    "SENSOR_REPORT_SPEC",
    "VOICE_SPEC",
    "WORLD_MODEL_SPEC",
    "all_builtin_specs",
]
