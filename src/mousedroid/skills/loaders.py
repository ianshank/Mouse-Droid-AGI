"""Concrete :class:`SkillLoaderProtocol` implementations.

Three loaders are provided so skills can be defined in:

* **YAML manifests** — ``config/skills/*.yaml`` for production deployments.
* **Markdown agents** — ``src/mousedroid/agents/*.md`` with optional
  YAML front-matter, mirroring the precedent set by ``agents/agent.md``.
* **Code-registered** — programmatic registration for unit tests and
  built-in skills wired from the factory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from mousedroid.logging.setup import get_logger
from mousedroid.skills.protocol import SkillLoaderProtocol, SkillSpec

_log = get_logger(__name__)


_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


def _spec_from_dict(raw: dict, *, source: str) -> SkillSpec:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        msg = f"Skill manifest missing required 'name' field (source={source})"
        raise ValueError(msg)
    tools = raw.get("tools") or raw.get("tool_names") or ()
    if not isinstance(tools, (list, tuple)):
        msg = f"Skill {name!r} 'tools' must be a list (source={source})"
        raise ValueError(msg)
    metadata = raw.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        msg = f"Skill {name!r} 'metadata' must be a dict (source={source})"
        raise ValueError(msg)
    return SkillSpec(
        name=name,
        description=str(raw.get("description", "")),
        tool_names=frozenset(str(t) for t in tools),
        system_prompt=str(raw.get("system_prompt", "")),
        source=source,
        metadata=metadata,
    )


class YAMLManifestLoader:
    """Loads skills from YAML files matching a glob.

    Each YAML file may contain a single skill (top-level dict) or a list
    of skills (top-level list). Empty files are skipped silently.
    """

    def __init__(self, manifest_glob: str, *, root: Path | None = None) -> None:
        self._manifest_glob = manifest_glob
        self._root = root if root is not None else Path()

    def load(self) -> Iterable[SkillSpec]:
        # ``Path.glob`` requires a relative pattern; normalise.
        for path in sorted(self._root.glob(self._manifest_glob)):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                _log.warning(
                    "skill_yaml_load_failed",
                    path=str(path),
                    error=str(exc),
                )
                continue
            if raw is None:
                continue
            entries = raw if isinstance(raw, list) else [raw]
            for entry in entries:
                if not isinstance(entry, dict):
                    _log.warning(
                        "skill_yaml_skipped_non_dict_entry",
                        path=str(path),
                    )
                    continue
                try:
                    yield _spec_from_dict(entry, source=f"manifest:{path}")
                except ValueError as exc:
                    _log.warning(
                        "skill_yaml_invalid",
                        path=str(path),
                        error=str(exc),
                    )


class MarkdownAgentLoader:
    """Loads skills from Markdown files with optional YAML front-matter.

    Files without front-matter are silently skipped so existing design
    docs (e.g. ``agents/agent.md``) are not accidentally registered.
    """

    def __init__(self, dirs: Sequence[Path]) -> None:
        self._dirs = tuple(Path(d) for d in dirs)

    def load(self) -> Iterable[SkillSpec]:
        for d in self._dirs:
            if not d.is_dir():
                _log.debug("skill_markdown_dir_missing", path=str(d))
                continue
            for path in sorted(d.glob("*.md")):
                content = path.read_text(encoding="utf-8")
                match = _FRONT_MATTER_RE.match(content)
                if match is None:
                    _log.debug(
                        "skill_markdown_no_front_matter",
                        path=str(path),
                    )
                    continue
                try:
                    raw = yaml.safe_load(match.group("yaml")) or {}
                except yaml.YAMLError as exc:
                    _log.warning(
                        "skill_markdown_yaml_invalid",
                        path=str(path),
                        error=str(exc),
                    )
                    continue
                if not isinstance(raw, dict):
                    _log.warning(
                        "skill_markdown_front_matter_not_dict",
                        path=str(path),
                    )
                    continue
                # Body becomes the system prompt unless explicitly provided.
                if "system_prompt" not in raw:
                    raw["system_prompt"] = match.group("body").strip()
                try:
                    yield _spec_from_dict(raw, source=f"markdown:{path}")
                except ValueError as exc:
                    _log.warning(
                        "skill_markdown_invalid",
                        path=str(path),
                        error=str(exc),
                    )


class CodeRegisteredLoader:
    """Programmatic loader — yields a fixed sequence of ``SkillSpec``s."""

    def __init__(self, specs: Sequence[SkillSpec]) -> None:
        self._specs = tuple(specs)

    def load(self) -> Iterable[SkillSpec]:
        yield from self._specs


# Static protocol checks.
_LOADER_CHECKS: tuple[SkillLoaderProtocol, ...] = (
    YAMLManifestLoader("none.yaml"),
    MarkdownAgentLoader([]),
    CodeRegisteredLoader([]),
)
del _LOADER_CHECKS


__all__ = ["CodeRegisteredLoader", "MarkdownAgentLoader", "YAMLManifestLoader"]
