"""AQA — F-050 narrative corrections: ONNX engine reachability, new_z consumers, HF artifact.

Pins the four false claims corrected by the ``narrative-correction-sweep`` over
the Jetson-ONNX delivery surfaces, so none of them can silently re-drift:

1. ``config/jetson_production.yaml`` cannot enable ``engine: onnx_trt`` --
   ``_build_onnx_world_model`` raises when ``cfc_hidden_dim <= 0``, the schema
   default is ``0``, and the production overlay carries no ``model:`` block.
2. Overlays without a ``world_model:`` block build plain ``RSSM``, not
   ``DualStreamRSSM``.
3. ADR-008's ``new_z`` exclusion is justified by non-reproducible sampling, not
   by an absence of consumers -- there are three in production.
4. ``get_safety_trace`` has no production caller, and ``build_planner`` does
   not exist (the real builder is ``build_agent``).

Every read is **whole-file and whitespace-normalized**: per
``.claude/skills/narrative-correction-sweep/SKILL.md``, a claim that reads as
one sentence rendered can be hard-wrapped across two source lines, so a
line-oriented match misses exactly the wrapped instance a reader would catch.

Presence-of-phrase alone is not the pin either: a *qualified* claim should
still contain the words and should still pass. Where a phrase legitimately
survives in corrected form, this module asserts the correction sits with it
rather than blacklisting the phrase.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ADR008 = _REPO_ROOT / "docs" / "architecture" / "ADR-008-world-model-onnx-engine.md"
_SCHEMA = _REPO_ROOT / "src" / "mousedroid" / "config" / "schema" / "world_model.py"
_FACTORY = _REPO_ROOT / "src" / "mousedroid" / "factory" / "world_model.py"
_COMPOSITE = _REPO_ROOT / "src" / "mousedroid" / "world_model" / "composite.py"
_EXPORT_SCRIPT = _REPO_ROOT / "scripts" / "export_dual_stream_rssm_onnx.py"
_ARCHITECTURE = _REPO_ROOT / "docs" / "architecture.md"
_PLANNING_NEXT_STEPS = _REPO_ROOT / "docs" / "planning" / "NEXT_STEPS.md"

# Surfaces that are historical record, not live narrative. A dated analysis, a
# CHANGELOG entry, a peer-review verdict and a change bundle SHOULD quote the
# claim they refute -- rewriting them erases what was wrong.
_HISTORICAL_PREFIXES = (
    "CHANGELOG.md",
    "docs/analysis/",
    "openspec/",
)


def _norm(path: Path) -> str:
    """Whole-file read with runs of whitespace collapsed to one space."""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _tracked(*globs: str) -> list[Path]:
    """Every tracked file matching ``globs``, from ``git ls-files``.

    Discovery rather than a hardcoded roster: the sweep skill's own failure
    case was a five-pattern directory list that silently missed 15 tracked
    docs. A new surface is covered the moment it is committed.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", *globs],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [_REPO_ROOT / line for line in result.stdout.splitlines() if line.strip()]


def _is_historical(path: Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return any(rel == p or rel.startswith(p) for p in _HISTORICAL_PREFIXES)


# --------------------------------------------------------------------------
# Claim 1 — jetson_production.yaml cannot enable engine: onnx_trt
# --------------------------------------------------------------------------


def test_production_overlay_still_has_no_model_block() -> None:
    """The ground truth the corrections rest on.

    If someone adds a ``model:`` block with ``cfc_hidden_dim > 0`` to the
    production overlay, the corrected prose becomes wrong in the other
    direction and this test says so before a reader is misled.
    """
    text = (_REPO_ROOT / "config" / "jetson_production.yaml").read_text(encoding="utf-8")
    assert not re.search(r"(?m)^model:", text), (
        "config/jetson_production.yaml gained a model: block -- the F-050 "
        "narrative corrections assert it has none; re-check every surface."
    )
    assert "cfc_hidden_dim" not in text


def test_dual_stream_overlay_is_the_enabling_one_and_self_gates() -> None:
    """``config/jetson_dual_stream.yaml`` sets 64 and gates it behind review."""
    text = _norm(_REPO_ROOT / "config" / "jetson_dual_stream.yaml")
    assert "cfc_hidden_dim: 64" in text
    assert "Only enable (64+) after human review of training metrics" in text


def test_factory_still_raises_when_cfc_disabled_under_onnx_engine() -> None:
    """The boot-time hard failure the prose now documents."""
    text = _norm(_FACTORY)
    assert "if cfg.model.cfc_hidden_dim <= 0:" in text
    assert "world_model.engine='onnx_trt' requires model.cfc_hidden_dim > 0" in text
    assert "raise ValueError(msg)" in text


def test_schema_default_cfc_hidden_dim_is_zero() -> None:
    """Read from source, not imported, so this pins with or without torch."""
    text = _norm(_SCHEMA)
    assert "cfc_hidden_dim: int = Field(0, ge=0" in text


def test_world_model_config_docstring_does_not_name_jetson_production_as_enabler() -> None:
    """``WorldModelConfig.__doc__`` must not tell operators to flip the engine there.

    Imported rather than read so the pin holds against the runtime object an
    operator actually introspects (``help(WorldModelConfig)``), not only the
    file. ``mousedroid.config.schema.world_model`` is pure Pydantic -- no torch.
    """
    from mousedroid.config.schema.world_model import WorldModelConfig

    doc = re.sub(r"\s+", " ", WorldModelConfig.__doc__ or "")
    assert doc, "WorldModelConfig lost its docstring"

    assert 'Flip ``engine="onnx_trt"`` in ``config/jetson_production.yaml``' not in doc, (
        "WorldModelConfig.__doc__ again names config/jetson_production.yaml as "
        "the enabling overlay; the factory raises ValueError there."
    )
    # Not a blanket ban on the filename: the docstring legitimately names it as
    # the overlay that CANNOT enable the engine. What must never return is the
    # filename presented as the place the flip happens.
    assert "cannot be enabled from ``config/jetson_production.yaml``" in doc
    assert "config/jetson_dual_stream.yaml" in doc
    assert "requires`` ``model.cfc_hidden_dim > 0" in doc or (
        "requires** ``model.cfc_hidden_dim > 0" in doc
    )


def test_world_model_config_docstring_does_not_promise_dualstream_for_bare_overlays() -> None:
    """Claim 2: overlays without a ``world_model:`` block build plain ``RSSM``."""
    from mousedroid.config.schema.world_model import WorldModelConfig

    doc = re.sub(r"\s+", " ", WorldModelConfig.__doc__ or "")
    assert "block load unchanged and continue to use the PyTorch" not in doc, (
        "WorldModelConfig.__doc__ again claims world_model-less overlays run "
        "DualStreamRSSM; cfc_hidden_dim defaults to 0, so they run RSSM."
    )
    assert "build the plain" in doc
    assert "rssm.RSSM" in doc
    assert "cfc_hidden_dim`` defaults to ``0" in doc


def test_adr008_does_not_claim_bare_yaml_instantiates_dualstream() -> None:
    """Same claim 2, in ADR-008's 'Public surface' section.

    Found by the sweep, not by the brief: the wrapped phrasing here
    ("continues to instantiate") differs from the schema docstring's
    ("continue to use"), which is why a remembered-file fix would have
    missed it.
    """
    text = _norm(_ADR008)
    assert "loads unchanged and continues to instantiate the PyTorch `DualStreamRSSM`" not in text
    assert "Those overlays instantiate the plain `RSSM`, **not** `DualStreamRSSM`" in text


def test_adr008_migration_points_at_dual_stream_overlay() -> None:
    """ADR-008 migration steps 1 and 2 must not send operators to production."""
    text = _norm(_ADR008)
    assert "--config config/jetson_production.yaml" not in text, (
        "ADR-008 migration step 1 again exports with the production overlay, "
        "which has no model: block and exports nothing."
    )
    assert "--config config/jetson_dual_stream.yaml" in text
    assert "Edit `config/jetson_dual_stream.yaml` — **not** `config/jetson_production.yaml`" in text
    assert "cannot enable this engine" in text


def test_export_script_docstring_points_at_dual_stream_overlay() -> None:
    """``scripts/export_dual_stream_rssm_onnx.py`` module docstring, claim 1."""
    text = _norm(_EXPORT_SCRIPT)
    assert "--config config/jetson_production.yaml" not in text, (
        "export script CLI example again names config/jetson_production.yaml"
    )
    assert "--config config/jetson_dual_stream.yaml" in text
    assert "``config/jetson_production.yaml`` carries no ``model:`` block" in text


def test_no_live_surface_invites_flipping_the_engine_in_production_overlay() -> None:
    """Repo-wide sweep: the imperative form must not survive anywhere live.

    Whitespace-normalized whole-file match over ``git ls-files``, so a hit that
    wraps across a line break is still caught. Historical surfaces are exempt
    -- they quote the claim in order to refute it.
    """
    claim = re.compile(
        r"(?:[Ff]lip|[Ss]et|[Ee]nable)[^.]{0,160}?`{1,2}engine[^.]{0,80}?"
        r"`{1,2}config/jetson_production\.yaml`{1,2}"
        r"|[Ee]dit `{1,2}config/jetson_production\.yaml`{1,2}:\s*```yaml[^`]{0,200}?onnx_trt",
    )
    offenders = [
        p.relative_to(_REPO_ROOT).as_posix()
        for p in _tracked("*.md", "*.py", "*.yaml", "*.yml", "*.sh")
        if not _is_historical(p) and claim.search(_norm(p))
    ]
    assert not offenders, (
        f"live surface(s) tell operators to enable onnx_trt from the production "
        f"overlay, which hard-fails at boot: {offenders}"
    )


# --------------------------------------------------------------------------
# Claim 3 — ADR-008's new_z equivalence reasoning
# --------------------------------------------------------------------------


def test_adr008_no_longer_claims_new_z_has_no_consumers() -> None:
    """The false justification, in both places ADR-008 stated it."""
    text = _norm(_ADR008)
    # Exact original wording; split only for line length -- the implicit
    # concatenation reproduces the shipped sentence byte-for-byte, which is
    # what the sweep skill means by proving the pin against what shipped.
    no_consumers = (
        "consumers that depend on the specific sample (none, in the current architecture)"
    )
    assert no_consumers not in text, "ADR-008 again claims new_z has no consumers; there are three."
    assert "no downstream consumer depends on the specific sample today" not in text, (
        "ADR-008's Negative-consequences bullet again asserts no new_z consumer."
    )


def test_adr008_names_all_three_new_z_consumers() -> None:
    """Each consumer named with the file that proves it."""
    text = _norm(_ADR008)
    assert "It is **not** justified by an absence of consumers" in text
    assert "`new_z` has three, all in production today" in text
    for proof in (
        "src/mousedroid/orchestrator/_world_model_state_mixin.py",
        "src/mousedroid/world_model/dual_stream_rssm.py",
        "src/mousedroid/agents/navigation.py",
        "src/mousedroid/orchestrator/_action_mixin.py",
    ):
        assert proof in text, f"ADR-008 no longer cites {proof} as a new_z consumer"
    assert "recurrent_input = torch.cat([z, prev_action], dim=-1)" in text
    assert "MCTSPlanner.plan(h, z)" in text
    assert "VLAObservation(h=..., z=...)" in text


def test_adr008_keeps_the_single_step_exclusion_conclusion() -> None:
    """The conclusion was right; only the reasoning was wrong.

    A correction that dropped the exclusion would be a regression: an unseeded
    sampler is not reproducible between two runs of the same engine either.
    """
    text = _norm(_ADR008)
    assert "`new_z` (the posterior Gaussian sample) is **intentionally excluded**" in text
    assert "The exclusion is correct for a **single-step** parity gate" in text


def test_adr008_records_that_multi_step_parity_is_impossible() -> None:
    """The consequence the corrected reasoning forces."""
    text = _norm(_ADR008)
    assert "Multi-step parity between the two engines is therefore impossible" in text.replace(
        "**", ""
    )
    assert "until the sampler's noise becomes an injected" in text
    assert "multi-step parity is impossible" in text.replace("**", "")


def test_new_z_consumers_still_exist_in_the_tree() -> None:
    """Ground truth for claim 3, so the doc pin cannot outlive the code."""
    orchestrator = _REPO_ROOT / "src" / "mousedroid" / "orchestrator"
    mixin = _norm(orchestrator / "_world_model_state_mixin.py")
    assert "self._h, self._z, _, _ = self._world_model.observe_step(" in mixin

    rssm = _norm(_REPO_ROOT / "src" / "mousedroid" / "world_model" / "dual_stream_rssm.py")
    assert "recurrent_input = torch.cat([z, prev_action], dim=-1)" in rssm

    nav = _norm(_REPO_ROOT / "src" / "mousedroid" / "agents" / "navigation.py")
    assert "self._planner.plan(h, z, n_simulations=budget)" in nav

    action = _norm(orchestrator / "_action_mixin.py")
    assert "VLAObservation(h=self._h, z=self._z)" in action


# --------------------------------------------------------------------------
# Claim 4 — get_safety_trace has no production caller; build_planner is fictional
# --------------------------------------------------------------------------


def test_get_safety_trace_has_no_production_caller() -> None:
    """Ground truth: the safety monitor never reaches for the CfC trace.

    Scans ``src/`` for a *call*, excluding the definitions and the composite's
    own delegation. A real caller appearing here means the corrected comments
    became false and want rewriting, not that this test wants relaxing.
    """
    definition_sites = {
        "src/mousedroid/world_model/protocol.py",
        "src/mousedroid/world_model/dual_stream_rssm.py",
        "src/mousedroid/world_model/composite.py",
    }
    callers = []
    for path in _tracked("src/**/*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in definition_sites:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?<!def )get_safety_trace\s*\(", text):
            callers.append(rel)
    assert not callers, (
        f"get_safety_trace gained production caller(s) {callers}; the F-050 "
        "comments in factory/world_model.py and composite.py say it has none."
    )

    safety_dir = _REPO_ROOT / "src" / "mousedroid" / "safety"
    for py in sorted(safety_dir.glob("*.py")):
        assert "get_safety_trace" not in py.read_text(encoding="utf-8"), (
            f"{py.name} now references get_safety_trace"
        )


def test_factory_comment_no_longer_blames_the_safety_monitor() -> None:
    """``factory/world_model.py`` retains the PyTorch engine for imagine_step."""
    text = _norm(_FACTORY)
    assert "the safety monitor's CfC inspection" not in text, (
        "factory/world_model.py again justifies the PyTorch engine with the "
        "safety monitor's CfC inspection; the monitor never calls it."
    )
    assert "MCTS rollouts on the # fallback planner path need the PyTorch graph" in text or (
        "fallback planner path need the PyTorch" in text
    )
    assert "has no production # caller today" in text or "no production" in text
    assert "mousedroid.safety.monitor never calls it" in text


def test_composite_comment_no_longer_claims_the_monitor_is_wired() -> None:
    """``composite.py``: delegation satisfies a Protocol, it wires no consumer."""
    text = _norm(_COMPOSITE)
    assert "keeps the safety monitor wired" not in text, (
        "composite.py again claims delegation keeps the safety monitor wired."
    )
    assert "keeps the protocol satisfied" in text
    assert "no production caller" in text
    assert "mousedroid.safety.monitor` never calls it" in text or (
        "safety.monitor` never invokes it" in text
    )


def test_build_planner_is_not_named_as_a_real_function() -> None:
    """ADR-008 named a builder that does not exist; ``build_agent`` is real."""
    text = _norm(_ADR008)
    assert "`build_planner(cfg, ...)` (when wired) keeps a separate PyTorch" not in text
    assert "there is no `build_planner` function in the tree" in text
    assert "src/mousedroid/factory/cognitive.py" in text
    assert "`build_agent`" in text


def test_build_agent_exists_and_build_planner_does_not() -> None:
    """Ground truth for the symbol swap."""
    cognitive = (_REPO_ROOT / "src" / "mousedroid" / "factory" / "cognitive.py").read_text(
        encoding="utf-8"
    )
    assert "def build_agent(" in cognitive

    defining = [
        p.relative_to(_REPO_ROOT).as_posix()
        for p in _tracked("src/**/*.py", "tools/**/*.py", "scripts/**/*.py")
        if "def build_planner(" in p.read_text(encoding="utf-8")
    ]
    assert not defining, f"build_planner now exists ({defining}); ADR-008 says it does not"


# --------------------------------------------------------------------------
# Hugging Face artifact availability
# --------------------------------------------------------------------------


def test_adr008_does_not_invite_relying_on_the_hub_fallback() -> None:
    """The repo publishes no ``.onnx``; ``onnx_path: null`` cannot boot."""
    text = _norm(_ADR008)
    assert "Or rely on the HF Hub fallback by leaving `onnx_path: null`" not in text, (
        "ADR-008 again invites onnx_path: null; the Hub repo has no artifact."
    )
    assert "`onnx_path` is **required in practice**" in text
    assert "currently publishes **no `.onnx` artifact**" in text
    assert "only `.gitattributes` and `README.md`" in text


def test_adr008_positive_consequence_qualifies_the_hub_fallback() -> None:
    """The 'Positive' bullet must not read as an available capability."""
    text = _norm(_ADR008)
    assert (
        "**HF Hub fallback** means deployments without a baked-in `.onnx` can auto-download "
        "on first boot." not in text
    ), "ADR-008 again presents the Hub auto-download as usable today."
    assert "Not usable today:" in text


def test_adr008_no_longer_calls_the_engine_one_config_flip_away() -> None:
    """Three preconditions, not one flip."""
    text = _norm(_ADR008)
    assert "**`observe_step` becomes one config flip away from <10ms**" not in text
    assert 'Not "one config flip"' in text


def test_architecture_doc_says_the_hf_repo_is_empty() -> None:
    """``docs/architecture.md`` claimed trained weights live there."""
    text = _norm(_ARCHITECTURE)
    assert "Trained weights at `ianshank/mousedroid-dual-stream-rssm` (experimental)" not in text, (
        "docs/architecture.md again claims trained weights are on the Hub."
    )
    assert "`ianshank/mousedroid-dual-stream-rssm` is reserved but **empty**" in text
    assert "no weights and no `observe_step.onnx`" in text


def test_planning_next_steps_marks_the_hf_upload_refuted() -> None:
    """``docs/planning/NEXT_STEPS.md`` recorded an upload that never happened.

    Qualified, not deleted: the record of what was claimed stays readable.
    """
    text = _norm(_PLANNING_NEXT_STEPS)
    upload = "HuggingFace upload: `ianshank/mousedroid-dual-stream-rssm` (experimental)"
    assert upload not in text, (
        "docs/planning/NEXT_STEPS.md again records an HF upload that never landed."
    )
    assert "**REFUTED as an upload**" in text
    assert "no weights and no `observe_step.onnx` were ever pushed" in text


def test_download_weights_script_flags_the_empty_repo() -> None:
    """A live operator script must not imply the repo carries weights."""
    text = _norm(_REPO_ROOT / "scripts" / "download_weights.sh")
    assert "All HuggingFace repos containing MouseDroid model weights" not in text, (
        "download_weights.sh again claims every listed repo contains weights."
    )
    assert '"ianshank/mousedroid-dual-stream-rssm" is currently EMPTY' in text


def test_world_model_config_docstring_states_no_artifact_is_published() -> None:
    """Operator-facing schema docstring carries the artifact truth too."""
    from mousedroid.config.schema.world_model import WorldModelConfig

    doc = re.sub(r"\s+", " ", WorldModelConfig.__doc__ or "")
    assert "No ``.onnx`` artifact is published yet" in doc
    assert "Hub fallback resolves nothing" in doc


def test_no_live_surface_claims_the_artifact_is_downloadable() -> None:
    """Repo-wide sweep for a bare availability claim about the dual-stream repo.

    Tuned against the tree before being trusted, per the skill's guardrail: it
    must not fire on the corrected text (which names the repo *and* says it is
    empty), on the future-tense plans to upload, or on the schema/CLI defaults
    that merely say where a push would go.
    """
    repo = r"ianshank/mousedroid-dual-stream-rssm"
    available = re.compile(
        rf"(?:[Tt]rained weights at|[Ww]eights (?:are )?(?:available|published) (?:at|on)|"
        rf"[Dd]ownload(?:able)? from|[Aa]uto-?downloads? from)\s*`?{re.escape(repo)}",
    )
    offenders = [
        p.relative_to(_REPO_ROOT).as_posix()
        for p in _tracked("*.md", "*.py", "*.yaml", "*.yml", "*.sh")
        if not _is_historical(p) and available.search(_norm(p))
    ]
    assert not offenders, (
        f"live surface(s) present the (empty) dual-stream Hub repo as carrying a "
        f"downloadable artifact: {offenders}"
    )
