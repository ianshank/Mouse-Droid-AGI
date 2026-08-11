# Design: NemoClaw Integration (OpenClaw via MCP)

## D1: Dual Memory Paths
Memory operates over two distinct paths:
1. `memory_transport: files` — The existing `MarkdownReplayExporter` writes episodes to disk.
2. `memory_transport: mcp` — The `MemoryResourceProvider` serves episodic samples and semantic retrievals via MCP.
If both are enabled, the MCP handlers read the same live memory tier the exporter pulls from. 

*F-PR6 Note: Episodic querying via MCP will be implemented as cursor-based deque iteration (insertion-time ordered) distinct from the existing priority-weighted sampling.*

## D2: Policy Single Source of Truth
The canonical source for skill whitelisting is `SkillSpec.tool_names`. A pure function translates the instantiated `SkillRegistry` into a human/LLM-readable policy document at startup.

## D3: IDE Agnostic
No dependency on `.claude/` hooks. All gating logic must be embedded directly in the `SkillDelegator` and `MissionDispatcher`.

## D4: Cross-Model Objectivity
Peer reviews and sandbox grading must be performed by an independent model. For instance, if an Anthropic model authored the code, a Gemini model must review it (or vice versa).

## D5: Sandbox as Live Policy Operation (Spike)
If NemoClaw/OpenShell proves mature, the deployment will utilize `openshell policy set` to dynamically manage the sandbox profile over the lifetime of a mission. If immature, this design gracefully degrades to static environment limits.

## D6: Baseline-Referenced Evaluation
A net-new `baselines.yaml` (schema-driven) will define acceptable thresholds (e.g., max memory query latency). Evaluations will assert against these values. *F-PR5 Note: Baselines represent on-device (droid-side) metric limits.*

## D7: Inherited Quality Standards
This change MUST respect the repo's existing toolchain gates:
- `mypy --strict` for all new modules
- Coverage gate: `fail_under = 93` (inherited from `pyproject.toml`)
- Property-based tests are required (as `hypothesis` is a confirmed dependency)
- C901 complexity max=15

## D8: Sub-project Exit Ramp
If the OpenShell spike (Phase 4) fails, Phases 1-3 (Memory, Transport Parity) still provide massive value and will be merged independently.

*F-PR4 Note: Configuration extensions in `OpenClawConfig` should utilize nested sub-models (e.g., `OpenClawMemoryConfig`) to manage complexity.*
