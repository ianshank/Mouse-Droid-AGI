# Peer Review: NemoClaw Integration (rev. D)

- **reviewed_artifact**: user-supplied OpenSpec change bundle `mouse-droid-nemoclaw-integration`
- **basis_commit**: 9e157f0
- **review_date**: 2026-08-11
- **method**: Objective factual verification via Sequential Thinking MCP, codebase grep, and toolchain assertion.
- **outcome**: Verified completely. All premises are correct. Implementation plan created with minor refinements.

## Round 1 (self-review) — summary
Findings 0–8: unverified Phase B premise; unknown OpenShell schema; exporter-demotion error; Antigravity git immaturity; missing features.yaml DAG entries; unsubstantiated estimates; no sandbox rollback; voice scope creep. Dispositioned in rev. A.

## Round 2 (external) — summary
F-IDs sequenced late (accepted — fixed a latent config-dependency bug); enforce-mode rollback.

## Round 3 (counter-findings)
- **C1**: Reviewer hard-coded 90%/95% coverage; repo's `pyproject.toml` uses 93%. (Justified)
- **C2**: `peer-review.md` was dropped. (Restored)
- **C3**: Proposal metadata was dropped. (Restored)
- **C4**: `baselines.md` proposed instead of `baselines.yaml`. (Rejected; YAML matches repo culture)

## Round 4 (Objective Peer Review)
- **F-PR1 (Minor)**: The quality gate template should strictly enforce the repo's 93% coverage threshold.
- **F-PR2 (Minor)**: Property-based tests should be fully enabled as `hypothesis>=6.80` is present in `pyproject.toml`.
- **F-PR3 (Medium)**: F-ID allocation was unspecified. F-024 is taken, F-025 and F-026 are used. Epic assigned to F-027.
- **F-PR4 (Minor)**: `OpenClawConfig` is becoming too large. Suggest nesting via `OpenClawMemoryConfig`.
- **F-PR5 (Low)**: Clarified that baselines are captured on the Orin Nano hardware, not the OpenClaw host.
- **F-PR6 (Medium)**: Episodic 'pagination' in the MCP resource should use cursor-based deque iteration, rather than overloading the existing priority-weighted `sample()`.
- **F-PR7 (Info)**: Adding `deployments/nemoclaw/` namespace is a clean extension.
- **Invariants**: Factory-first, Schema-driven, Asyncio-everywhere (with `to_thread`), and Test-pyramid invariants validated.
