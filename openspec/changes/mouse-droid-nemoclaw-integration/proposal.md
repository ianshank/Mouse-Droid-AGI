# Proposal: NemoClaw Integration (OpenClaw via MCP)

- **owner**: Ian Cruickshank
- **created**: 2026-08-10
- **basis_commit**: 9e157f0
- **rev**: D

## Objective
Integrate the NemoClaw sandbox and OpenShell capability into the MouseDroid ecosystem via the Model Context Protocol (MCP), establishing a secure, observable, and strictly governed agentic operations layer.

## Context
The existing OpenClaw integration in MouseDroid is partially built: `OpenClawConfig`, four builtin skills (with paired docs), `MissionDispatcher`, and `MarkdownReplayExporter` exist. However, there is no sandbox security model (OpenShell) or memory retrieval (MCP) implemented, and the work is untracked by the feature harness.

## Scope
1. **Governance Foundations**: Allocate F-IDs, establish baseline schema.
2. **Live Memory**: Implement episodic iteration and semantic MCP handlers.
3. **Transport Parity**: Ensure identical gating logic across REST and MCP channels.
4. **Sandbox Integration (Spike)**: Test and potentially integrate NemoClaw/OpenShell.
5. **Evaluations**: Baseline-referenced testing and observability metrics.

## Dependencies
- Model Context Protocol (MCP) Python SDK
- Hypothesis (for property-based testing)
- NemoClaw/OpenShell (Spike dependent)
