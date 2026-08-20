# Claude Workforce Surface — Model Context Protocol (MCP) Evaluation Notes

> **Context**: Surface evaluation record for Model Context Protocol (MCP) servers
> in the MouseDroid repository.
> **Governance Rule**: D-7 ("Earn a place, not use-all") — every MCP server represents
> attack surface, context budget overhead, and lifecycle complexity.

---

## 1. Selection Philosophy: "Earn a Place"

Adding an MCP server into `.mcp.json` requires clear architectural justification:

1. It must provide capabilities that native agent tools (Read, Grep, Glob, Bash, Edit) cannot provide.
2. It must be secretless (no credential literals; environment variable expansion only).
3. It must not violate edge autonomy or introduce uncontrolled network dependencies.

---

## 2. Adopted MCP Servers (Active in `.mcp.json`)

### `mousedroid` (In-Tree Robot Server)

* **Specification**: Defined in `docs/MCP_OPERATOR_GUIDE.md` and implemented in `src/mousedroid/mcp/`.
* **Transport**: `stdio` (embedded subprocess).
* **Security & Safety**:
  * `MOUSEDROID_MOCK_HARDWARE`: Defaults to `${MOUSEDROID_MOCK_HARDWARE:-true}` to guarantee motion safety in workstation developer sessions.
  * Exposes telemetry inspection, sensor validation probes, motor health queries, and diagnostics.
* **Justification**: Canonical interface for agents to inspect and interact with the MouseDroid robot runtime.

### `github` (GitHub Operations)

* **Package**: `@modelcontextprotocol/server-github` via `npx`.
* **Authentication**: `${GITHUB_PERSONAL_ACCESS_TOKEN:-${GITHUB_TOKEN:-}}` (secretless expansion).
* **Capabilities**: Issue triage, PR reviews, comment tracking, release inspection.
* **Justification**: Essential for agentic workflow orchestration, peer review closeouts, and openspec issue linking.

---

## 3. Evaluate-First MCP Servers (Recorded Decisions; Not Added)

### `grafana` (Observability & Dashboards)

* **Candidate Capability**: Direct querying of Grafana dashboards, Prometheus metrics panels, and alert state.
* **Evaluation Criteria**:
  * Requires a live, reachable Grafana instance and operational API keys.
  * MouseDroid currently inspects metrics via the local `/metrics` Prometheus endpoint (`TelemetryServer`), golden characterization tests, and PromQL verification harnesses.
* **Status**: **DEFERRED (Evaluate-First)**.
* **Revisit Trigger**: When an external Grafana Cloud or production telemetry dashboard server is provisioned for multi-rover fleet operations.

### `huggingface` (Model & Dataset Hub)

* **Candidate Capability**: Hugging Face Hub dataset sync, model weights downloads, model card inspection.
* **Evaluation Criteria**:
  * Relevant for physical AI, VLA (Vision-Language-Action), and offline RL policy retraining.
  * Model weights and ONNX artifacts are currently stored in versioned local asset paths and Jetson container images.
* **Status**: **DEFERRED (Evaluate-First)**.
* **Revisit Trigger**: Post-unfreeze of capability streams (F-008 / F-022 VLA distillation pipeline).

---

## 4. Explicitly Rejected MCP Servers

| Server Category | Reason for Rejection |
| --- | --- |
| **Filesystem / Memory MCPs** | Native IDE and agent tools (Read, Grep, Glob, Write, SQLite memory) provide deterministic, lower-latency access without protocol serialization overhead. |
| **Browser MCPs** | Edge robotics runtime and autonomous firmware do not require interactive web browsing. Unnecessary attack surface. |
| **Database Direct-Exec MCPs** | Database access is mediated through schema-driven repository abstractions (`src/mousedroid/`) rather than arbitrary agent SQL execution. |
