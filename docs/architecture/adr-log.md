# Architecture Decision Log

ADRs record significant, hard-to-reverse decisions. They are immutable once accepted — a reversal is a
*new* ADR that supersedes the old one. New ADRs use [`adr/TEMPLATE.md`](adr/TEMPLATE.md).

> **Numbering note.** The sequence starts at **ADR-004** — ADR-001/002/003 were never filed (early decisions
> predate the ADR practice). ADR-007/008/009 numbering was reconciled across parallel branches (see each doc's
> own numbering note).

| ADR | Title | Status | Date | Area |
|-----|-------|--------|------|------|
| [004](ADR-004-prebuilt-ai-containers.md) | Pre-built AI Containers | Accepted¹ | 2026-03 | Deployment / containers |
| [005](ADR-005-gpu-pretraining-pipeline.md) | GPU Pre-Training Pipeline on Jetson Orin Nano | Proposed | 2026-03-13 | Training |
| [006](ADR-006-telemetry-server.md) | Telemetry Server | Accepted | 2026-03 | Telemetry |
| [007](ADR-007-hailo8-accelerator.md) | Hailo-8 Neural Accelerator for Perception Offload | Accepted | 2026-04 | Perception / hardware |
| [008](ADR-008-world-model-onnx-engine.md) | World-Model ONNX Engine (Tier B2) | Accepted | 2026-05-16 | World model |
| [009](ADR-009-isaac-lab-phase-b.md) | Isaac Lab Phase B (Real-Env Wiring) | Accepted | 2026-05-16 | Simulation |
| [010](ADR-010-cloud-weight-update-ota.md) | Closed-Loop Cloud Retraining + OTA Weight Updates (Tier C1) | Accepted | 2026-05-16 | Learning / OTA |
| [011](ADR-011-mission-closed-loop-safety-projection.md) | Mission Closed-Loop + Safety Projection (Tier C2) | Accepted | 2026-05-16 | Safety |
| [012](ADR-012-spec-driven-harness.md) | Adopt the Spec-Driven Harness (HARNESS_SPEC v2.1) | Accepted | 2026-06-14 | Process / harness |
| [013](ADR-013-f-number-namespaces.md) | F-Number Namespaces + Findings-Only Audit Posture | Accepted | 2026-07-03 | Process |
| [014](ADR-014-cyclomatic-complexity-gate.md) | Cyclomatic-Complexity Gate + Enterprise-Hardening Refactor | Accepted | 2026-07-05 | Code quality / CI |
| [015](ADR-015-bounded-context-latent-memory.md) | Bounded-Context Latent Memory + Corrupted-History Drift Training | Accepted | 2026-07-23 | World model / memory |
| [016](ADR-016-autonomous-orchestrator-disposition.md) | AutonomousOrchestrator Disposition | Accepted | 2026-08-23 | Orchestrator / architecture |
| [l4t-container](ADR-l4t-container.md) | L4T Container Deployment for MouseDroid | Proposed | 2026-03-11 | Deployment |

¹ ADR-004 carries no explicit Status field in its source (predates the template); treated as Accepted — it is
implemented and widely referenced.
