# MouseDroid Documentation

Start with the root [`README.md`](../README.md) for the project overview. This page is the navigation hub
for the `docs/` tree.

## Start here

- [CHARTER.md](CHARTER.md) — the project constitution (vision, scope, invariants, roadmap); sits above the
  other guidance docs.
- [../AGENTS.md](../AGENTS.md) · [../SKILLS.md](../SKILLS.md) · [../CLAUDE.md](../CLAUDE.md) — behavioural
  contracts and project facts for agentic contributors.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — how to build, test, and submit changes.

## Architecture

- [architecture/c4-overview.md](architecture/c4-overview.md) — **canonical** C4 diagram index (Context →
  Container → Component), routing to the per-area component diagrams.
- [architecture.md](architecture.md) — single-page prose walkthrough (Levels 1–4 with all sub-diagrams).
- [architecture/adr-log.md](architecture/adr-log.md) — Architecture Decision Log (ADR-004…015 + l4t-container);
  new ADRs use [architecture/adr/TEMPLATE.md](architecture/adr/TEMPLATE.md).

## Reference

- [glossary.md](glossary.md) — MSE-6, RSSM, MCTS, BDI, and other terms.
- [api-reference.md](api-reference.md) — telemetry REST/WebSocket surface + MCP tools.
- [testing.md](testing.md) — pytest tiers, the 85% coverage gate, the spec harness.
- [deployment.md](deployment.md) — flash/deploy/service, NVMe setup, probe-first bring-up.
- [development.md](development.md) — developer environment and local loop.
- [training.md](training.md) — the offline GPU training pipeline.

## Operator runbooks

`runbooks/` is the **canonical** operator-procedure home. (`operations/` and `operator/` hold older
Jetson smoke docs of overlapping scope, kept for now — see the note below.)

- [runbooks/jetson-full-bringup.md](runbooks/jetson-full-bringup.md) — full rover bring-up.
- [runbooks/jetson-full-validation.md](runbooks/jetson-full-validation.md) — the full validation pipeline.
- [runbooks/jetson-rover-smoke.md](runbooks/jetson-rover-smoke.md) — USB-C rover smoke test.
- [runbooks/jetson-claude-pilot-deploy.md](runbooks/jetson-claude-pilot-deploy.md) — cloud/local LLM pilot deploy.
- [runbooks/jetson-on-device-learning.md](runbooks/jetson-on-device-learning.md) — on-device incremental learning.
- [runbooks/jetson-alayaworld-spike.md](runbooks/jetson-alayaworld-spike.md) — AlayaWorld distillation spike.
- [runbooks/claude-code-on-jetson.md](runbooks/claude-code-on-jetson.md) · [runbooks/mlflow-local-ui.md](runbooks/mlflow-local-ui.md)
  · [runbooks/secret-scanning.md](runbooks/secret-scanning.md) · [runbooks/history-purge.md](runbooks/history-purge.md).

Failure-recovery **playbooks** live in [playbooks/](playbooks/) (camera, LiDAR, ESP32, GPIO, voice, bring-up, replay).

> **Consolidation note.** `operations/jetson_smoke_runbook.md` and `operator/JETSON_SMOKE_RUNBOOK.md` overlap
> `runbooks/jetson-rover-smoke.md` but are **not** duplicates — each covers different hardware (USB-C rover vs
> IMX500/NVMe/Hailo vs the SMOKE_REPORT template). A future content-merge into `runbooks/` is tracked; for now
> all three are indexed here.

## Operator guides

- [MCP_OPERATOR_GUIDE.md](MCP_OPERATOR_GUIDE.md) · [operator/JETSON_REMOTE_LLM_SETUP.md](operator/JETSON_REMOTE_LLM_SETUP.md)
- [jetson-runner-setup.md](jetson-runner-setup.md) — self-hosted GitHub Actions runner on the Jetson.
- [hailo_model_compilation.md](hailo_model_compilation.md) — Hailo-8 HEF model compilation.

## Product & planning

- **PRDs** — [prd/](prd/) (GPU pretraining, L4T container, prebuilt LLM container).
- **Planning** — [planning/](planning/) and [superpowers/](superpowers/) (dated design plans + specs).
  The root [`NEXT_STEPS.md`](../NEXT_STEPS.md) is the **canonical** forward roadmap; `planning/NEXT_STEPS.md`
  is a legacy v0.3.0-era snapshot.
- **Analysis** — [analysis/](analysis/) (coverage, test-suite, validation checklists, spikes).

## Skills

- [openclaw_skills/README.md](openclaw_skills/README.md) — the OpenClaw skill catalog.

## Related work

- [related-work.md](related-work.md) — academic / industry prior art referenced by the design.
