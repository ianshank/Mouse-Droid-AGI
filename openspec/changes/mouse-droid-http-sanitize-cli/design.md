# Design — HTTP gateway constructor sanitise + CLI filter threading

## D-1. Constructor symmetry, not a CLI-only patch

Anthropic and llama_cpp already self-build when `None`. Patching only the
two scripts would leave every future forgetful caller open. The constructor
is the close-every-caller fix; the CLIs still pass the shared factory
instance so REST + MCP + probes share one envelope.

## D-2. Keep HTTP never-raises

Anthropic lets `InjectionRejected` propagate (caller error). The HTTP
gateway historically maps sanitiser exceptions to a neutral `GoalVector`.
F-037 does not flip that contract — a CLI that used to skip the filter
must not start crashing the probe on a rejected payload.

## D-3. Explicit filter still wins

When the factory / orchestrator passes an instance, it is stored as-is.
Self-build is only the `None` path.

## D-4. Frozen arm backend stays frozen

`arm/.../anthropic_backend.py` still has no sanitize. F-008 freeze stays
until hardware is `done`.
