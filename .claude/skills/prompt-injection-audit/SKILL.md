---
name: prompt-injection-audit
description: Security audit workflow for testing and fuzzing natural language mission inputs against the pre-egress prompt injection defense filter.
---

# Prompt Injection Audit Skill

Workflow for fuzzing and validating prompt injection filters before egress to remote LLM gateways.

## Target Paths

- Composite LLM Gateway: `src/mousedroid/llm_gateway/composite_gateway.py`
- Security Filter: `src/mousedroid/security/injection_filter.py`
- Security Tests: `tests/security/test_pre_egress_injection_sanitization.py`

## Execution Steps

1. Initialize `CompositeLLMGateway` with `enable_injection_filter=True`.
2. Fuzz with jailbreak payloads (e.g. system prompt overrides, instruction bypasses).
3. Confirm rejected payloads yield e-stop goal vectors with `is_safe=False`.
4. Validate that legitimate navigation commands pass without false positives.
