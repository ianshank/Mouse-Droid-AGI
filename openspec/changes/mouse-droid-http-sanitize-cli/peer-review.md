# Peer review — HTTP gateway constructor sanitise + CLI filter threading

## Verdict table

| Claim | Verdict |
|---|---|
| Orchestrator already threads `build_injection_filter` | **CONFIRMED** — `factory/orchestrator.py` |
| Anthropic / llama_cpp already self-build when None | **CONFIRMED** |
| `translate_mission.py` / `ask_rover.py` called `build_llm_gateway(settings)` with no filter | **CONFIRMED** |
| None-skip was a documented unit-test seam, not a production orchestrator hole | **CONFIRMED** — still a live CLI hole |
| Frozen arm anthropic backend stays out of scope | **CONFIRMED** — F-008 freeze |

## What survives review unchanged

- HTTP never-raises still swallows sanitiser exceptions.
- Explicit factory filter instance is stored unchanged.
- No CHARTER §3 carve-out.

## Load-bearing pins

1. `OpenAICompatibleLLMGateway(cfg)._injection_filter` is a `RegexInjectionFilter`.
2. Default payload `ignore previous instructions...` never POSTs.
3. CLI `build_llm_gateway` kwargs include the `build_injection_filter` return value.
