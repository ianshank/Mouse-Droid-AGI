"""Security primitives shared across NL command channels.

Currently houses :class:`~mousedroid.security.injection_filter.RegexInjectionFilter`
and its protocol so the LLM gateway, the OpenClaw REST endpoint, and any
future channel adapter all enforce the same prompt-injection envelope
without duplicating regex patterns.
"""
