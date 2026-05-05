# mousedroid-voice

**Channel:** REST + MCP
**Actuation:** no — only the speaker is touched.

Trigger a Piper TTS phrase from the personality phrase bank, or play a
short operator-supplied string verbatim.

## Sample prompts

* "Greet the operator"
* "Acknowledge the mission"
* "Say: 'I see you, scoundrel'"

## Schema

Exactly one of `event` or `text` must be supplied:

* `event` (string) — phrase-bank event name
  (`"greeting"`, `"obstacle_detected"`, etc.).
* `text` (string ≤512 chars) — free-form text rendered by the same
  Piper TTS pipeline.
* `valence` (-1..1, optional) — personality-driven inflection hint.

## Refusal modes

The voice handler refuses text longer than 512 characters, plus the
shared envelope (rate limit, bearer auth, prompt injection).
