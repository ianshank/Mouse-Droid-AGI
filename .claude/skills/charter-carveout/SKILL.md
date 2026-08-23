---
name: charter-carveout
description: Decide whether a change needs a ratified CHARTER.md Section 3 carve-out before it can land, and shape one in the house format if so — using the three existing ratified carve-outs as worked examples.
status: active
---

# Charter Carve-Out

`docs/CHARTER.md` §3 draws three hard lines. Crossing one without a ratified
carve-out is a scope violation, not a bug to route around silently.

## Why this exists

CHARTER §6 already says the right thing in the abstract — "escalate, don't
unilaterally implement" — but gives no procedure for *recognizing* that a
change needs escalating in the first place. `config-guardian` guards schema
shape, `security-scanner` guards leaks, `openspec-change` authors bundles —
none of them owns "does this specific change need a charter amendment, and
what does the entry look like." This skill is that missing procedure.

## The decision procedure

Ask, of the change under consideration:

1. Does it let the rover or arm move (a real actuator command, not a
   simulated or logged one) **without** an explicit, human-gated
   authorization path — i.e. does the default posture stop being no-motion?
2. Does it place LLM inference or training work **inside** the 30 Hz hot
   loop, rather than dispatched off-loop at a slow-cadence seam — native
   async I/O for I/O-bound work (e.g. an LLM gateway's API call), or
   `asyncio.to_thread` for blocking/CPU-bound work (e.g. a torch operation)?
3. Does it change runtime behaviour by editing source, rather than through a
   YAML field or `MOUSEDROID_*__*` environment variable that defaults to the
   pre-existing, safe behaviour?

**Any "yes" needs a ratified carve-out before the change can land.** Read the
three existing ones in `docs/CHARTER.md:76-110` (Jetson + USB-C smoke
validation #106, cloud LLM egress #107, on-device incremental learning #135)
as the literal template — each is: a bold title with a PR number and date,
prose describing the bounded expansion of scope, and the specific
`Field(...)` name(s) whose safe default is the compensating control.

**If in doubt, treat it as needing a carve-out.** The two failure modes here
are not symmetric. Proposing an unneeded carve-out costs one human "no,
that's already covered" — visible, cheap, and easily corrected in review. A
change that skips a carve-out it actually needed produces no CHARTER.md diff
at all for a reviewer to notice; the violation ships silently inside
whatever PR needed it. When the three questions above are genuinely
ambiguous, resolve the ambiguity toward escalating, not toward proceeding.

## What to do with the answer

- **None of the three apply:** proceed. No carve-out needed — but **state
  this explicitly, with the reasoning, in the PR or commit description**
  that relies on it (e.g. "no CHARTER §3 carve-out needed: this change adds
  a config field, gates a new default-OFF path, and never touches the hot
  loop or actuation"). A silent "no carve-out needed" conclusion is exactly
  as unreviewable as a silently-skipped carve-out — the point of writing it
  down is to give a human reviewer something to check, not to create paper
  compliance.
- **One or more apply:** this is a human decision per CHARTER §6, not
  something this skill — or any agent — authorizes on its own. Surface the
  tradeoff (what scope expands, what the compensating gate would be) and
  stop. **Never add a carve-out bullet to `docs/CHARTER.md` unilaterally.**
  If the human approves, shape the entry following the three existing
  carve-outs' exact format, and record the deviation in the landing
  feature's `features.yaml` `notes:` field as well as the CHARTER.md prose —
  not only in a PR body, which is easy to lose track of once merged.

## Worked examples already in the tree

- `#107` (cloud LLM egress) shows the fullest shape: a `Literal[...]` field
  defaulting to the pre-existing safe backend, a pre-egress filter named and
  cited by file path, and an explicit statement of what the filter does and
  does not defend against.
- `#135` (on-device learning) shows a **soak-gated** carve-out — the config
  path exists and defaults safely, but is additionally held off the live
  rover pending a separate operational gate. Use this shape when "off by
  default in config" isn't itself sufficient assurance yet.

## Guardrails

- This skill documents the decision procedure and the entry's shape; it does
  not grant authority to create or approve a carve-out. That authority stays
  with the human decider named in CHARTER §6.
- A "no carve-out needed" conclusion is a claim like any other in this
  codebase's discipline — it should be checkable, not merely asserted. State
  the reasoning inline; do not treat silence as an implicit yes.
- Re-read `docs/CHARTER.md` §3 and §6 directly before applying this
  procedure to a specific change — this skill summarizes them, it does not
  replace reading the source.
