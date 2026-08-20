---
description: Author and maintain OpenSpec change bundles in the repository's house formats
status: active
---

# OpenSpec Change Authoring

Scaffold, author, and validate change bundles under `openspec/changes/`.

## Workflow

1. **Reserve an F-number** in `features.yaml` per ADR-013. Burned numbers
   (F-009 through F-014) must never be reused; start at the next available.

2. **Scaffold the change directory:**
   ```bash
   mkdir -p openspec/changes/<change-id>
   touch openspec/changes/<change-id>/{proposal,design,tasks,peer-review}.md
   ```

3. **Author the four files** using the house delta format:
   - `proposal.md` — problem statement, scope, and success criteria
   - `design.md` — technical design with D-sections (D-1 through D-N)
   - `tasks.md` — bold phase labels, N.M numbering, flat checkboxes
   - `peer-review.md` — adversarial review record

4. **Task ordering is binding:** each task lands green before the next starts.
   Deviations from task wording are recorded inline — declared, not silent.

5. **Validate the bundle:**
   - Every backtick-wrapped repo path must resolve on disk
   - Run `python tools/validate_skill_commands.py` for path hygiene
   - Check `features.yaml` parses and the new entry has correct fields

6. **Update the project registry** (`openspec/project.md`):
   proposed → in progress → implemented, with the landing commit SHA.

## Guardrails

- No hardcoded host/IP addresses in change bundles
- Status vocabulary: `active`, `frozen`, `deferred` only
- `implemented_in` must be a hex commit SHA, never a branch name
