# Agent operating instructions (performance-critical)

- **Honor existing versions first**: scan all local metadata (lockfiles, pyproject, flake, README) before suggesting or pinning anything; never invent a new pin if a version already exists in context.
- **Upgrade discipline**:
  - Read the full changelog/release notes for the target version and treat it as the source of truth.
  - Map API changes: verify function names, arguments, and behaviors against current usage to preempt breakage.
  - Optimize for catching regressions; if any ambiguity or possible breaking change remains, stop and ask for user feedback before proceeding.

## Pythonic development

- Prefer context managers (`with` blocks) for resource management instead of try/except patterns when handling files or other closable resources.
