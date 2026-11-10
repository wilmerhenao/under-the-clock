---
name: verify-before-submit
description: >
  Pre-submit gate: targeted test evidence, patch hygiene, then submit_patch.
---

# Verify before submit

1. Run **only** the targeted test node or a short assertion-based repro for this fix. Printed booleans and swallowed exceptions are not passing evidence.
2. `git status` / `git diff`: no scratch under `/workspace`, no test/config edits.
3. Scratch belongs in `/tmp`.
4. On passing assertions and a clean diff → `submit_patch` → short summary.
5. If the targeted test fails → fix source again; do not edit tests.
