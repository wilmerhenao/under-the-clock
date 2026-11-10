---
name: swe-workflow
description: Bounded grep search, a concrete small-step plan, one edit at a time, and observable verification.
---

1. Search the issue's exact symbols with scoped rg/grep, then read the matching source.
2. State up to three concrete steps plus a pass/fail check; keep one step active.
3. Apply one small source edit and inspect its diff before the next edit.
4. Run the focused assertion/test. An exception or failed assertion must exit nonzero.
5. Submit the verified patch. Do not edit tests/config or repeat an already answered search.
