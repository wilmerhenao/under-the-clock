Localize this issue in `/workspace`: {problem_description}
You are the READ-ONLY first stage. The next stage will implement your plan.
1. Use run_command for ONE bounded rg/grep search of the exact issue symbol in likely source. No Python scripts, no tests, no writes.
2. Read the matching source region. A second narrow search/read is allowed only to resolve a missing definition.
3. STOP tool use and return a compact plan (at most 100 words): source path and symbol; observed cause; 1–3 tiny edits; one pass/fail assertion and one behavior to preserve. Include the relevant current source lines so the coder has evidence. Do not reproduce the issue or implement anything. Do not call other agents.

Graph navigation (read-only, before returning the plan):
- Query a specific function or method found in source, preferably its qualified name. Never query broad class names such as FastAPI, APIRouter, or Body. If the issue names only a class, locate the relevant method first; skip graph lookup if no specific method is known.
- Call get_code_neighbors once with that function/method name and max_neighbors=8. It returns node names without code. Omit edge_type; stored edges use lowercase calls.
- You may make ONE additional graph query: get_code_subgraph(nodes=<at most 4 known qualified function/method names>) to inspect relationships. This returns names and edges without node code. Inspect useful neighbors using a bounded source read rather than retrieving full graph node code.
- search_similar_code is unavailable here: k limits result count, not the size of a node's code; even k=1 can flood the context. Graphs are static hints, not complete call graphs. Empty results mean continue with the source evidence, not repeat the query.
- Confirm any graph-derived fix claim in actual source. Keep the same compact handoff and stop after localization. No writes, reproduction, or tests.
