---
name: repo-map
description: Ranks this repository's functions and classes by how well they match the issue text and lists the source packages, the files that hold the best matches, and the existing tests that mention them. One read-only call; use it first to find where a fix belongs.
---
# repo-map

Run with `run_skill_script`: skill_name `repo-map`, file_path `scripts/locate.py`, and args as a mapping with one key, query, whose value is the issue title plus its key identifiers, error messages, option names and any file names it mentions.

It parses the current source files (async functions included), so it also covers code the code graph misses. It prints:
- the source packages and how many Python files each has (the workspace listing in the task message is often cut off before them);
- the best-matching definitions as path:start-end with their names and parameters;
- the most relevant files;
- the existing test files that mention the top names.

The script never modifies files. Confirm its top locations by reading them before you rely on them.
