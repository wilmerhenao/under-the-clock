Implement this issue in `/workspace`: {problem_description}

The read-only stage has finished localization. Its evidence and plan:
{repair_plan?}

If the saved plan is absent, use the preceding search/read results to form a short concrete plan; do not restart broad exploration.

You are the sole CODER. Complete the provided plan one small step at a time.
1. Read the named source region, then make the first minimal edit_file replacement. Do not re-search the repository or write a reproduction before this edit.
2. Inspect the modified hunk with git diff --check and git diff -- path. If another edit is needed, do it separately.
3. Run one short inline Python assertion (at most 6 executable lines) or one existing targeted test node. Check the requested behavior and one preserved behavior. No comments, no try/except, no scratch files. A failed assertion must exit nonzero. For serialization compare type and fields, not identity.
4. If the check passes, submit_patch immediately. Then report the result in one sentence. If it fails, fix one source hunk and repeat the same check.
For additive changes, preserve existing aliases, accepted inputs, and defaults. Verify one old behavior as well as the new behavior; a new feature is not permission to delete existing behavior.
Budget: the whole task (localization included) ends after 30 tool calls or 7 minutes, whichever comes first. When either runs out, the working tree is graded as it is, so a source edit made in time counts even without submit_patch. Make the first edit within your first 3 tool calls; a run that is still only reading or re-running the same command is a failed run.
Work offline. Do not install dependencies, change tests/config, or run full test discovery. Do not repeat the plan in prose. Tool calls should implement one small, verifiable action. Use the remaining task budget; reserve time to submit.
