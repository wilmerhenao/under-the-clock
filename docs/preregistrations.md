# Pre-registered analysis plans

The paper calls five tests confirmatory: the healing criteria, the two same-session comparisons, the role-aware
score's next-action test with the speculative-decoding test, and the tolerant parser's task session. Their plans are copied below verbatim from the project's
dated logbook, each with the entry it comes from, the time that entry was written and when the run it governs was
pushed and ran. The logbook's internal names: `c70` / `c55` = the 70% / 55% cuts; `c70kd` / `c55kd` (served as
`c70+kd~`) = the cut healed with the full model as teacher (KL) and merged into the 4-bit weights by stochastic
rounding; `full2` = a second run of the full model; `c55role` = the 55% cut chosen by the role-aware score; `G11a` =
*Short-thought*; arm a / arm b of a healing run = the KL arm / the next-token arm; `sub-v9c` = the agent bundle.

## 1. Healing the cut from the full model (Section 4)

Entry B176, written 08:00 UTC 2026-11-04. The healing notebook (`gemma-4-31b-agent-pruned-model-teacher-healing` v1)
was queued at 07:51 UTC the same day and started running at 15:57 UTC 2026-11-05, so the plan predates every result
of the run, though not its push. Results: B186.

- **Pre-registered reading:** start eval should reproduce part 1 / serve v3 (teacher ~0.258 own nats; student ~+0.08).
  Healing works if arm a's own-token damage falls below +0.04 and its KL falls by half; the teacher matters if arm a's
  KL ends clearly below arm b's at equal steps. LoRA's serving cost decides between LoRA and N26 (scale-only healing).


## 2. Same-session comparison, 70% cut (Section 4)

Entry B192, written 04:10 UTC 2026-11-07; `g4dev-eval-rg-same4-gpu` v1 pushed 08:17 UTC 2026-11-07. Results: B197.

- **Pre-registered analysis of same4 (before its push):** analysis set = per block, the tasks every model run finished;
  both blocks pooled. (1) First the in-design A/A: full vs full2, discordant tasks and exact McNemar. (2) Primary: c70 vs
  the full model, per task d = c70 - (full + full2)/2, two-sided exact sign-flip permutation test of mean d = 0; also
  c70 vs full and vs full2 by exact McNemar (both must point the same way). (3) Secondary: c70kd vs c70 (McNemar) and vs
  the full model (as in 2). (4) Descriptive per model: timeouts, task time, malformed-call share (unambiguous `read_file`
  counts and tool-error share first; the regex detector has false positives on code inside edit/write calls), loop
  sessions. Detectable: an effect the size of session B's (~11% of tasks) gives ~14/2 discordant (p ~ 0.004); half of it
  is not detectable, so no equivalence claims. **Gate for c70kd:** serve-v5's `c70+kd~` ran with rc 0 and served damage
  <= +0.04 (the review: stochastic rounding is unbiased in the weights, not in the loss; it adds variance ~|dW| s^2 per
  weight); else MODELS = full, c70, full2.

## 3. Same-session comparison, 55% cut (Section 4)

Entry B199 (21:35 UTC 2026-11-07). The analysis is B192's plan (section 2 above, written 04:10 UTC 2026-11-07);
`g4dev-eval-rg-same55-gpu` v1 was pushed at 21:32 UTC 2026-11-07, after the gate below was checked. Results: B200.

- **Gate (push only if all hold):** heal-c55 COMPLETE with arm a's held-out own-token damage <= +0.06 (two thirds of
  the damage recovered; B187's "healing works" bar is +0.04) and its adapter exported; no GPU session of ours running;
  rebuild without diff. Otherwise no push, and the user is told.
- **Pre-registered analysis (as B192):** analysis set = per block, the tasks every run finished; A/A full vs full2
  first; primary: c55 vs mean(full, full2), exact sign-flip; secondary: c55kd vs c55 (McNemar) and vs mean(full,
  full2); descriptive: timeouts, task time, calls, malformed (`eval_swallowed.py`), loops; cross-session: c55 vs
  same4's c70 on the shared tasks (descriptive only, different sessions).


## 4. The role-aware score and its next-action test; the speculative-decoding test (Section 4)

Entry B202, written 03:40 UTC 2026-11-09 (one amendment at 03:50 UTC, before any serve-v6 data, marked in the text).
`g4dev-prune-score4-gpu` v1 then `g4dev-prune-serve6-gpu` v1 (pushed 05:26 UTC 2026-11-09; results B203-B204);
Run 2' = `g4dev-prune-serve7-gpu` (v1 pushed ~06:33 UTC failed on a bug of ours, v2 pushed 07:00 UTC 2026-11-09;
results B205-B206).

- **Adopted GPU plan (at most ~5 h of 4 x L4, one session at a time; nothing pushed after ~12:00 UTC 11-09):**
  Run 1 = score-v4 then serve-v6 (below); Run 2 (one block of 56 tasks, c55 vs c55role, one-sided exact McNemar
  p < 0.05) only if Run 1's primary reaches p < 0.01; otherwise Run 2' (ngram speculative decoding speed test).
- **Score (fixed before any data):** per group g, MS_g = mean square of the channel's input to down_proj over the
  positions of g; `role_ff = sqrt(mean_g [MS_g / median_channels(MS_g)]) x ||down_proj column||`, the five roles with
  equal weight, per layer (the median is over the layer's channels). The plain scores (agent_ff, agent_own_ff) are
  measured again in the same pass.
- **Run 1a, `g4dev-prune-score4-gpu` v1 (run score-v4):** score-v1/v3's 300 measurement rows, seed 0, 8,192 tokens,
  bf16 model on 4 x L4, score-v3 attached. **Check (pre-registered):** same examples and token counts as score-v3
  (2,374,829 tokens, 170,242 own), recomputed agent_ff rank correlation >= 0.999 in every layer, same down_proj norms;
  if not, role_ff is not used. Output `scores.npz` = every score-v3 array unchanged (so `agent_ff`, and c55, are
  exactly serve-v5's and same55's) + role_ff + each role's own score + the re-measured plain scores. Expected ~1.8 h.
- **Run 1b, serve-v6 (to be pushed when 1a passes its check):** c55 (agent_ff, as before) vs c55role (role_ff), both
  single channels at 55% (multiple of 512, down_proj re-quantized with the same activation weighting), on 80 held-out
  rows x 4 states (~200 states), greedy, 512 tokens; no T 0.2 pass, no full_b; served nats per role of the token added
  to the held-out logprob pass. **Amended 03:50 UTC, before any serve-v6 data:** the full model stays in (its per-role
  served loss is the reference for both cuts; +~20 min); the logprob pass runs on serve-v5's 40 rows (`QUALITY_ROWS`),
  the states on all 80; analysis `kernels/analysis/role_test.py` (written and tested on synthetic output before the run).
  - **Primary:** malformed share (unparsed or a swallowed parameter, `serve_next_action.py`, of the states with a
    call) of c55role at most half of c55's, and one-sided Fisher exact p < 0.05.
  - **Secondary:** recorded-call agreement c55role >= c55 + 0.05; served damage (agent_own_nats) c55role <= c55 + 0.03.
  - **Descriptive:** served nats per role; kept-channel overlap; each role's top-5% channels kept by each cut; the 109
    states shared with serve-v5 (c55 should reproduce serve-v5's c55 outputs).
  - Reported either way. Prior (rebuttal): ~25% for <= 5/100; likeliest 15% -> 8-10%.
- **Run 2' prepared (04:15 UTC, not pushed):** `kernels/prune-serve7-gpu` (run serve-v7, `build_serve.py --v7`): the full
  model, then the same weights with n-gram speculative decoding (`speculative_config` {method ngram, 4 tokens, lookup
  2-4}) as `full_b`; next calls at serve-v5's ~109 states; decode speed at 2k / 16k / 8 streams and on 16 real agent
  turns at 1 and 8 streams; no quality pass. **Criteria (rebuttal, fixed now):** behaviour kept if the speculative
  engine's next calls agree with plain greedy's at >= 0.70 (the numerics floor's lower bound) and malformed calls are
  <= 3/100; speed reported as same-session ratios. Analysis `kernels/analysis/spec_test.py`. CPU tests: serve test
  modes V6 and V7 (synthetic data) pass; the fake vLLM now returns `token_ids` as vLLM does.

## 5. The tolerant tool-call parser on the 70% cut (Section 5)

Entry B209, written 20:20 UTC 2026-11-10 and committed before `g4dev-eval-rg-tol70-gpu` v1 was pushed at
20:25 UTC; the analysis script (`kernels/analysis/tol70_report.py`) was committed at 20:30, before any result. Results:
B211.

- **Pre-registered analysis (fixed before the push):** analysis set = the block-0 tasks `c70tol` finishes.
  1. Primary: `c70tol` vs same4's c70 on those tasks: solved, discordant tasks, exact McNemar (two-sided). This is a
     cross-session comparison, so the A/A floor applies (6 of 82 tasks flip between identical arms; 7-14% across
     sessions): a difference counts as recovery only if `c70tol` solves >= 4 more than c70 with discordant tasks at
     least 3:1. Also reported: the share of the cut's gap closed, (`c70tol` - 8) / (14.5 - 8).
  2. Mechanism (descriptive): calls the plugin rewrote, as a share of the segment's calls (the cut's own malformed
     rate; same4's c70 wrote 20.7%), and the `registered` marker; malformed calls left in the traces
     (`eval_swallowed.py`); timeouts, call-cap endings and loops (`eval_loops.py`) against same4's c70 and full runs.
  3. Reading: (a) recovery with fewer timeouts: a tolerant parser is a cheap, training-free alternative to healing for
     self-hosted agents, and the paper says so with the cross-session caveat; (b) rewrites near c70's rate but no
     recovery: the cut loses tasks beyond call syntax, and healing does more than fix syntax; (c) the plugin rarely
     fires (< 5% of calls): the session's failures differ from the next-action test's, reported as such.
- **Gates:** the self-test passes before slicing (else the run stops after ~7 min); the server writes its marker; a
  segment that finishes no task stops the run; no other GPU session of ours running at push.
