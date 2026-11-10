# Under the Clock: Shorter Steps and Fewer Bytes for Tool-Using Agents

**Subtitle:** Making Gemma 4 faster per step bought no tasks beyond noise; breaking its tool calls lost many

## Abstract

In the Gemma 4 Developer Agent competition, Gemma 4 31B, served in 4 bits on four L4 GPUs, gets 8 minutes and 24 tool
calls to fix a repository issue, so shorter or faster steps buy more of them. We shortened steps by fine-tuning and sped
them up by pruning, and traced each to tasks solved. Fine-tuning on the model's own successes mostly changed how long it
reasons before each call and gained nothing on unseen repositories; a stronger teacher's preferences halved the slowest
steps but kept the agent working after it had submitted. Keeping 70% of the MLP channels decodes 1.15 times faster for
little loss, yet the pruned model solved 20 of 112 tasks against 32 and 34 for the full model: in 24% of its calls, one
string argument swallowed the next. Average loss missed this; a 15-minute next-action test caught it. Healed by the full
model and merged into its 4-bit weights, the pruned model kept its speed and solved 32; on 56 tasks, a parser that
accepts its wrong quotes did as well without training. Under a clock, how an agent writes its calls and when it stops
mattered more than how fast it decodes. Code:
[github.com/wilmerhenao/under-the-clock](https://github.com/wilmerhenao/under-the-clock).

![Figure 1](figs/fig_teaser.png)
*Figure 1. Three models' next call after one held-out transcript. The 70% cut (70% of MLP channels kept) ends the path
with a backtick instead of Gemma's string delimiter `<|"|>`, swallowing `start_line`; healed and merged, it writes the
full model's call.*

## 1. Introduction

A coding agent works in steps: it reasons, calls a tool and reads the result. Under a fixed clock, shorter steps (fewer
tokens) or faster ones (cheaper tokens) buy more of them. We pulled both levers, fine-tuning (Section 3) and pruning
(Section 4), and asked what each buys in tasks solved. The answer turned less on how fast the model runs than on how the
agent reasons, writes its calls and stops: fine-tuning moved reasoning length before behavior, and pruning broke the
tool calls (Figure 1) in a way average loss missed. Section 5 repairs the pruned model, by training and by parsing, and
Section 6 shows how much of any comparison between agents is noise.

## 2. Setting

**Episodes.** The competition's harness runs a declarative agent (YAML, prompts, optional LoRA adapters) in a sandbox at
the repository's pre-fix commit, with shell, file, edit, patch and code-graph tools. An *episode*, one attempt at a
task, gets 8 minutes, 24 tool calls, a 4,096-token thinking budget and temperature 0.2. It ends when the agent answers
without a tool call after submitting a patch (it *stops*), at the 24-call cap, or at the clock (a *timeout*). Hidden
tests then grade the diff, as in SWE-bench (arXiv:2310.06770), but a timeout loses the task: none of our 2,027 timed-out
episodes passed, even after a submitted patch. An episode *loops* when it repeats one call, arguments included, three
times in a row (repeated empty submits aside).

**Tasks.** We compare models on the 71 tasks of the competition's public repositories (*development* tasks) and on
verified tasks from *unseen* repositories, outside every training pool. The scorer serves only its own 4-bit (W4A16)
checkpoint plus LoRA adapters, so a pruned model cannot be submitted; we study pruning for self-hosted agents on the
same model, harness, hardware and clock.

**Hardware and comparisons.** We trained on a TPU v5e-8 and served the competition's Gemma 4 31B checkpoint
(arXiv:2607.02770) on four L4 GPUs with vLLM. A *paired session* runs two *arms* (say, the base model and an adapter) on
one server, 8 tasks each at a time on a 16-minute clock for the doubled load, and compares them by the tasks one solved
and the other did not (exact McNemar test; Section 6 measures the noise). There a model call that writes n tokens takes
about a + n·b seconds (Figure 2), and model calls fill 98% of an episode: fine-tuning acts on n, pruning on b and on the
prompt reading inside a. Models with different weights cannot share a server; in a *same-session comparison* each serves
the same tasks alone, 8 at a time on the 8-minute clock, in ABCD-DCBA order over two blocks of 56 tasks.

![Figure 2](figs/fig_framework.png)
*Figure 2. The agent under the clock and the two levers.*

## 3. Lever 1: fine-tuning moves length first

**Pipeline.** We ran Gemma through the harness on SWE-rebench-style task banks (arXiv:2505.20411) and FreshSWE-Py
(kaggle.com/datasets/moshegolderson/freshswe-py, issues filed after the model's cutoff), never on development or unseen
repositories, and kept the runs whose diffs passed the hidden tests, rendered as the scorer sees them. On these we
trained rank-16 LoRA adapters (JAX/Tunix) with the loss on tool calls only, each named after what it supervises (Table
1).

| adapter | trained on | step text vs base (p90) | solved vs base (only-this / only-base) |
|---|---|---|---|
| *Imitation* | stronger models' public runs (SWE-smith, SWE-Gym, R2E-Gym, SWE-Zero), whole turns | 0.24x | 16 vs 26 of 103 (4/14), p 0.031 |
| *Thought-first* | own runs, turns that open with a thought | 1.72x | 19 vs 27 (0/8), p 0.008 |
| *Every-call* | own runs (422), every call | 1.11x | 16 vs 22 (2/8) |
| ***Short-thought*** | 458 runs, calls after at most 1,500 characters of reasoning | **0.90x** | **25 vs 18 (8/1), p 0.039**; unseen: 17 vs 19 of 64 (2/4) |
| *Short-thought+* | the same, +153 runs | 1.03x of *Short-thought* | vs *Short-thought*: 18 vs 18 dev; 27 vs 35 of 125 unseen (1/9), p 0.021 |

*Table 1. Each adapter against the base in one paired session (development tasks unless marked); step text: 90th
percentile of characters per step.*

**Length moved both ways.** *Imitation* wrote short, narrated thoughts and looped (26 consecutive re-reads of one file)
while its validation loss fell from 1.05 to 0.37. *Thought-first* thought at length on every turn and made 1,171 model
calls to the base's 1,709 in the same time.

**One filter reversed the long steps.** *Every-call* lengthened steps and lost to the base. With the same recipe on 36
more runs, *Short-thought* supervised a call only if at most 1,500 characters of reasoning preceded it (59% of calls):
steps got shorter, timeouts fell from the base's 38 to 24, and it solved 25 tasks to the base's 18. No reasoning token
was trained, so length moved only through which calls were supervised, much as training on short chains shortens
reasoning models (arXiv:2505.17813). The gain is exploratory: one session, against a low draw of a base that solved
17-27 of these tasks across sessions. On unseen repositories *Short-thought* tied the base, and more data
(*Short-thought+*) lost there (77 timeouts vs 53), as on the public leaderboard (0.10 vs 0.18; base 0.15).

**A teacher at decision points.** An *agent state* is the transcript up to the agent's next turn. At 717 states, mostly
from slow or failed episodes, the stronger Gemini 3.8 Flash judged Gemma's next step and rewrote the ones it found
wasteful or wrong (589 preference pairs, $9.21). Trained on them from the base with DPO (arXiv:2305.18290),
*Teacher-DPO* took on the teacher's terse style (90th-percentile step 172 tokens; *Short-thought*, 514) but looped in 36
of 98 episodes and lost to *Short-thought* (20 vs 27). *Teacher-DPO-2* started from *Short-thought* and added a
next-token loss on the teacher's step (the RPO objective, arXiv:2405.16436): it looped in only 6 of 70 episodes, no
longer lost (20 vs 17) and halved the slow steps (56 vs 125 s at the 90th percentile), yet was not faster. In 38 of the
65 episodes where it submitted a patch, it kept calling tools until the clock took the task, though the patch changed in
only one. Without the preference term, the same stage stopped normally but looped (31 of 103). The teacher's data
shortens steps and brings loops; the preference term prevents the loops but breaks the stop.

## 4. Lever 2: prune what the agent does not use

Each of the model's 60 layers has a gated MLP with 21,504 hidden channels, about 70% of the layer's weights. We score
each channel by the size (RMS) of its activation on agent trajectories times the norm of its output weights and keep the
top share in every layer, without retraining. A cut is named by the share it keeps: the *70% cut* removes 30% of the
channels, 21% of the layer's weights. Masking channels in the 16-bit model, we measure the *damage*: the rise in loss
(nats per token) on *agent tokens*, the thought and call of each turn with at most 1,500 characters of reasoning, in 40
held-out trajectories from other tasks (full model: 0.26). The 70% cut costs +0.080 when channels are scored on agent
text, +0.533 when drawn at random (one seed) and +0.867 when scored on WikiText-103: each score keeps the channels its
own text uses.

**Real 4-bit cuts.** We then cut the competition's 4-bit checkpoint itself and served each cut in vLLM, keeping a
multiple of 512 channels per layer as the 4-bit kernels require (69% for the 70% cut). Removing single channels splits
the output projection's 4-bit scale groups of 32 inputs, so we re-quantize it; removing whole groups avoids that but
loses twice as much. The 85%, 70% and 55% cuts decode 1.05, 1.15 and 1.24 times faster for a served damage of +0.025,
+0.084 and +0.188, within 0.005 nats of what masking predicted. Removing 6 whole layers by block influence
(arXiv:2403.03853) costs +0.925 for 1.07x (Figure 3), and lossless n-gram speculative decoding sped up real agent turns
only 1.05-1.07x.

![Figure 3](figs/fig_speed.png)
*Figure 3. Decode time follows the bytes read (left). At equal bytes, damage depends on what is removed (right).*

**Tasks under the clock.** In the same-session comparison on 112 unseen-repository tasks (Figure 4), the 70% cut solved
20 against 32 and 34 for two runs of the full model, which differed from each other on 8. It solved no task that either
full run missed and missed 17 that at least one solved (two-sided sign-flip test, p 1.5e-5), losing them at the clock:
72 timeouts against 43-44. The failure is in the call syntax. Gemma 4 writes a call as
`call:name{key:<|"|>text<|"|>,...}`, and the cut often closes a string with a backtick or a JSON quote instead of
`<|"|>` (Figure 1), so the string runs on and swallows the next argument. This broke 24% of its calls, against 2-3% for
the full model: the call fails, the agent resends it, and 42 of its episodes looped (full model: 4 and 7).

![Figure 4](figs/fig_tasks.png)
*Figure 4. Same-session comparisons on 112 unseen-repository tasks. Grey band: two runs of the full model.*

## 5. Repairing the cut

**Healing from the full model.** Healing trains a rank-16 LoRA on the cut, here for two epochs on 628 agent trajectories
in two arms on identical batches: one imitates the full model by minimizing the KL divergence from it at every token,
the other minimizes the plain next-token loss. The KL arm met its pre-registered bar (damage below +0.04, KL halved):
damage fell from +0.085 to +0.021 and KL from 0.098 to 0.042, in 2.4 TPU hours. The next-token arm fit the data instead:
its loss fell below the full model's (-0.078) and its KL rose to 0.243. Served through vLLM's LoRA path, the adapter
costs the whole speedup (1.02x against 1.15x), so it must be merged into the 4-bit weights. Rounding to the nearest
4-bit value would erase the repair, whose changes average 5% of a quantization step (as Scale-QLoRA, arXiv:2609.04526,
found), so we round stochastically: up or down at random, right on average. From one draw, the merged cut keeps the
repair (+0.022 nats) and the speed (1.15x).

In the same-session comparison the healed 70% cut solved 32 of 112, within noise of the full model (one task below its
mean; 95% interval -7.5 to +5; with no margin fixed in advance, not non-inferiority) and 12 more than the unhealed cut
(p 0.0075). Its malformed calls fell to 3%. Faster per call (16.9 s against 19.0-19.3 s at equal length), it had 36
timeouts against 43-44, but most of the difference turned into endings at the 24-call cap (21 against 12-16), which pass
one time in five: under the cap its speed was worth about one task. Healing the 55% cut (1.24x) won back 14 of the 22
tasks it had lost, short of the full model (p 0.007).

**A next-action test.** Task sessions take hours, so we built a 15-minute check: at 109 held-out agent states, each
model writes its next call greedily, and we compare it with the full model's (Figure 5). The same weights with other
batching agree on 78% of calls, a floor set by numerics (arXiv:2506.09501); sampling at the agent's temperature agrees
on 64%; the cuts, healed or not, on 28-42%. Exact agreement does not track tasks (the unhealed 70% and 55% cuts, which
solved 20 and 13, agree on 29% and 28%); the malformed share does. The test is a gate, not a predictor, and a thin one:
the 70% cut wrote 8 malformed calls of 100 against the full model's 2 of 102 (p 0.06), a rate its sessions compounded to
24% by resending each failed call.

![Figure 5](figs/fig_nextaction.png)
*Figure 5. The next-action test at 109 held-out states (agreement over the 102 where the full model calls; malformed
shares over each model's calls). Grey band: the same weights with other batching.*

**A tolerant parser.** The cut's slips are regular: a backtick, a JSON quote or an escaped quote where `<|"|>` should
close a string, or a stray quote after a key. A vLLM tool-parser plugin that accepts them repairs 53 of the cuts' 68
malformed calls in the next-action test. Served with it, the unhealed 70% cut solved 15 of the first block's 56 tasks,
against 8 for the same cut there in the same-session comparison (9 tasks gained, 2 lost; p 0.065) and 14 and 15 for the
full model: the recovery we pre-registered for this cross-session test (at least four more tasks, at least 3:1). Its
timeouts fell from 37 to 20 and its loops from 20 episodes to 4. The parser rewrote 8% of its calls, the next-action
test's rate, and left none malformed.

**A better score?** Scoring channels on the agent's own tokens rather than whole trajectories, as RAC does for reasoning
chains (arXiv:2509.12464), changed nothing at 70% (Figure 5). Weighting five token roles equally (after SKIM, Section
7), the closing quote among them (0.08% of positions, where the 55% cut's loss rose most), halved the wrongly closed
strings, but calls then lost their first token (Figure 6), and the pre-registered test failed (malformed calls 12/112 vs
20/122, one-sided Fisher p 0.14).

![Figure 6](figs/fig_failure_moves.png)
*Figure 6. Scoring by token role (55% cut, 141 states). The closing quote recovers and the opener breaks.*

## 6. Measuring agents under noise

Between sessions the same model moved by up to ten tasks and flipped 7-14% of the same unseen tasks (agent reruns
elsewhere vary by 2-6 points, arXiv:2602.07150). Two identical full-model arms on one server, our *A/A floor*, flipped 6
of 82 tasks (2/4), as many as across sessions: pairing controls load and time of day but not these flips, so only
lopsided splits such as 8/1 carry signal. Pairing also couples the arms: one that writes less speeds up its rival.
Single-session p-values near 0.04, among about twenty adapter comparisons, are exploratory; the same-session, healing,
role and parser tests were pre-registered in dated entries we release.

## 7. Related work

Fine-tuning on a model's own filtered successes dates to expert iteration (arXiv:1705.08439), STaR (arXiv:2203.14465)
and ReST-EM (arXiv:2312.06585); SWE-Gym (arXiv:2412.21139) found it weak for software agents. Agents overthink
(arXiv:2502.08235) and, under wall-clock budgets, fill spare time with repeated actions (On the Clock,
arXiv:2610.10833), as *Teacher-DPO-2* did; *Teacher-DPO*'s collapse resembles likelihood displacement
(arXiv:2410.08847).

Perplexity misses what compression breaks (arXiv:2310.01382), for agents too (ACBench, arXiv:2505.19433), and 4-bit
quantization hides amplified agent failures behind flat scores (arXiv:2607.27275); flips and KL track it better
(arXiv:2407.09141). Our channel score is FLAP's Wanda-sp (arXiv:2312.11983), and calibration data steer such scores
(arXiv:2311.09755). Reasoning models are pruned with calibration on their own chains (RAC; SSGR, arXiv:2511.18864;
OBC-Prune, arXiv:2609.17890), without tools or a clock. Concurrent SKIM (arXiv:2611.07558) prunes tool-calling agents by
per-role saliency and heals on-policy, under a grammar that enforces the call format (as XGrammar, arXiv:2411.15100,
can); our parser repairs the format after decoding instead.

Minitron (arXiv:2407.14679) is our cut-then-distill template, and healing after pruning goes back to arXiv:2403.17887.
QA-LoRA (arXiv:2309.14717) and LoTA-QAF (arXiv:2505.18724) train adapters that merge into quantized weights losslessly.

## 8. Limitations and conclusion

With one model, harness and competition, the mechanisms (supervision shapes reasoning length, cuts break rare syntax
tokens, preference terms move stopping) may generalize, but the numbers need not. The fine-tuning gain held on
development repositories only, and we did not try policy-gradient training (arXiv:2502.18449). Each cut has one mask,
without reconstruction, one healing run and one rounding draw. The token-level tests cover short-reasoning turns within
8,192 tokens, on held-out tasks that share repositories with training rows. We kept the clock and the call cap fixed.
The healed cut's result is one session of 112 tasks at our own load; the parser's is one block of 56, compared across
sessions, on the 70% cut only.

Before asking what a faster model buys an agent, measure the agent: its tokens per step, which fine-tuning moves first,
and its time per token, which follows the bytes read. Before any long run, check cheaply whether it still writes its
calls and stops as it did; on a clock, those decided every result we measured.

**Disclosure.** The author works in Advanced Engineering at Invesco. This work, outside that role, used no Invesco
resources.
