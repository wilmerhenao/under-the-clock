# Under the Clock: Shorter Steps and Fewer Bytes for Tool-Using Agents

**Subtitle:** Making Gemma 4 faster per step bought no tasks beyond noise, and breaking its tool calls lost many

## Abstract

In the Gemma 4 Developer Agent competition, a coding agent works under a clock. Gemma 4 31B, in 4 bits on four L4 GPUs,
gets 8 minutes per repository issue. We made each step faster in two ways and traced each to tasks solved. Fine-tuning
on the model's own verified trajectories changed the agent's reasoning length before each tool call more than its
actions. Supervising only calls preceded by short reasoning solved more dev-set tasks (25 vs 18), though not on unseen
repositories, and a teacher-guided preference stage halved the slow steps (125 to 56 seconds at the 90th percentile) but
broke the agent's stopping. Pruning 30% of the MLP channels, chosen on agent trajectories, costs +0.08 nats per agent
token and decodes 1.15 times faster on one stream (1.1 at 8). Yet in one session on the 8-minute clock the cut solved 20
of 112 tasks against 32 and 34 for two full-model runs, winning none of the tasks that separated them (p 1.5e-5). Its
string arguments ran into one another in 24% of its calls (full model: 2-3%), and it looped on the failures. Healed with
the full model as teacher and merged into the 4-bit weights by stochastic rounding, the cut kept its speed and solved 32
(-1 task, 95% interval -7.5 to +5). With 45% pruned, healing won back 14 of the 22 lost tasks. Average per-token loss
ranked the cuts but neither sized nor located the failure. A 15-minute next-action test located it, and showed a
role-aware channel score moving it instead of removing it. Code and notebooks:
[github.com/wilmerhenao/under-the-clock](https://github.com/wilmerhenao/under-the-clock).

![Figure 1](figs/fig_teaser.png)
*Figure 1. One held-out agent state, three models. The 70% cut (70% of MLP channels kept) ends the path with a backtick
instead of Gemma's string delimiter `<|"|>` and swallows `start_line`. Healed and merged, it writes the full model's
call.*

## 1. Introduction

An agent that writes fewer tokens per step, or writes each faster, gets more steps under a clock. Does it solve more
tasks? We pulled two levers on seconds per step, fine-tuning and pruning (Figure 2). How the agent reasons, writes tool
calls and stops outweighed every gain in model speed we measured. Our contributions:

1. Fine-tuning moves reasoning length before decisions. A teacher at the agent's decision points shortened steps and
   broke stopping.
2. An agent-chosen cut has the 4-bit loss and speed that its bf16 mask and bytes predict, yet fails through malformed
   tool calls (Figure 1) that average per-token loss ranks but does not locate.
3. Healed from the full model and merged into the 4-bit weights by stochastic rounding, the cut keeps its speed and
   stays within noise of the full model's task rate.
4. We measure agents under noise with an A/A floor, a same-session comparison of checkpoints and a next-action test.

## 2. Setting

**Harness.** The competition's harness compiles a declarative agent (YAML, prompts, optional LoRA adapters) and runs it
in a sandbox holding the repository at its pre-fix commit, with shell, file, edit, patch and three code-graph tools.
Hidden tests grade the agent's diff, submitted or not, as in SWE-bench (arXiv:2310.06770). We allow each *episode* (one
attempt at a task) 8 minutes and 24 tool calls, a 4,096-token thinking budget and temperature 0.2. At 14,336 tokens the
model summarizes older events (compaction). Evaluation repositories are outside our training pools (two, found later
under other owners, change no result). The scorer serves one 4-bit (W4A16) checkpoint with LoRA adapters under 3 GiB, so
a pruned model cannot be submitted. We study pruning for self-hosted agents on the competition's model, harness,
hardware and clock.

**Hardware and clocks.** Training ran on a TPU v5e-8, serving on 4x L4 with vLLM (tensor parallel 4) and the
competition's checkpoint of Gemma 4 31B (arXiv:2607.02770, one scale per 32 inputs). *Paired sessions* (Section 3) put
two arms on one server, 8 concurrent tasks each, on a clock doubled to 16 minutes. At 16 concurrent requests a model
call writing n tokens takes a + n·b seconds (Figure 2), and calls take 98% of episode time. Fine-tuning acts on n,
pruning on b. For checkpoints that cannot share a server, the *same-session comparison* (Section 4) runs them one at a
time on the same tasks, 8 at once on the 8-minute clock, in ABCD-DCBA order over two blocks, so neither order nor time
favours a model.

![Figure 2](figs/fig_framework.png)
*Figure 2. The agent under the clock and the two levers.*

## 3. Lever 1: fine-tuning moves length first

**Pipeline.** We ran Gemma through the harness in private Kaggle Benchmarks tasks on SWE-rebench-style task banks and
FreshSWE-Py (issues filed after the model's cutoff), excluding the public (development) repositories, and kept the 732
trajectories whose diffs passed the hidden tests. Each is re-rendered as the scorer sees it. A compaction simulator
matched the token counts of all 82 captured scorer requests, and Gemma writes the summaries. A rank-16 LoRA on all
projections is trained on TPU (JAX/Tunix, fp32 factors) with the loss on tool-call bodies only.

| adapter | data and supervision | step text vs base (p90, full sessions) | solved vs base (discordant: only-this / only-base) |
|---|---|---|---|
| *Imitation* | stronger models' trajectories, whole turns | 0.24x | 16 vs 26 of 103 (4/14), p 0.031 |
| *Thought-first* | own runs, only turns opening with a thought | 1.72x | 19 vs 27 (0/8), p 0.008 |
| *Every-call* | own runs, compaction-aware, every call | 1.11x | 16 vs 22 (2/8) |
| ***Short-thought*** | *Every-call*'s rows, calls after thoughts <= 1,500 chars | **0.90x** | **25 vs 18 (8/1), p 0.039**; unseen: 17 vs 19 of 64 (2/4) |
| *Short-thought+* | *Short-thought*'s recipe, +153 runs | 1.03x of *Short-thought* | dev: 18 vs *Short-thought* 18; unseen: 27 vs *Short-thought* 35 of 125 (1/9), p 0.021 |

**Length moved in both directions.** *Imitation* wrote short narrated thoughts and looped (26 consecutive re-reads of
one file) while its validation loss fell from 1.05 to 0.37. *Thought-first* trained only on turns that opened with a
thought, which selected long ones. Served, it thought long on every turn, made 1,171 model calls to the base's 1,709 in
the same time, and lost eight tasks, winning none.

**One filter reversed *Every-call*'s longer steps.** *Short-thought*, with *Every-call*'s rows and hyperparameters,
supervised a call only if at most 1,500 characters of reasoning preceded it (59% of calls): shorter steps, timeouts 38
to 24, and 25 vs 18 solved. No reasoning token was trained, so length moved only through which calls were supervised.
Our reading is that the adapter generalizes the deliberation that preceded them. The gain is exploratory. On unseen
repositories *Short-thought* tied the base, with 90th-percentile steps still 0.89 times the base's, and more data
(*Short-thought+*) lost there (77 timeouts vs *Short-thought*'s 53), as on the public leaderboard (*Short-thought* 0.18,
*Short-thought+* 0.10, base 0.15).

**A teacher at decision points.** At 717 agent states (Figure 1), mostly from slow or failed episodes, the stronger
Gemini 3.8 Flash judged Gemma's next step or wrote its own (589 preference pairs, $9.21). Trained from the base
(*Teacher-DPO*), Gemma took the teacher's style (90th-percentile step 172 tokens, *Short-thought*'s 514) but looped (36
of 98 episodes) and lost to *Short-thought* (20 vs 27). A second stage on *Short-thought* computed the preference in
each training step (*Teacher-DPO-2*, DPO plus a next-token loss on the teacher's step). It removed the losses to
*Short-thought* (20 vs 17, 5/2) and kept steps short (229 vs 610 tokens, 56 vs 125 s at the 90th percentile). Yet the
agent was not faster. Once a patch is submitted, an episode ends at the first response without a tool call, and
*Teacher-DPO-2*, after submitting early, kept calling tools until the clock. Of its episode time, 52% came after a first
patch that then changed in 1 of 65 episodes. The next-token loss alone kept steps short and stops normal, but the agent
looped (31 of 103). The teacher's data shortens steps, and the preference term prevents loops but breaks the stop.

## 4. Lever 2: prune what the agent does not use

Each of the model's 60 layers has a gated MLP with 21,504 hidden channels (about 70% of the layer's weights). We score
each channel by the RMS of its activation over 300 agent trajectories (2.37 M tokens) times the norm of its
down-projection weights. A cut is an exact mask, without retraining. Its *damage* is the rise in nats per token on
*agent tokens*, the thought and call of each short-reasoning turn (at most 1,500 characters of reasoning, 17% of the
agent's text), in 40 held-out trajectories from other tasks. The 70% cut (70% of channels kept per layer, 21% of layer
weights removed) costs +0.080 with agent scores, +0.867 with WikiText-103 scores and +0.533 at random.

**Real 4-bit cuts.** We cut the competition's 4-bit checkpoint and served each cut in vLLM. Single channels split the
down projection's quantization groups of 32, so the projection is re-quantized (activation-weighted) or cut in whole
groups. The 85%, 70% and 55% cuts decode 1.05, 1.15 and 1.24 times faster (1.03-1.14 at 8 streams) for a served damage
of +0.025, +0.084 and +0.188. Removing 6 layers by block influence (arXiv:2403.03853) costs +0.925 at 1.07x (Figure 3).
Every single-channel cut loses in vLLM what its bf16 mask predicted, within 0.005 nats. Whole groups lose twice as much.

![Figure 3](figs/fig_speed.png)
*Figure 3. Decode time follows the bytes read (left). At equal bytes, damage depends on what is removed (right).*

**Tasks under the clock.** In the same-session comparison (112 unseen-repository tasks, Figure 4), the 70% cut solved 20
tasks against 32 and 34 for two runs of the full model, which differed on only 8 tasks (3/5). The cut won none of the 12
and 14 tasks that separated it from them (exact paired sign-flip test, p 1.5e-5). It lost them at the clock (72 timeouts
against 43-44), and none of our 1,944 timed-out episodes passed the hidden tests. The failure is in the call syntax
(Figure 1): a string closed with a backtick or a JSON quote instead of Gemma's delimiter swallows the next argument. Of
the cut's calls, 24% were malformed, against 2-3% for the full model. The call fails, the agent resends it, and 42
episodes looped, 31 until the clock (full-model runs: 4 and 7).

![Figure 4](figs/fig_tasks.png)
*Figure 4. Same-session comparisons on 112 unseen-repository tasks. Grey band: two runs of the full model.*

**Healing from the full model** (mask and adapter off). A rank-16 LoRA is trained for two epochs on 628 agent
trajectories in two arms on identical batches. The KL arm minimizes KL(full || cut) at every token, the next-token arm
the next-token loss. A removed channel gets no gradient, so the adapter slices exactly onto the 4-bit cut. The KL arm
met its pre-registered bar, damage below +0.04 and KL halved (2.4 TPU hours). In the trainer's bf16 evaluation the damage fell from +0.085 to +0.021
and the KL on agent tokens from 0.098 to 0.042. The next-token arm reached -0.078, below the full model, while its KL
rose to 0.243. It fit the data, not the model. Served through vLLM's LoRA path, the adapter costs the whole speedup
(1.02x against 1.15x). Rounding W + dW to the nearest 4-bit value would erase the repair, since its deltas average 5% of
a grid step (Scale-QLoRA, arXiv:2609.04526, saw the same). So we use stochastic rounding (up or down at random, exact in
expectation), and the merged cut keeps the repair (+0.022 nats) and the speed (1.15x).

In the same-session comparison the merged cut solved 32 of 112, within noise of the full model (-1 task, 95% interval
-7.5 to +5, discordant 5/5 and 6/8; against the unhealed cut 15/3, p 0.0075). It made 3% malformed calls and 36 timeouts
against 43-44 (seven fewer, worth about 3 tasks, since full-model episodes ending after 7 minutes pass 43% of the time).
Its calls took 16.9 s against 19.0-19.3 s at equal length. On the 55% cut (1.24x), healing recovers the same share of
the damage (+0.185 to +0.047, missing its +0.04 bar) and 14 of the 22 lost tasks, still short of the full model (p
0.007). Malformed calls fell from 43% to 13% (full model 1-2%, Figure 4).

**A next-action test.** At 102 held-out agent states (starts of short-reasoning turns in 40 trajectories), each model's
greedy next call is compared with the full model's (Figure 5). Two engines with the same weights agree on 78% of calls,
sampling at the agent's temperature on 64%, and the cuts, healed or not, on 28-42%. Exact agreement does not track tasks
(the cuts solving 20 and 13 agree on 29% and 28%). The malformed share does, and so does agreement with the call the
agent recorded (full 0.40, healed 0.34, 70% 0.28, 55% 0.24). The test is a gate, not a predictor.

![Figure 5](figs/fig_nextaction.png)
*Figure 5. The next-action test at 102 held-out states. Grey band: two engines with the same weights.*

**Two scores for pruning reasoning.** Scoring channels only on agent tokens, in the spirit of RAC's calibration on the
model's own chains (arXiv:2509.12464), changed nothing at 70% (Figure 5). Scoring by token role (after SKIM, Section 6)
gives an equal vote to text read, prose, argument values, call syntax and the closing quote (0.08% of positions). Split
by role, the 55% cut's loss rose most on that quote (+0.50 nats, against +0.19 overall). At 141 states the role-scored
cut halved both that rise and the wrongly closed strings, but its calls lost their opener (their first token) instead
(Figure 6). In all, 20% of its attempts failed against 19% (pre-registered test, parsed calls only: 11% vs 16%, p 0.14).
A format-enforcing grammar would block the failures the role score adds but not those it removes (inference).

![Figure 6](figs/fig_failure_moves.png)
*Figure 6. Scoring by token role (55% cut, 141 states). The closing quote recovers and the opener breaks.*

**Speed without a cut.** N-gram speculative decoding kept the full model's calls (89% the same next call, malformed
2/103) but sped up real agent turns only 1.05-1.07x. Unlike a cut's, its gain depends on the text (3.4x on a synthetic
continuation, Figure 7). We did not test Gemma 4's own draft model (arXiv:2607.02770).

![Figure 7](figs/fig_speculation.png)
*Figure 7. N-gram speculative decoding against plain decoding (full model, same session).*

## 5. Measuring agents under noise

**Pairing and the noise floor.** Between sessions the same model moved by up to ten tasks and flipped 7-14% of the same
unseen tasks. Paired sessions, compared by discordant tasks (exact McNemar), separated adapters. Two identical
full-model arms on one server, the A/A floor, flipped 6 of 82 tasks (2/4), as many as across sessions. Pairing controls
load and time of day but leaves the flips, so only lopsided splits such as 8/1 carry signal.

**Exploratory and confirmatory.** Single-session p-values near 0.04, among about twenty adapter comparisons, are
exploratory, and we claim only directions that held across sessions or mechanisms. The same-session, role and
speculation tests are confirmatory, with analysis set, primary test and gates fixed before each run.

## 6. Related work

Fine-tuning on a model's own filtered successes dates to expert iteration (arXiv:1705.08439), STaR (arXiv:2203.14465)
and ReST-EM (arXiv:2312.06585). SWE-Gym (arXiv:2412.21139) found it weak for software agents. SWE-smith
(arXiv:2504.21798), R2E-Gym (arXiv:2504.07164) and SWE-rebench (arXiv:2505.20411) supply tasks. Cuadron et al.
(arXiv:2502.08235) measured overthinking in agents, and On the Clock (arXiv:2610.10833) agents under wall-clock budgets.
Our preference stages use DPO (arXiv:2305.18290) and IPO (arXiv:2310.12036). *Teacher-DPO*'s collapse matches likelihood
displacement (arXiv:2410.08847).

Perplexity misses what compression breaks (arXiv:2310.01382), for agents too (ACBench, arXiv:2505.19433, worst on JSON
outputs), and flips and KL track it better (arXiv:2407.09141). Quantization amplifies agents' existing failures
(arXiv:2607.27275), and we trace them to call syntax. Our channel score is a structured Wanda (arXiv:2306.11695), FLAP's
Wanda-sp (arXiv:2312.11983). Reasoning models are pruned with calibration on their own chains (RAC; SSGR,
arXiv:2511.18864; RESP, arXiv:2512.02185; OBC-Prune, arXiv:2609.17890), without tools or a clock. Concurrent work, SKIM
(arXiv:2611.07558), prunes tool-calling agents by per-role saliency and heals on-policy, under a grammar that enforces
the format.

Minitron (arXiv:2407.14679) is our cut-then-distil template. LLM-Pruner (arXiv:2305.11627) retrains, D-Pruner
(arXiv:2405.06275) calibrates on a target domain, and self-data distillation (arXiv:2410.09982) heals with the unpruned
model's outputs. We heal with an off-policy forward KL (on-policy: arXiv:2306.13649, arXiv:2306.08543) and merge by
stochastic rounding, whereas QA-LoRA (arXiv:2309.14717) trains merge-ready adapters and PEQA (arXiv:2305.14152) tunes
the quantization scales. Speculative decoding (arXiv:2211.17192, arXiv:2302.01318) speeds decoding without changing the
model, and SuffixDecoding (arXiv:2411.04975) targets agents. Agent reruns vary by 2-6 points (arXiv:2602.07150; error
bars: arXiv:2411.00640).

## 7. Limitations and conclusion

With one model, harness and competition, the mechanisms (supervised actions condition reasoning length, cuts break rare
syntax tokens, preference terms move stopping) may generalize but the numbers need not. The fine-tuning gain held on
development repositories only. We did not try policy-gradient training. Each cut has one mask and one healing run. The
token-level tests cover short-reasoning turns within 8,192 tokens, on held-out tasks that share repositories with
training rows. No affordable control changes time with behaviour fixed. The healed cut's result is one session of 112
tasks, at our own load of 8 tasks at a time, not a documented scorer setting.

Before asking what a faster model buys an agent, measure the agent: its tokens per step, which fine-tuning moves first,
and its time per token, which follows the bytes read. Before any long run, check cheaply whether it still writes its
calls and stops as it did. On a clock, those decided every result we measured. Healed by its full model, the 70% cut
chosen by the agent's use stayed within noise of the full model's task rate at 1.15 times its decode speed, but the 55%
cut did not.

**Disclosure.** The author works in Advanced Engineering at Invesco. This work, outside that role, used no Invesco
resources.
