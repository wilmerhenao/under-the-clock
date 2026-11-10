"""Turn the built TPU trainer script (kernels/train-g9-tpu/train-g9-tpu.py) into the publishable notebook
kernels/g4dev-trainer-tpu/: explanatory markdown, the three modules as %%writefile cells (byte-identical to
kernels/train/ apart from a few comment/help-text edits below), and the driver split at its section markers. The code
that runs is the built script's, cell by cell; check() verifies that.
usage: python3 kernels/analysis/make_trainer_notebook.py [g9|g10]  (after build_one.py train-g9-tpu / train-g10-tpu)
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILT = ROOT / 'kernels/train-g9-tpu/train-g9-tpu.py'
OUT = ROOT / 'kernels/g4dev-trainer-tpu'
SLUG = 'gemma-4-31b-agent-lora-tpu-trainer'
TITLE = 'Gemma 4 31B Agent LoRA: TPU Trainer'
MODULES = [('LORA_TPU', 'lora_tpu.py'), ('TRAIN_PY', 'train_lora.py'), ('RUNNER', 'train_tpu_runner.py')]
# comment / help-text edits only (references to internal run logs)
SCRUB = {
    'train_tpu_runner.py': [('Robustness for a run without a preceding canary (user 10-14):',
                             'Robustness (the run has to succeed in one long TPU session):')],
    'train_lora.py': [(', B26-B33)', ')'), ('(B38: a teacher edit', '(e.g. a teacher edit'), ('run never trained, B43)',
                                                                                            'run never trained)')],
    'driver': [("# exact versions resolved by canary v2 (10-13), so a later run gets the same stack",
                "# exact versions, so every run gets the same stack"),
               ("  # rb2: uv's 30 s default timed out on a large wheel", "  # uv's 30 s default can time out on a large wheel"),
               ("(the 62 GB read took 38 min page by page in canary v2)", "(page by page, the 62 GB read can take ~40 min)")],
}

INTRO = """# Gemma 4 31B Agent LoRA: TPU Trainer

This notebook fine-tunes **Gemma 4 31B** (`gemma-4-31b-it`, QAT checkpoint) with a rank-16 **LoRA adapter** for the
Gemma 4 Developer Agent competition, on a single **Kaggle TPU v5e-8** session, in JAX with
[Tunix](https://github.com/google/tunix)'s Gemma 4 model and [Qwix](https://github.com/google/qwix) LoRA. The adapter
it writes uses the Hugging Face PEFT layout, so it goes straight into a submission's `adapters/` folder.

Run it with **Run All** on a TPU v5e-8 with Internet on (the first cell installs a pinned JAX stack). A full run takes
about 1.5-2.5 hours.

## The recipe, and why it is expected to help

**1. Train on the agent's own verified successes (rejection-sampling fine-tuning).**
The training rows are runs of the *same* model, with the *same* system prompt, tools and budget as the submission,
kept only when the task's hidden tests passed. Fine-tuning on them raises the probability of the actions that led to
verified fixes. Because the rows come from the model's own distribution, there is no teacher/student mismatch: the
tool-call syntax, the thought style and the turn structure are exactly what the served model produces. This is the
"expert iteration" / STaR / ReST idea: sample, keep what a verifier accepts, fine-tune, repeat.

**2. Keep only the fast solves.**
The agent works under a hard budget (here 24 tool calls and 8 minutes per task), and a run that is still exploring
after about 15 calls rarely finishes in time. The successful runs that resolved the task within **14 tool calls**
show exactly the behaviour the budget rewards: go straight to the files named in the issue, edit early, run one
targeted test, submit. `KEEP_MAX_CALLS` applies this filter; longer successes are dropped rather than imitated.

**3. Loss on the tool calls only, not on the thoughts.**
Gemma 4 thinks before acting. Training on the thought text teaches thought *length* as much as content, and longer
thoughts mean fewer tool calls before the time cap. So the loss covers only the bodies of the tool calls
(`TRAIN_ON = 'actions'`); thoughts, channel markers and tool results are context. A tool call that the model's own next
thought calls a mistake is masked too (`MASK_MISTAKES`).

**4. A small, controlled dose.**
Rank 16 on every attention and MLP projection, learning rate 5e-5 with warmup and cosine decay, two epochs, and an
adapter saved after every epoch. A second arm restarts from the *same* initialisation and data order at 1e-4 for one
epoch. Training loss is not a quality signal here (the model's own tool calls are already highly probable, and the
loss falls toward zero as the rows are memorised), so choose among the saved adapters with a paired evaluation: base
model vs adapter on the same held-out tasks.

**5. fp32 master weights for the LoRA parameters.**
Qwix creates the LoRA matrices in the base weights' dtype (bf16). With lr 5e-5, most Adam updates to `lora_a` are
smaller than half a bf16 ulp and round away, so `A` stays frozen. Here the LoRA parameters and optimizer state are
fp32 and are cast to bf16 only where the forward pass reads them.

## How 31B parameters fit on 8 x 16 GB

* **Weights:** the bf16 checkpoint (`gemma-4-31b-it-qat-q4_0-unquantized`, the QAT model's unquantized weights, so the
  adapter matches the served `w4a16` model) is sharded over a 2 x 4 mesh: FSDP over 2 devices, tensor parallelism
  over 4. Tensor parallelism stops at 4 because the full-attention layers have 4 KV heads.
* **Activations:** decoder-level rematerialisation; rows are packed into two length buckets (8,192 and 12,288 tokens),
  batch 2.
* **Attention:** Pallas splash attention with block 512 (block 1024 with head dim 512 exceeds the 16 MB scoped VMEM).
* **Loss:** the 262k-token vocabulary makes full logits huge, so logits are computed only at the labelled positions,
  512 at a time.
* **Throughput:** about 3,100 tokens/s; one epoch over ~400 rows of ~10k tokens takes ~26 minutes.

## Safety nets for a long TPU session

* Attention ladder: a step that fails in the attention kernel is retried with splash block 256, then eager attention
  capped at 8,192 tokens.
* Memory: an out-of-memory length is demoted 12,288 -> 8,192 -> 4,096 (rows truncated, the kept-label fraction logged).
* A non-finite step applies no update; five in a row stop the run.
* The schedule is time-aware: after the first steps the plan is cut to what fits the time left, at an epoch boundary
  when possible, and the cosine horizon follows the cut.
* Exports are atomic and refuse non-finite tensors; the runner is supervised by a deadline and a no-progress watchdog,
  and nothing raises after training starts, so the notebook's outputs always survive.

## Inputs

* **Model:** Kaggle Models `google/gemma-4`, Transformers, `gemma-4-31b-it-qat-q4_0-unquantized`.
* **Training rows:** a JSONL file (`render/sft.jsonl`), one verified run per line, rendered with the Gemma 4 chat
  template: `text` (the full conversation), `spans` (character ranges of the model's turns), `messages`, `tool_calls`
  (the run's number of tool calls) and `n_tokens`. To train on your own runs, attach a dataset with the same fields
  and set `SFT_NAME` to its path suffix.
* **Reference adapter (optional):** any PEFT adapter for this model, used only to check the exported key names and
  shapes.

## Outputs

* `adapter_out/`: the last adapter of the first arm (`adapter_model.safetensors`, `adapter_config.json`);
  `adapters/ep0`, `adapters/ep1`: one per epoch.
* `adapter_out_b/`, `adapters_b/ep0`: the second arm.
* `tpu/log.jsonl` (every stage and step), `tpu/report.json` (summary), `tpu/data_stats.json`.

To use an adapter in a submission, put its two files under `adapters/<name>/` in `submission.zip` and add
`adapter: <name>` after the `model:` line of `agent.yaml`.
"""

SECTIONS = {
    'settings': """## Settings

`MODE = 'train'` runs the full recipe (`'canary'` runs two steps per length bucket to measure compile time, step time
and memory). `DEADLINE_S` bounds the whole notebook, `TRAIN_CAP_MIN` the training itself. `LR`, `EPOCHS` and `RANK`
define the first arm, `PHASE2_LR` / `PHASE2_EPOCHS` the second (0 turns it off). `KEEP_MAX_CALLS` is the fast-solve
filter (0 keeps every row). `PINS` fixes every package version.
""",
    'setup': """## Logging and helpers

Every stage appends one JSON line to `tpu/log.jsonl`, so a run can be followed live and summarised afterwards.
""",
    'modules': """## The three modules

* `lora_tpu.py`: the model wrapper that returns hidden states, Qwix LoRA with fp32 master weights, the loss at
  labelled positions, the jitted train step (gradient clipping, non-finite skip, the learning rate as an argument),
  the sharded checkpoint loader and the PEFT export.
* `train_lora.py`: tokenisation and label masking. It is shared with a GPU trainer; here it only runs with
  `--dump-npz`, which writes the token ids and labels of every row, so both trainers see identical targets.
* `train_tpu_runner.py`: the training loop (length buckets, schedule, fallbacks, exports, the second arm). It runs in
  its own Python 3.11 environment.
""",
    'env': """## 0. Environment

Tunix needs Python 3.11, so a separate virtual environment is created with `uv` and the pinned JAX/Tunix/Qwix stack
(plus CPU PyTorch, which the tokenisation script imports).
""",
    'inputs': """## 1. Inputs and the fast-solve filter

Locate the checkpoint, the rows and the reference adapter, then keep the rows of runs that finished within
`KEEP_MAX_CALLS` tool calls (if that would leave less than a quarter of the rows, every row is kept).
""",
    'prefetch': """## 1b. Read-ahead

Reading 58 GB of mounted weights page by page is slow. 32 threads read the files ahead in the background, so the
loader later finds them in the page cache.
""",
    'dump': """## 2. Tokens and labels

The rows are tokenised with the model's own tokenizer and chat template; labels are set only inside the tool-call
bodies of the model's turns.
""",
    'train': """## 3. Training

The runner is started as a supervised subprocess: it is stopped at the deadline or after `STALL_MIN` minutes without a
new log line. Whatever happens, the adapters exported so far stay in the output.
""",
    'report': """## 4. Report

A summary built from the log: steps, loss per epoch, step time per length, tokens/s, compile times, peak HBM, every
export and every fallback event.
""",
}


INTRO_G10 = """# Gemma 4 31B Agent LoRA: Compaction-Aware Trainer

This notebook fine-tunes **Gemma 4 31B** (`gemma-4-31b-it`, QAT checkpoint) with a rank-16 **LoRA adapter** for the
Gemma 4 Developer Agent competition on one **Kaggle TPU v5e-8** session (JAX, [Tunix](https://github.com/google/tunix)'s
Gemma 4 model, [Qwix](https://github.com/google/qwix) LoRA). The adapter is written in the Hugging Face PEFT layout and
goes straight into a submission's `adapters/` folder.

Run it with **Run All** on a TPU v5e-8 with Internet on (the environment cell installs a pinned JAX stack). A full run
takes about 4 hours.

## The problem this adapter targets: life after context compaction

The competition harness (Google ADK) keeps the conversation short. Before each model call it checks the size of the
previous request; once that reaches **14,336 tokens**, it asks the same model to summarise everything except the last
few events (about three tool calls and their results). From then on the agent sees only:

```
system prompt
user:  For context: [model] said: <the model's own summary>
       <the last ~3 tool calls and results>
```

The original task message is gone. In long runs the first compaction comes around the 14th tool call and about half
of all steps happen after it. That is where agents struggle: after a compaction, exact repeats of earlier tool calls
roughly triple and most file reads re-read a file the agent had already read. Runs that never recover end at the time
limit, often with no edit at all.

A training set built by cutting each run at a fixed length below the trigger contains only the *start* of every run,
so an adapter trained on it never sees a single post-compaction step. This notebook trains on both phases.

## The recipe, and why it is expected to help

**1. The agent's own verified successes (rejection-sampling fine-tuning).** The rows come from runs of the same model
with the same prompt, tools and budget as the submission, kept only when the task's hidden tests passed. Fine-tuning on
them raises the probability of actions that led to verified fixes, with no teacher/student style mismatch.

**2. Each run is shown exactly as the harness showed it.** A run is split into segments with a simulator of the
harness's compaction (same trigger, same split rule that never separates a tool call from its result, same summariser
prompt, same message format), checked token-for-token against requests captured from the real harness. The first
segment is the run up to its first compaction; every later segment starts from `For context: [model] said: <summary>`
and the retained calls, and continues with what the agent actually did next: how a successful agent re-orients from
its own summary, checks its earlier edits and finishes.

**3. Summaries written by Gemma 4 31B itself.** Every summary was generated by the competition model from the
harness's own summariser prompt, so the post-compaction context is the kind of text the model really sees. Each
compaction has two independent summaries; the second is used for an extra copy of the post-compaction segments, so
the adapter learns to act on the content of a summary rather than on one particular wording.

**4. Loss on the tool calls only.** Thoughts, summaries, tool results and the retained calls (already trained in the
previous segment) are context. Training on thought text teaches thought *length*, and longer thoughts mean fewer tool
calls before the time limit. Only turns that open with a thought are trained, as the served model always thinks; a
tool call that the model's own next thought calls a mistake is masked.

**5. Two arms from the same initialisation.** Arm a trains on every segment (two epochs, lr 5e-5, cosine). Arm b
restarts from the same initial weights and trains on the post-compaction segments only. Comparing them with a paired
evaluation (base model vs adapter on the same held-out tasks) shows whether the pre-compaction segments help, hurt or
do nothing. Training loss is not a quality signal here: the model's own tool calls are already highly probable.

**6. fp32 master weights for LoRA.** Qwix creates the LoRA matrices in the base weights' dtype (bf16); with lr 5e-5 most
Adam updates to `lora_a` would round away. LoRA parameters and optimizer state are fp32 and cast to bf16 where the
forward pass reads them.

## How 31B parameters fit on 8 x 16 GB

* **Weights:** the bf16 checkpoint (`gemma-4-31b-it-qat-q4_0-unquantized`, the QAT model's unquantized weights, so the
  adapter matches the served `w4a16` model), sharded over a 2 x 4 mesh: FSDP over 2 devices, tensor parallelism over 4
  (the full-attention layers have 4 KV heads).
* **Lengths:** a segment ends after the response whose request crossed the trigger, plus its tool results, so
  segments reach ~24k tokens; four buckets (8,192, 12,288, 16,384 and 24,576 tokens) cover them; batch 2,
  decoder-level rematerialisation.
* **Attention:** Pallas splash attention with block 512 (block 1024 with head dim 512 exceeds the 16 MB scoped VMEM).
* **Loss:** logits only at the labelled positions, 512 at a time (the vocabulary has 262k tokens).
* **Throughput:** about 3,100 tokens/s.

## Safety nets for a long TPU session

* Attention ladder: a step that fails in the attention kernel is retried with splash block 256, then eager attention
  capped at 8,192 tokens.
* Memory: an out-of-memory length is demoted to the next shorter one (rows truncated, the kept-label fraction logged).
* A non-finite step applies no update; five in a row stop the run.
* Time-aware schedule: the plan is cut to what fits the time left, at an epoch boundary when possible.
* Exports are atomic and refuse non-finite tensors; the runner is supervised by a deadline and a no-progress watchdog,
  and nothing raises after training starts, so the outputs always survive.

## Inputs

* **Model:** Kaggle Models `google/gemma-4`, Transformers, `gemma-4-31b-it-qat-q4_0-unquantized`.
* **Training rows:** `sft.jsonl`, one segment per line, rendered with the Gemma 4 chat template and the harness's tool
  schema: `text`, `spans` (character ranges of the trained model turns), `source` (`onpolicy/pre` for the first
  segment of a run, `onpolicy/post` after a compaction), `instance_id`, `n_tokens`. Any dataset with these fields works;
  set `SFT_NAME` to its path suffix.
* **Reference adapter (optional):** any PEFT adapter for this model, used only to check the exported key names and
  shapes.

## Outputs

* `adapter_out/` (arm a, last), `adapters/ep0`, `adapters/ep1` (per epoch); `adapter_out_b/`, `adapters_b/ep<N>` (arm b).
* `tpu/log.jsonl` (every stage and step), `tpu/report.json` (summary), `tpu/data_stats.json`.

To use an adapter in a submission, put its two files under `adapters/<name>/` in `submission.zip` and add
`adapter: <name>` after the `model:` line of `agent.yaml`.
"""

SECTIONS_G10 = {
    'settings': """## Settings

`MODE = 'train'` runs the full recipe (`'canary'` runs two steps per length bucket to measure compile time, step time
and memory). `DEADLINE_S` bounds the whole notebook, `TRAIN_CAP_MIN` the training itself. `LR`, `EPOCHS` and `RANK`
define arm a; `PHASE2_LR` / `PHASE2_EPOCHS` define arm b and `PHASE2_SOURCE` restricts it to the rows whose `source`
contains that text (`/post`: post-compaction segments). `BUCKETS` are the padded lengths. `PINS` fixes every package
version.
""",
    'inputs': """## 1. Inputs

Locate the checkpoint, the segment rows and the reference adapter. (`KEEP_MAX_CALLS > 0` would keep only runs solved
within that many tool calls; it is off here.)
""",
}
VARIANTS = {
    'g9': {},
    'g10': {'BUILT': ROOT / 'kernels/train-g10-tpu/train-g10-tpu.py', 'OUT': ROOT / 'kernels/g4dev-trainer-g10-tpu',
            'SLUG': 'gemma-4-31b-agent-lora-compaction-aware-trainer',
            'TITLE': 'Gemma 4 31B Agent LoRA: Compaction-Aware Trainer', 'INTRO': INTRO_G10,
            'SECTIONS': {**SECTIONS, **SECTIONS_G10}},
}
VARIANTS['g10v2'] = {**VARIANTS['g10'], 'BUILT': ROOT / 'kernels/train-g10v2-tpu/train-g10v2-tpu.py',
                      'OUT': ROOT / 'kernels/g4dev-trainer-g10v2-tpu',
                      'SLUG': 'gemma-4-31b-agent-lora-compaction-trainer-v2',
                      'TITLE': 'Gemma 4 31B Agent LoRA: Compaction Trainer v2',  # Kaggle titles: <= 50 chars
                      'INTRO': INTRO_G10.replace("""segments reach ~24k tokens; four buckets (8,192, 12,288, 16,384 and 24,576 tokens) cover them; batch 2,
  decoder-level rematerialisation.""", """segments reach ~24k tokens; buckets of 8,192, 12,288, 16,384, 20,480 and 24,576 tokens, batch 2,
  decoder-level rematerialisation. At 24,576 the step needs ~28 MB more than a v5e chip has, so the memory fallback moves
  those rows to 20,480 (truncating only their last ~4k tokens).""")}

INTRO_G11 = VARIANTS['g10v2']['INTRO']
for _old, _new in [
    ('# Gemma 4 31B Agent LoRA: Compaction-Aware Trainer', '# Gemma 4 31B Agent LoRA: Short-Thought Trainer'),
    ('A full run\ntakes about 4 hours.', 'A full run\ntakes about 3.5 hours.'),
    ("""**4. Loss on the tool calls only.** Thoughts, summaries, tool results and the retained calls (already trained in the
previous segment) are context. Training on thought text teaches thought *length*, and longer thoughts mean fewer tool
calls before the time limit. Only turns that open with a thought are trained, as the served model always thinks; a
tool call that the model's own next thought calls a mistake is masked.""",
     """**4. Short-thought turns only.** The loss falls on the tool call of a model turn (its reasoning, like tool results and
summaries, is context); the served model always thinks before it acts. Token-weighted fine-tuning on raw runs is dominated by the few turns with very long thoughts:
in these runs a sixth of the turns hold about 60% of all thought text, and an adapter trained on all of them learns to
think longer. Under an 8-minute budget per task, longer thoughts mean fewer tool calls before the time limit. Here a
turn is trained only if its thought is at most 1,500 characters (about the base model's own 90th percentile per step);
longer turns stay in the context, untrained, so the adapter learns the decisions without the length. Summaries, tool
results and the retained calls (already trained in the previous segment) are context; only turns that open with a
thought are trained."""),
    ("""**5. Two arms from the same initialisation.** Arm a trains on every segment (two epochs, lr 5e-5, cosine). Arm b
restarts from the same initial weights and trains on the post-compaction segments only. Comparing them with a paired
evaluation (base model vs adapter on the same held-out tasks) shows whether the pre-compaction segments help, hurt or
do nothing.""",
     """**5. Two arms from the same initialisation.** Arm a trains on every segment (two epochs, lr 5e-5, cosine). Arm b
restarts from the same initial weights and trains only on runs that resolved their task within 14 tool calls (the
median run): the most budget-efficient behaviour in the data. Comparing each with the base model in a paired
evaluation (same held-out tasks) shows whether the efficient runs alone carry the useful signal."""),
    ("""`source` (`onpolicy/pre` for the first
  segment of a run, `onpolicy/post` after a compaction)""",
     """`source` (`onpolicy/pre` for the first
  segment of a run, `onpolicy/post` after a compaction, with `/fast` appended for runs solved within 14 tool calls)"""),
]:
    assert INTRO_G11.count(_old) == 1, _old[:60]
    INTRO_G11 = INTRO_G11.replace(_old, _new)
VARIANTS['g11'] = {**VARIANTS['g10'], 'BUILT': ROOT / 'kernels/train-g11-tpu/train-g11-tpu.py',
                    'OUT': ROOT / 'kernels/g4dev-trainer-g11-tpu',
                    'SLUG': 'gemma-4-31b-agent-lora-short-thought-trainer',
                    'TITLE': 'Gemma 4 31B Agent LoRA: Short-Thought Trainer', 'INTRO': INTRO_G11,
                    'SECTIONS': {**SECTIONS, **SECTIONS_G10, 'settings': SECTIONS_G10['settings'].replace(
                        "(`/post`: post-compaction segments)", "(`/fast`: runs solved within 14 tool calls)")}}
VARIANTS['g12'] = {**VARIANTS['g11'], 'BUILT': ROOT / 'kernels/train-g12-tpu/train-g12-tpu.py',
                    'OUT': ROOT / 'kernels/g4dev-trainer-g12-tpu',
                    'SLUG': 'gemma-4-31b-agent-lora-short-thought-trainer-v2',
                    'TITLE': 'Gemma 4 31B Agent LoRA: Short-Thought Trainer v2',
                    'INTRO': INTRO_G11.replace('# Gemma 4 31B Agent LoRA: Short-Thought Trainer',
                                               '# Gemma 4 31B Agent LoRA: Short-Thought Trainer v2')}
VARIANTS['g13'] = {**VARIANTS['g11'], 'BUILT': ROOT / 'kernels/train-g13-tpu/train-g13-tpu.py',
                    'OUT': ROOT / 'kernels/g4dev-trainer-g13-tpu',
                    'SLUG': 'gemma-4-31b-agent-lora-short-thought-trainer-v3',
                    'TITLE': 'Gemma 4 31B Agent LoRA: Short-Thought Trainer v3',
                    'INTRO': INTRO_G11.replace('# Gemma 4 31B Agent LoRA: Short-Thought Trainer',
                                               '# Gemma 4 31B Agent LoRA: Short-Thought Trainer v3')}
VARIANTS['g14'] = {**VARIANTS['g11'], 'BUILT': ROOT / 'kernels/train-g14-tpu/train-g14-tpu.py',
                    'OUT': ROOT / 'kernels/g4dev-trainer-g14-tpu',
                    'SLUG': 'gemma-4-31b-agent-lora-short-thought-trainer-v4',
                    'TITLE': 'Gemma 4 31B Agent LoRA: Short-Thought Trainer v4',
                    'INTRO': INTRO_G11.replace('# Gemma 4 31B Agent LoRA: Short-Thought Trainer',
                                               '# Gemma 4 31B Agent LoRA: Short-Thought Trainer v4')}
INTRO_G15 = INTRO_G11.replace('# Gemma 4 31B Agent LoRA: Short-Thought Trainer',
                              '# Gemma 4 31B Agent LoRA: Preference Trainer')
_old5 = INTRO_G15[INTRO_G15.index('**5. Two arms from the same initialisation.**'):]
_old5 = _old5[:_old5.index('\n\n') if '\n\n' in _old5 else len(_old5)]
INTRO_G15 = INTRO_G15.replace(_old5, """**5. Preferences on top of the fine-tuning rows.** Besides the solved runs, the data holds pairs of runs on the same
task: a solved run against a failed one, and a fast solved run against a slow one (at least 3 tool calls and 25% fewer;
a faster run that skipped the final test run is not preferred). Each epoch the pair rows get signed weights from how
well the adapter already separates them: DPO for solved-vs-failed pairs, IPO (a margin pulled to a fixed target
instead of pushed to infinity) for the noisier fast-vs-slow pairs. The solved side keeps its ordinary fine-tuning
weight as well. One arm, two epochs, lr 5e-5.""")
VARIANTS['g15'] = {**VARIANTS['g11'], 'BUILT': ROOT / 'kernels/train-g15-tpu/train-g15-tpu.py',
                   'OUT': ROOT / 'kernels/g4dev-trainer-g15-tpu',
                   'SLUG': 'gemma-4-31b-agent-lora-preference-trainer',  # the 56-char slug was refused (HTTP 400)
                   'TITLE': 'Gemma 4 31B Agent LoRA: Preference Trainer', 'INTRO': INTRO_G15}

INTRO_G16 = """# Gemma 4 31B Agent: Clock-Calibrated Preferences

This notebook trains the LoRA adapter behind our paper-track entry: **preference optimisation for a coding agent that
works against a wall clock.** It fine-tunes **Gemma 4 31B** (QAT checkpoint) on one **Kaggle TPU v5e-8** (JAX,
[Tunix](https://github.com/google/tunix)'s Gemma 4 model, [Qwix](https://github.com/google/qwix) LoRA, rank 16) from
pairs of the agent's own runs, and writes a Hugging Face PEFT adapter that drops into a competition submission.

The idea in one line: **prefer runs that solve the task, and among solved runs prefer the faster one, by an amount set
by how much faster it was.** Outcome preferences use DPO; speed preferences use IPO with a per-pair target derived from
a measured model of serving time ("clock-calibrated IPO"). Everything runs as one weighted fine-tuning loop, without a
second copy of the model in memory.

The notebook trains **two adapters back to back** on the same data, from the same starting point, for the same
number of steps: **arm a** with clock-calibrated targets, **arm b** with one uniform target (plain IPO). The pair is
the ablation: the only difference is how the preference pressure is distributed across pairs.

Run it with **Run All** on a TPU v5e-8 with Internet on (the environment cell installs a pinned JAX stack). A full run
takes about 8.5 hours (the Kaggle TPU session limit is 9).

## Why time, and why preferences

In the Gemma 4 Developer Agent competition every task gets **8 minutes**; the agent loops *think -> call a tool ->
read the result*. We timed 7,980 model calls from 500 agent sessions on repositories the model had never seen (four
L4 GPUs, the competition's vLLM stack, 8 sessions at a time):

* **About 98% of a session is the model, not the tools.** Tool execution has a session median of 1-2 seconds.
* **A call costs about 16.7 s plus 0.139 s per generated token** (least squares; prompt length adds nothing measurable,
  thanks to prefix caching). Calls with under 100 tokens take ~9 s; calls with 2,000+ tokens ~6 minutes.
* **Running out of time is fatal.** 249 of the 500 sessions hit the time limit; **none** of them solved its task, even
  the 166 that had already edited the code. Sessions that finished in time solved 51%.

So the model loses tasks less for lack of skill than for lack of time. Supervised fine-tuning on solved runs helped
mainly by shortening the model's reasoning (an adapter trained only on short-thought turns solved 25 vs 18 of 71 tasks
over the base model, with timeouts down from 38 to 24). This notebook adds the next step: teach the model *directly*
which of two behaviours is better, including "same answer, less time".

## Where the pairs come from (no teacher, no human labels)

The agent (the competition model, the competition prompt, tools and budget) re-attempts tasks it has already solved,
through the official harness, with hidden tests as the verifier; each attempt costs about one cent of model-API time.
Re-running a task gives pairs for free:

* **Outcome pairs** - a task with a passing and a failing run: the passing run with the fewest tool calls is *chosen*,
  the failing run closest to it in length is *rejected*.
* **Efficiency pairs** - a task with only passing runs: the run with the fewest tool calls is chosen and the slowest is
  rejected, if they differ by at least 3 calls and 25%. A faster run that skipped the final test while the slower one
  ran it is not preferred, so the model cannot learn "skip verification to be fast". A pair is also dropped when the
  run with fewer calls is *not* faster by the clock model below (it wrote more tokens): about one in eight.

Re-attempts are scheduled hardest-first (tasks whose best solve needed the most calls fail most often, which yields the
most outcome pairs). Each run is rendered exactly as the harness showed it, including the summaries written when the
context is compacted, and each side carries its **estimated serving time**
`T = 16.7 s x model calls + 0.139 s x generated tokens` from the timing model above.

## The objective

**What is trained.** The loss falls on the agent's *actions*: the tool-call tokens of model turns whose reasoning is
at most 1,500 characters. Reasoning text, tool results and summaries are context. A preference therefore says "act as
the faster run acted, given what it had seen", and the time gap decides how strongly.

For a pair, let `m` be the difference between the chosen and the rejected run of the mean (per trained token)
log-ratio `log pi(y|x) - log pi_ref(y|x)`; `pi_ref` is the starting point (the LoRA with B = 0, i.e. the base model).

* **Outcome pairs: DPO** (Rafailov et al. 2023). Gradient weight `a = beta * sigmoid(-beta * m)`, beta = 2: pushed until
  the pair is clearly separated.
* **Efficiency pairs: clock-calibrated IPO.** IPO (Azar et al. 2023) regresses the margin to a target `t` with
  `(m - t)^2` instead of pushing it to infinity, which suits noisy preferences: two passing runs of the same task often
  differ by chance. Here each pair gets its own target from its time gap,
  `t = tau * ln(T_rejected / T_chosen)`, clipped to `[t_lo, t_hi]`: a pair whose runs differ by a few seconds asks for
  almost no margin, a pair where one run took twice as long asks for a larger one. Weight
  `a = clip((t - m) / t_mean, -1, 3)`, with `t_mean` the mean target over efficiency pairs: this is the IPO gradient
  `-dL/dm = 2 (t - m)` up to a constant, scaled so the *average* pair starts at weight 1. A pair with a large time gap
  starts above 1 and a near-tie below 1; every pair's weight falls to 0 at its target and turns negative past it.
* **Ablation (arm b): uniform IPO.** Every efficiency pair gets the same target `t` (the arm-a mean) and weight
  `a = clip((t - m) / t, -1, 1)`. Same total push at the start, spread evenly instead of by time gap.
* **Plus supervised fine-tuning:** the chosen side keeps its ordinary SFT weight (1); every chosen run passed the
  hidden tests, so the policy stays anchored to verified behaviour. The data holds pair rows only.

## How it runs on a single TPU v5e-8

DPO normally needs a frozen reference model next to the policy. With LoRA the reference is the policy with `B = 0`, so
reference log-probabilities are computed once, with a forward pass, before training. The trainer then keeps the
ordinary token-weighted cross-entropy loop and gives every pair row a **signed weight**: chosen rows `1 + a`, rejected
rows `-a` (an unlikelihood term; a rejected side already well below the reference stops receiving weight). At the
start of every epoch the margins are recomputed with the current adapter and the weights refreshed, so each pair's
pressure fades as it is satisfied. This is a coarse-grained version of the DPO/IPO gradient (weights refreshed per
epoch rather than per step) and costs one forward pass over the pair rows per epoch.

Each refresh is logged (`dpo_refresh` lines in `tpu/log.jsonl`): margins and accuracy per pair kind, and for the
efficiency pairs the correlation between achieved margin and target - whether the model learned *more* preference
where the time gap was larger.

## Inherited from our earlier adapters

* **Short-thought turns only:** a model turn is trained only if its reasoning is at most 1,500 characters, and then only
  on its tool call; longer turns stay as untrained context, so the adapter learns the decisions without the verbosity.
* **Compaction-aware rows:** runs are split exactly where the harness compacts the conversation (trigger 14,336 tokens),
  and later segments start from the model's own summary, as at serving time.
* **fp32 LoRA master weights;** loss computed only at labelled positions, 512 logits at a time (262k vocabulary).
* **Memory:** bf16 base weights sharded FSDP 2 x tensor-parallel 4; splash attention (block 512); decoder-level
  rematerialisation; length buckets of 8k to 24k tokens; batch 2. The forward-only reference pass donates the model's
  buffers, otherwise XLA keeps a second copy of the weights and a v5e chip runs out of memory.
* **Safety nets:** attention-kernel fallbacks, out-of-memory length demotion, non-finite steps skipped, a time-aware
  schedule, atomic exports, and a deadline/no-progress watchdog; nothing raises after training starts.

## Inputs

* **Model:** Kaggle Models `google/gemma-4`, Transformers, `gemma-4-31b-it-qat-q4_0-unquantized`.
* **Rows:** `sft.jsonl` - one rendered segment per line (`text`, `spans` = character ranges of trained model turns,
  `source`, `n_tokens`, ...). Pair rows have `source` `dpo:<task>~<kind>:c|r` (`kind` = `fail` or `speed`).
* **Pair costs:** `pairs.json` next to it: `{pair id: {kind, c: {calls, turns, gen_tokens, est_s}, r: {...}}}`.
* **Reference adapter (optional):** any PEFT adapter for this model, used only to check exported key names and shapes.

## Outputs

* Arm a (clock-calibrated): `adapter_out/` (final), `adapters/ep<N>/` (per epoch).
* Arm b (uniform IPO): `adapter_out_b/`, `adapters_b/ep<N>/`.
* `tpu/log.jsonl` (every stage, step and refresh, tagged by arm), `tpu/report.json`, `tpu/data_stats.json`.

To use the adapter, put its two files under `adapters/<name>/` in `submission.zip` and add `adapter: <name>` after
the `model:` line of `agent.yaml`.
"""
SECTIONS_G16 = {**VARIANTS['g15']['SECTIONS'],
    'settings': """## Settings

`MODE = 'train'` runs the full recipe (`'canary'` runs two steps per length bucket to measure compile time, step time
and memory). `DEADLINE_S` bounds the whole notebook, `TRAIN_CAP_MIN` the training itself; `LR`, `EPOCHS` and `RANK`
set the LoRA run, `BUCKETS` the padded lengths and `PINS` every package version. `DPO` holds the preference settings:
`beta` (DPO for outcome pairs), `sft_chosen` (the chosen side's SFT weight), `reject_floor` (a rejected side whose mean
log-ratio is below it gets no more weight), and for efficiency pairs `loss_speed = 'ipo'` with
`ipo_target_speed = 'clock'`: target `ipo_tau_speed * ln(T_slow / T_fast)` clipped to `[ipo_lo_speed, ipo_hi_speed]`
(`ipo_fallback_speed` when a pair has no cost record); `ipo_norm_speed = 'mean'` scales weights by the mean target and
`ipo_amax` caps them. Arm b is the `PHASE2_*` block: after arm a is exported the LoRA is reset to the same initial
values and the optimizer to zero, `PHASE2_DPO` overrides the preference settings (one uniform target, weights scaled
per pair), `PHASE2_A_SHARE` is arm a's share of the training time and `PHASE2_MATCH_A` makes arm b run exactly arm a's
steps (same rows, same order). The reference log-probabilities are computed once and serve both arms.
""",
    'report': """## 4. Report

A summary built from the log: steps, loss per epoch, step time per length, tokens/s, compile times, peak HBM, exports
and fallback events. The preference diagnostics are the `dpo_refresh` lines of `tpu/log.jsonl`, one per epoch:
margins and accuracy per pair kind (`fail`, `speed`), the mean weight, how many rejected sides hit the floor, and for
the efficiency pairs the target and weight distributions, the share of pairs that reached their target and the
correlation between achieved margin and target. The `dpo` lines (one per arm) give each arm's targets and weight scale;
`plan` and `done` lines are tagged with the arm (`phase` a or b).
"""}
VARIANTS['g16'] = {**VARIANTS['g15'], 'BUILT': ROOT / 'kernels/train-g16-tpu/train-g16-tpu.py', 'SECTIONS': SECTIONS_G16,
                   'OUT': ROOT / 'kernels/g4dev-trainer-g16-tpu',
                   'SLUG': 'gemma-4-31b-agent-clock-calibrated-preferences',
                   'TITLE': 'Gemma 4 31B Agent: Clock-Calibrated Preferences', 'INTRO': INTRO_G16}
INTRO_G17V2 = """# Gemma 4 31B Agent: In-Step Teacher Preferences

This notebook trains the second stage of our paper-track adapter: **a stronger model corrects the agent at the states
the agent itself reaches, and the agent is trained to prefer the correction over what it actually did** - with a
preference objective that stays bounded. It fine-tunes **Gemma 4 31B** (QAT checkpoint) on one **Kaggle TPU v5e-8**
(JAX, [Tunix](https://github.com/google/tunix)'s Gemma 4 model, [Qwix](https://github.com/google/qwix) LoRA, rank 16),
starting from our best supervised adapter, and writes Hugging Face PEFT adapters that drop into a competition
submission.

The notebook trains **two adapters back to back**, from the same starting adapter, on the same pairs, for the same
steps: **arm a** with the preference term (in-step DPO plus supervised loss on the teacher's step), **arm b** with the
supervised loss on the teacher's step only (`beta = 0`). The pair is the ablation: does pushing down the agent's own
step add anything to imitating the teacher's?

Run it with **Run All** on a TPU v5e-8 with Internet on (the environment cell installs a pinned JAX stack). A full run
takes about 6.5 hours.

## Stage 1 (the starting adapter): fast, verified solves

The starting point is our short-thought adapter: Gemma 4 31B fine-tuned on its *own* agent runs that passed the
task's hidden tests within 14 tool calls, with the loss on the tool calls of turns whose reasoning was short. The agent
works under a hard budget (24 tool calls and 8 minutes per task) and a model call costs about **16.7 s + 0.139 s per
generated token**, so most failures are timeouts, not wrong ideas; that adapter solved 25 vs 18 of 71 held-out tasks
over the base model, with timeouts down from 38 to 24. What it cannot learn from successes alone is *what to do
instead* at the steps where the agent wastes its budget.

## Stage 2 data: a teacher labels the agent's own decision points

From pairs of the agent's runs on the same tasks (a faster and a slower or failed run), we sampled 1,000 **decision
points**: the exact context the agent saw at one step (its prompt, task, earlier steps with their tool results, and
any summaries written when the context was compacted) together with the step it actually took. Sampling favoured steps
that usually waste time: long reasoning, a repeated call, still exploring after 8 calls without an edit, a full test
suite run.

A stronger model (**Gemini 3.8 Flash**, through Kaggle's model proxy) saw each decision point as a readable transcript,
with the agent's budget and the cost model above, and

1. rated the agent's next step **good, slow** (right call, reasoning much longer than needed), **wasteful** (a call
   that does not move the fix forward) or **wrong** (a mistake), and
2. unless it was good, wrote the step it would take instead, in the agent's place: a **first-person plan of 200-700
   characters** (what the evidence shows, then what to do next and why) and **exactly one tool call**.

A teacher step is kept only if it is executable from that state: a known tool with valid arguments; an edit whose
`old_string` appears in what the agent had actually seen; no edits to test files; no bare full-suite `pytest`. **589
pairs** remain (400 wasteful, 164 slow, 25 wrong; 138 keep the agent's own call with a shorter plan). The teacher's
plans have a median of 322 characters; the agent's rejected reasoning a median of 2,359 (p90 6,587).

Each pair is two training rows that share the agent's context and differ only in the final step:

* **chosen** = the teacher's step, rendered in Gemma's own format (`<|channel>thought ... <channel|>` and the tool
  call), and
* **rejected** = the step Gemma actually generated.

Only that final step is trained, reasoning included: the decision lives in the plan, and a call trained without its
plan would only teach the model to follow whatever plan it wrote. Because every state is one the agent really reached
(its own context, compactions included), this is DAgger-style correction rather than imitation of teacher trajectories
(which, in an earlier experiment, put the model in states and a narration style it never produces, and hurt).

## The objective: in-step DPO with an SFT anchor

For a pair, let `r_c` and `r_r` be the mean, per trained token, log-ratio `log pi(y|x) - log pi_ref(y|x)` of the chosen
and the rejected step, and `m = r_c - r_r`. The reference `pi_ref` is the starting adapter: its log-probabilities are
computed once, with a forward pass, before training. Each training step holds **one pair** (two rows) and minimises

    L = -log sigmoid(beta * m)  +  sft * NLL(chosen step) / tokens

with `beta = 2` and `sft = 1` (DPO, Rafailov et al. 2023, length-normalised; plus the chosen side's supervised loss, as
in RPO, Pang et al. 2024). The gradient on the margin is `beta * sigmoid(-beta * m)`: 1 at the start, and fading as the
pair separates.

**Why "in-step".** Our first attempt used a cheaper approximation: the margin of every pair was computed once per epoch
and turned into fixed row weights (chosen `1 + a`, rejected `-a`) for the whole epoch. With actions-only rows that
worked, because chosen and rejected tool calls overlap. With reasoning in the rows it did not: the agent's own long
reasoning is a very distinctive span, and with a fixed negative weight for a whole epoch nothing stopped the model from
driving it down - after one epoch the rejected steps sat at **-70 nats per token** below the reference. Here the margin
comes from the same forward pass as the gradient, so the preference term switches itself off as soon as a pair is
separated. A rejected log-ratio can also be held at a floor (`instep_floor`, off here).

**Arm b (`beta = 0`)** keeps the identical batches and the supervised term and drops the preference term: plain
supervised fine-tuning on the teacher's steps at the agent's states.

## How it runs on a single TPU v5e-8

* **Start from the stage-1 adapter:** its PEFT tensors are mapped back onto the Qwix LoRA parameters (the exact
  inverse of the export; checked bit for bit on load), so the reference and both arms start from it.
* **Memory:** bf16 base weights sharded FSDP 2 x tensor-parallel 4; a pair batch is 2 rows, one per FSDP shard, padded
  to the longer row's length bucket (8k to 20k tokens); splash attention (block 512); decoder-level rematerialisation;
  fp32 LoRA master weights; loss computed only at labelled positions, 512 logits at a time (262k vocabulary). The
  forward-only reference pass donates the model's buffers (otherwise XLA keeps a second copy of the weights).
* **A pair is trained whole or not at all:** if a length had to be demoted after an out-of-memory error, a pair that
  no longer fits is skipped rather than trained on a truncated step.
* **Safety nets:** attention-kernel fallbacks, out-of-memory length demotion, non-finite steps skipped, a time-aware
  schedule, atomic exports, a deadline/no-progress watchdog; nothing raises after training starts.

## Inputs

* **Model:** Kaggle Models `google/gemma-4`, Transformers, `gemma-4-31b-it-qat-q4_0-unquantized`.
* **Pairs:** `sft.jsonl` - one row per side: `text` (the agent's rendered context plus the step), `spans` (the trained
  final step), `train_on: all`, `source` `dpo:<state>~teacher:c|r`.
* **Starting adapter:** the stage-1 PEFT adapter (`INIT_ADAPTER`, a path pattern under `/kaggle/input`).
* **Reference adapter (optional):** any PEFT adapter for this model, used only to check exported key names and shapes.

## Outputs

* Arm a (in-step DPO + SFT): `adapter_out/` (final), `adapters/ep<N>/` (per epoch).
* Arm b (SFT on the teacher's steps only): `adapter_out_b/`, `adapters_b/ep<N>/`.
* `tpu/log.jsonl` (every stage and step, tagged by arm; pair steps carry their margin), `tpu/report.json`.

To use an adapter, put its two files under `adapters/<name>/` in `submission.zip` and add `adapter: <name>` after
the `model:` line of `agent.yaml`.
"""
SECTIONS_G17V2 = {**VARIANTS['g15']['SECTIONS'],
    'settings': """## Settings

`MODE = 'train'` runs the full recipe (`'canary'` runs two steps per length bucket to measure compile time, step time
and memory). `DEADLINE_S` bounds the whole notebook, `TRAIN_CAP_MIN` the training itself; `LR`, `EPOCHS` and `RANK`
set the LoRA run, `BUCKETS` the padded lengths and `PINS` every package version. `INIT_ADAPTER` is the starting
adapter. `DPO` holds the preference settings: `beta`, `sft_chosen` (the weight of the chosen step's supervised loss) and
`in_step = True` (each pair is one batch and its margin comes from that step's forward pass; `instep_floor` would hold a
rejected side below that log-ratio). Arm b is the `PHASE2_*` block: after arm a is exported the LoRA is reset to the
starting adapter and the optimizer to zero, `PHASE2_DPO` overrides the preference settings (`beta = 0`),
`PHASE2_A_SHARE` is arm a's share of the training time and `PHASE2_MATCH_A` makes arm b run exactly arm a's steps (same
pairs, same order). The reference log-probabilities are computed once and serve both arms.
""",
    'report': """## 4. Report

A summary built from the log: steps, loss per epoch, step time per length, tokens/s, compile times, peak HBM, exports
and fallback events. The preference diagnostics are in `tpu/log.jsonl`: every pair step logs its margin `m` and both
log-ratios (`ratio_c`, `ratio_r`), and a `pair_epoch` line per arm and epoch summarises them (margin percentiles,
accuracy = share of pairs with `m > 0`, chosen and rejected log-ratio percentiles), measured as each pair was met,
before its own update. The `dpo` lines (one per arm) give each arm's settings; `init_adapter` confirms the starting
adapter's round trip.
"""}
VARIANTS['g17v2'] = {**VARIANTS['g15'], 'BUILT': ROOT / 'kernels/train-g17v2-tpu/train-g17v2-tpu.py',
                     'SECTIONS': SECTIONS_G17V2, 'OUT': ROOT / 'kernels/g4dev-trainer-g17v2-tpu',
                     'SLUG': 'gemma-4-31b-agent-in-step-teacher-preferences',
                     'TITLE': 'Gemma 4 31B Agent: In-Step Teacher Preferences', 'INTRO': INTRO_G17V2}
INTRO_HEAL = """# Gemma 4 31B Agent: Healing a Pruned Model with Its Teacher

**In one paragraph.** We made Gemma 4 31B smaller by deleting 30% of the hidden units ("channels") of every MLP layer:
the ones our coding agent hardly uses. (We measured which channels are active while the model does agent work and kept
the busiest 70%.) Served in 4 bits on four L4 GPUs, the cut model generates text about 15% faster, but it predicts the
agent's own tokens a little worse: +0.08 nats per token. This notebook tries to win that quality back cheaply by
letting the original, uncut model act as a **teacher**: the cut model learns to imitate the full model's predictions on
agent transcripts. Only a small LoRA adapter (rank 16) is trained; the model's weights stay frozen.

## Two arms: with and without the teacher

* **Arm a (teacher, knowledge distillation).** At every token of an agent transcript, the cut model is trained to
  match the full model's whole probability distribution over the next token (it minimises KL(full || cut)). The
  teacher tells the student not just which token came next, but how likely every other token was.
* **Arm b (no teacher).** Exactly the same batches in the same order, but the cut model learns from the text alone:
  the usual next-token loss on the tokens that actually came next.

Comparing a with b answers the question: does the teacher matter, or does any fine-tuning on agent text heal the
cut equally well?

## The trick that makes it fit on one TPU

The cut model lives *inside* the full model: a 0/1 mask multiplies the outputs of each MLP's hidden units, and a
deleted unit contributes exactly nothing (the same as removing its weights). The teacher is the very same weights with
the mask set to all ones and the adapter switched off. So there is one copy of the 62 GB model in memory, and each
training step runs the teacher once (no gradients) and the student once (with gradients).

Because a deleted unit always outputs zero, its adapter weights never receive a gradient. The finished adapter can
therefore be cut to the smaller model's shape exactly and served with the 4-bit cut model.

## How it is scored

40 held-out agent transcripts from tasks the training never sees (the same 40 the pruning study measured). For each
model we report the average surprise in nats per token (lower is better) on the tokens the agent wrote itself, and on
all tokens:

* the full model (the teacher): the target;
* the cut model before any healing: the damage to repair;
* the cut model after each epoch of each arm.

For the cut model we also report how far its whole next-token distribution is from the full model's: the KL divergence
KL(full || cut) per token, again on the agent's own tokens and on all tokens. Surprise can fall below the full model's
(fine-tuning on the agent's own text sharpens the model toward what the agent wrote), so surprise alone cannot say
whether the cut model became the full model again; the KL can: 0 means identical predictions.

## Inputs

* **Model:** Kaggle Models `google/gemma-4`, Transformers, `gemma-4-31b-it-qat-q4_0-unquantized` (bf16).
* **Agent transcripts:** `g4dev-g11-data/sft.jsonl`: our agent's own runs (`text`, and `spans` marking what the agent
  wrote). One task in ten is held out (`md5(instance_id) % 10 == 0`); everything else is training data.
* **Which channels to keep:** the pruning study's channel scores (`scores.npz`, key `agent_ff`): per layer the 14,848
  highest-scoring of 21,504 channels (70%, rounded to a multiple of 512, which the 4-bit GPU kernels need).
* **Reference adapter (optional):** any PEFT adapter for this model, used only to check exported key names and shapes.

## Outputs

* Arm a (teacher): `adapter_out/` (final), `adapters/ep<N>/` (per epoch).
* Arm b (no teacher): `adapter_out_b/`, `adapters_b/ep<N>/`.
* `tpu/heal_mask.npz` (the kept channels, so the 4-bit model can be cut identically), `tpu/heal_data.json` (data
  counts), `tpu/log.jsonl` (every step; the held-out scores are the `heal_eval` lines), `tpu/report.json`.

The adapters are for the full-size model; to serve one with the cut model, keep the rows of `gate_proj`/`up_proj`
`lora_B` and the columns of `down_proj` `lora_A` listed in `heal_mask.npz`.
"""
SECTIONS_HEAL = {**SECTIONS,
    'settings': """## Settings

`HEAL` holds the healing setup: `scores` (where the pruning study's channel scores are), `keep` (share of channels kept
per layer) and `multiple` (the kept count is rounded to it), `kd` (True: arm a learns from the teacher), `temp` (the
distillation temperature; 1 = the teacher's distribution as it is), `positions` (`'all'`: every token of a transcript
is a training target), `eval_rows` and `eval_len` (the held-out rows and their length). `PHASE2_HEAL = {'kd': False}`
makes arm b the no-teacher control; `PHASE2_MATCH_A` makes it run exactly arm a's batches; `PHASE2_A_SHARE` splits the
training time between the arms. `LR`, `EPOCHS`, `RANK` and `BUCKETS` (padded lengths) are as in our other trainers.
""",
    'modules': """## The four modules

* `lora_tpu.py`: the model wrapper, Qwix LoRA with fp32 master weights, the training steps and the PEFT export. For
  healing: `install_ff_mask` (the cut as a 0/1 mask), `teacher_view` (the same buffers with masks 1 and LoRA 0, i.e.
  the original model), `make_distill_step` (KL to the teacher at every target position, 512 positions at a time over
  the 262k-token vocabulary) and `make_eval_rows` (held-out log-likelihood of student or teacher, and the student's KL
  to the teacher).
* `train_lora.py`: the shared tokenisation script of our SFT trainers (not used for healing, kept for the same stack).
* `train_tpu_runner.py`: the training loop (length buckets, schedule, fallbacks, exports, the second arm, and the
  held-out evaluations before, per epoch and after each arm).
* `heal_data.py`: the training rows (every token a target), the 40 held-out rows and the channel mask.
""",
    'inputs': """## 1. Inputs

Locate the checkpoint, the agent transcripts, the channel scores and the reference adapter. A missing input stops the
notebook here, before the 62 GB load.
""",
    'dump': """## 2. Training rows, held-out rows and the cut

`heal_data.py` tokenises the transcripts with the model's tokenizer, splits them by task exactly as the pruning study
did, and writes three files: the training rows (cut at the longest bucket, every token a target), the 40 held-out rows
(with a flag on each token the agent wrote itself) and the channel mask (layers x channels, 1 = kept). The counts land
in `tpu/heal_data.json`.
""",
    'report': """## 4. Report

A summary built from the log: steps, loss per epoch (arm a: KL to the teacher in nats per token; arm b: next-token
loss), step times, tokens/s, compile times, peak memory, exports and fallback events, and under `heal` every held-out
evaluation: `own_nats` (tokens the agent wrote) and `all_nats` (every token), for the teacher and for the cut model,
and the cut model's `kl_own` / `kl_all` (mean KL(full || cut) per token).
""",
}
VARIANTS['heal'] = {**VARIANTS['g15'], 'BUILT': ROOT / 'kernels/heal-c70-tpu/heal-c70-tpu.py',
                    'SECTIONS': SECTIONS_HEAL, 'OUT': ROOT / 'kernels/g4dev-heal-c70-tpu',
                    'SLUG': 'gemma-4-31b-agent-pruned-model-teacher-healing',
                    'TITLE': 'Gemma 4 31B Agent: Pruned Model Teacher Healing', 'INTRO': INTRO_HEAL,  # title slug = id (<= 50 chars)
                    'MODULES': MODULES + [('HEAL_DATA', 'heal_data.py')]}
# 2026-11-06 (B187): the same recipe on the deeper cut (55% of the channels kept). Same notebook text, except the
# opening paragraph, what the first run found, and the kept count.
INTRO_HEAL55 = INTRO_HEAL
for _old, _new in [
    ("# Gemma 4 31B Agent: Healing a Pruned Model with Its Teacher", "# Gemma 4 31B Agent: Healing a Deeper Pruning Cut"),
    ("""**In one paragraph.** We made Gemma 4 31B smaller by deleting 30% of the hidden units ("channels") of every MLP layer:
the ones our coding agent hardly uses. (We measured which channels are active while the model does agent work and kept
the busiest 70%.) Served in 4 bits on four L4 GPUs, the cut model generates text about 15% faster, but it predicts the
agent's own tokens a little worse: +0.08 nats per token. This notebook tries to win that quality back cheaply by
letting the original, uncut model act as a **teacher**: the cut model learns to imitate the full model's predictions on
agent transcripts. Only a small LoRA adapter (rank 16) is trained; the model's weights stay frozen.""",
     """**In one paragraph.** This is the second healing run, on a deeper cut. We made Gemma 4 31B smaller by deleting 45% of
the hidden units ("channels") of every MLP layer: the ones our coding agent hardly uses. (We measured which channels
are active while the model does agent work and kept the busiest 55%.) Served in 4 bits on four L4 GPUs, this cut model
generates text about 24% faster, but it predicts the agent's own tokens clearly worse: +0.19 nats per token, more than
twice the damage of the milder cut. This notebook tries to win that quality back cheaply by letting the original, uncut
model act as a **teacher**: the cut model learns to imitate the full model's predictions on agent transcripts. Only a
small LoRA adapter (rank 16) is trained; the model's weights stay frozen.

## What the first run found

The first run (notebook "Gemma 4 31B Agent: Pruned Model Teacher Healing") healed the milder cut, which keeps 70% of
the channels. Learning from the teacher took the damage from +0.085 to +0.021 nats per token and halved the distance
to the full model. Learning from the text alone pushed the loss even lower, below the full model's own, while the cut
model's predictions moved *further away* from the full model's: it learned the transcripts, not the model. This run
asks the same two questions of a cut that removes half as much again: how much of a larger damage does the teacher win
back, and does the teacher still matter?"""),
    ("""per layer the 14,848
  highest-scoring of 21,504 channels (70%, rounded to a multiple of 512, which the 4-bit GPU kernels need).""",
     """per layer the 11,776
  highest-scoring of 21,504 channels (55%, rounded to a multiple of 512, which the 4-bit GPU kernels need)."""),
]:
    assert INTRO_HEAL55.count(_old) == 1, _old[:60]
    INTRO_HEAL55 = INTRO_HEAL55.replace(_old, _new)
VARIANTS['heal55'] = {**VARIANTS['heal'], 'BUILT': ROOT / 'kernels/heal-c55-tpu/heal-c55-tpu.py',
                      'OUT': ROOT / 'kernels/g4dev-heal-c55-tpu',
                      'SLUG': 'gemma-4-31b-agent-healing-a-deeper-pruning-cut',
                      'TITLE': 'Gemma 4 31B Agent: Healing a Deeper Pruning Cut', 'INTRO': INTRO_HEAL55}
# 2026-11-07 (B193): the 70% cut again, healed only where the agent writes.
INTRO_HEAL70OWN = INTRO_HEAL
for _old, _new in [
    ("# Gemma 4 31B Agent: Healing a Pruned Model with Its Teacher", "# Gemma 4 31B Agent: Healing Where the Agent Writes"),
    ("""**In one paragraph.** We made Gemma 4 31B smaller by deleting 30% of the hidden units ("channels") of every MLP layer:
the ones our coding agent hardly uses. (We measured which channels are active while the model does agent work and kept
the busiest 70%.) Served in 4 bits on four L4 GPUs, the cut model generates text about 15% faster, but it predicts the
agent's own tokens a little worse: +0.08 nats per token. This notebook tries to win that quality back cheaply by
letting the original, uncut model act as a **teacher**: the cut model learns to imitate the full model's predictions on
agent transcripts. Only a small LoRA adapter (rank 16) is trained; the model's weights stay frozen.""",
     """**In one paragraph.** This is the third healing run. We made Gemma 4 31B smaller by deleting 30% of the hidden units
("channels") of every MLP layer, the ones our coding agent hardly uses; served in 4 bits it generates text about 15%
faster. The first healing run taught this cut model to imitate the full model on *every* token of the agent's
transcripts. That repaired most of the average loss, yet the cut agent still solved fewer tasks, and the reason was in
what it *writes*: about one tool call in five was malformed (a file path that swallowed the next argument), the agent
retried the broken call, and the session ran out of time. 93% of a transcript is text the agent only reads (tool
output, the task, the prompt), so the first run spent most of its effort there. This run puts all of it on the 7% the
agent writes itself: its reasoning and its tool calls.

## What the first run found

The first run (notebook "Gemma 4 31B Agent: Pruned Model Teacher Healing") took the cut's damage on the agent's tokens
from +0.085 to +0.021 nats per token and cut malformed tool calls from 21% to 4% of calls in real agent sessions, but
the healed agent won back only 2 of the 7 tasks the cut had lost. Learning from the text alone (no teacher) fit the
transcripts better than the full model itself while moving away from it."""),
    ("""Comparing a with b answers the question: does the teacher matter, or does any fine-tuning on agent text heal the
cut equally well?""",
     """Both arms learn only at the tokens the agent wrote. Comparing a with b asks again whether the teacher matters; comparing
this run with the first asks whether healing should look at what the agent writes rather than at everything it
reads."""),
    ("""per layer the 14,848
  highest-scoring of 21,504 channels (70%, rounded to a multiple of 512, which the 4-bit GPU kernels need).""",
     """per layer the 14,848
  highest-scoring of 21,504 channels (70%, rounded to a multiple of 512, which the 4-bit GPU kernels need): the same
  cut as the first run."""),
]:
    assert INTRO_HEAL70OWN.count(_old) == 1, _old[:60]
    INTRO_HEAL70OWN = INTRO_HEAL70OWN.replace(_old, _new)
VARIANTS['heal70own'] = {**VARIANTS['heal'], 'BUILT': ROOT / 'kernels/heal-c70own-tpu/heal-c70own-tpu.py',
                         'OUT': ROOT / 'kernels/g4dev-heal-c70own-tpu',
                         'SLUG': 'gemma-4-31b-agent-healing-where-the-agent-writes',
                         'TITLE': 'Gemma 4 31B Agent: Healing Where the Agent Writes', 'INTRO': INTRO_HEAL70OWN}
# 2026-11-07 (B192/B195): the long TPU run after the review: the 70% cut healed on its tool calls only (every turn).
INTRO_HEAL70CALLS = INTRO_HEAL
for _old, _new in [
    ("# Gemma 4 31B Agent: Healing a Pruned Model with Its Teacher", "# Gemma 4 31B Agent: Healing the Tool Calls"),
    ("""**In one paragraph.** We made Gemma 4 31B smaller by deleting 30% of the hidden units ("channels") of every MLP layer:
the ones our coding agent hardly uses. (We measured which channels are active while the model does agent work and kept
the busiest 70%.) Served in 4 bits on four L4 GPUs, the cut model generates text about 15% faster, but it predicts the
agent's own tokens a little worse: +0.08 nats per token. This notebook tries to win that quality back cheaply by
letting the original, uncut model act as a **teacher**: the cut model learns to imitate the full model's predictions on
agent transcripts. Only a small LoRA adapter (rank 16) is trained; the model's weights stay frozen.""",
     """**In one paragraph.** This is the third healing run. We made Gemma 4 31B smaller by deleting 30% of the hidden units
("channels") of every MLP layer, the ones our coding agent hardly uses; served in 4 bits it generates text about 15%
faster. The first healing run taught this cut model to imitate the full model on *every* token of the agent's
transcripts. That repaired most of the average loss, yet the healed agent solved no more tasks than the cut one. What
cost the tasks was in the tool calls: about one call in five had a broken text argument (a file path that swallowed the
next argument), the agent sent the broken call again, and the session ran out of time. Tool calls are a small part of a
transcript (most of it is text the agent only reads: tool output, the task, the prompt), so the first run spent most of
its effort elsewhere. This run trains only on the tool calls, in every turn the agent took.

## What the first run found

The first run (notebook "Gemma 4 31B Agent: Pruned Model Teacher Healing") took the cut's damage on the agent's tokens
from +0.085 to +0.021 nats per token and its broken tool calls from 19% to 4% of calls in real agent sessions (the full
model's own rate is 1-4%), but the healed agent solved about as many tasks as the unhealed cut (15 vs 14 of 71).
Learning from the text alone (no teacher) fit the transcripts better than the full model itself while moving away
from it."""),
    ("""Comparing a with b answers the question: does the teacher matter, or does any fine-tuning on agent text heal the
cut equally well?""",
     """Both arms learn only at the tokens of the agent's tool calls (a call's name and arguments, through the marker that
ends it), so neither arm is trained on the agent's reasoning: training on reasoning changes how long the agent thinks,
which is a different effect. Comparing a with b asks again whether the teacher matters; comparing this run with the
first asks whether healing should focus on the calls. Each arm trains four epochs and is saved and evaluated after every
epoch: epoch 2 is the like-for-like comparison with the first run (two epochs), epoch 4 shows what more training buys."""),
    ("""per layer the 14,848
  highest-scoring of 21,504 channels (70%, rounded to a multiple of 512, which the 4-bit GPU kernels need).""",
     """per layer the 14,848
  highest-scoring of 21,504 channels (70%, rounded to a multiple of 512, which the 4-bit GPU kernels need): the same
  cut as the first run."""),
    ("""KL(full || cut) per token, again on the agent's own tokens and on all tokens.""",
     """KL(full || cut) per token, again on the agent's own tokens, on all tokens and on the tool-call tokens (calls) this run
trains."""),
]:
    assert INTRO_HEAL70CALLS.count(_old) == 1, _old[:60]
    INTRO_HEAL70CALLS = INTRO_HEAL70CALLS.replace(_old, _new)
SECTIONS_HEAL70CALLS = dict(SECTIONS_HEAL)
for _key, _old, _new in [
    ('settings', """distillation temperature; 1 = the teacher's distribution as it is), `positions` (`'all'`: every token of a transcript
is a training target)""",
     """distillation temperature; 1 = the teacher's distribution as it is), `positions` (`'calls'`: only the tool-call tokens
of every turn are training targets; rows without a call are dropped)"""),
    ('modules', """* `heal_data.py`: the training rows (every token a target), the 40 held-out rows and the channel mask.""",
     """* `heal_data.py`: the training rows (tool-call tokens as targets), the 40 held-out rows (the same as the first run's,
  with a tool-call mask) and the channel mask."""),
]:
    assert SECTIONS_HEAL70CALLS[_key].count(_old) == 1, (_key, _old[:60])
    SECTIONS_HEAL70CALLS[_key] = SECTIONS_HEAL70CALLS[_key].replace(_old, _new)
VARIANTS['heal70calls'] = {**VARIANTS['heal'], 'BUILT': ROOT / 'kernels/heal-c70calls-tpu/heal-c70calls-tpu.py',
                           'OUT': ROOT / 'kernels/g4dev-heal-c70calls-tpu',
                           'SLUG': 'gemma-4-31b-agent-healing-the-tool-calls',
                           'TITLE': 'Gemma 4 31B Agent: Healing the Tool Calls', 'INTRO': INTRO_HEAL70CALLS,
                           'SECTIONS': SECTIONS_HEAL70CALLS}
# 2026-11-09 (B207): before the push, the calls now include the `<|tool_call>` opener (serve-v6 showed cut models
# dropping it), and the first run's task result is brought up to date (same4, B197).
for _old, _new in [
    ("""but the healed agent solved about as many tasks as the unhealed cut (15 vs 14 of 71).""",
     """but served through vLLM's adapter path, which costs the whole speedup, the healed agent solved about as many tasks as
the unhealed cut (15 vs 14 of 71). Merged into the 4-bit weights and compared in one session on 112 tasks, it later
solved 32 against 20 for the unhealed cut and 32-34 for two runs of the full model."""),
    ("""(a call's name and arguments, through the marker that
ends it)""",
     """(the marker that opens a call, its name and
arguments, through the marker that ends it)"""),
]:
    assert INTRO_HEAL70CALLS.count(_old) == 1, _old[:60]
    INTRO_HEAL70CALLS = INTRO_HEAL70CALLS.replace(_old, _new)
VARIANTS['heal70calls']['INTRO'] = INTRO_HEAL70CALLS
# 2026-11-09 (B207): the same tool-call healing on the deeper cut (55% of the channels kept, as heal-c55), where healing
# every token left 13% of the calls broken (B200). Same notebook text as heal70calls except the opening, what the
# earlier runs found, the run compared with, and the kept count.
INTRO_HEAL55CALLS = INTRO_HEAL70CALLS
_i = INTRO_HEAL55CALLS.index('**In one paragraph.**')
_j = INTRO_HEAL55CALLS.index('from it.', _i) + len('from it.')
INTRO_HEAL55CALLS = (INTRO_HEAL55CALLS[:_i] + """**In one paragraph.** This is the fourth healing run. We made Gemma 4 31B smaller by deleting 45% of the hidden
units ("channels") of every MLP layer, the ones our coding agent hardly uses; served in 4 bits it generates text about
24% faster. Cut this deep, the agent broke 43% of its tool calls (the full model: 1-2%): a text argument closed with the
wrong quote and swallowed the next argument, or the call lost the marker that opens it. The second healing run taught
the cut model to imitate the full model on *every* token of the agent's transcripts. It won back 14 of the 22 tasks the
cut had lost, but 13% of its calls stayed broken. Tool calls are a small part of a transcript (most of it is text the
agent only reads: tool output, the task, the prompt), so that run spent most of its effort elsewhere. This run trains
only on the tool calls, in every turn the agent took, from the marker that opens a call to the marker that ends it.

## What the earlier runs found

On the milder cut (70% of the channels kept), healing on every token, merged into the 4-bit weights, brought the
broken calls back to the full model's rate and solved 32 of 112 tasks against 20 for the unhealed cut and 32-34 for
the full model. On this deeper cut the same recipe took the damage on the agent's tokens from +0.185 to +0.047 nats per
token (74% of it) and the broken calls from 43% to 13%; the healed agent solved 27 tasks against 13 for the unhealed
cut and 37 and 33 for two runs of the full model. A later test showed that choosing the channels differently only
moves the breakage around the call (fewer wrong quotes, more missing openers); training is what can repair it.""" +
                     INTRO_HEAL55CALLS[_j:])
for _old, _new in [
    ("# Gemma 4 31B Agent: Healing the Tool Calls", "# Gemma 4 31B Agent: Deeper Cut Tool Call Healing"),
    ("""comparing this run with the
first asks whether healing should focus on the calls.""",
     """comparing this run with the
deeper-cut run (every token) asks whether healing should focus on the calls."""),
    ("""epoch 2 is the like-for-like comparison with the first run (two epochs)""",
     """epoch 2 is the like-for-like comparison with the deeper-cut run (two epochs)"""),
    ("""per layer the 14,848
  highest-scoring of 21,504 channels (70%, rounded to a multiple of 512, which the 4-bit GPU kernels need): the same
  cut as the first run.""",
     """per layer the 11,776
  highest-scoring of 21,504 channels (55%, rounded to a multiple of 512, which the 4-bit GPU kernels need): the same
  cut as the deeper-cut run."""),
]:
    assert INTRO_HEAL55CALLS.count(_old) == 1, _old[:60]
    INTRO_HEAL55CALLS = INTRO_HEAL55CALLS.replace(_old, _new)
VARIANTS['heal55calls'] = {**VARIANTS['heal70calls'], 'BUILT': ROOT / 'kernels/heal-c55calls-tpu/heal-c55calls-tpu.py',
                           'OUT': ROOT / 'kernels/g4dev-heal-c55calls-tpu',
                           'SLUG': 'gemma-4-31b-agent-deeper-cut-tool-call-healing',
                           'TITLE': 'Gemma 4 31B Agent: Deeper Cut Tool Call Healing', 'INTRO': INTRO_HEAL55CALLS}
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'g9'
globals().update(VARIANTS[VARIANT])

def cell(kind, text):
    c = {'cell_type': kind, 'metadata': {}, 'source': text.strip('\n').splitlines(True)}
    if kind == 'code':
        c.update({'execution_count': None, 'outputs': []})
    return c


def scrub(text, key):
    for old, new in SCRUB.get(key, []):
        assert text.count(old) == 1, (key, old)
        text = text.replace(old, new)
    return text


def build():
    src = BUILT.read_text()
    tree = ast.parse(src)
    body = src[ast.get_source_segment(src, tree.body[0]).__len__():].lstrip('\n')  # drop the module docstring
    lines = body.splitlines(True)
    embeds = {}
    for var, _ in MODULES:
        i = next(k for k, x in enumerate(lines) if x.startswith(f'{var} = '))
        embeds[var] = ast.literal_eval(lines[i].split(' = ', 1)[1])
        lines[i] = ''
    body = scrub(''.join(lines), 'driver')
    head, rest = body.split('\nT0 = time.time()\n', 1)
    rest = 'T0 = time.time()\n' + rest
    marks = ['# ---- 0. environment', '# ---- 1. inputs', '# ---- 1b. parallel read-ahead', '# ---- 2. tokens',
             '# ---- 3. train', '\n\ndef median(']
    parts, cur = [], rest
    for m in marks:
        a, b = cur.split(m, 1)
        parts.append(a)
        cur = m.lstrip('\n') + b
    parts.append(cur)
    setup, env, inputs, prefetch, dump, train, report = parts
    cells = [cell('markdown', INTRO), cell('markdown', SECTIONS['settings']), cell('code', head),
             cell('markdown', SECTIONS['setup']), cell('code', setup), cell('markdown', SECTIONS['modules'])]
    for var, name in MODULES:
        text = scrub(embeds[var], name)
        assert not text.startswith('\n')
        cells.append(cell('code', f'%%writefile /tmp/g8/{name}\n' + text))
    cells.append(cell('code', '\n'.join(f"{var} = (WORK / '{name}').read_text()" for var, name in MODULES)))
    for key, code in (('env', env), ('inputs', inputs), ('prefetch', prefetch), ('dump', dump), ('train', train),
                      ('report', report)):
        cells += [cell('markdown', SECTIONS[key]), cell('code', code)]
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:02d}'
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{SLUG}.ipynb').write_text(json.dumps(nb, indent=1))
    meta = json.loads((BUILT.parent / 'kernel-metadata.json').read_text())
    meta.update({'id': f'verracodeguacas/{SLUG}', 'title': TITLE, 'code_file': f'{SLUG}.ipynb', 'kernel_type': 'notebook',
                 'is_private': True})
    (OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    return nb, embeds, src


def check(nb, embeds, src):
    """The notebook's code, with each %%writefile cell turned back into its embedded string, must equal the built
    script's code apart from the docstring, the scrubbed comments and the order of the embed assignments."""
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    files = {}
    for c in code:
        if c.startswith('%%writefile '):
            first, text = c.split('\n', 1)
            files[Path(first.split()[1]).name] = text + '\n' if not text.endswith('\n') else text
    for var, name in MODULES:
        orig = (ROOT / 'kernels/train' / name).read_text()
        assert embeds[var] == orig, f'{name}: built embed differs from kernels/train'
        diff = [(a, b) for a, b in zip(orig.splitlines(), files[name].splitlines()) if a != b]
        assert len(orig.splitlines()) == len(files[name].splitlines()) and len(diff) == len(SCRUB.get(name, [])), name
        for a, b in diff:  # the only differing lines are comments / help strings
            assert re.sub(r'\s+', '', a) != re.sub(r'\s+', '', b)
    drv = '\n'.join(c for c in code if not c.startswith('%%writefile ') and not c.startswith('LORA_TPU = (WORK'))
    ref = scrub(src, 'driver')
    ast.parse(drv)
    mods = '|'.join(v for v, _ in MODULES)
    norm = lambda s: [x for x in s.splitlines() if x.strip() and not re.match(rf'^({mods}) = ', x)]  # noqa: E731
    ref_lines = norm(ref[len(ast.get_source_segment(ref, ast.parse(ref).body[0])):])
    assert norm(drv) == ref_lines, 'driver code differs from the built script'
    return {'cells': len(nb['cells']), 'code_cells': len(code), 'driver_lines': len(ref_lines),
            'modules': {n: len(files[n].splitlines()) for _, n in MODULES}}


if __name__ == '__main__':
    nb, embeds, src = build()
    print(check(nb, embeds, src))
