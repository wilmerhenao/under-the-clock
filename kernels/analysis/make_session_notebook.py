#!/usr/bin/env python3
"""One TPU session, two experiments back to back (user 10-30: "run them back to back right now because this takes
forever"; B157): part 1 trains G17 (teacher-guided step preferences, kernels/train-g17-tpu), part 2 frees the TPU,
part 3 evaluates G17, G11a and G16a on the same unseen-repo tasks (kernels/tpu-eval-g17). Both drivers are embedded
unchanged and run as subprocesses; this notebook only sets their deadlines from the session clock, drops an eval arm
whose adapter is missing, and summarises the two reports.
usage: python3 kernels/analysis/build_one.py train-g17-tpu tpu-eval-g17 && python3 kernels/analysis/make_session_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = 'gemma-4-31b-agent-teacher-guided-step-preferences'
TITLE = 'Gemma 4 31B Agent: Teacher-Guided Step Preferences'
OUTDIR = ROOT / 'kernels/g4dev-session-g17-tpu'
TRAIN = ROOT / 'kernels/train-g17-tpu/train-g17-tpu.py'
EVAL = ROOT / 'kernels/tpu-eval-g17/tpu-eval-g17.py'
STATS = ROOT / 'runs/g17-data/stats.json'


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(keepends=True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': s.strip('\n').splitlines(keepends=True)}


def intro(st):
    pairs = st.get('pairs', 'N')
    by = ', '.join(f"{st.get('pairs_' + k, 0)} {k}" for k in ('slow', 'wasteful', 'wrong'))
    return f"""
# {TITLE}

This notebook runs one experiment of our paper track from start to finish in a single Kaggle **TPU v5e-8** session.
It **trains** a LoRA adapter for **Gemma 4 31B** from step-level preferences labelled by a stronger model, frees the
TPU, and **evaluates** the new adapter against two earlier ones on coding tasks from repositories none of them has
seen.

## The idea

Gemma 4 31B solves many repository-level coding tasks, but it loses most of the others to the clock. Every task in the
Gemma 4 Developer Agent competition gets 8 minutes, and one model call costs about **17 s plus 0.14 s per generated
token**. A 2,000-token deliberation costs almost five minutes. In our timing of 500 sessions, none of the 249 that ran
out of time solved its task.

A stronger model can tell which of Gemma's steps waste time and what to do instead. The obvious way to use it is to
fine-tune on the stronger model's own trajectories, and that failed for us. An adapter trained mostly on public teacher
trajectories solved 16 tasks against the base model's 26. It imitated the teachers' one-line, narration-style
"thoughts", lost track of what it had already seen, and looped. Two things went wrong:

* the teacher's trajectories go through states that Gemma never reaches;
* its way of reasoning is not Gemma's.

So here **the teacher never drives**:

1. **The states are Gemma's.** Decision points come from Gemma's own agent runs, mostly runs that were slow or failed.
   Each state is the exact context Gemma saw, compaction summaries included.
2. **The teacher labels one step.** Gemini (3.8 Flash, through Kaggle's model proxy) reads the transcript and Gemma's
   actual next step. It rates that step *good*, *slow* (right call, far too much deliberation), *wasteful*
   (re-reading, broad exploration, throwaway scripts, full test suites) or *wrong*. Unless the step is good, it writes
   the step it would take instead: a short first-person plan and one tool call.
3. **A preference at that step.** Each step rated slow, wasteful or wrong gives two training rows with the same context
   and different final steps. The *chosen* row ends with the teacher's step, rendered in Gemma's own format. The
   *rejected* row ends with Gemma's step. Only that final step is trained, plan included, because the decision lives
   in the plan.

This run uses **{pairs} pairs** ({by}). The *slow* pairs keep the same tool call and only shorten the plan: they teach
Gemma to stop deliberating once the next move is clear. That is our "prefer shorter runs" objective applied to a single
step.

Nothing is executed to check a teacher step. A step is dropped if:

* the tool or its arguments are unknown;
* it edits text that Gemma had never seen verbatim (in a file read, an earlier edit or a command's output);
* it edits a test file;
* it runs a bare `pytest`;
* its plan falls outside 80 to 1,200 characters.

## Plan of the session (8.5 hours)

| Part | What | Time box |
|---|---|---|
| 1 | Train G17 on the step pairs (DPO + SFT on the chosen step) | about 3.5 h |
| 2 | Free the TPU (the trainer exits; lock files and scratch removed) | minutes |
| 3 | Evaluate G17, G11a and G16a on the same 40 unseen-repo tasks | the rest |

**Arms of the evaluation:**

* **G17**, this notebook's adapter.
* **G11a**: supervised fine-tuning on Gemma's own solved runs, restricted to steps whose reasoning is short. It is our
  best adapter so far.
* **G16a**: run-level preferences (pass over fail, fast over slow) with clock-calibrated IPO targets.

Run it with **Run All** on a TPU v5e-8 with Internet on.
"""


PART1 = """
## Part 1: training (G17)

The trainer is the one from our *Clock-Calibrated Preferences* notebook, embedded unchanged. It uses:

* rank-16 LoRA on the bf16 weights of the QAT checkpoint;
* JAX, [Tunix](https://github.com/google/tunix)'s Gemma 4 model and [Qwix](https://github.com/google/qwix) LoRA;
* FSDP 2 x tensor parallel 4 with splash attention;
* learning rate 5e-5 and two epochs.

Preferences are trained as **signed per-token weights** inside the ordinary cross-entropy loop:

* The reference model is the LoRA with B = 0, i.e. the base model. Its log-probabilities are computed once.
* At the start of every epoch each pair's margin `m` is recomputed. `m` is the difference between the chosen and the
  rejected step of the mean log-ratio to the reference.
* Each pair then gets the DPO weight `a = beta * sigmoid(-beta * m)` with beta = 2.
* The chosen step is trained with weight `1 + a` (SFT plus preference). The rejected step gets `-a` until its
  log-ratio falls below -0.1.

The rows carry `train_on: all`, so the plan is trained together with the call. In our earlier adapters only the call
was trained.

The cell writes the driver to `/kaggle/working/session/train.py` and runs it with this session's time box. Its log is
`/kaggle/working/tpu/log.jsonl`. The adapter is written to `/kaggle/working/adapter_out/`, with per-epoch copies in
`/kaggle/working/adapters/ep<N>/`.
"""

PART2 = """
## Part 2: free the TPU

The trainer runs in its own process, so its TPU memory is released when it exits. This cell only removes what could
block the next process: the libtpu lock file, leftover processes and the trainer's scratch directory.
"""

PART3 = """
## Part 3: evaluation

[vllm-tpu](https://github.com/vllm-project/tpu-inference) serves Gemma 4 31B (bf16, tensor parallel 8, the
competition's gemma4 tool and reasoning parsers, thinking on), with each arm's LoRA merged into the weights.

The agent side is the competition's own:

* the official harness, tools and sub-v9c prompt;
* an 8-minute, 24-call budget, with time scaled x2 for the shared server;
* 8 tasks at a time.

The tasks are the same 40 for every arm: verified tasks from repositories outside every training set
(`debmalya84/repograph-verified-swe-tasks`), checked by gold and empty-patch controls. Per arm the report gives tasks
solved, timeouts, wall time, tool calls and a per-call latency fit. Arms run in order (G17, G11a, G16a), so if time runs
short, the last arm is the one cut.
"""

SETTINGS = """
import glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

T0 = time.time()
SESSION_DEADLINE_S = int(8.5 * 3600)  # the whole notebook (Kaggle stops a TPU session at 9 h)
TRAIN_MIN = 215                         # part 1 time box, minutes (its training cap is 205)
REPORT_RESERVE_MIN = 8                  # kept at the end for part 4
WORK = Path('/kaggle/working/session')
WORK.mkdir(parents=True, exist_ok=True)

# Quota guard for this push: Kaggle stops a session when the weekly TPU quota runs out, and only 4.35 h were left
# before the weekly refresh when this version was queued. If the session starts so early that it would run out before
# the refresh, the notebook shrinks to the quota and spends it on training (the adapter is the expensive part; the
# evaluation can be rerun). After the refresh the full plan runs.
import datetime
QUOTA_H_LEFT_AT_PUSH, QUOTA_REFRESH_UTC = 4.35, datetime.datetime(2026, 10, 31, 0, 0)
_to_refresh_s = (QUOTA_REFRESH_UTC - datetime.datetime.utcnow()).total_seconds()
if _to_refresh_s > QUOTA_H_LEFT_AT_PUSH * 3600 - 900:
    SESSION_DEADLINE_S = int(QUOTA_H_LEFT_AT_PUSH * 3600 - 900)
    print(f'quota guard: {_to_refresh_s / 3600:.1f} h to the refresh, session shortened to {SESSION_DEADLINE_S / 3600:.2f} h')
EVAL_RESERVE_S = 3 * 3600 if SESSION_DEADLINE_S >= 7 * 3600 else 600  # time kept for part 3 when part 1 starts


def left_s():
    return SESSION_DEADLINE_S - (time.time() - T0)


def set_define(src, var, value):
    lines = src.splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if l.startswith(var + ' = '))
    lines[i] = f'{var} = {value!r}\\n'
    return ''.join(lines)


def run_part(name, src, deadline_s):
    path = WORK / f'{name}.py'
    path.write_text(set_define(src, 'DEADLINE_S', int(deadline_s)))
    t = time.time()
    with open(WORK / f'{name}.log', 'w') as log:
        try:
            rc = subprocess.run([sys.executable, str(path)], stdout=log, stderr=subprocess.STDOUT,
                                timeout=deadline_s + 600).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    tail = (WORK / f'{name}.log').read_text(errors='replace').splitlines()[-15:]
    print(f'{name}: rc {rc}, {(time.time() - t) / 60:.1f} min, {left_s() / 60:.0f} min left in the session')
    print('\\n'.join(tail))
    return rc
"""

RUN1 = """
TRAIN_PY = {train!r}

train_rc = run_part('train', TRAIN_PY, min(TRAIN_MIN * 60, left_s() - EVAL_RESERVE_S))
"""

RUN2 = """
for pat in ('train_tpu_runner', 'vllm'):
    subprocess.run(['pkill', '-f', pat], capture_output=True)
for f in glob.glob('/tmp/libtpu_lockfile*'):
    os.remove(f)
shutil.rmtree('/tmp/g8', ignore_errors=True)  # the trainer's scratch (token dump, venv); adapters stay in /kaggle/working
time.sleep(20)
print('adapter written:', os.path.exists('/kaggle/working/adapter_out/adapter_model.safetensors'),
      '| disk free GB:', round(shutil.disk_usage('/tmp').free / 2**30), '| session min left:', round(left_s() / 60))
"""

RUN3 = """
EVAL_PY = {eval!r}

arms = {arms!r}
if not os.path.exists('/kaggle/working/adapter_out/adapter_model.safetensors'):
    arms = [a for a in arms if not a['adapter'].startswith('/kaggle/working')]  # part 1 left no adapter: skip its arm
    print('no G17 adapter: evaluating', [a['name'] for a in arms])
eval_rc = run_part('eval', set_define(EVAL_PY, 'ARMS', arms), left_s() - REPORT_RESERVE_MIN * 60 - 300)
"""

PART4 = """
## Part 4: report

The training lines to read are:

* `dpo_refresh`: one per epoch, with margins and the share of pairs ordered correctly;
* `done`: steps and stop reason.

For the evaluation, read each arm's tasks solved, timeouts and minutes. Full logs are in `/kaggle/working/tpu/log.jsonl`
and `/kaggle/working/tpu_eval/report.json`.
"""

REPORT = """
def lines(path, stages):
    return [json.loads(l) for l in open(path) if l.strip() and json.loads(l).get('stage') in stages] if os.path.exists(path) else []

for x in lines('/kaggle/working/tpu/log.jsonl', {'dpo', 'plan', 'done', 'dpo_refresh', 'fatal', 'runner_exit'}):
    keep = ('stage', 't_min', 'phase', 'pairs', 'steps', 'stop', 'steps_ok', 'min', 'tag', 'margin_p10_50_90', 'acc',
            'by_kind', 'weight_mean', 'rejected_frozen', 'error', 'rc')
    print(json.dumps({k: x[k] for k in keep if k in x})[:600])
rep = json.load(open('/kaggle/working/tpu_eval/report.json')) if os.path.exists('/kaggle/working/tpu_eval/report.json') else {}
for name, r in (rep.get('arms') or {}).items():
    tasks = r.get('tasks') or []
    timeouts = sum(1 for t in tasks if 'timeout' in str(t.get('error') or '').lower())
    print(f"{name}: solved {r.get('n_resolved')} of {r.get('n_tasks')}, finished {r.get('n_finished')}, timeouts {timeouts}, "
          f"wall {r.get('wall_min')} min, latency {r.get('latency')}")
print('eval stages:', [s.get('stage') for s in rep.get('stages', [])][-12:])
"""


def main():
    train_src, eval_src = TRAIN.read_text(), EVAL.read_text()
    arms = eval(next(l for l in eval_src.splitlines() if l.startswith('ARMS = '))[len('ARMS = '):])  # noqa: S307
    st = json.loads(STATS.read_text()) if STATS.exists() else {}
    cells = [md(intro(st)), code(SETTINGS), md(PART1), code(RUN1.format(train=train_src)), md(PART2), code(RUN2),
             md(PART3), code(RUN3.format(eval=eval_src, arms=arms)), md(PART4), code(REPORT)]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f'{SLUG}.ipynb').write_text(json.dumps(nb, indent=1))
    tm = json.loads((TRAIN.parent / 'kernel-metadata.json').read_text())
    em = json.loads((EVAL.parent / 'kernel-metadata.json').read_text())
    meta = {'id': f'verracodeguacas/{SLUG}', 'title': TITLE, 'code_file': f'{SLUG}.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': True, 'enable_internet': True,
            'machine_shape': 'TpuV5E8',
            'dataset_sources': sorted(set(tm['dataset_sources']) | set(em['dataset_sources'])),
            'competition_sources': ['gemma-4-developer-agent'],
            'kernel_sources': sorted(set(tm['kernel_sources']) | set(em['kernel_sources'])),
            'model_sources': sorted(set(tm['model_sources']) | set(em['model_sources'])),
            'docker_image': tm['docker_image']}
    assert tm['docker_image'] == em['docker_image'], (tm['docker_image'], em['docker_image'])
    (OUTDIR / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print({'cells': len(cells), 'chars': sum(len(''.join(c['source'])) for c in cells), 'arms': [a['name'] for a in arms],
           'datasets': meta['dataset_sources'], 'kernels': meta['kernel_sources']})


if __name__ == '__main__':
    main()
