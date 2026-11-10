"""Which tokens would recipe v2 train? (CPU, no internet; the tokenizer only, no model weights are loaded)

Runs our trainer (`train_lora.py`, embedded unchanged) with `--labels-only` on the mix7 training lines at the 12k cap:
- `actions` + `--skip-kinds summary` (recipe v2): every trainable run must be one tool call (from `call:` through
  `<|tool_response>`), nothing inside a thought may be trainable, and lines without a tool call are dropped
  (they would have crashed training: an int loss has no backward).
- `all` without skipping (G2b's recipe on mix7) for comparison: how many trainable tokens action-only keeps.
Decision: whether the masking is right before the GPU validation run (B35, lora_plan §8 Q2).
Outputs: /kaggle/working/labels/{actions,all}/labels_audit.json and train_log.jsonl.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

TRAIN_LORA_PY = None  # @embed ../train/train_lora.py
MAX_TOKENS = 12288  # @define
MASK_MISTAKES = True  # @define (actions mode: also mask calls the next thought calls a mistake)
DATA_GLOB = '/kaggle/input/**/g4dev-train-mix7-cpu/mix/train.jsonl'  # @define
OUT = Path('/kaggle/working/labels')
OUT.mkdir(parents=True, exist_ok=True)
script = Path('/kaggle/working/train_lora.py')
script.write_text(TRAIN_LORA_PY)

data = sorted(glob.glob(DATA_GLOB, recursive=True))[0]
tok_dir = None
for c in sorted(glob.glob('/kaggle/input/**/tokenizer.json', recursive=True)):
    if 'w4a16' in c or tok_dir is None:
        tok_dir = str(Path(c).parent)
print('data', data, '| tokenizer', tok_dir, flush=True)
res = {}
for mode, skip in (('actions', 'summary'), ('all', '')):
    r = subprocess.run([sys.executable, str(script), '--model', tok_dir, '--data', data, '--out', str(OUT / mode),
                        '--labels-only', '--train-on', mode, '--max-tokens', str(MAX_TOKENS),
                        *(['--skip-kinds', skip] if skip else []), *(['--mask-mistakes'] if MASK_MISTAKES and mode == 'actions' else [])], capture_output=True, text=True)
    print(mode, 'rc', r.returncode, (r.stdout[-1500:] + r.stderr[-3000:]), flush=True)
    f = OUT / mode / 'labels_audit.json'
    res[mode] = json.loads(f.read_text())['summary'] if f.exists() else {'rc': r.returncode, 'stderr': r.stderr[-3000:]}
(OUT / 'summary.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
