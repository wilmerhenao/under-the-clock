#!/usr/bin/env python3
"""Kaggle script kernel of pruner part 2 (kernels/prune-serve-gpu/): cut the competition's 4-bit model by part 1's
scores and measure vLLM speed + quality on 4 x L4 (private, no internet). The cutter and the per-model benchmark are
embedded as strings. usage: python3 kernels/colab/prune/build_serve.py [--v4]
--v4 (serve-v2, kernels/prune-serve2-gpu/): the healed cut models of part 3 (c70 + the teacher adapter, c70 + the
no-teacher adapter, both cut from the healing notebook's full-size adapters; the plain c70 again as the reference in
the same session), removing the 6 layers part 1 measured as least harmful (d6m), and width + depth together (c70d6m).
Mounts score-v3 (same channel scores as score-v1, plus the measured layer order) and the healing notebook."""
import sys
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'kernels/colab/prune'
src = (HERE / 'prune_serve.py').read_text()
V4B = '--v4b' in sys.argv    # serve-v2b (kernels/prune-serve2b-gpu/): serve-v2's two LoRA models again, after its fix
# serve-v5 (kernels/prune-serve5-gpu/, B192): the next-action test at 4 states per held-out example (~160), malformed
# calls, the full model sampled at T 0.2 (its own agreement ceiling), all-token loss; cuts: c70 again, c70 + healing
# adapter a merged into the 4-bit weights (no LoRA), c70 chosen on what the agent writes (agent_own_ff), and c85 / c55
# (how the behavioural damage grows with the cut)
V5 = '--v5' in sys.argv
# serve-v6 (kernels/prune-serve6-gpu/, B202): the role-aware score's next-action test: c55 (agent_ff, exactly serve-v5's
# and same55's) vs c55role (role_ff from score-v4), the full model as the reference; ~200 states (80 held-out examples x
# 4), quality on serve-v5's 40 rows with the loss per role of the token
V6 = '--v6' in sys.argv
# serve-v7 (kernels/prune-serve7-gpu/, B202 Run 2'): the speed tool that keeps behaviour: the full model, then the same
# weights with n-gram speculative decoding as 'full_b' (FULL_REPLICA); next calls at serve-v5's ~109 states, decode
# speed on 16 real agent turns at 1 and 8 streams, no quality pass (same weights)
V7 = '--v7' in sys.argv
# serve-v8 (kernels/prune-serve8-gpu/, B207): the last behaviour bench: the deeper cut healed on every token (heal-c55,
# as same55) against the same cut healed on its tool calls (B207 Run A: arm a, and arm b without the teacher), all
# merged into the 4-bit weights; serve-v6's 141 states and per-role logprob pass; the unhealed c55 is serve-v6's
V8 = '--v8' in sys.argv
HEAL55_NB = 'gemma-4-31b-agent-healing-a-deeper-pruning-cut'
CALLS55_NB = 'gemma-4-31b-agent-deeper-cut-tool-call-healing'
V4 = '--v4' in sys.argv or V4B or V5 or V6 or V7 or V8
HEAL_NB = 'gemma-4-31b-agent-pruned-model-teacher-healing'
if V4:
    cuts_v4 = f"""CUTS = [{{'name': 'c70', 'keep': 0.70, 'drop': 0, 'group': 1}},
        {{'name': 'c70+kd', 'keep': 0.70, 'drop': 0, 'group': 1, 'adapter': '**/{HEAL_NB}/adapter_out'}},
        {{'name': 'c70+lm', 'keep': 0.70, 'drop': 0, 'group': 1, 'adapter': '**/{HEAL_NB}/adapter_out_b'}},
        {{'name': 'd6m', 'keep': 1.0, 'drop': 6, 'group': 32, 'order': 'measured'}},
        {{'name': 'c70d6m', 'keep': 0.70, 'drop': 6, 'group': 1, 'order': 'measured'}}]"""
    if V5:
        cuts_v4 = f"""CUTS = [{{'name': 'c70', 'keep': 0.70, 'drop': 0, 'group': 1}},
        {{'name': 'c70+kd~', 'keep': 0.70, 'drop': 0, 'group': 1, 'merge': '**/{HEAL_NB}/adapter_out'}},
        {{'name': 'c70own', 'keep': 0.70, 'drop': 0, 'group': 1, 'score': 'agent_own_ff'}},
        {{'name': 'c85', 'keep': 0.85, 'drop': 0, 'group': 1}},
        {{'name': 'c55', 'keep': 0.55, 'drop': 0, 'group': 1}}]"""
    if V6:
        cuts_v4 = """CUTS = [{'name': 'c55', 'keep': 0.55, 'drop': 0, 'group': 1},
        {'name': 'c55role', 'keep': 0.55, 'drop': 0, 'group': 1, 'score': 'role_ff'}]"""
    if V7:
        cuts_v4 = "CUTS = []"
    if V8:
        cuts_v4 = f"""CUTS = [{{'name': 'c55kd~', 'keep': 0.55, 'drop': 0, 'group': 1, 'merge': '**/{HEAL55_NB}/adapter_out'}},
        {{'name': 'c55calls~', 'keep': 0.55, 'drop': 0, 'group': 1, 'merge': '**/{CALLS55_NB}/adapter_out'}},
        {{'name': 'c55callsb~', 'keep': 0.55, 'drop': 0, 'group': 1, 'merge': '**/{CALLS55_NB}/adapter_out_b'}}]"""
    if V4B:  # serve-v2 measured full and c70; its two LoRA engines ran out of memory at start-up (B188)
        cuts_v4 = f"""CUTS = [{{'name': 'c70+kd', 'keep': 0.70, 'drop': 0, 'group': 1, 'adapter': '**/{HEAL_NB}/adapter_out'}},
        {{'name': 'c70+lm', 'keep': 0.70, 'drop': 0, 'group': 1, 'adapter': '**/{HEAL_NB}/adapter_out_b'}}]"""
    m = re.search(r"^CUTS = \[.*?\]\]?$", src, re.M | re.S)
    src = src[:m.start()] + cuts_v4 + src[m.end():]
    run = 'serve-v8' if V8 else 'serve-v7' if V7 else 'serve-v6' if V6 else 'serve-v5' if V5 else 'serve-v2b' if V4B else 'serve-v2'
    src = src.replace("RUN_NAME = 'serve-v1'  # v4: 'serve-v2' (set by the builder)", f"RUN_NAME = '{run}'")
    assert f"RUN_NAME = '{run}'" in src and ("'c55role'" in src if V6 else 'CUTS = []' in src if V7 else
                                             "'c55calls~'" in src if V8 else "'c70+kd" in src)
    if V5:
        # BENCH_TIMEOUT_MIN 60 (review, 11-07): the full model now writes 2 x ~160 turns, ~35 min by serve-v2b's 181 s
        # per 40 turns, and a full-model timeout ends the run; FULL_REPLICA: the in-session numerics floor (full_b)
        for old, new in (("TURNS_PER_ROW = 1 ", "TURNS_PER_ROW = 4 "), ("SAMPLE_TURNS_FULL = 0.0 ", "SAMPLE_TURNS_FULL = 0.2 "),
                         ("BENCH_TIMEOUT_MIN = 20 ", "BENCH_TIMEOUT_MIN = 60 "),
                         ("FULL_REPLICA = {} ", "FULL_REPLICA = {'PRUNE_MAX_BATCHED': 512, 'PRUNE_GPU_UTIL': 0.75} ")):
            assert src.count(old) == 1, old
            src = src.replace(old, new)
    if V7:
        spec = '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 4}'
        for old, new in (("TURNS_PER_ROW = 1 ", "TURNS_PER_ROW = 4 "), ("BENCH_TIMEOUT_MIN = 20 ", "BENCH_TIMEOUT_MIN = 40 "),
                         ("FULL_REPLICA = {} ", f"FULL_REPLICA = {{'PRUNE_SPEC': {spec!r}}} "),
                         ("BENCH_ENV = {} ", "BENCH_ENV = {'PRUNE_SKIP_QUALITY': '1', 'PRUNE_TURN_SPEED': '16'} ")):
            assert src.count(old) == 1, old
            src = src.replace(old, new)
    if V6 or V8:
        for old, new in (("TURNS_PER_ROW = 1 ", "TURNS_PER_ROW = 4 "), ("BENCH_TIMEOUT_MIN = 20 ", "BENCH_TIMEOUT_MIN = 60 "),
                         ("HELDOUT_ROWS = 40 ", "HELDOUT_ROWS = 80 "), ("QUALITY_ROWS = 0 ", "QUALITY_ROWS = 40 "),
                         ("QUALITY_BY_ROLE = False ", "QUALITY_BY_ROLE = True ")):
            assert src.count(old) == 1, old
            src = src.replace(old, new)
head = (f"SLICE_SRC = {(HERE / 'slice_model.py').read_text()!r}\n"
        f"BENCH_SRC = {(HERE / 'serve_bench.py').read_text()!r}\n"
        f"ROLES_SRC = {(HERE / 'token_roles.py').read_text()!r}\n\n")
gpu_image = re.search(r"^GPU_IMAGE = '([^']+)'", (ROOT / 'kernels/smoke/build.py').read_text(), re.M).group(1)
out = ROOT / ('kernels/prune-serve8-gpu' if V8 else 'kernels/prune-serve7-gpu' if V7 else 'kernels/prune-serve6-gpu' if V6 else 'kernels/prune-serve5-gpu' if V5 else 'kernels/prune-serve2b-gpu' if V4B else 'kernels/prune-serve2-gpu'
              if V4 else 'kernels/prune-serve-gpu')
out.mkdir(exist_ok=True)
(out / 'prune_serve.py').write_text(head + src)
slug = ('g4dev-prune-serve8-gpu' if V8 else 'g4dev-prune-serve7-gpu' if V7 else 'g4dev-prune-serve6-gpu' if V6 else 'g4dev-prune-serve5-gpu' if V5 else 'g4dev-prune-serve2b-gpu' if V4B else 'g4dev-prune-serve2-gpu' if V4
        else 'g4dev-prune-serve-gpu')
meta = {'id': f'verracodeguacas/{slug}', 'title': slug, 'code_file': 'prune_serve.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'machine_shape': 'NvidiaL4',
        'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse', 'verracodeguacas/g4dev-g11-data'],
        'competition_sources': ['gemma-4-developer-agent'],
        'kernel_sources': (['verracodeguacas/g4dev-prune-score3-gpu', f'verracodeguacas/{HEAL55_NB}',
                            f'verracodeguacas/{CALLS55_NB}'] if V8 else
                           ['verracodeguacas/g4dev-prune-score4-gpu'] if V6 else ['verracodeguacas/g4dev-prune-score3-gpu'] if V7 else
                           ['verracodeguacas/g4dev-prune-score3-gpu', f'verracodeguacas/{HEAL_NB}'] if V4
                           else ['verracodeguacas/g4dev-prune-score-gpu']),
        'model_sources': ['google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2'], 'docker_image': gpu_image}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
compile(head + src, 'prune_serve.py', 'exec')
print(out)
