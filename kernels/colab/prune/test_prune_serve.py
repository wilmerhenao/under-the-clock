#!/usr/bin/env python3
"""CPU test of the pruner part 2 kernel (kernels/prune-serve-gpu/prune_serve.py as built): a tiny random Gemma 4 in
the competition's 4-bit format (real tokenizer), fake part-1 scores, synthetic agent rows, and a small stand-in for
vLLM (same calls as serve_bench.py uses, computed with transformers). Checks that every cut is made, served and
measured, and that the quality numbers equal the hook-masked full model's.
usage: PRUNE_TOK_DIR=<dir with the real config.json + tokenizer> python3 kernels/colab/prune/test_prune_serve.py
PRUNE_TEST_V4=1: the --v4 build (serve-v2): healing adapters cut to c70 and served through the LoRA path (a zero
adapter must leave c70's numbers unchanged, a non-zero one must change them; the cut must match the training mask), the
c70 checkpoint reused by the adapter cuts, the measured layer order (d6m, exact: equals the hook-masked full model) and
width + depth (c70d6m).
PRUNE_TEST_V6=1: the --v6 build (serve-v6, B202): full, c55 (agent_ff) and c55role (role_ff); the quality pass on
QUALITY_ROWS rows with the loss per role of the token, which must equal the in-process full model's per-role loss.
Without PRUNE_TOK_DIR (or without runs/sft-convert-smoke-cpu/sft/sft_v1.jsonl) the tokenizer, config and rows are
synthetic (synth_agent.py).
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# PRUNE_TEST_ABORT=1: two cuts in a row hang (c70, c85): both are killed and the cuts after them are skipped
ABORT = os.environ.get('PRUNE_TEST_ABORT') == '1'
V4 = os.environ.get('PRUNE_TEST_V4') == '1'
V6 = os.environ.get('PRUNE_TEST_V6') == '1'
V7 = os.environ.get('PRUNE_TEST_V7') == '1'   # serve-v7: the full model, then with n-gram speculation as full_b
V8 = os.environ.get('PRUNE_TEST_V8') == '1'   # serve-v8: three healing adapters merged into the c55 cut
ROOT = HERE.parents[2]
tmp = Path(tempfile.mkdtemp())
REAL_ROWS = ROOT / 'runs/sft-convert-smoke-cpu/sft/sft_v1.jsonl'
SYNTH = not os.environ.get('PRUNE_TOK_DIR') or not REAL_ROWS.exists()
if SYNTH:
    sys.path.insert(0, str(HERE))
    import synth_agent  # noqa: E402
    SYNTH_ROWS = synth_agent.rows(60)
    TOK = synth_agent.tokenizer_dir(SYNTH_ROWS, tmp / 'tok')
else:
    TOK = Path(os.environ['PRUNE_TOK_DIR'])
subprocess.run([sys.executable, str(HERE / 'build_serve.py'), *(['--v4'] if V4 else []), *(['--v6'] if V6 else []),
                *(['--v7'] if V7 else []), *(['--v8'] if V8 else [])], check=True, capture_output=True)
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import Gemma4Config, Gemma4ForConditionalGeneration  # noqa: E402

# 1. tiny model (real tokenizer and vocabulary), 4-bit feed-forward weights in compressed-tensors format
c = json.load(open(TOK / 'config.json'))
c['text_config'].update(hidden_size=64, intermediate_size=128, num_hidden_layers=14, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, num_global_key_value_heads=1, global_head_dim=32,
                        layer_types=None)
c['vision_config'].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2, head_dim=16)
torch.manual_seed(0)
model = Gemma4ForConditionalGeneration(Gemma4Config(**{k: v for k, v in c.items()
                                                       if k not in ('architectures', 'transformers_version')}))
for lyr in model.model.language_model.layers:
    lyr.layer_scalar.fill_(0.5 + torch.rand(1).item())
model = model.float().eval()
w4 = tmp / 'w4'
model.save_pretrained(tmp / 'w16')
sd = {}
from safetensors.torch import load_file  # noqa: E402
for fn in sorted((tmp / 'w16').glob('*.safetensors')):
    sd.update(load_file(fn))
for k in [k for k in sd if '.mlp.' in k and 'language_model' in k]:
    w = sd.pop(k).float()
    gw = w.reshape(w.shape[0], -1, 32)
    sc = (gw.abs().amax(-1) / 7).clamp(min=1e-8)
    q = torch.round(gw / sc[..., None]).clamp(-8, 7)
    model.get_parameter(k).data.copy_((q * sc[..., None]).reshape(w.shape))   # the reference: same 4-bit values
    b = k[:-len('.weight')]
    sd[b + '.weight_packed'] = pack_to_int32(q.reshape(w.shape).to(torch.int8), 4)
    sd[b + '.weight_scale'] = sc.float()
    sd[b + '.weight_shape'] = torch.tensor(w.shape)
w4.mkdir()
save_file(sd, str(w4 / 'model.safetensors'), metadata={'format': 'pt'})
cq = json.load(open(tmp / 'w16/config.json'))
cq['quantization_config'] = {
    'quant_method': 'compressed-tensors', 'format': 'pack-quantized', 'quantization_status': 'compressed',
    'config_groups': {'group_0': {'format': 'pack-quantized', 'targets': ['Linear'], 'input_activations': None,
                                  'output_activations': None, 'weights': {
                                      'num_bits': 4, 'type': 'int', 'symmetric': True, 'strategy': 'group',
                                      'group_size': 32, 'dynamic': False, 'actorder': None, 'observer': 'minmax'}}},
    'ignore': ['lm_head', 're:.*vision.*', 're:.*self_attn.*', 're:.*embed_vision.*']}
json.dump(cq, open(w4 / 'config.json', 'w'))
for f in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    if (TOK / f).exists():
        shutil.copy(TOK / f, w4 / f)

# 2. fake part-1 outputs and synthetic agent rows (40 tasks: 1 in 10 is held out)
p1 = tmp / 'part1'
p1.mkdir()
rng = np.random.default_rng(0)
np.savez_compressed(p1 / 'scores.npz', agent_ff=rng.random((14, 128)), agent_bi=rng.random(14),
                    down_norm=rng.random((14, 128)) + 0.5, role_ff=rng.random((14, 128)))
ORDER_M = [7, 3, 10, 2, 5, 11, 8, 1, 4, 6, 9, 12]  # a measured layer order (part 1's layer sweep)
json.dump({'run': 'score-test', 'settings': {'MAX_TOKENS': 6144},
           'layer_sweep': {'order_least_damage_first': ORDER_M}}, open(p1 / 'report.json', 'w'))
src = SYNTH_ROWS if SYNTH else [json.loads(x) for x in open(REAL_ROWS)]
r0 = random.Random(0)
rows = []
for k in range(40):
    r = dict(r0.choice(src))
    r['instance_id'] = f'task_{k}'
    rows.append(r)
(tmp / 'sft.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

# 3. a stand-in for vLLM: the calls serve_bench.py makes, computed with transformers on CPU
fake = tmp / 'fakevllm' / 'vllm'
fake.mkdir(parents=True)
(fake / 'inputs.py').write_text('def TokensPrompt(prompt_token_ids):\n    return {"prompt_token_ids": prompt_token_ids}\n')
(fake / 'lora').mkdir()
(fake / 'lora' / '__init__.py').write_text('')
(fake / 'lora' / 'request.py').write_text('''class LoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path):
        self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path
''')
(fake / '__init__.py').write_text('''
import torch
from types import SimpleNamespace as NS
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


class SamplingParams:
    def __init__(self, max_tokens=16, temperature=0, ignore_eos=False, prompt_logprobs=None, **kw):
        self.max_tokens, self.prompt_logprobs = min(max_tokens, 3), prompt_logprobs


def _child(hang_file=None):
    if hang_file:                     # a worker that never finishes (as when one vLLM worker fails and the rest wait)
        import os, time
        open(hang_file, 'w').write(str(os.getpid()))
        time.sleep(10**6)


class LLM:
    def __init__(self, model, **kw):
        # like vLLM: start a worker with "spawn" (it imports the calling script again; an unguarded script fails here)
        import multiprocessing as mp
        import os
        import time
        hang = [h for h in os.environ.get('PRUNE_FAKE_HANG', '').split(',') if h]
        name = os.path.basename(model.rstrip('/'))[len('cut_'):]
        if model.rstrip('/').endswith(tuple('cut_' + h for h in hang)):
            w = mp.get_context('spawn').Process(target=_child, args=(os.environ['PRUNE_FAKE_HANG_PIDFILE'] + '.' + name,))
            w.start()
            time.sleep(10**6)
        w = mp.get_context('spawn').Process(target=_child)
        w.start()
        w.join()
        assert w.exitcode == 0, 'spawned worker failed'
        if isinstance(kw.get('speculative_config'), dict):   # like vLLM 0.19: it adds its ModelConfig to the caller's dict
            kw['speculative_config']['target_model_config'] = object()
        from transformers import CompressedTensorsConfig
        self.m = Gemma4ForConditionalGeneration.from_pretrained(
            model, dtype=torch.float32, quantization_config=CompressedTensorsConfig(run_compressed=False)).eval()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tm, cfg = self.m.model.language_model, self.m.config.get_text_config()
        self.cap = cfg.final_logit_softcapping

    def _logp(self, h):
        x = self.m.lm_head(h)
        if self.cap:
            x = torch.tanh(x / self.cap) * self.cap
        return torch.log_softmax(x.float(), -1)

    @torch.no_grad()
    def _merge(self, path):
        """vLLM's LoRA path, as merged weights: W += (alpha / r) B @ A for every PEFT tensor pair."""
        import json, os
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(path, 'adapter_config.json')))
        t = load_file(os.path.join(path, 'adapter_model.safetensors'))
        mods = dict(self.m.named_modules())
        for k, A in t.items():
            if not k.endswith('.lora_A.weight'):
                continue
            B = t[k.replace('lora_A', 'lora_B')]
            name = k[len('base_model.model.'):-len('.lora_A.weight')]
            W = mods[name].weight
            assert W.shape == (B.shape[0], A.shape[1]), (name, W.shape, B.shape, A.shape)
            W += (cfg['lora_alpha'] / cfg['r']) * (B.float() @ A.float())

    @torch.no_grad()
    def generate(self, prompts, sp, use_tqdm=False, lora_request=None):
        if lora_request is not None and getattr(self, 'merged', None) != lora_request.lora_path:
            assert getattr(self, 'merged', None) is None, 'one adapter per engine'
            self._merge(lora_request.lora_path)
            self.merged = lora_request.lora_path
        outs = []
        for p in prompts:
            ids = list(p['prompt_token_ids']) if isinstance(p, dict) else self.tok(p, add_special_tokens=False)['input_ids']
            h = self.tm(input_ids=torch.tensor([ids])).last_hidden_state[0]
            pl = None
            if sp.prompt_logprobs is not None:
                pl = [None]
                for t in range(1, len(ids)):
                    pl.append({ids[t]: NS(logprob=float(self._logp(h[t - 1])[ids[t]]))})
            new = []
            for _ in range(sp.max_tokens):
                nxt = int(self._logp(h[-1]).argmax())
                new.append(nxt)
                h = self.tm(input_ids=torch.tensor([ids + new])).last_hidden_state[0]
            outs.append(NS(prompt_logprobs=pl, outputs=[NS(text=self.tok.decode(new), token_ids=new)]))
        return outs
''')

# 3b. (V4) healing outputs: the training mask of c70 and two full-size PEFT adapters trained on it (removed gate/up rows
# exactly 0): 'kd' all zero (served, it must change nothing), 'lm' random
heal_env = {}
if V4:
    sys.path.insert(0, str(HERE))
    from safetensors.numpy import save_file as save_np  # noqa: E402
    from slice_model import make_plan as _mk  # noqa: E402
    plan70 = _mk(np.load(p1 / 'scores.npz'), 0.7, multiple=32, group=1)
    hmask = np.zeros((14, 128), np.uint8)
    for i, idx in plan70['ff_idx'].items():
        hmask[i, idx] = 1
    (tmp / 'heal' / 'tpu').mkdir(parents=True)
    np.savez(tmp / 'heal' / 'tpu' / 'heal_mask.npz', mask=hmask)
    ar = np.random.default_rng(5)
    for arm, scale in (('kd', 0.0), ('lm', 0.05)):
        ad = {}
        for i in range(14):
            for mod, (o, n) in {'mlp.gate_proj': (128, 64), 'mlp.up_proj': (128, 64), 'mlp.down_proj': (64, 128),
                                'self_attn.q_proj': (model.model.language_model.layers[i].self_attn.q_proj.weight.shape[0], 64)}.items():
                A = ar.standard_normal((4, n)).astype(np.float32) * 0.3
                B = ar.standard_normal((o, 4)).astype(np.float32) * scale
                if mod in ('mlp.gate_proj', 'mlp.up_proj'):
                    B[hmask[i] == 0] = 0
                pre = f'base_model.model.model.language_model.layers.{i}.{mod}'
                ad[pre + '.lora_A.weight'], ad[pre + '.lora_B.weight'] = A, B
        d_ = tmp / 'heal' / f'adapter_{arm}'
        d_.mkdir()
        save_np(ad, str(d_ / 'adapter_model.safetensors'))
        json.dump({'r': 4, 'lora_alpha': 8, 'peft_type': 'LORA'}, open(d_ / 'adapter_config.json', 'w'))
        heal_env['PRUNE_ADAPTER_C70_' + arm.upper()] = str(d_)
    heal_env['PRUNE_HEAL_MASK'] = str(tmp / 'heal' / 'tpu' / 'heal_mask.npz')

if V8:
    sys.path.insert(0, str(HERE))
    from safetensors.numpy import save_file as save_np  # noqa: E402
    from slice_model import make_plan as _mk  # noqa: E402
    ar = np.random.default_rng(8)
    masks = {}
    for k, keep in (('c55', 0.55), ('c70', 0.7)):
        pl = _mk(np.load(p1 / 'scores.npz'), keep, multiple=32, group=1)
        masks[k] = np.zeros((14, 128), np.uint8)
        for i, idx in pl['ff_idx'].items():
            masks[k][i, idx] = 1
    for nb, arms in (('heal55', (('C55KDM', 'adapter_out', 0.0),)),
                     ('calls55', (('C55CALLSM', 'adapter_out', 0.05), ('C55CALLSBM', 'adapter_out_b', 0.1)))):
        (tmp / nb / 'tpu').mkdir(parents=True)
        np.savez(tmp / nb / 'tpu' / 'heal_mask.npz', mask=masks['c55'])
        for env_name, sub, scale in arms:
            ad = {}
            for i in range(14):
                for mod, (o, n) in {'mlp.gate_proj': (128, 64), 'mlp.up_proj': (128, 64), 'mlp.down_proj': (64, 128)}.items():
                    A = ar.standard_normal((4, n)).astype(np.float32) * 0.3
                    B = ar.standard_normal((o, 4)).astype(np.float32) * scale
                    if mod in ('mlp.gate_proj', 'mlp.up_proj'):
                        B[masks['c55'][i] == 0] = 0
                    pre = f'base_model.model.model.language_model.layers.{i}.{mod}'
                    ad[pre + '.lora_A.weight'], ad[pre + '.lora_B.weight'] = A, B
            d_ = tmp / nb / sub
            d_.mkdir()
            save_np(ad, str(d_ / 'adapter_model.safetensors'))
            json.dump({'r': 4, 'lora_alpha': 8, 'peft_type': 'LORA'}, open(d_ / 'adapter_config.json', 'w'))
            heal_env['PRUNE_ADAPTER_' + env_name] = str(d_)
    (tmp / 'wrong' / 'tpu').mkdir(parents=True)
    np.savez(tmp / 'wrong' / 'tpu' / 'heal_mask.npz', mask=masks['c70'])
    heal_env['PRUNE_HEAL_MASK'] = str(tmp / 'wrong' / 'tpu' / 'heal_mask.npz')

# 4. run the built kernel script
out = tmp / 'out'
env = dict(os.environ, PRUNE_TEST='1', PRUNE_MODEL=str(w4), PRUNE_SCORES=str(p1 / 'scores.npz'),
           PRUNE_ROWS=str(tmp / 'sft.jsonl'), PRUNE_OUT=str(out), PRUNE_TMP=str(tmp / 'work'),
           PRUNE_FAKE_HANG='' if (V4 or V6 or V7 or V8) else ('c70,c85' if ABORT else 'c85'),
           PRUNE_FAKE_HANG_PIDFILE=str(tmp / 'hung_worker.pid'), PRUNE_BENCH_TIMEOUT_MIN='2.5',
           PYTHONPATH=str(fake.parent) + os.pathsep + os.environ.get('PYTHONPATH', ''), **heal_env)
(tmp / 'work').mkdir()
p = subprocess.run([sys.executable, str(ROOT / ('kernels/prune-serve8-gpu/prune_serve.py' if V8 else
                                                 'kernels/prune-serve7-gpu/prune_serve.py' if V7 else
                                                 'kernels/prune-serve6-gpu/prune_serve.py' if V6 else
                                                 'kernels/prune-serve2-gpu/prune_serve.py' if V4 else
                                                 'kernels/prune-serve-gpu/prune_serve.py'))], env=env,
                   capture_output=True, text=True, cwd=tmp)
print(p.stdout[-3000:])
assert p.returncode == 0, p.stderr[-4000:]
rep = json.load(open(out / 'report.json'))
if V8:
    M = rep['models']
    assert list(M) == ['full', 'c55kd~', 'c55calls~', 'c55callsb~'], list(M)
    assert all(r['rc'] == 0 and 'quality' in r and r.get('turns') for r in M.values()), {n: r.get('log_tail') for n, r in M.items()}
    assert rep['run'] == 'serve-v8' and rep['data']['quality_rows'] == 3 and rep['data']['turns_per_row'] == 4
    assert 'reused_cut_of' not in M['c55kd~'] and M['c55calls~']['reused_cut_of'] == 'c55kd~' \
        and M['c55callsb~']['reused_cut_of'] == 'c55calls~'
    for n, nb in (('c55kd~', 'heal55'), ('c55calls~', 'calls55'), ('c55callsb~', 'calls55')):
        r = M[n]
        assert r['adapter_cut_matches_training_mask'] is True, r
        assert r['adapter_mask'] == str(tmp / nb / 'tpu' / 'heal_mask.npz'), r['adapter_mask']   # its own notebook's mask
        assert r['adapter_slice']['removed_rows_lora_b_absmax'] == 0.0 and 'merge' in r and not r.get('adapter'), r
        assert r['plan_summary']['ff_channels'] == 64 and list(r['quality']['by_role']) == ['read', 'thought', 'args', 'syntax', 'close']
    q = {n: M[n]['quality']['agent_own_nats'] for n in M}
    print('v8 quality', q, {n: M[n]['merge'] for n in M if n != 'full'})
    assert len({q['c55kd~'], q['c55calls~'], q['c55callsb~']}) == 3, q    # three different merged models
    d = {t['model']: t for t in rep['table']}
    assert set(d) == set(M) and all('role_damage' in t for t in d.values())
    assert not list((tmp / 'work').glob('cut_*')) and not list((tmp / 'work').glob('merged_*')), 'cut models are deleted'
    print('TEST OK (v8)', tmp)
    sys.exit(0)
if V7:
    M = rep['models']
    assert list(M) == ['full', 'full_b'] and all(r['rc'] == 0 for r in M.values()), {n: r.get('log_tail') for n, r in M.items()}
    for n in M:
        b = json.load(open(out / f'bench_{n}.json'))
        assert 'quality' not in b and b['turns'] and set(b['turn_speed']) == {'streams1', 'streams8'}, b.keys()
        assert b['turn_speed']['streams1']['tokens'] > 0 and 'speed' in b
    sc = json.load(open(out / 'bench_full_b.json'))['engine']['speculative_config']
    assert sc['method'] == 'ngram' and 'speculative_config' not in json.load(open(out / 'bench_full.json'))['engine']
    assert all(not t['ok'] for t in rep['table']) and rep['run'] == 'serve-v7'
    print('TEST OK (v7)', tmp)
    sys.exit(0)
if V6:
    M = rep['models']
    assert list(M) == ['full', 'c55', 'c55role'], list(M)
    assert all(r['rc'] == 0 and 'quality' in r and r.get('turns') for r in M.values()), {n: r.get('log_tail') for n, r in M.items()}
    assert rep['run'] == 'serve-v6' and rep['data']['quality_rows'] == 3 and rep['data']['turns_per_row'] == 4
    names = ['read', 'thought', 'args', 'syntax', 'close']
    tpr = rep['data']['quality_tokens_per_role']
    assert list(tpr) == names and all(v > 0 for v in tpr.values()), tpr
    for n, r in M.items():
        br = r['quality']['by_role']
        assert list(br) == names and {g: v['tokens'] for g, v in br.items()} == tpr, (n, br)
        assert sum(v['tokens'] for v in br.values()) == r['quality']['all_tokens'] and 'by_role_error' not in r['quality']
    pl = {n: M[n]['plan_summary'] for n in ('c55', 'c55role')}
    assert pl['c55']['ff_channels'] == pl['c55role']['ff_channels'] == 64, pl
    assert M['c55']['quality']['agent_own_nats'] != M['c55role']['quality']['agent_own_nats']   # other channels kept
    d = {t['model']: t for t in rep['table']}
    assert set(d['c55role']['role_damage']) == set(names) and d['full']['role_damage'] == {g: 0.0 for g in names}
    # the per-role loss of the full model equals the in-process model's, token by token roles from the bench data
    data = json.load(open(tmp / 'work/bench_data.json'))
    TM, cap = model.model.language_model, model.config.get_text_config().final_logit_softcapping
    acc = {}
    with torch.no_grad():
        for r in data['heldout'][:3]:
            h = TM(input_ids=torch.tensor([r['ids']])).last_hidden_state[0]
            for t in range(1, len(r['ids'])):
                x = model.lm_head(h[t - 1])
                a = acc.setdefault(names[r['roles'][t]], [0.0, 0])
                a[0] -= float(torch.log_softmax(torch.tanh(x / cap) * cap, -1)[r['ids'][t]])
                a[1] += 1
    for g, (tot, n) in acc.items():
        assert abs(tot / n - M['full']['quality']['by_role'][g]['nats']) < 2e-3, (g, tot / n, M['full']['quality']['by_role'][g])
    assert len(data['heldout']) == 3 and all(len(r['roles']) == len(r['ids']) for r in data['heldout'])
    assert not list((tmp / 'work').glob('cut_*')), 'cut models are deleted after measuring'
    print('TEST OK (v6)', tmp, {g: M['c55role']['quality']['by_role'][g]['nats'] for g in names})
    sys.exit(0)
if V4:
    M = rep['models']
    assert list(M) == ['full', 'c70', 'c70+kd', 'c70+lm', 'd6m', 'c70d6m'], list(M)
    assert all(r['rc'] == 0 and 'quality' in r and 'speed' in r for r in M.values()), {n: r.get('log_tail') for n, r in M.items()}
    assert M['c70+kd']['reused_cut_of'] == 'c70' and M['c70+lm']['reused_cut_of'] == 'c70+kd' and 'reused_cut_of' not in M['c70']
    for n in ('c70+kd', 'c70+lm'):
        assert M[n]['adapter_cut_matches_training_mask'] is True and M[n]['lora_rank'] == 4, M[n]
        assert M[n]['adapter_slice']['removed_rows_lora_b_absmax'] == 0.0 and M[n]['adapter'], M[n]
    q = {n: M[n]['quality']['agent_own_nats'] for n in M}
    print('v4 quality', q)
    assert q['c70+kd'] == q['c70'] and q['c70+lm'] != q['c70'], q   # a zero adapter changes nothing, a real one does
    assert M['d6m']['dropped_layers'] == ORDER_M[:6] == M['c70d6m']['dropped_layers'], (M['d6m']['dropped_layers'])
    assert M['c70d6m']['plan_summary']['layers'] == 8 and M['c70d6m']['plan_summary']['ff_channels'] == 96
    assert rep['layer_orders']['measured'][:6] == ORDER_M[:6]
    assert not list((tmp / 'work').glob('cut_*')), 'cut models are deleted after measuring'
    CHECK = (('d6m', 1.0),)
else:
    assert list(rep['models']) == ['full', 'b70', 'c70', 'c85', 'b85', 'c55', 'd6'], list(rep['models'])
    CHECK = (('b85', 0.85), ('d6', 1.0))


def killed(name, wait_s=20):
    """The hung worker has exited (gone, or a zombie waiting to be reaped, which holds no memory), within wait_s."""
    import time
    pid = int((tmp / f'hung_worker.pid.{name}').read_text())
    for _ in range(wait_s * 4):
        try:
            if open(f'/proc/{pid}/stat').read().split(')')[-1].split()[0] == 'Z':
                return True
        except FileNotFoundError:
            return True
        time.sleep(0.25)
    return False


if ABORT and not V4:
    assert rep['models']['b70']['rc'] == 0 and rep['models']['full']['rc'] == 0
    assert rep['models']['c70']['rc'] == 'timeout' and rep['models']['c85']['rc'] == 'timeout'
    assert all(rep['models'][n]['rc'] == 'skipped' for n in ('b85', 'c55', 'd6')), rep['models']
    assert killed('c70') and killed('c85'), 'hung workers left alive'
    assert [t['model'] for t in rep['table'] if not t['ok']] == ['c70', 'c85', 'b85', 'c55', 'd6'], rep['table']
    assert not list((tmp / 'work').glob('cut_*')), 'cut models are deleted after measuring'
    print('TEST OK (abort after two failures in a row)', tmp)
    sys.exit(0)
if not V4:
    assert all((out / f'bench_{n}.log').exists() and (out / f'bench_{n}.json').exists() == (n != 'c85') for n in rep['models'])
    # c85 hangs on purpose: it is killed with its worker after the time limit, and the other cuts still run
    assert rep['models']['c85']['rc'] == 'timeout' and 'log_tail' in rep['models']['c85']
    assert killed('c85'), 'the hung worker is still alive'
    for name, r in rep['models'].items():
        if name != 'c85':
            assert r['rc'] == 0 and 'speed' in r and 'quality' in r and 'sanity' in r, (name, r.get('log_tail'))
    assert [t['model'] for t in rep['table'] if not t['ok']] == ['c85'], rep['table']
    d = {t['model']: t for t in rep['table']}
    assert rep['data']['heldout_own_tokens'] > 0 and d['full']['agent_own_nats'] > 0
    assert d['full']['agent_damage'] == 0 and all(d[n]['agent_damage'] != 0 for n in ('b70', 'c70', 'd6'))
    assert rep['models']['d6']['plan_summary']['layers'] == 8 and rep['models']['b70']['plan_summary']['ff_channels'] == 96
    assert rep['models']['c55']['plan_summary']['ff_channels'] == 64 and 'requant_rel_err' in rep['models']['c70']['plan_summary'] and isinstance(json.load(open(tmp / 'work/bench_data.json'))['sanity'], list)
    assert not list((tmp / 'work').glob('cut_*')), 'cut models are deleted after measuring'

# 5. the quality number of a cut equals the hook-masked full (4-bit) model's, on the same held-out tokens
sys.path.insert(0, str(HERE))
from slice_model import make_plan, plan_masks  # noqa: E402
data = json.load(open(tmp / 'work/bench_data.json'))
z = np.load(p1 / 'scores.npz')
TM = model.model.language_model
cap = model.config.get_text_config().final_logit_softcapping
for name, keep in CHECK:
    drop = rep['models'][name]['dropped_layers']
    plan = make_plan(z, keep, drop=drop, multiple=32, group=32)
    mask, dset = plan_masks(plan)
    hooks = []
    for li, lyr in enumerate(TM.layers):
        hooks.append(lyr.mlp.down_proj.register_forward_pre_hook(lambda m, a, li=li: (a[0] * mask[li],)))
        hooks.append(lyr.register_forward_hook(
            lambda m, a, kw, o, li=li: a[0] * m.layer_scalar if li in dset else None, with_kwargs=True))
    tot, n = 0.0, 0
    with torch.no_grad():
        for r in data['heldout']:
            h = TM(input_ids=torch.tensor([r['ids']])).last_hidden_state[0]
            for t in r['own']:
                x = model.lm_head(h[t - 1])
                tot -= float(torch.log_softmax(torch.tanh(x / cap) * cap, -1)[r['ids'][t]])
                n += 1
    for hk in hooks:
        hk.remove()
    want = tot / n
    got = rep['models'][name]['quality']['agent_own_nats']
    print(f'{name} quality: served cut {got:.4f} vs masked full {want:.4f}')
    assert abs(got - want) < 1e-3, (name, got, want)
print('TEST OK', tmp)
