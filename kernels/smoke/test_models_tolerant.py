"""CPU test of train_gpu.run_models with the B209 options (BLOCK_IDS, a model served with the tolerant tool-call parser).

usage: python3 kernels/smoke/test_models_tolerant.py <path to vLLM 0.19.1's vllm/tool_parsers/gemma4_tool_parser.py>
run_models is taken from train_gpu.py's source and run with stubs: adk_submission's VllmServer (records its config and,
like vLLM's --tool-parser-plugin, imports the plugin file in the "server", then sends one malformed call through the
registered parser), slice_model, the eval harness subprocess, transformers' tokenizer and vLLM's parser registry (whose
Gemma 4 parser uses vLLM 0.19.1's own regex and argument parser from the given file)."""
import ast
import glob as _glob
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARSER = sys.argv[1]


def load_vllm_parser_functions(path):
    stubs = {}
    for name in ('vllm', 'vllm.entrypoints', 'vllm.entrypoints.chat_utils', 'vllm.entrypoints.openai',
                 'vllm.entrypoints.openai.chat_completion', 'vllm.entrypoints.openai.chat_completion.protocol',
                 'vllm.entrypoints.openai.engine', 'vllm.entrypoints.openai.engine.protocol',
                 'vllm.entrypoints.openai.responses', 'vllm.entrypoints.openai.responses.protocol', 'vllm.logger',
                 'vllm.tokenizers', 'vllm.tool_parsers', 'vllm.tool_parsers.abstract_tool_parser', 'vllm.tool_parsers.utils'):
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: type(attr, (), {})
        stubs[name] = m
    stubs['vllm.logger'].init_logger = lambda *a, **k: types.SimpleNamespace(exception=print, info=print)
    import re
    stubs['regex'] = re
    old = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location('vllm_g4', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


G4 = load_vllm_parser_functions(PARSER)


# ---- fake vLLM registry and Gemma 4 parser (vLLM's own regex and argument parser) ----
class ToolParserManager:
    tool_parsers = {}

    @classmethod
    def register_module(cls, name=None, force=True, module=None):
        cls.tool_parsers[name] = module
        return module

    @classmethod
    def get_tool_parser(cls, name):
        return cls.tool_parsers[name]


class Gemma4ToolParser:
    tool_call_start_token = '<|tool_call>'

    def __init__(self, tokenizer, tools=None):
        self.tok = tokenizer

    def extract_tool_calls(self, model_output, request):
        import re
        calls = re.findall(r'<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>', model_output, re.DOTALL)
        tc = [types.SimpleNamespace(function=types.SimpleNamespace(name=n, arguments=json.dumps(G4._parse_gemma4_args(a))))
              for n, a in calls]
        return types.SimpleNamespace(tools_called=bool(tc), tool_calls=tc, content=None)


m_abs = types.ModuleType('vllm.tool_parsers.abstract_tool_parser')
m_abs.ToolParserManager = ToolParserManager
m_g4 = types.ModuleType('vllm.tool_parsers.gemma4_tool_parser')
m_g4.Gemma4ToolParser = Gemma4ToolParser
sys.modules.update({'vllm': types.ModuleType('vllm'), 'vllm.tool_parsers': types.ModuleType('vllm.tool_parsers'),
                    'vllm.tool_parsers.abstract_tool_parser': m_abs, 'vllm.tool_parsers.gemma4_tool_parser': m_g4})
tf = types.ModuleType('transformers')
tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: {'path': p})
sys.modules['transformers'] = tf

# ---- fake adk_submission: the server "loads" the plugin like --tool-parser-plugin and parses one malformed call ----
CONFIGS, SERVER_EVENTS = [], []


class VllmConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        CONFIGS.append(kw)


class VllmServer:
    def __init__(self, config, adapter_manifest=None):
        self.config, self.base_url, self.effective_model_path = config, 'http://127.0.0.1:8000/v1', config.model

    def start(self):
        ea = list(self.config.extra_args or [])
        if '--tool-parser-plugin' in ea:
            path = ea[ea.index('--tool-parser-plugin') + 1]
            spec = importlib.util.spec_from_file_location('gemma4_tolerant_server', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)                     # registers 'gemma4_tolerant', writes the marker
            cls = ToolParserManager.get_tool_parser(self.config.tool_call_parser)
            req = types.SimpleNamespace(tools=[types.SimpleNamespace(function=types.SimpleNamespace(
                name='read_file', parameters={'properties': {'filepath': {}, 'start_line': {}}}))])
            r = cls(None).extract_tool_calls('<|tool_call>call:read_file{filepath:<|"|>a.py`,start_line:3}<tool_call|>', req)
            SERVER_EVENTS.append(json.loads(r.tool_calls[0].function.arguments))
        return self

    def stop(self):
        pass


adk = types.ModuleType('adk_submission')
adk.VllmConfig, adk.VllmServer = VllmConfig, VllmServer
sys.modules['adk_submission'] = adk

# ---- run_models from train_gpu.py's source ----
src = (HERE / 'train_gpu.py').read_text()
tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_models')
code = compile(ast.Module(body=[fn], type_ignores=[]), 'train_gpu.run_models', 'exec')

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    work, out, inp = d / 'working', d / 'working' / 'train_eval', d / 'input'
    out.mkdir(parents=True)
    (inp / 'k' / 'prune' / 'score-v3').mkdir(parents=True)
    import numpy as np
    np.savez(inp / 'k' / 'prune' / 'score-v3' / 'scores.npz', agent_ff=np.ones((2, 8)), down_norm=np.ones((2, 8)))
    ids = [f't{i}' for i in range(10)]
    data = d / 'data'
    data.mkdir()
    (data / 'tasks.jsonl').write_text('\n'.join(json.dumps({'instance_id': i, 'repo': f'o/r{i}'}) for i in ids))
    splits = d / 'splits.json'
    splits.write_text(json.dumps({'repograph': ids}))
    stages = []
    calls = []

    def fake_run(cmd, timeout=None, **kw):
        calls.append(cmd)
        o = Path(cmd[cmd.index('--out') + 1])
        o.mkdir(parents=True, exist_ok=True)
        (o / 'summary_smoke.json').write_text(json.dumps({'status': 'complete', 'healthy': True, 'n_tasks': 3,
                                                          'n_finished': 3, 'n_resolved': 1}))
        return types.SimpleNamespace(returncode=0)

    slice_src = ('def make_plan(z, keep, **k):\n    return {"ff_idx": {}, "kept_layers": []}\n'
                 'def slice_checkpoint(src, dst, plan, **k):\n    return {"shards": 1}\n'
                 'def slice_adapter(*a, **k):\n    return {}\n'
                 'def merge_lora_w4(*a, **k):\n    return {}\n')
    fake_glob = types.SimpleNamespace(glob=lambda pat, recursive=False: _glob.glob(
        pat.replace('/kaggle/input/', str(inp) + '/'), recursive=recursive))
    ns = {'WORK': work, 'OUT': out, 'EVAL_SET': 'repograph', 'BLOCKS': 1, 'BLOCK_TASKS': 3, 'BLOCK_EXCLUDE_REPOS': [],
          'BLOCK_IDS': {'block0': ['t7', 't2', 't5']}, 'SEG_CAP_MIN': 66, 'DEADLINE_S': 5760, 'T0': time.time(),
          'EVAL_CONC': 8, 'EVAL_BUNDLE': 'sub-v9c', 'BUNDLES': {'sub-v9c': {'agent.yaml': 'name: x\n'}},
          'EVAL_TIME_SCALE': 1.0, 'EVAL_PY': 'print(1)\n', 'SLICE_SRC': slice_src,
          'TOLERANT_SRC': (HERE.parent / 'colab' / 'prune' / 'gemma4_tolerant.py').read_text(),
          'MODELS': [{'name': 'c70tol', 'cut': {'scores': '**/prune/score-v*/scores.npz', 'keep': 0.7, 'multiple': 512,
                                                'group': 1}, 'tool_parser': 'gemma4_tolerant'}],
          'server': None, 'stage': lambda name, **kw: stages.append((name, kw)), 'json': json, 'time': time, 'os': os,
          'sys': sys, 'Path': Path, 'glob': fake_glob,
          'subprocess': types.SimpleNamespace(run=fake_run, TimeoutExpired=TimeoutError)}
    work.mkdir(parents=True, exist_ok=True)
    exec(code, ns)
    ns['run_models'](str(d / 'model_full'), str(data), str(splits), ['--x'])

    names = [s[0] for s in stages]
    st = dict(stages)
    want = {'filepath': 'src/a.py', 'start_line': 70, 'end_line': 170}
    assert st['tolerant_selftest']['got'] == {'backtick': want, 'json_key': want, 'clean': want}, st['tolerant_selftest']
    assert st['models_blocks']['sizes'] == {'block0': 3}, st['models_blocks']
    assert json.loads((work / 'blocks.json').read_text()) == {'block0': ['t7', 't2', 't5']}
    cfg = CONFIGS[-1]
    assert cfg['tool_call_parser'] == 'gemma4_tolerant' and cfg['extra_args'] == ['--tool-parser-plugin',
                                                                                  str(work / 'gemma4_tolerant.py')], cfg
    assert SERVER_EVENTS == [{'filepath': 'a.py', 'start_line': 3}], SERVER_EVENTS
    assert st['tolerant_rewrites'] == {'model': 'c70tol', 'registered': 1, 'rewritten': 1}, st['tolerant_rewrites']
    assert '--task-ids-key' in calls[0] and calls[0][calls[0].index('--task-ids-key') + 1] == 'block0'
    assert names[-1] == 'models_done', names
    print('stages:', names)
    print('run_models with BLOCK_IDS and the tolerant parser: ok')
