"""CPU tests of the tolerant Gemma 4 tool-call parser (gemma4_tolerant.py, B209).

usage: python3 kernels/colab/prune/test_gemma4_tolerant.py [path to vLLM 0.19.1's vllm/tool_parsers/gemma4_tool_parser.py]
With the path, vLLM's own argument parser checks the rewritten calls (its vLLM imports are stubbed); without it, that
part is skipped. The plugin's registration and its extract_tool_calls override run against a fake vLLM."""
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gemma4_tolerant as gt  # noqa: E402

Q = '<|"|>'
RF = {'filepath', 'start_line', 'end_line'}
EF = {'filepath', 'old_string', 'new_string'}
CASES = [  # (tool keys, malformed body, rewritten body)
    (RF, f'filepath:{Q}src/a.py`,start_line:70,end_line:170', f'filepath:{Q}src/a.py{Q},start_line:70,end_line:170'),
    (RF, f'filepath:{Q}a.py{Q},start_line":153,end_line":160', f'filepath:{Q}a.py{Q},start_line:153,end_line:160'),
    (RF, f'filepath:{Q}m/merge.py\\", start_line: 145', f'filepath:{Q}m/merge.py{Q}, start_line: 145'),
    (RF, f'filepath:{Q}K/query.py`', f'filepath:{Q}K/query.py{Q}'),
    (EF, f'filepath:{Q}r.py{Q},old_string:{Q}x = 1)`,new_string:{Q}x = 2{Q}',
     f'filepath:{Q}r.py{Q},old_string:{Q}x = 1){Q},new_string:{Q}x = 2{Q}'),
]
UNCHANGED = [  # well-formed calls, code with quotes and backticks inside strings, and a failure the rule does not cover
    (RF, f'filepath:{Q}src/a.py{Q},start_line:70'),
    (EF, f'filepath:{Q}r.py{Q},old_string:{Q}print("a", end="")  # `x`{Q},new_string:{Q}print("b"){Q}'),
    (RF, f'filepath:{Q}g.py`,start_line=190,end_line=230'),
]


def test_rewrite():
    for keys, bad, good in CASES:
        assert gt.rewrite_args(bad, keys) == good, (bad, gt.rewrite_args(bad, keys))
        assert gt.rewrite_args(good, keys) == good, good          # idempotent on its own output
    for keys, body in UNCHANGED:
        assert gt.rewrite_args(body, keys) == body, (body, gt.rewrite_args(body, keys))
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'parameters': {'properties': {k: {} for k in RF}}}},
             {'type': 'function', 'function': {'name': 'edit_file', 'parameters': {'properties': {k: {} for k in EF}}}}]
    params = gt.tool_params(tools)
    assert params == {'read_file': RF, 'edit_file': EF}
    text = ('I will read it.<|tool_call>call:read_file{' + CASES[0][1] + '}<tool_call|>'
            '<|tool_call>call:edit_file{' + UNCHANGED[1][1] + '}<tool_call|>')
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, 'rw.jsonl')
        out, n = gt.rewrite_output(text, params, log)
        assert n == 1 and out.count('<|tool_call>') == 2 and CASES[0][2] in out and UNCHANGED[1][1] in out
        rows = [json.loads(x) for x in open(log)]
        assert len(rows) == 1 and rows[0]['tool'] == 'read_file'
    print('rewrite: ok')


def test_vllm_args(parser_path):
    """vLLM's own _parse_gemma4_args on the rewritten bodies (its imports stubbed: only the pure functions run)."""
    stubs = {}
    for name in ('vllm', 'vllm.entrypoints', 'vllm.entrypoints.chat_utils', 'vllm.entrypoints.openai',
                 'vllm.entrypoints.openai.chat_completion', 'vllm.entrypoints.openai.chat_completion.protocol',
                 'vllm.entrypoints.openai.engine', 'vllm.entrypoints.openai.engine.protocol',
                 'vllm.entrypoints.openai.responses', 'vllm.entrypoints.openai.responses.protocol', 'vllm.logger',
                 'vllm.tokenizers', 'vllm.tool_parsers', 'vllm.tool_parsers.abstract_tool_parser',
                 'vllm.tool_parsers.utils', 'regex'):
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: type(attr, (), {})                          # any imported name: a dummy class
        stubs[name] = m
    stubs['vllm.logger'].init_logger = lambda *a, **k: types.SimpleNamespace(exception=print, info=print,
                                                                              warning=print, debug=print)
    import re as _re
    stubs['regex'] = _re
    old = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location('vllm_gemma4_parser', parser_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    want = [{'filepath': 'src/a.py', 'start_line': 70, 'end_line': 170}, {'filepath': 'a.py', 'start_line': 153,
            'end_line': 160}, {'filepath': 'm/merge.py', 'start_line': 145}, {'filepath': 'K/query.py'},
            {'filepath': 'r.py', 'old_string': 'x = 1)', 'new_string': 'x = 2'}]
    for (keys, bad, good), w in zip(CASES, want):
        before, after = mod._parse_gemma4_args(bad), mod._parse_gemma4_args(good)
        assert after == w, (good, after)
        assert before != w, (bad, before)                                         # vLLM alone gets these wrong
    print('vLLM argument parser on the rewritten calls: ok')


def test_plugin_with_fake_vllm():
    """Registration under 'gemma4_tolerant' and extract_tool_calls handing the rewritten text to vLLM's parser."""
    seen, registry = [], {}

    class FakeGemma4ToolParser:
        tool_call_start_token = '<|tool_call>'

        def __init__(self, *a, **k):
            pass

        def extract_tool_calls(self, model_output, request):
            seen.append(model_output)
            return 'parsed'

    class FakeManager:
        @classmethod
        def register_module(cls, name=None, force=True, module=None):
            registry[name] = module
            return module

    a = types.ModuleType('vllm.tool_parsers.abstract_tool_parser')
    a.ToolParserManager = FakeManager
    g = types.ModuleType('vllm.tool_parsers.gemma4_tool_parser')
    g.Gemma4ToolParser = FakeGemma4ToolParser
    mods = {'vllm': types.ModuleType('vllm'), 'vllm.tool_parsers': types.ModuleType('vllm.tool_parsers'),
            'vllm.tool_parsers.abstract_tool_parser': a, 'vllm.tool_parsers.gemma4_tool_parser': g}
    old = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    sys.modules.pop('gemma4_tolerant', None)
    try:
        plugin = importlib.import_module('gemma4_tolerant')
        cls = registry.get('gemma4_tolerant')
        assert cls is not None and issubclass(cls, FakeGemma4ToolParser), registry
        req = types.SimpleNamespace(tools=[types.SimpleNamespace(function=types.SimpleNamespace(
            name='read_file', parameters={'properties': {k: {} for k in RF}}))])
        with tempfile.TemporaryDirectory() as d:
            os.environ['G4DEV_TOLERANT_LOG'] = os.path.join(d, 'rw.jsonl')
            out = cls().extract_tool_calls('<|tool_call>call:read_file{' + CASES[0][1] + '}<tool_call|>', req)
            n_logged = sum(1 for _ in open(os.environ['G4DEV_TOLERANT_LOG']))
            os.environ.pop('G4DEV_TOLERANT_LOG')
        assert out == 'parsed' and seen == ['<|tool_call>call:read_file{' + CASES[0][2] + '}<tool_call|>'], seen
        assert n_logged == 1
        cls().extract_tool_calls('no call here', req)
        assert seen[-1] == 'no call here'
        assert plugin.rewrite_args is gt.rewrite_args or plugin.rewrite_args(CASES[0][1], RF) == CASES[0][2]
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop('gemma4_tolerant', None)
    print('plugin registration and override (fake vLLM): ok')


if __name__ == '__main__':
    test_rewrite()
    if len(sys.argv) > 1:
        test_vllm_args(sys.argv[1])
    else:
        print('vLLM argument parser: skipped (no path given)')
    test_plugin_with_fake_vllm()
    print('all ok')
