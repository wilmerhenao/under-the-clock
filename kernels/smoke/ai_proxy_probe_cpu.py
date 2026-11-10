"""Does a Kaggle notebook get the AI-quota model proxy (Kaggle Benchmarks), and does it serve Gemma 4 31B?

From outside Kaggle, `kaggle benchmarks auth` mints a token for a proxy that lists google/gemma-4-31b but answers 404
for it (2026-10-07 16:40 UTC; Gemini 3 Flash and DeepSeek-R1 answer). Inside a notebook the proxy comes from the
environment (MODEL_PROXY_URL, MODEL_PROXY_API_KEY, LLMS_AVAILABLE). This kernel reports whether those are set, the
proxy host, and the model names; then a 1-line request per candidate model; then, for Gemma, the scorer's own
requests from our replay captures: 3 compaction-summary requests and 2 agent steps (6 tools, ~7k and ~15k tokens).
Answers: can our training mix get Gemma-written summaries, and can the harness generate on-policy trajectories on CPU
with Gemma behind the proxy (tool calls, thinking, seconds and dollars per step)?

The key is never printed or written. Output: /kaggle/working/probe/probe.json.
"""
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

PROBE_INPUTS = ''  # @embed ../../runs/ai-proxy-probe/probe_inputs.json
MODELS = ['google/gemma-4-31b', 'google/gemma-4-26b-a4b', 'zai/glm-5', 'qwen/qwen3-coder-480b-a35b-instruct',
          'google/gemini-3-flash-preview', 'deepseek-ai/deepseek-r1-0528']
FULL_MODEL = 'google/gemma-4-31b'  # @define (the model that gets the summary and agent-step requests)
OUT = Path('/kaggle/working/probe')
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()

url, key = os.environ.get('MODEL_PROXY_URL', ''), os.environ.get('MODEL_PROXY_API_KEY', '')
report = {'env': {'MODEL_PROXY_URL_set': bool(url), 'MODEL_PROXY_API_KEY_set': bool(key),
                  'proxy_host': urlparse(url).netloc if url else None,
                  'MODEL_PROXY_EXPIRY_TIME': os.environ.get('MODEL_PROXY_EXPIRY_TIME'),
                  'LLM_DEFAULT': os.environ.get('LLM_DEFAULT'), 'LLM_DEFAULT_EVAL': os.environ.get('LLM_DEFAULT_EVAL'),
                  'LLMS_AVAILABLE': [x for x in os.environ.get('LLMS_AVAILABLE', '').split(',') if x],
                  'other_names': sorted(k for k in os.environ if 'PROXY' in k or 'BENCHMARK' in k or k.startswith('LLM'))}}
try:
    import kaggle_benchmarks  # noqa: F401 - only the version is wanted
    report['env']['kaggle_benchmarks'] = getattr(kaggle_benchmarks, '__version__', '?')
except Exception as e:  # noqa: BLE001
    report['env']['kaggle_benchmarks'] = f'import failed: {type(e).__name__}'
print(json.dumps(report['env'], indent=1), flush=True)


def save():
    report['seconds'] = round(time.time() - T0, 1)
    (OUT / 'probe.json').write_text(json.dumps(report, indent=1))


def post(body, timeout=300):
    t = time.time()
    try:
        r = requests.post(url.rstrip('/') + '/openapi/chat/completions', json=body,
                          headers={'Authorization': f'Bearer {key}'}, timeout=timeout)
        out = {'status': r.status_code, 'seconds': round(time.time() - t, 1)}
        try:
            out['json'] = r.json()
        except ValueError:
            out['text'] = r.text.replace(key, '<redacted>')[:2000]
    except Exception as e:  # noqa: BLE001 - the type only, never the request
        out = {'status': None, 'error': type(e).__name__, 'seconds': round(time.time() - t, 1)}
    return out


def brief(res, keep=6000):
    j = res.get('json') or {}
    ch = (j.get('choices') or [{}])[0]
    msg = ch.get('message') or {}
    reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
    return {'status': res['status'], 'seconds': res['seconds'], 'finish': ch.get('finish_reason'),
            'msg_keys': sorted(msg), 'content': (msg.get('content') or '')[:keep], 'reasoning': reasoning[:keep],
            'reasoning_chars': len(reasoning), 'content_chars': len(msg.get('content') or ''),
            'tool_calls': msg.get('tool_calls'), 'usage': j.get('usage'),
            'error': res.get('error') or res.get('text') or j.get('message') or (j.get('error') or {}).get('message')}


def ask(base, variants):
    """Try the request as captured, then without vLLM-only fields: -> (brief, variant name)."""
    last = None
    for name, drop in variants:
        body = {k: v for k, v in base.items() if k not in drop}
        b = brief(post(body))
        if b['status'] == 200:
            return b, name
        last = (b, name)
    return last


if not (url and key):
    report['verdict'] = 'no proxy environment in this notebook'
    save()
    raise SystemExit(0)

report['models'] = {}
for m in MODELS:
    b = brief(post({'model': m, 'messages': [{'role': 'user', 'content': 'Reply with the word OK.'}], 'max_tokens': 256}))
    report['models'][m] = {k: b[k] for k in ('status', 'seconds', 'finish', 'content', 'reasoning_chars', 'usage', 'error')}
    print(m, b['status'], b['seconds'], 's |', (b['content'] or '')[:60].replace('\n', ' '), '|', b['usage'], '|',
          str(b['error'])[:160], flush=True)
save()

if report['models'].get(FULL_MODEL, {}).get('status') == 200:
    inputs = json.loads(PROBE_INPUTS)
    vllm_only = ('chat_template_kwargs', 'thinking_token_budget')
    variants = [('as_captured', ()), ('no_vllm_fields', vllm_only),
                ('minimal', vllm_only + ('temperature', 'top_p', 'max_completion_tokens'))]
    tools_kw = {'type': 'function', 'function': {'name': 'run_command', 'description': 'Run a shell command.',
                'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}
    b, v = ask({'model': FULL_MODEL, 'tools': [tools_kw], 'max_tokens': 1024, 'chat_template_kwargs': {'enable_thinking': True},
                'messages': [{'role': 'system', 'content': 'You are a software engineering agent. Use the tools.'},
                             {'role': 'user', 'content': 'List the Python files under /workspace/src.'}]}, variants)
    report['tool_call_toy'] = dict(b, variant=v)
    print('tool_call_toy', b['status'], v, '| tool_calls', b['tool_calls'], '| reasoning chars', b['reasoning_chars'],
          '| err', str(b['error'])[:200], flush=True)
    report['summaries'] = []
    for s in inputs['summaries']:
        b, v = ask(dict(s['request'], model=FULL_MODEL, max_tokens=4096), variants)
        report['summaries'].append(dict(b, variant=v, capture=s['capture'], turn=s['turn'],
                                        template_summary=s['template_summary']))
        print('summary', s['capture'], b['status'], v, b['seconds'], 's | content', b['content_chars'], 'reasoning',
              b['reasoning_chars'], '|', b['usage'], '|', str(b['error'])[:160], flush=True)
        save()
    report['agent_steps'] = []
    for s in inputs['agent_steps']:
        b, v = ask(dict(s['request'], model=FULL_MODEL), variants)
        report['agent_steps'].append(dict(b, variant=v, capture=s['capture'], turn=s['turn'],
                                          prompt_tokens_gemma=s['prompt_tokens_gemma']))
        print('agent_step', s['capture'], s['turn'], b['status'], v, b['seconds'], 's | tool_calls',
              json.dumps(b['tool_calls'])[:300] if b['tool_calls'] else None, '| content', b['content_chars'],
              'reasoning', b['reasoning_chars'], '|', b['usage'], '|', str(b['error'])[:160], flush=True)
        save()
report['verdict'] = 'see models / summaries / agent_steps'
save()
print('done in', round(time.time() - T0), 's')
