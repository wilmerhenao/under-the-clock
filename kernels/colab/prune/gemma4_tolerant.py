"""A tolerant Gemma 4 tool-call parser for vLLM (B209): vLLM's own `gemma4` parser after two rewrites of each call's
argument string, the rule measured offline by kernels/analysis/tolerant_parse.py (53 of the 70% / 55% cuts' 68 malformed
next calls become well-formed, B208).

Gemma 4 writes a call as <|tool_call>call:NAME{key:<|"|>text<|"|>,num:3}<tool_call|>. A pruned model often closes a
string with a backtick or a JSON quote instead of <|"|> (filepath:<|"|>a.py`,start_line:70), so vLLM's parser reads the
rest of the call into that string, or writes a JSON-style key (start_line":151), which arrives as an unknown parameter.
The rewrites, applied only inside a call:
  (1) inside an open <|"|> string, a backtick, a double quote or an escaped double quote (\\") directly before
      ', <one of this tool's parameters>:' or before the end of the arguments becomes <|"|>;
  (2) a double quote between a parameter name and its colon is dropped.
Parameter names come from the tools in the request (the harness's tool schema). Everything else is vLLM's parser
unchanged. Each rewritten call is appended to the JSON-lines file named by $G4DEV_TOLERANT_LOG (if set).

Use: vllm serve ... --tool-parser-plugin gemma4_tolerant.py --tool-call-parser gemma4_tolerant
The rewrite functions import nothing from vLLM, so they are tested on CPU (test_gemma4_tolerant.py)."""
import json
import os
import re
import time

Q = '<|"|>'


def rewrite_args(body, keys):
    """The argument string of one call (between '{' and '}') with the two rewrites; `keys` = the tool's parameter names."""
    keys = sorted({k for k in keys if k}, key=len, reverse=True)
    if not keys:
        return body
    alt = '|'.join(map(re.escape, keys))
    body = re.sub(r'\b(' + alt + r')"\s*:', r'\1:', body)                                  # (2) start_line": 151
    nxt = re.compile(r'\s*,\s*(?:' + alt + r')\s*:')
    out, i, open_s = [], 0, False
    while i < len(body):
        if body.startswith(Q, i):
            open_s = not open_s
            out.append(Q)
            i += len(Q)
            continue
        ch = body[i]
        w = 2 if body.startswith('\\"', i) else 1                                           # \" closes as well
        if open_s and (ch in '`"' or w == 2) and (nxt.match(body, i + w) or body[i + w:].strip() == ''):   # (1)
            if body.find(Q, i + w) < 0 or nxt.match(body, i + w):
                out.append(Q)
                open_s = False
                i += w
                continue
        out.append(ch)
        i += 1
    return ''.join(out)


CALL = re.compile(r'<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>', re.DOTALL)   # vLLM 0.19.1's own regex


def tool_params(tools):
    """{tool name: set of parameter names} from an OpenAI-style tool list (objects or dicts)."""
    out = {}
    for t in tools or []:
        f = getattr(t, 'function', None) if not isinstance(t, dict) else t.get('function')
        if f is None:
            continue
        name = getattr(f, 'name', None) if not isinstance(f, dict) else f.get('name')
        params = (getattr(f, 'parameters', None) if not isinstance(f, dict) else f.get('parameters')) or {}
        out[name] = set((params.get('properties') or {}).keys())
    return out


def rewrite_output(text, params, log_path=None):
    """Every complete call in `text` rewritten; returns (new text, number of calls changed)."""
    every = set().union(*params.values()) if params else set()
    changed = []

    def fix(m):
        name, body = m.group(1), m.group(2)
        new = rewrite_args(body, params.get(name) or every)
        if new == body:
            return m.group(0)
        changed.append({'tool': name, 'before': body[:600], 'after': new[:600]})
        return '<|tool_call>call:' + name + '{' + new + '}<tool_call|>'

    out = CALL.sub(fix, text)
    if changed and log_path:
        try:
            with open(log_path, 'a') as f:
                for c in changed:
                    f.write(json.dumps({'t': round(time.time(), 1), **c}) + '\n')
        except OSError:
            pass
    return out, len(changed)


try:  # the vLLM part: only where vLLM is installed (the serving kernel)
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    from vllm.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

    class Gemma4TolerantToolParser(Gemma4ToolParser):
        """vLLM's Gemma 4 parser on the rewritten text (non-streaming requests; the harness does not stream)."""

        def extract_tool_calls(self, model_output, request):
            if self.tool_call_start_token in (model_output or ''):
                model_output, _ = rewrite_output(model_output, tool_params(getattr(request, 'tools', None)),
                                                 os.environ.get('G4DEV_TOLERANT_LOG'))
            return super().extract_tool_calls(model_output, request)

    ToolParserManager.register_module(name='gemma4_tolerant', module=Gemma4TolerantToolParser)
    if os.environ.get('G4DEV_TOLERANT_LOG'):  # proof that the serving process loaded the plugin
        try:
            with open(os.environ['G4DEV_TOLERANT_LOG'], 'a') as _f:
                _f.write(json.dumps({'t': round(time.time(), 1), 'event': 'registered', 'pid': os.getpid()}) + '\n')
        except OSError:
            pass
except ImportError:  # CPU tests and the offline analysis
    pass
