# %% [markdown]
# ### The role of every token (score-v4, B202)
#
# What kind of text the model is about to write at each position of an agent example:
#
# * **0 read:** everything the environment wrote (the prompt, the issue, tool output) and the chat template's markers
# * **1 thought:** the agent's own prose and thoughts (inside its turns, outside its tool calls)
# * **2 args:** argument values inside its tool calls (string contents, numbers, true/false, lists)
# * **3 syntax:** the rest of a call: `<|tool_call>`, `call:NAME{`, keys, colons, commas, braces, the quote that
#   opens a string `<|"|>`, `<tool_call|>`, and the `<|tool_response>` stop marker the agent writes after a call
# * **4 close:** the quote `<|"|>` that closes a string argument. If the model misses it, the string swallows the
#   next argument (`a.py",start_line:350` becomes one file name) and the call breaks: the malformed calls of the cut
#   models (B200).

# %%
import re
import numpy as np

ROLE_NAMES = ('read', 'thought', 'args', 'syntax', 'close')
Q = '<|"|>'
CALL_OPEN, CALL_END, RESP_OPEN = '<|tool_call>', '<tool_call|>', '<|tool_response>'
MODEL_TURN = re.compile(r'<\|turn>model\n(.*?)(?=<\|turn>|\Z)', re.S)      # as build_sft.trainable_spans
TOOL_RESP = re.compile(r'<\|tool_response>.*?<tool_response\|>', re.S)
KEY = re.compile(r'\s*[A-Za-z_]\w*\s*:\s*')


def _call_roles(role, text, s, e):
    """Roles inside one tool call text[s:e] ('<|tool_call>call:NAME{...}<tool_call|>', or an unclosed call up to e)."""
    role[s:e] = 3
    masked = list(text[s:e])          # string arguments blanked out, so commas and braces inside them do not count
    i = text.find(Q, s, e)
    while i >= 0:
        j = text.find(Q, i + len(Q), e)
        role[i + len(Q):(j if j >= 0 else e)] = 2
        end = j + len(Q) if j >= 0 else e
        if j >= 0:
            role[j:end] = 4
        masked[i - s:end - s] = ['\0'] * (end - i)
        if j < 0:
            break
        i = text.find(Q, end, e)
    body = ''.join(masked)
    p = body.find('{')
    if p < 0:
        return
    p += 1
    while p < len(body):              # key: value pairs; a value that is not a string is an argument value too
        km = KEY.match(body, p)
        if not km:
            return
        p = km.end()
        if p < len(body) and body[p] == '\0':
            while p < len(body) and body[p] == '\0':
                p += 1
        else:
            depth, k = 0, p
            while k < len(body):
                ch = body[k]
                if ch in '{[':
                    depth += 1
                elif ch in '}]':
                    if depth == 0:
                        break
                    depth -= 1
                elif ch == ',' and depth == 0:
                    break
                k += 1
            for t in range(p, k):
                if body[t] != '\0':
                    role[s + t] = 2
            p = k
        if p < len(body) and body[p] == ',':
            p += 1
        else:
            return


def char_roles(text):
    """Role (0-4, ROLE_NAMES) of every character of an agent example."""
    role = np.zeros(len(text), np.int8)
    for m in MODEL_TURN.finditer(text):
        s, e = m.start(1), m.end(1)
        segs, cursor = [], s
        for tr in TOOL_RESP.finditer(text, s, e):
            if tr.start() > cursor:   # model text, then the stop marker it wrote; back-to-back responses (parallel
                segs.append((cursor, tr.start()))   # calls) are all the environment's
            cursor = tr.end()
        if e > cursor:
            segs.append((cursor, e))
        for a, b in segs:
            role[a:b] = 1
            k = text.find(CALL_OPEN, a, b)
            while k >= 0:
                c = text.find(CALL_END, k, b)
                stop = c + len(CALL_END) if c >= 0 else b
                _call_roles(role, text, k, stop)
                k = text.find(CALL_OPEN, stop, b)
            if b < e and text.startswith(RESP_OPEN, b):
                role[b:b + len(RESP_OPEN)] = 3
    return role


def position_roles(starts, role, n):
    """Role of the token that each of the first n positions predicts (position t -> token t + 1; -1: no next token).
    starts: where each token of the whole example begins (character offsets); role: char_roles of its text."""
    starts = np.asarray(starts, dtype=np.int64)
    out = np.full(n, -1, np.int64)
    m = min(n, len(starts) - 1)
    if m > 0 and len(role):
        out[:m] = role[np.clip(starts[1:m + 1], 0, len(role) - 1)]
    return out
