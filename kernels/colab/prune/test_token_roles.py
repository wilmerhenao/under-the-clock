#!/usr/bin/env python3
"""Unit test of token_roles (score-v4, B202): the role of every character of a synthetic Gemma 4 agent example.
usage: python3 kernels/colab/prune/test_token_roles.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_roles import ROLE_NAMES, Q, char_roles, position_roles  # noqa: E402

R = {n: i for i, n in enumerate(ROLE_NAMES)}
parts = []          # (text, expected role name per character, or a list of (text, role) pieces)


def add(text, role):
    parts.append((text, role))


add('<bos><|turn>system\nYou are a coding agent.<turn|>\n<|turn>user\nFix the bug in a.py<turn|>\n<|turn>model\n', 'read')
add('<|channel>thought\nLet me read it, then {edit} it.<channel|>', 'thought')
add('<|tool_call>call:read_file{filepath:' + Q, 'syntax')
add('a.py', 'args')
add(Q, 'close')
add(',start_line:', 'syntax')
add('3', 'args')
add(',end_line:', 'syntax')
add('40', 'args')
add('}<tool_call|>', 'syntax')
add('<|tool_response>', 'syntax')                        # the stop marker the agent writes after its call
add('response:read_file{output:' + Q + 'def f(x, y): {pass}' + Q + '}<tool_response|>', 'read')
add('Now edit.', 'thought')
add('<|tool_call>call:edit_file{filepath:' + Q, 'syntax')
add('a.py', 'args')
add(Q, 'close')
add(',old_string:' + Q, 'syntax')
add('x, y}', 'args')                                      # a comma and a brace inside a string are part of the value
add(Q, 'close')
add(',new_string:' + Q, 'syntax')
add('z', 'args')
add(Q, 'close')
add(',allow_multiple:', 'syntax')
add('false', 'args')
add('}<tool_call|><|tool_call>call:get_code_subgraph{nodes:', 'syntax')
add('[', 'args')
add(Q, 'syntax')
add('a', 'args')
add(Q, 'close')
add(',', 'args')                                          # inside a list value
add(Q, 'syntax')
add('b', 'args')
add(Q, 'close')
add(']', 'args')
add('}<tool_call|>', 'syntax')
add('<|tool_response>', 'syntax')
add('response:edit_file{ok:true}<tool_response|><|tool_response>response:get_code_subgraph{}<tool_response|>', 'read')
add('Done.<turn|>\n', 'thought')                          # back-to-back responses above: the second marker is theirs
add('<|turn>user\nContinue. <|tool_call>call:x{a:' + Q + 'not the agent' + Q + '}<tool_call|><turn|>\n', 'read')
add('<|turn>model\n', 'read')
add('<|tool_call>call:run_command{command:' + Q, 'syntax')
add('ls -la', 'args')                                     # unclosed string and call at the end of the example

text = ''.join(t for t, _ in parts)
want = np.concatenate([np.full(len(t), R[r], np.int8) for t, r in parts])
got = char_roles(text)
assert got.shape == want.shape and got.dtype == np.int8
bad = np.nonzero(got != want)[0]
assert not len(bad), [(int(i), text[max(0, i - 20):i + 20], ROLE_NAMES[got[i]], ROLE_NAMES[want[i]]) for i in bad[:5]]
assert int((got == R['close']).sum()) == 6 * len(Q)       # six closed strings by the agent; the user's one is read

# position t takes the role of token t + 1; the last position has no next token
role = np.array([0, 0, 1, 1, 4, 3], np.int8)
assert position_roles([0, 2, 4, 5], role, 4).tolist() == [1, 4, 3, -1]
assert position_roles([0, 2, 4, 5], role, 2).tolist() == [1, 4]          # truncated: the next token still counts
assert position_roles([0], role, 1).tolist() == [-1] and char_roles('').shape == (0,)
assert (char_roles('no chat markup at all') == 0).all()
print('TOKEN_ROLES OK', {n: int((got == i).sum()) for n, i in R.items()})
