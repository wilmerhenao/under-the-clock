"""Convert swegemma harness traces (ATIF SessionTrace JSON) into Gemma 4 SFT examples.

Each resolved rollout becomes one example: the OpenAI-style message list the agent actually exchanged
(system prompt, task prompt, assistant turns with `reasoning` + `tool_calls`, tool results), rendered with the
served model's own chat template (tools + enable_thinking=True), plus character spans of the tokens the
model produced. Only those spans are trained on; system/user text and tool-response bodies are masked.

Trace layout (adk-eval-core 1.0): an agent step with N tool calls stores the first result inline in
`observation` and the remaining N-1 as following `system` steps (each tagged with tool_name).
"""
import argparse
import glob
import json
import re
from pathlib import Path

SANDBOX_RE = re.compile(r'/tmp/swegemma_sandbox_[A-Za-z0-9_-]+/(workspace|tmp)')


def clean(text):
    """Map the subprocess sandbox's host paths back to the paths the agent (and the official scorer) uses."""
    return SANDBOX_RE.sub(lambda m: '/' + m.group(1), text) if isinstance(text, str) else text


def trace_to_messages(trace):
    steps = trace['steps']
    msgs, pending_reasoning = [], []
    i = 0
    while i < len(steps):
        s = steps[i]
        src, ev = s['source'], (s.get('extra') or {}).get('event_type')
        if src == 'system' and ev == 'system_instruction':
            msgs.append({'role': 'system', 'content': s.get('message', '')})
        elif src == 'user':
            msgs.append({'role': 'user', 'content': clean(s.get('message', ''))})
        elif src == 'system' and ev == 'continuation_nudge':
            msgs.append({'role': 'user', 'content': s.get('message', '')})
        elif src == 'agent' and s.get('tool_calls'):
            calls = s['tool_calls']
            reasoning = '\n\n'.join([*pending_reasoning, s.get('message') or '']).strip()
            pending_reasoning = []
            msg = {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': c['tool_call_id'], 'type': 'function',
                 'function': {'name': c['function_name'], 'arguments': c.get('arguments') or {}}} for c in calls]}
            if reasoning:
                msg['reasoning'] = reasoning
            msgs.append(msg)
            observations = [s.get('observation')] if s.get('observation') else []
            j = i + 1
            while len(observations) < len(calls) and j < len(steps) and steps[j]['source'] == 'system' \
                    and steps[j].get('observation'):
                observations.append(steps[j]['observation'])
                j += 1
            for c, obs in zip(calls, observations):
                msgs.append({'role': 'tool', 'tool_call_id': c['tool_call_id'], 'name': c['function_name'],
                             'content': clean((obs or {}).get('content', ''))})
            for c in calls[len(observations):]:  # result never recorded (e.g. session ended mid-turn)
                msgs.append({'role': 'tool', 'tool_call_id': c['tool_call_id'], 'name': c['function_name'],
                             'content': ''})
            i = j - 1
        elif src == 'agent' and ev == 'thinking':
            if s.get('message'):
                pending_reasoning.append(s['message'])
        elif src == 'agent' and ev in ('text', 'final'):
            if s.get('message'):
                msgs.append({'role': 'assistant', 'content': clean(s['message'])})
        i += 1
    return msgs


MODEL_TOOL_RESPONSE = re.compile(r'<\|tool_response>.*?<tool_response\|>', re.S)


def trainable_spans(text):
    """Character spans the model generated: every `<|turn>model` turn, minus tool-response bodies.

    The `<|tool_response>` opening token is the model's stop marker (it ends the tool call), so it stays
    trainable; everything after it up to and including `<tool_response|>` came from the environment.
    """
    spans = []
    for m in re.finditer(r'<\|turn>model\n(.*?)(?=<\|turn>|\Z)', text, re.S):
        start, end = m.start(1), m.end(1)
        cursor = start
        for tr in MODEL_TOOL_RESPONSE.finditer(text, start, end):
            # Only the marker right after model text is generated (it's the stop token after the last tool
            # call); back-to-back response blocks for parallel calls are written by the environment.
            if tr.start() > cursor:
                spans.append((cursor, tr.start() + len('<|tool_response>')))
            cursor = tr.end()
        if end > cursor:
            spans.append((cursor, end))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='results dirs (traces/ + task_results.jsonl), optionally "DIR::LABELS.jsonl" to take '
                         '`resolved` from a re-verification of that run\'s patches')
    ap.add_argument('--split', required=True)
    ap.add_argument('--subset', default='dev')
    ap.add_argument('--tokenizer', required=True, help='dir with tokenizer.json, tokenizer_config.json, chat_template.jinja')
    ap.add_argument('--tools', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=32768)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tok.chat_template = Path(a.tokenizer, 'chat_template.jinja').read_text()
    tools = json.load(open(a.tools))
    keep = set(json.load(open(a.split))[a.subset])

    examples, skipped = [], []
    for spec in a.runs:
        rdir, _, label_file = spec.partition('::')
        rdir = Path(rdir)
        label_file = Path(label_file) if label_file else rdir / 'task_results.jsonl'
        labels = {json.loads(line)['instance_id']: json.loads(line) for line in open(label_file) if line.strip()}
        source = f'{rdir.parent.parent.name}/{rdir.parent.name}'
        for iid, row in sorted(labels.items()):
            if iid not in keep or not row.get('resolved'):
                continue
            tf = rdir / 'traces' / f'trace_{iid}.json'
            if not tf.exists():
                skipped.append((iid, source, 'no trace'))
                continue
            msgs = trace_to_messages(json.load(open(tf)))
            if not any(m['role'] == 'assistant' and m.get('tool_calls') for m in msgs):
                skipped.append((iid, source, 'no tool calls'))
                continue
            text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=True)
            n_tokens = len(tok(text, add_special_tokens=False)['input_ids'])
            if n_tokens > a.max_tokens:
                skipped.append((iid, source, f'{n_tokens} tokens'))
                continue
            spans = trainable_spans(text)
            examples.append({'instance_id': iid, 'source': source, 'n_tokens': n_tokens,
                             'n_trainable_chars': sum(e - s for s, e in spans), 'n_chars': len(text),
                             'text': text, 'spans': spans})
    with open(a.out, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    print(json.dumps({'examples': len(examples), 'unique_tasks': len({e['instance_id'] for e in examples}),
                      'tokens_total': sum(e['n_tokens'] for e in examples),
                      'tokens_max': max((e['n_tokens'] for e in examples), default=0),
                      'trainable_char_frac': round(sum(e['n_trainable_chars'] for e in examples)
                                                   / max(1, sum(e['n_chars'] for e in examples)), 3),
                      'by_source': {s: sum(e['source'] == s for e in examples) for s in {e['source'] for e in examples}},
                      'skipped': skipped}, indent=1))


if __name__ == '__main__':
    main()
