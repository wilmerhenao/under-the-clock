"""Build the G11 training rows (private Kaggle dataset g4dev-g11-data) from the G10 rows (g4dev-g10-data v2).

G10 trained the tool call of every turn that opens with a thought (TRAIN_ON 'actions': the loss is on the call body,
the thought is context; B154 corrects earlier notes that said thoughts were trained). 16% of those turns have thoughts
over 4,000 chars and carry 59% of all thought chars (p90 6,015 chars vs the base model's ~1,300-char per-step text p90
on the canaries, B91). G10a came out with 1.58x
longer per-step text and more timeouts on dev. G11 keeps a turn trained only if its thought is at most MAX_THOUGHT
chars; longer turns stay in the context, untrained. Rows left with no trained turn are dropped.
Runs that resolved within FAST_CALLS tool calls get '/fast' appended to their source, for the trainer's arm b
(PHASE2_SOURCE = '/fast').
usage: python3 kernels/analysis/g11_data.py <g10 sft.jsonl> <out dir>
"""
import collections
import json
import re
import sys
from pathlib import Path

MAX_THOUGHT = 1500
FAST_CALLS = 14  # the median run of the G10 data
THOUGHT = re.compile(r'<\|channel>thought\n(.*?)<channel\|>', re.S)

README = """# G11 segments (private)

The G10 on-policy segments (verified Gemma 4 31B runs, shown as the harness shows them around compaction) with the
trained turns limited to those whose thought is at most %d characters; longer turns are context only. `source` gains
`/fast` for runs that resolved the task within %d tool calls. Fields as in g4dev-g10-data.
""" % (MAX_THOUGHT, FAST_CALLS)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    st, kept_th, all_th = collections.Counter(), [], []
    with open(out / 'sft.jsonl', 'w') as fo:
        for line in open(src):
            if not line.strip():
                continue
            r = json.loads(line)
            spans = []
            for a, b in r['spans']:
                m = THOUGHT.match(r['text'], a)
                n = len(m.group(1)) if m and m.end() <= b else 0
                all_th.append(n)
                if n <= MAX_THOUGHT:
                    spans.append([a, b])
                    kept_th.append(n)
            st['turns_all'] += len(r['spans'])
            st['turns_kept'] += len(spans)
            old = sum(b - a for a, b in r['spans'])
            new = sum(b - a for a, b in spans)
            st['span_chars_all'] += old
            st['span_chars_kept'] += new
            if not spans:
                st['rows_dropped_no_turn'] += 1
                continue
            r['n_trainable'] = round(r.get('n_trainable', 0) * new / max(old, 1))  # estimate: kept share of the chars
            r['spans'] = spans
            if r.get('calls', 99) <= FAST_CALLS:
                r['source'] += '/fast'
            st['rows:' + r['source']] += 1
            st['tokens:' + r['source']] += r['n_tokens']
            fo.write(json.dumps(r) + '\n')

    def q(v):
        v = sorted(v)
        return {p: v[min(len(v) - 1, len(v) * p // 100)] for p in (50, 75, 90, 99)} if v else {}
    stats = {'max_thought': MAX_THOUGHT, 'fast_calls': FAST_CALLS, 'counts': dict(sorted(st.items())),
             'thought_chars_all': q(all_th), 'thought_chars_kept': q(kept_th)}
    (out / 'stats.json').write_text(json.dumps(stats, indent=1))
    (out / 'README.md').write_text(README)
    (out / 'dataset-metadata.json').write_text(json.dumps(
        {'title': 'g4dev-g11-data', 'id': 'verracodeguacas/g4dev-g11-data', 'licenses': [{'name': 'other'}]}))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
