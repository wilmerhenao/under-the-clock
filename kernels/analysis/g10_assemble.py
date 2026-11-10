"""Assemble the G10 training dataset (private Kaggle dataset g4dev-g10-data) from the g4dev-g10-mix-cpu output.

Input: <mix>/train.jsonl (chain 0: every segment of every run, one line each, `part` 0 = up to the first compaction) and
<mix>/train_chains.jsonl (chain 1: the same runs with an independent Gemma summary at every compaction).
Output: <out>/sft.jsonl, one segment per line with `source` 'onpolicy/pre' (part 0) or 'onpolicy/post' (after a
compaction; chain 0 and chain 1); lines with no trained span, or longer than MAX_TOKENS, are left out and counted.
Also <out>/stats.json and a README.
usage: python3 kernels/analysis/g10_assemble.py <mix dir> <out dir> [--version-note TEXT]
"""
import collections
import json
import sys
from pathlib import Path

MAX_TOKENS = 24576  # the longest TPU bucket
KEEP = ('instance_id', 'repo', 'ref', 'text', 'spans', 'n_tokens', 'n_trainable', 'kind', 'part', 'chain', 'calls', 'resolved', 'pair_kind', 'pair_side', 'pair_cost')

README = """# G10 on-policy segments (private)

Verified runs of Gemma 4 31B on repository-level coding tasks, under the competition harness's prompt, tools and
budget, shown the way the harness shows them around context compaction: one line per segment. `source` is
`onpolicy/pre` for a run's first segment (up to its first compaction) and `onpolicy/post` for each later segment
(`For context: [model] said: <summary>` + the retained calls + what the agent did next). Summaries were written by
Gemma 4 31B from the harness's summariser prompt; `chain` 0 and 1 are two independent summaries of the same
compaction (chain 1 only for post-compaction segments). `spans` are the trained character ranges: tool-call turns that
open with a thought, excluding retained turns already trained in the previous segment.

Fields: instance_id, repo, ref, source, chain, part, text, spans, n_tokens, n_trainable, kind, calls.
"""


def main():
    mix, out = Path(sys.argv[1]), Path(sys.argv[2])
    note = sys.argv[sys.argv.index('--version-note') + 1] if '--version-note' in sys.argv else ''
    out.mkdir(parents=True, exist_ok=True)
    st = collections.Counter()
    runs, post_runs = set(), set()
    tokens = collections.Counter()
    with open(out / 'sft.jsonl', 'w') as fo:
        for name, chain_default in (('train.jsonl', 0), ('train_chains.jsonl', None)):
            f = mix / name
            if not f.exists():
                st[f'missing:{name}'] += 1
                continue
            for line in open(f):
                if not line.strip():
                    continue
                x = json.loads(line)
                if x.get('kind', 'segment') != 'segment':
                    st['skip_summary_kind'] += 1
                    continue
                part = int(x.get('part', 0))
                chain = int(x.get('chain', chain_default if chain_default is not None else 0))
                if chain_default is None and part == 0:
                    st['skip_chain_pre_duplicate'] += 1  # identical to chain 0's first segment
                    continue
                if not x.get('spans'):
                    st['skip_no_trained_span'] += 1
                    continue
                if x['n_tokens'] > MAX_TOKENS:
                    st['skip_over_max_tokens'] += 1
                    continue
                src = 'onpolicy/pre' if part == 0 else 'onpolicy/post'
                row = {k: x[k] for k in KEEP if k in x}
                row.update({'source': src, 'chain': chain, 'part': part})
                fo.write(json.dumps(row) + '\n')
                st[f'lines:{src}:chain{chain}'] += 1
                tokens[src] += x['n_tokens']
                runs.add(x['instance_id'])
                if part:
                    post_runs.add(x['instance_id'])
    stats = {'counts': dict(sorted(st.items())), 'tokens': dict(tokens), 'runs': len(runs),
             'runs_with_post_segments': len(post_runs), 'max_tokens': MAX_TOKENS, 'note': note}
    (out / 'stats.json').write_text(json.dumps(stats, indent=1))
    (out / 'README.md').write_text(README)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
