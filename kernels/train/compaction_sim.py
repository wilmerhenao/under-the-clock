"""The scorer's view of a conversation under ADK 1.36.1 token-threshold compaction, for training data.

swegemma's `scripts/inference.py` sets `EventsCompactionConfig(compaction_interval=5, overlap_size=2,
token_threshold=14336, event_retention_size=5)`. Before every model call ADK reads the prompt tokens of the latest
model response; at >= 14,336 it summarizes every event since the last compaction except the last 5 (moved earlier
so that no retained tool result loses its call: 3 call/result pairs in practice) with a tool-less request to the same
model (`apps/llm_event_summarizer.py`): a fixed instruction + one `author: text` line per text part (task message,
the agent's thoughts and visible text; tool calls and tool results are dropped), joined by a literal backslash-n.
A later compaction starts from `model: <previous summary>`. The agent then sees the system prompt, one user message
`For context:` + `[model] said: <summary>` (`flows/llm_flows/contents.py`), and the retained events.
Validated on the captured requests of `g4dev-format-parity-cpu` (summary prompts byte-identical, post-compaction
messages identical, trigger at the first request >= 14,336 tokens).

Summaries in training rows are made by `summarize()` from the summarizer's own input only (the model writes them at
deployment): task line with repository and budget, the problem statement (shortened), files mentioned, the latest notes.
"""
import re

PREFIX = ('The following is a conversation history between a user and an AI agent. Please summarize the conversation, '
          'focusing on key information and decisions made, as well as any unresolved questions or tasks. The summary '
          'should be concise and capture the essence of the interaction.\\n\\n')
THRESHOLD = 14336
RETAIN = 5
AGENT = 'swe_agent'
PS_CHARS = 1200
NOTES = 5
NOTE_CHARS = 200
FILES = 8
FINDINGS = True  # notes keep the sentences that state findings (mix5 on); False = first 200 characters (mix4)
PATH_RX = re.compile(r'(?<![\w/.-])((?:/workspace/)?(?:[\w.-]+/)*[\w-]+\.(?:py|pyi|pyx|cfg|toml|ini|txt|rst|md|json|yaml|yml|'
                     r'c|h|cc|cpp|js|ts|sh))(?![\w/])')


def text_of(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    return ''.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type', 'text') == 'text')


def to_events(conv):
    """Messages after the system message -> ADK-like events: a user message, one model response (consecutive assistant
    messages, e.g. a text message then a tool-call message, are one response and one model turn in the template), or the
    tool messages answering it (parallel responses are one event). Calls without ids get positional ids."""
    events, k = [], 0
    for m in conv:
        role = m['role']
        if role in ('tool', 'assistant') and events and events[-1]['kind'] == role:
            events[-1]['msgs'].append(m)
            continue
        events.append({'kind': role, 'msgs': [m]})
    pending = []
    for ev in events:
        if ev['kind'] == 'assistant':
            ids = []
            for tc in [tc for m in ev['msgs'] for tc in (m.get('tool_calls') or [])]:
                ids.append(tc.get('id') or f'call_{k}')
                k += 1
            ev['calls'], ev['resps'] = set(ids), set()
            pending = ids
        elif ev['kind'] == 'tool':
            ev['calls'] = set()
            ev['resps'] = {m.get('tool_call_id') or (pending[j] if j < len(pending) else f'orphan_{j}')
                           for j, m in enumerate(ev['msgs'])}
        else:
            ev['calls'], ev['resps'] = set(), set()
    return events


def event_lines(ev):
    """Text parts as the summarizer formats them: thought first, then visible text (ADK's LiteLLM conversion)."""
    if ev['kind'] == 'user':
        t = text_of(ev['msgs'][0].get('content'))
        return [f'user: {t}'] if t else []
    out = []
    if ev['kind'] == 'assistant':
        for m in ev['msgs']:
            r = m.get('reasoning') or m.get('reasoning_content')
            if r:
                out.append(f'{AGENT}: {r}')
            t = text_of(m.get('content'))
            if t:
                out.append(f'{AGENT}: {t}')
    return out


def response_starts(msgs):
    """Indices of the assistant messages that start a model response (one request each)."""
    return [i for i, m in enumerate(msgs) if m['role'] == 'assistant' and (i == 0 or msgs[i - 1]['role'] != 'assistant')]


def safe_split(cands, retain=RETAIN):
    """ADK `_safe_token_compaction_split_index`: the latest split <= len - retain that orphans no retained response."""
    initial = len(cands) - retain
    if initial <= 0:
        return 0
    unmatched = set()
    for i in range(len(cands) - 1, -1, -1):
        unmatched |= cands[i]['resps']
        unmatched -= cands[i]['calls']
        if not unmatched and i <= initial:
            return i
    return 0


def prompt_of(lines):
    """The summarizer request text for these input lines (what a summarize_fn's model would be sent)."""
    return PREFIX + '\\n'.join(lines)


def summary_prompt(seed, to_compact):
    lines = ([f'model: {seed}'] if seed else []) + [x for ev in to_compact for x in event_lines(ev)]
    return prompt_of(lines), lines


def for_context(summary):
    return {'role': 'user', 'content': [{'type': 'text', 'text': 'For context:'},
                                        {'type': 'text', 'text': f'[model] said: {summary}'}]}


def flat(events):
    return [m for ev in events for m in ev['msgs']]


def simulate(system_msg, conv, counts_fn, summarize_fn, threshold=THRESHOLD, retain=RETAIN):
    """-> list of ('segment', messages, n_old_assistant) and ('summary', prompt, summary) in order.

    counts_fn(messages) -> prompt tokens of the request for each model response in `messages` (`response_starts`), or
    None. A segment's first n_old_assistant responses were already trained in the previous segment (retained)."""
    events = to_events(conv)
    out, seed, start, n_old = [], None, 0, 0
    while True:
        ctx_events = events[start:]
        head = [system_msg] + ([for_context(seed)] if seed else [])
        counts = counts_fn(head + flat(ctx_events))
        if counts is None:
            return None
        a_pos = [start + i for i, ev in enumerate(ctx_events) if ev['kind'] == 'assistant']
        if len(counts) != len(a_pos):
            return None
        cut = None
        for i in range(n_old + 1, len(a_pos)):  # checks before each new call read the previous new request's tokens
            if counts[i - 1] < threshold:
                continue
            cands = events[start:a_pos[i]]
            split = safe_split(cands, retain) if len(cands) > retain else 0
            if split > 0:
                cut = (a_pos[i], split)
                break
        if cut is None:
            out.append(('segment', head + flat(ctx_events), n_old))
            return out
        j, split = cut
        out.append(('segment', head + flat(events[start:j]), n_old))
        prompt, lines = summary_prompt(seed, events[start:start + split])
        new = summarize_fn(lines)
        out.append(('summary', prompt, new))
        seed, start = new, start + split
        n_old = sum(1 for ev in events[start:j] if ev['kind'] == 'assistant')


# ------------------------------------------------------------------ deterministic summaries
def _clip(t, n):
    t = ' '.join(t.split())
    if len(t) <= n:
        return t
    cut = t[:n]
    j = max(cut.rfind('. '), cut.rfind('? '), cut.rfind('! '), cut.rfind(': '))
    return cut[:j + 1] if j >= n // 3 else cut.rstrip() + '...'


FIND_RX = re.compile(r'(\.py:\d+|\bline \d+|\bthe (?:bug|issue|problem|cause|fix)\b|root cause|\b(?:fails?|failed|failing|'
                     r'passe[sd]|works|error|exception|traceback|assert\w*)\b|\binstead of\b|\bbecause\b|\bfound\b|\bfixed\b)', re.I)
INTENT_RX = re.compile(r"^(?:now,? |ok(?:ay)?,? |great,? )?(?:let me|let's|i'll|i will|i need to|i should|next,? i|first,? i|now i)\b",
                       re.I)


def _note(t, n):
    """The sentences of an agent message that state findings (file:line, bug/cause, test results, errors), in order, up to
    n characters; the plain clip when no sentence states one (pure intentions: "Let me look at ...")."""
    t = ' '.join(t.split())
    sents = [x for x in re.split(r'(?<=[.!?])\s+', t) if x]
    scored = []
    for i, x in enumerate(sents):
        sc = 2 * len(FIND_RX.findall(x)) + x.count('`') // 2 + (1 if PATH_RX.search(x) else 0) - (2 if INTENT_RX.match(x) else 0)
        if sc > 0:
            scored.append((sc, i, x))
    if not scored:
        return _clip(t, n)
    chosen, total = [], 0
    for sc, i, x in sorted(scored, key=lambda y: (-y[0], y[1])):
        if chosen and total + len(x) + 1 > n:
            break
        chosen.append((i, x))
        total += len(x) + 1
    out = ' '.join(x for _, x in sorted(chosen))
    return out if len(out) <= n else _clip(out, n)


def _shorten_block(t, n):
    t = t.strip()
    if len(t) <= n:
        return t, False
    cut = t[:n]
    j = cut.rfind('\n')
    return (cut[:j] if j >= n // 2 else cut).rstrip() + '\n[...]', True


def parse_summary(s):
    """Our own summary format -> (head, problem, files, notes)."""
    head = s.split('\n', 1)[0]
    m = re.search(r'\nProblem statement[^\n]*:\n(.*?)\n(?:Files mentioned so far: |My notes so far)', s, re.S)
    problem = m.group(1) if m else ''
    f = re.search(r'\nFiles mentioned so far: ([^\n]*)', s)
    files = [x for x in (f.group(1).split(', ') if f else []) if x]
    n = s.split('\nMy notes so far, oldest first:\n', 1)
    notes = [x[2:] for x in n[1].split('\n') if x.startswith('- ')] if len(n) == 2 else []
    return head, problem, files, notes


def summarize(lines):
    """Summary from the summarizer's input lines only (what the model sees when it writes it)."""
    head, problem, files, notes, shortened = None, '', [], [], False
    texts = []
    for x in lines:
        author, _, text = x.partition(': ')
        if author == 'model' and head is None:
            head, problem, files, notes = parse_summary(text)
        elif author == 'user' and head is None:
            repo = re.search(r'for repository (\S+?)\.\n', text)
            ps = re.search(r'Problem Statement:\n(.*?)\n\n## Task Budget', text, re.S)
            tm = re.search(r'Time allowance: ([^\n]+)', text)
            cl = re.search(r'Tool calls allowance: ([^\n]+)', text)
            head = (f'Task: resolve the issue in repository {repo.group(1) if repo else "(unknown)"} (code in /workspace)'
                    + (f'; budget {tm.group(1)}, {cl.group(1)}' if tm and cl else '') + '.')
            problem, shortened = _shorten_block(ps.group(1) if ps else text, PS_CHARS)
        elif author == AGENT:
            texts.append(text)
    for t in texts:
        for p in PATH_RX.findall(t):
            p = p[len('/workspace/'):] if p.startswith('/workspace/') else p
            if p in files:
                files.remove(p)
            files.append(p)
        c = _note(t, NOTE_CHARS) if FINDINGS else _clip(t, NOTE_CHARS)
        if c and (not notes or c != notes[-1]):
            notes.append(c)
    files, notes = files[-FILES:], notes[-NOTES:]
    out = [head or 'Task: resolve the issue in the repository (code in /workspace).',
           'Problem statement' + (' (shortened)' if shortened or problem.endswith('[...]') else '') + ':', problem]
    if files:
        out.append('Files mentioned so far: ' + ', '.join(files))
    out.append('My notes so far, oldest first:')
    out += [f'- {n}' for n in notes] or ['- (none yet)']
    return '\n'.join(out)
