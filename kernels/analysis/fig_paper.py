"""Paper figures (B207 / paper v5): the teaser, the framework diagram and the result figures, drawn at the arXiv text
width (5.5 in) from the measured files, in the palette of fig_prune_speed.py (dataviz reference palette, light mode).
Writes PNG (300 dpi, for the Kaggle writeup) and PDF into docs/paper/figs/. With --twocol it writes the PDFs of the
two-column arXiv version into docs/paper/arxiv/figs/: the five data figures redrawn at the column width (3.25 in), the
teaser and the framework diagram as drawn (they span both columns).
usage: <python with matplotlib> kernels/analysis/fig_paper.py [--twocol] [name ...]   (default: all)
Sources: serve-v5 (B196) runs/prune-serve5-gpu/v1, serve-v6 (B204) runs/prune-serve6-gpu/v1, serve-v7 v2 (B206)
runs/prune-serve7-gpu/v2; the same-session task counts are B197 / B200 (measured; quoted below)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_next_action import first_call  # noqa: E402

OUT = 'docs/paper/figs'
V5, V6, V7 = 'runs/prune-serve5-gpu/v1', 'runs/prune-serve6-gpu/v1', 'runs/prune-serve7-gpu/v2'
W = 5.5                                    # single-column text width, inches (the teaser and framework canvases)
COL = 3.25                                 # two-column arXiv layout: column width, inches
TWO = '--twocol' in sys.argv
if TWO:
    OUT = 'docs/paper/arxiv/figs'
# palette (as fig_prune_speed.py): text and axis tokens, categorical slots 1-3
SURF, INK, INK2, MUTED, GRID, AXIS = '#ffffff', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
FULL = '#6f6e69'                           # the full model: neutral ink, not a series hue
OK_C, BAD_C = '#13804f', '#c0392b'         # check / cross marks, with the glyph (never colour alone)
TINT = {'state': '#fdf3d0', 'full': '#efeee9', 'cut': '#fde5d8', 'healed': '#dbe9fa', 'lever': '#dff3e8',
        'meas': '#f1f0ec', 'eq': '#ffffff'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 8.5,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 6,
                     'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'pdf.fonttype': 42})
MONO = {'family': 'DejaVu Sans Mono'}
CW = 0.602                                 # DejaVu Sans Mono advance, in em


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    exts = ('pdf',) if TWO else ('png', 'pdf')
    for ext in exts:
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('wrote', *(f'{OUT}/{name}.{ext}' for ext in exts))


def canvas(h):
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, fc, ec=AXIS, lw=0.8, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def arrow(ax, a, b, color=MUTED, lw=0.9, style='-|>', rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def mono_line(ax, x, y, parts, size):
    """Draw [(text, colour, highlight?)] left to right in the monospace font from x (inches) at baseline-centre y."""
    cw = CW * size / 72
    for text, colour, hl in parts:
        if hl:
            ax.add_patch(Rectangle((x - 0.005, y - 0.055), cw * len(text) + 0.01, 0.11, fc='#fbd9d3', ec='none'))
        ax.text(x, y, text, fontsize=size, color=colour, va='center', ha='left', fontdict=MONO)
        x += cw * len(text)
    return x


# ------------------------------------------------------------------------------------------------ teaser
def teaser():
    """One held-out agent state (serve-v5 state 98, the structlog repository), three models' greedy next calls."""
    i = 98
    calls = {n: first_call(json.load(open(f'{V5}/prune/serve-v5/bench_{n}.json'))['turns'][i])
             for n in ('full', 'c70', 'c70+kd~')}
    rec = json.load(open(f'{V5}/prune/serve-v5/report.json'))['turn_state_recorded_calls'][i]
    ptok = json.load(open(f'{V5}/prune/serve-v5/report.json'))['turn_state_prompt_tokens'][i]
    assert calls['full'] == calls['c70+kd~'] == rec.strip(), (calls, rec)
    q = '<|"|>'
    head = 'call:read_file{filepath:' + q
    assert calls['c70'].startswith(head + 'src/structlog/dev.py`'), calls['c70']
    h = 2.22
    fig, ax = canvas(h)
    # the state
    box(ax, 0.02, 0.11, 1.40, 2.00, TINT['state'], ec='#e3c766')
    ax.text(0.11, 1.98, 'Held-out agent state', fontsize=7.0, weight='bold', color=INK, va='top')
    ax.text(0.11, 1.74, 'structlog repository:\n'
                        f'the {ptok:,}-token transcript\nup to the next agent turn', fontsize=6.3, color=INK2,
            va='top', linespacing=1.4)
    ax.text(0.11, 1.03, "The agent's recorded\nnext call:", fontsize=6.3, color=INK2, va='top', linespacing=1.4)
    ax.text(0.11, 0.62, 'read_file(dev.py,\n  start_line=151)', fontsize=6.2, color=INK, va='top', fontdict=MONO,
            linespacing=1.4)
    rows = [('full', 'Full model', '4-bit weights', TINT['full'], '#cfcdc4', True, 1.86, 0.50),
            ('c70', '70% cut', 'keeps 70% of channels', TINT['cut'], '#f2b598', False, 1.13, 0.62),
            ('c70+kd~', '70% cut, healed', 'merged into 4 bits', TINT['healed'], '#a9c8f0', True, 0.36, 0.50)]
    speed = {'full': '1.00× decode', 'c70': '1.15× decode', 'c70+kd~': '1.15× decode'}
    s = 6.4
    for key, title, sub, fc, ec, ok, y, bh in rows:
        arrow(ax, (1.42, 1.11), (1.55, y), color=MUTED)
        box(ax, 1.55, y - 0.25, 1.12, 0.50, fc, ec=ec)
        ax.text(2.11, y + 0.12, title, fontsize=6.9, weight='bold', color=INK, ha='center', va='center')
        ax.text(2.11, y - 0.04, sub, fontsize=5.8, color=INK2, ha='center', va='center')
        ax.text(2.11, y - 0.16, speed[key], fontsize=5.8, color=INK2, ha='center', va='center')
        arrow(ax, (2.67, y), (2.80, y), color=MUTED)
        top = y + 0.25 if bh == 0.50 else y + 0.31
        box(ax, 2.80, top - bh, 2.36, bh, '#ffffff', ec=ec)
        l1 = top - 0.15
        mono_line(ax, 2.87, l1, [(head, INK, False)], s)
        if key == 'c70':
            tail = calls['c70'][len(head):]                       # src/structlog/dev.py`,start_line:210}
            k = tail.index('`')
            mono_line(ax, 2.87, l1 - 0.19, [(tail[:k], INK, False), (tail[k:], BAD_C, True)], s)
            ax.text(2.87, l1 - 0.38, 'backtick instead of <|"|>: start_line is swallowed',
                    fontsize=5.5, color=BAD_C, va='center')
        else:
            mono_line(ax, 2.87, l1 - 0.19, [(calls[key][len(head):], INK, False)], s)
        ax.text(5.33, y, '\u2713' if ok else '\u2717', fontsize=12, color=OK_C if ok else BAD_C, ha='center',
                va='center', weight='bold')
    save(fig, 'fig_teaser')


# --------------------------------------------------------------------------------------------- framework
def framework():
    h = 2.95
    fig, ax = canvas(h)
    # the agent loop under the clock
    y0, bh = 2.12, 0.50
    items = [(0.02, 1.02, 'Repository issue', 'pre-fix commit', TINT['state'], '#e3c766'),
             (1.30, 1.42, 'Agent: Gemma 4 31B', '4-bit, vLLM on 4 × L4', TINT['full'], '#cfcdc4'),
             (3.00, 1.18, 'Sandbox tools', 'shell, files, edit, graph', TINT['meas'], '#cfcdc4'),
             (4.46, 1.02, 'Hidden tests', 'grade the patch', TINT['state'], '#e3c766')]
    for x, w, t, s, fc, ec in items:
        box(ax, x, y0, w, bh, fc, ec=ec)
        ax.text(x + w / 2, y0 + 0.32, t, fontsize=7.0, weight='bold', color=INK, ha='center', va='center')
        ax.text(x + w / 2, y0 + 0.13, s, fontsize=6.3, color=INK2, ha='center', va='center')
    arrow(ax, (1.04, y0 + 0.25), (1.30, y0 + 0.25))
    arrow(ax, (2.72, y0 + 0.33), (3.00, y0 + 0.33))           # tool call
    arrow(ax, (3.00, y0 + 0.17), (2.72, y0 + 0.17))           # tool output
    ax.text(2.86, y0 + 0.43, 'call', fontsize=5.8, color=MUTED, ha='center')
    ax.text(2.86, y0 + 0.02, 'result', fontsize=5.8, color=MUTED, ha='center')
    arrow(ax, (4.18, y0 + 0.25), (4.46, y0 + 0.25))
    ax.text(4.32, y0 + 0.31, 'patch', fontsize=5.8, color=MUTED, ha='center')
    # the clock over the loop
    ax.plot([1.30, 4.18], [2.80, 2.80], color=INK2, lw=0.8)
    for x in (1.30, 4.18):
        ax.plot([x, x], [2.75, 2.85], color=INK2, lw=0.8)
    ax.text(2.74, 2.86, '8 minutes and 24 calls per task: every step spends the clock', fontsize=6.6, color=INK2,
            ha='center', va='bottom')
    # the step-time model
    box(ax, 1.30, 1.50, 2.88, 0.42, TINT['eq'], ec=INK2, lw=0.9)
    ax.text(3.28, 1.815, 'seconds per step  ≈  a  +  ', fontsize=8.2, color=INK, ha='right', va='center',
            weight='bold')
    for x, t, c in ((3.35, 'n', GREEN), (3.45, '·', INK), (3.55, 'b', GREEN)):
        ax.text(x, 1.815, t, fontsize=8.2, color=c, ha='center', va='center', weight='bold')
    ax.text(2.74, 1.655, 'a = 16.7 s per call,  n = tokens written,  b = 0.139 s per token', fontsize=6.0,
            color=INK2, ha='center', va='center')
    ax.text(2.74, 1.565, 'fit at 16 concurrent requests; one stream alone: b = 0.026 s', fontsize=5.4, color=MUTED,
            ha='center', va='center')
    arrow(ax, (2.01, 2.12), (2.01, 1.92), color=MUTED, style='-')
    # the two levers
    for x, title, lines, target in (
            (0.02, 'Lever 1: fine-tuning (Section 3)  →  n',
             ['LoRA on the agent\'s own verified runs', 'supervise calls made after short reasoning',
              'a teacher at the agent\'s decision points'], 3.08),
            (2.80, 'Lever 2: pruning (Section 4)  →  b',
             ['keep 70% or 55% of MLP channels, by agent scores', 'serve the cut in 4 bits',
              'heal: the full model teaches (KL); merge into 4 bits'], 3.37)):
        box(ax, x, 0.50, 2.68, 0.80, TINT['lever'], ec='#9fd6bb')
        ax.text(x + 0.10, 1.17, title, fontsize=7.0, weight='bold', color=INK, va='center')
        for k, line in enumerate(lines):
            ax.text(x + 0.10, 0.96 - 0.17 * k, '• ' + line, fontsize=6.2, color=INK2, va='center')
    arrow(ax, (2.10, 1.30), (2.10, 1.50), color=GREEN)    # lever 1 -> n (tokens per step)
    arrow(ax, (3.40, 1.30), (3.40, 1.50), color=GREEN)    # lever 2 -> b (time per token)
    # how it is measured
    box(ax, 0.02, 0.04, 5.46, 0.34, TINT['meas'], ec='#d7d5cc')
    ax.text(0.12, 0.21, 'Measured under noise (Section 5):', fontsize=6.6, weight='bold', color=INK, va='center')
    ax.text(1.98, 0.21, 'A/A floor  ·  paired sessions  ·  same-session comparisons  ·  next-action test',
            fontsize=6.4, color=INK2, va='center')
    save(fig, 'fig_framework')


FIGS = {'teaser': teaser, 'framework': framework}

# ------------------------------------------------------------------------------------------ data figures
def style(ax, xgrid=True):
    ax.set_facecolor(SURF)
    ax.grid(True, axis='x' if xgrid else 'y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, pad=3)


TASK_ROWS = [  # label, solved of 112, malformed calls % of calls, kind (measured: B197 same4, B200 same55)
    ('full model, run 1', 32, 2.3, 'full'), ('full model, run 2', 34, 3.3, 'full'),
    ('70% cut', 20, 24.2, 'cut'), ('70% cut, healed', 32, 3.0, 'healed'),
    ('full model, run 1', 37, 0.9, 'full'), ('full model, run 2', 33, 1.9, 'full'),
    ('55% cut', 13, 43.4, 'cut'), ('55% cut, healed', 27, 13.4, 'healed')]


def tasks():
    """Same-session comparisons on 112 unseen-repository tasks (8-minute clock, one model per server): solved tasks
    and malformed tool calls. Measured, B197 (same4) and B200 (same55)."""
    rows = TASK_ROWS
    col = {'full': FULL, 'cut': ORANGE, 'healed': BLUE}
    y = [8.6, 7.8, 7.0, 6.2, 3.6, 2.8, 2.0, 1.2]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw={'width_ratios': [1.25, 1], 'wspace': 0.06})
    for ax in (a, b):
        style(ax)
        ax.set_ylim(0.6, 9.6)
        ax.set_yticks(y)
    a.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    b.set_yticklabels([])
    for k, (lab, solved, mal, kind) in enumerate(rows):
        a.plot([0, solved], [y[k], y[k]], color=col[kind], lw=1.0, alpha=0.35, solid_capstyle='butt')
        a.scatter(solved, y[k], s=34, color=col[kind], zorder=3, edgecolors=SURF, linewidths=1.2)
        a.text(solved + 1.2, y[k], f'{solved}', fontsize=6.8, color=INK, va='center')
        b.barh(y[k], mal, height=0.5, color=col[kind])
        b.text(mal + 0.8, y[k], f'{mal:.1f}%', fontsize=6.8, color=INK, va='center')
    a.axvspan(32, 34, ymin=(5.8 - 0.6) / 9.0, ymax=(9.0 - 0.6) / 9.0, color=GRID, alpha=0.6, lw=0)
    a.axvspan(33, 37, ymin=(0.8 - 0.6) / 9.0, ymax=(4.0 - 0.6) / 9.0, color=GRID, alpha=0.6, lw=0)
    a.set_xlim(0, 44)
    b.set_xlim(0, 52)
    a.set_xlabel('tasks solved (of 112)')
    b.set_xlabel('malformed tool calls (% of calls)')
    a.set_title('Tasks solved in one session')
    b.set_title('Broken tool calls')
    for ax in (a,):
        ax.text(-0.5, 9.35, 'Session A: the 70% cut', fontsize=6.8, weight='bold', color=INK2, transform=ax.transData,
                ha='left')
        ax.text(-0.5, 4.35, 'Session B: the 55% cut', fontsize=6.8, weight='bold', color=INK2, transform=ax.transData,
                ha='left')
    save(fig, 'fig_tasks')


def nextaction():
    """serve-v5 (B196): 102 held-out agent states where the full model made a call."""
    n = json.load(open(f'{V5}/next_action.json'))['models']
    order = [('full_b', 'same weights, other batching', 'full'), ('full@T0.2', 'full model sampled at T 0.2', 'full'),
             ('c85', '85% cut', 'cut'), ('c70', '70% cut', 'cut'), ('c55', '55% cut', 'cut'),
             ('c70own', '70% cut, agent-token score', 'alt'), ('c70+kd~', '70% cut, healed', 'healed')]
    col = {'full': FULL, 'cut': ORANGE, 'healed': BLUE, 'alt': GREEN}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.06})
    ys = list(range(len(order)))[::-1]
    for ax in (a, b):
        style(ax)
        ax.set_yticks(ys)
        ax.set_ylim(-0.7, len(order) - 0.3)
    a.set_yticklabels([o[1] for o in order], fontsize=6.8)
    b.set_yticklabels([])
    fb = n['full_b']['vs_full']['parsed']
    a.axvspan(fb['ci95'][0], fb['ci95'][1], color=GRID, alpha=0.6, lw=0)
    a.text(fb['ci95'][1] - 0.01, -0.62, 'numerics floor', fontsize=5.8, color=MUTED, ha='right', va='bottom')
    for (key, lab, kind), yy in zip(order, ys):
        v = n[key]['vs_full']['parsed']
        a.plot(v['ci95'], [yy, yy], color=col[kind], lw=1.4, solid_capstyle='round')
        a.scatter(v['share'], yy, s=30, color=col[kind], zorder=3, edgecolors=SURF, linewidths=1.0)
        a.text(v['ci95'][1] + 0.015, yy, f"{v['share']:.2f}", fontsize=6.6, color=INK, va='center')
        m = n[key]['malformed']
        pct = 100 * m['malformed'] / m['calls']
        b.barh(yy, pct, height=0.5, color=col[kind])
        b.text(pct + 0.3, yy, f"{m['malformed']}/{m['calls']}", fontsize=6.6, color=INK, va='center')
    a.set_xlim(0, 1.0)
    a.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    b.set_xlim(0, 18)
    a.set_xlabel('same next call as the full model (95% interval)')
    b.set_xlabel('malformed next calls (%)')
    a.set_title('Agreement with the full model')
    b.set_title('Broken calls')
    save(fig, 'fig_nextaction')


def failure_moves():
    """serve-v6 (B204): the 55% cut by the plain score vs by the role score, 141 held-out states."""
    rt = json.load(open(f'{V6}/role_test.json'))['descriptive']['role_nats']
    ca = json.load(open(f'{V6}/call_attempts.json'))['failure_kinds']
    roles = ['read', 'thought', 'args', 'syntax', 'close']
    names = {'read': 'text\nread', 'thought': 'agent\nprose', 'args': 'argument\nvalues', 'syntax': 'call\nsyntax',
             'close': 'closing\nquote'}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.5})
    style(a, xgrid=False)
    style(b)
    x = list(range(len(roles)))
    for off, key, c, lab in ((-0.2, 'c55', ORANGE, 'plain score'), (0.2, 'c55role', GREEN, 'role score')):
        vals = [rt[key][r] - rt['full'][r] for r in roles]
        a.bar([i + off for i in x], vals, width=0.38, color=c, label=lab)
        for i, v in zip(x, vals):
            if roles[i] in ('read', 'close'):                  # selective labels: the two roles that move
                a.text(i + off, v + 0.012, f'{v:+.2f}', fontsize=5.8, color=INK, ha='center', va='bottom')
    a.set_xticks(x)
    a.set_xticklabels([names[r] for r in roles], fontsize=5.9)
    a.set_ylim(0, 0.6)
    a.set_ylabel('loss rise (nats per token)')
    a.set_title('Loss rise by token role')
    a.legend(frameon=False, loc='upper left', fontsize=6.4, handlelength=1.0)
    a.spines['bottom'].set_visible(True)
    kinds = [('wrong delimiter', ['string closed wrongly']),
             ('missing opener', ['opener: <|channel>call:', 'opener: other syntax']),
             ('quoted key', ['JSON quote on a key']), ('other', ['= for :', 'empty'])]
    ys = list(range(len(kinds)))[::-1]
    for off, key, c in ((0.18, 'c55', ORANGE), (-0.18, 'c55role', GREEN)):
        f = ca[key]['failed']
        for (lab, ks), yy in zip(kinds, ys):
            v = sum(f.get(k, 0) for k in ks)
            b.barh(yy + off, v, height=0.34, color=c)
            b.text(v + 0.3, yy + off, str(v), fontsize=6.2, color=INK, va='center')
    b.set_yticks(ys)
    b.set_yticklabels([k[0] for k in kinds], fontsize=6.8)
    b.set_xlim(0, 22)
    b.set_xlabel('failed call attempts (of about 125)')
    b.set_title('How the calls fail')
    b.text(0.98, 0.04, f"total: plain {ca['c55']['failed_total']}/{ca['c55']['attempts']},"
                       f" role {ca['c55role']['failed_total']}/{ca['c55role']['attempts']}",
           fontsize=5.8, color=MUTED, transform=b.transAxes, ha='right', va='bottom')
    save(fig, 'fig_failure_moves')


FIGS.update({'tasks': tasks, 'nextaction': nextaction, 'failure_moves': failure_moves})


def speed():
    """B175 (serve-v1 v3): decode time and served damage against the weight bytes read per token, for the full model,
    single-channel cuts, whole 32-channel blocks and 6 removed layers (as fig_prune_speed.py, redrawn at text width)."""
    import numpy as np
    d = 'runs/prune-serve-gpu/v3/prune/serve-v1/'
    gib = 1.073741824
    size = {'full': 21.668, 'c85': 20.11, 'c70': 18.29, 'c55': 16.74, 'b85': 20.11, 'b70': 18.29, 'd6': 20.16}  # GiB, B175; full = its model.safetensors, 23,265,352,448 bytes (B208)
    rep = {r['model']: r for r in json.load(open(d + 'report.json'))['table']}
    P = {}
    for m, g in size.items():
        b = json.load(open(d + f'bench_{m}.json'))
        P[m] = {'gb': g * gib, 'ms': 1000 / b['speed']['p2k']['decode_tok_s'], 'damage': rep[m]['agent_damage']}
    kind = {'full': 'full', 'c85': 'ch', 'c70': 'ch', 'c55': 'ch', 'b85': 'blk', 'b70': 'blk', 'd6': 'lay'}
    mk = {'full': ('D', FULL, FULL), 'ch': ('o', ORANGE, ORANGE), 'blk': ('s', 'none', ORANGE), 'lay': ('^', GREEN, GREEN)}
    lab_a = {'full': ('full model', -6, 4, 'right'), 'c85': ('85%', -6, -9, 'right'), 'c70': ('70%', 6, -7, 'left'),
             'c55': ('55%', 6, -7, 'left'), 'b85': ('85%, blocks', -6, 4, 'right'), 'b70': ('70%, blocks', -6, 4, 'right'),
             'd6': ('6 layers', 6, -8, 'left')}
    lab_b = {'full': ('full model', 0, 6, 'center'), 'c85': ('85%', 5, -9, 'left'), 'c70': ('70%', -6, -9, 'right'),
             'c55': ('55%', -6, 2, 'right'), 'b85': ('85%, blocks', 5, 9, 'left'), 'b70': ('70%, blocks', 6, 2, 'left'),
             'd6': ('6 layers', -6, -3, 'right')}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={'wspace': 0.32})

    def pts(ax, key, lab):
        for m, q in P.items():
            marker, fc, ec = mk[kind[m]]
            ax.scatter(q['gb'], q[key], s=26, marker=marker, facecolors=fc, edgecolors=ec, linewidths=1.0, zorder=4)
            t, dx, dy, ha = lab[m]
            ax.annotate(t, (q['gb'], q[key]), xytext=(dx, dy), textcoords='offset points', ha=ha, color=INK2, fontsize=6.0)
    for ax in (a, b):
        style(ax)
        ax.grid(True, axis='both', color=GRID, lw=0.5)
        ax.spines['left'].set_visible(True)
        ax.set_xlim(17.0, 24.4)
        ax.set_xlabel('weights read per token (GB)')
    fit = ['full', 'c85', 'c70', 'c55']
    k, c0 = np.polyfit([P[m]['gb'] for m in fit], [P[m]['ms'] for m in fit], 1)
    xs = np.linspace(17.2, 24.2, 50)
    a.plot(xs, k * xs + c0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    a.text(0.97, 0.05, f'fit: {k:.2f} ms per GB + {c0:.1f} ms', transform=a.transAxes, color=INK2, fontsize=6.0,
           ha='right', va='bottom')
    pts(a, 'ms', lab_a)
    a.set_ylim(19.5, 27.0)
    a.set_ylabel('decode time per token (ms)')
    a.set_title('Decode time follows the bytes')
    pts(b, 'damage', lab_b)
    b.set_ylim(-0.04, 1.0)
    b.set_ylabel('damage (nats per agent token)')
    b.set_title('Damage depends on what is cut')
    hs = [plt.Line2D([], [], ls='', marker='D', ms=4.5, mfc=FULL, mec=FULL, label='full model'),
          plt.Line2D([], [], ls='', marker='o', ms=4.5, mfc=ORANGE, mec=ORANGE, label='single channels'),
          plt.Line2D([], [], ls='', marker='s', ms=4.5, mfc='none', mec=ORANGE, label='whole 32-channel blocks'),
          plt.Line2D([], [], ls='', marker='^', ms=4.5, mfc=GREEN, mec=GREEN, label='6 layers removed')]
    b.legend(handles=hs, loc='center left', bbox_to_anchor=(0.0, 0.56), frameon=False, fontsize=6.0,
             handletextpad=0.3, borderaxespad=0.2)
    save(fig, 'fig_speed')


def speculation():
    """serve-v7 v2 (B206): n-gram speculative decoding over plain decoding, same weights and session."""
    sp = json.load(open(f'{V7}/spec_test.json'))['speedup']
    b5 = {n: json.load(open(f'{V7}/prune/serve-v7/bench_{n}.json'))['turns_s'] for n in ('full', 'full_b')}
    bars = [('synthetic, 2k context', sp['p2k']), ('synthetic, 16k context', sp['p16k']),
            ('synthetic, 8 streams', sp['streams8']), ('agent turns, 1 stream', sp['turns_streams1']),
            ('agent turns, 8 streams', sp['turns_streams8']), ('109-state pass, 8 streams', b5['full'] / b5['full_b'])]
    fig, ax = plt.subplots(figsize=(W, 1.75))
    style(ax)
    ys = list(range(len(bars)))[::-1]
    for (lab, v), yy in zip(bars, ys):
        c = AXIS if lab.startswith('synthetic') else FULL
        ax.barh(yy, v, height=0.55, color=c)
        ax.text(v + 0.03, yy, f'{v:.2f}\u00d7', fontsize=6.6, color=INK, va='center')
    ax.axvline(1.0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.8)
    ax.set_xlim(0, 3.8)
    ax.set_xlabel('speed with n-gram speculation / without (same weights, same session)')
    ax.set_title('Speculative decoding: fast on a synthetic continuation, not on agent turns')
    save(fig, 'fig_speculation')


FIGS.update({'speed': speed, 'speculation': speculation})


# ------------------------------------------------------------- column-width versions (two-column arXiv layout)
def tasks_col():
    col = {'full': FULL, 'cut': ORANGE, 'healed': BLUE}
    y = [8.6, 7.8, 7.0, 6.2, 3.6, 2.8, 2.0, 1.2]
    fig = plt.figure(figsize=(COL, 2.6))
    a = fig.add_axes([0.30, 0.10, 0.37, 0.80])
    b = fig.add_axes([0.70, 0.10, 0.285, 0.80])
    for ax in (a, b):
        style(ax)
        ax.set_ylim(0.6, 9.6)
        ax.set_yticks(y)
        ax.tick_params(axis='x', labelsize=6.3)
    a.set_yticklabels([r[0] for r in TASK_ROWS], fontsize=6.4)
    b.set_yticklabels([])
    for k, (lab, solved, mal, kind) in enumerate(TASK_ROWS):
        a.plot([0, solved], [y[k], y[k]], color=col[kind], lw=1.0, alpha=0.35, solid_capstyle='butt')
        a.scatter(solved, y[k], s=24, color=col[kind], zorder=3, edgecolors=SURF, linewidths=1.0)
        a.text(solved + 1.6, y[k], f'{solved}', fontsize=6.3, color=INK, va='center')
        b.barh(y[k], mal, height=0.5, color=col[kind])
        b.text(mal + 1.2, y[k], f'{mal:.1f}', fontsize=6.3, color=INK, va='center')
    a.axvspan(32, 34, ymin=(5.8 - 0.6) / 9.0, ymax=(9.0 - 0.6) / 9.0, color=GRID, alpha=0.6, lw=0)
    a.axvspan(33, 37, ymin=(0.8 - 0.6) / 9.0, ymax=(4.0 - 0.6) / 9.0, color=GRID, alpha=0.6, lw=0)
    a.set_xlim(0, 46)
    a.set_xticks([0, 20, 40])
    b.set_xlim(0, 58)
    b.set_xticks([0, 25, 50])
    a.set_title('Solved (of 112)', fontsize=7.2)
    b.set_title('Malformed calls (%)', fontsize=7.2)
    for yy, t in ((9.35, 'Session A: 70% cut'), (4.35, 'Session B: 55% cut')):
        a.text(-0.5, yy, t, fontsize=6.4, weight='bold', color=INK2, ha='left')
    save(fig, 'fig_tasks')


def nextaction_col():
    n = json.load(open(f'{V5}/next_action.json'))['models']
    order = [('full_b', 'full, other batching', 'full'), ('full@T0.2', 'full, sampled at T 0.2', 'full'),
             ('c85', '85% cut', 'cut'), ('c70', '70% cut', 'cut'), ('c55', '55% cut', 'cut'),
             ('c70own', '70% cut, agent tokens', 'alt'), ('c70+kd~', '70% cut, healed', 'healed')]
    col = {'full': FULL, 'cut': ORANGE, 'healed': BLUE, 'alt': GREEN}
    fig = plt.figure(figsize=(COL, 2.3))
    a = fig.add_axes([0.335, 0.165, 0.36, 0.715])
    b = fig.add_axes([0.75, 0.165, 0.235, 0.715])
    ys = list(range(len(order)))[::-1]
    for ax in (a, b):
        style(ax)
        ax.set_yticks(ys)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.tick_params(axis='x', labelsize=6.3)
    a.set_yticklabels([o[1] for o in order], fontsize=6.4)
    b.set_yticklabels([])
    fb = n['full_b']['vs_full']['parsed']
    a.axvspan(fb['ci95'][0], fb['ci95'][1], color=GRID, alpha=0.6, lw=0)
    a.text(fb['ci95'][1] - 0.01, -0.62, 'numerics floor', fontsize=5.6, color=MUTED, ha='right', va='bottom')
    for (key, lab, kind), yy in zip(order, ys):
        v = n[key]['vs_full']['parsed']
        a.plot(v['ci95'], [yy, yy], color=col[kind], lw=1.3, solid_capstyle='round')
        a.scatter(v['share'], yy, s=22, color=col[kind], zorder=3, edgecolors=SURF, linewidths=0.9)
        a.text(v['ci95'][1] + 0.02, yy, f"{v['share']:.2f}", fontsize=6.2, color=INK, va='center')
        m = n[key]['malformed']
        pct = 100 * m['malformed'] / m['calls']
        b.barh(yy, pct, height=0.5, color=col[kind])
        b.text(pct + 0.5, yy, f"{m['malformed']}/{m['calls']}", fontsize=6.2, color=INK, va='center')
    a.set_xlim(0, 1.12)
    a.set_xticks([0, 0.5, 1.0])
    b.set_xlim(0, 26)
    b.set_xticks([0, 10, 20])
    a.set_xlabel('share, 95% interval', fontsize=6.4)
    b.set_xlabel('malformed (%)', fontsize=6.4)
    a.set_title('Same next call as the full model', fontsize=7.2)     # spans both panels; b has none
    save(fig, 'fig_nextaction')


def failure_moves_col():
    rt = json.load(open(f'{V6}/role_test.json'))['descriptive']['role_nats']
    ca = json.load(open(f'{V6}/call_attempts.json'))['failure_kinds']
    roles = ['read', 'thought', 'args', 'syntax', 'close']
    names = {'read': 'text\nread', 'thought': 'agent\nprose', 'args': 'argument\nvalues', 'syntax': 'call\nsyntax',
             'close': 'closing\nquote'}
    fig = plt.figure(figsize=(COL, 3.35))
    a = fig.add_axes([0.135, 0.615, 0.85, 0.325])
    b = fig.add_axes([0.285, 0.095, 0.70, 0.305])
    style(a, xgrid=False)
    style(b)
    x = list(range(len(roles)))
    for off, key, c, lab in ((-0.2, 'c55', ORANGE, 'plain score'), (0.2, 'c55role', GREEN, 'role score')):
        vals = [rt[key][r] - rt['full'][r] for r in roles]
        a.bar([i + off for i in x], vals, width=0.38, color=c, label=lab)
        for i, v in zip(x, vals):
            if roles[i] in ('read', 'close'):                  # selective labels: the two roles that move
                a.text(i + off, v + 0.012, f'{v:+.2f}', fontsize=5.8, color=INK, ha='center', va='bottom')
    a.set_xticks(x)
    a.set_xticklabels([names[r] for r in roles], fontsize=6.2)
    a.set_ylim(0, 0.6)
    a.tick_params(axis='y', labelsize=6.3)
    a.set_ylabel('nats per token', fontsize=6.4)
    a.set_title('Loss rise by token role (55% cut)', fontsize=7.2)
    a.legend(frameon=False, loc='upper center', ncol=2, fontsize=6.3, handlelength=1.0, columnspacing=1.2)
    a.spines['bottom'].set_visible(True)
    kinds = [('wrong delimiter', ['string closed wrongly']),
             ('missing opener', ['opener: <|channel>call:', 'opener: other syntax']),
             ('quoted key', ['JSON quote on a key']), ('other', ['= for :', 'empty'])]
    ys = list(range(len(kinds)))[::-1]
    for off, key, c in ((0.18, 'c55', ORANGE), (-0.18, 'c55role', GREEN)):
        f = ca[key]['failed']
        for (lab, ks), yy in zip(kinds, ys):
            v = sum(f.get(k, 0) for k in ks)
            b.barh(yy + off, v, height=0.34, color=c)
            b.text(v + 0.3, yy + off, str(v), fontsize=6.0, color=INK, va='center')
    b.set_yticks(ys)
    b.set_yticklabels([k[0] for k in kinds], fontsize=6.4)
    b.tick_params(axis='x', labelsize=6.3)
    b.set_xlim(0, 22)
    b.set_xlabel('failed call attempts (of about 125)', fontsize=6.4)
    b.set_title('How the calls fail', fontsize=7.2)
    b.text(0.98, 0.04, f"total: plain {ca['c55']['failed_total']}/{ca['c55']['attempts']},"
                       f" role {ca['c55role']['failed_total']}/{ca['c55role']['attempts']}",
           fontsize=5.7, color=MUTED, transform=b.transAxes, ha='right', va='bottom')
    save(fig, 'fig_failure_moves')


def speed_col():
    """speed() stacked: decode time above, damage below (B175)."""
    import numpy as np
    d = 'runs/prune-serve-gpu/v3/prune/serve-v1/'
    gib = 1.073741824
    size = {'full': 21.668, 'c85': 20.11, 'c70': 18.29, 'c55': 16.74, 'b85': 20.11, 'b70': 18.29, 'd6': 20.16}  # GiB, B175; full = its model.safetensors, 23,265,352,448 bytes (B208)
    rep = {r['model']: r for r in json.load(open(d + 'report.json'))['table']}
    P = {}
    for m, g in size.items():
        bj = json.load(open(d + f'bench_{m}.json'))
        P[m] = {'gb': g * gib, 'ms': 1000 / bj['speed']['p2k']['decode_tok_s'], 'damage': rep[m]['agent_damage']}
    kind = {'full': 'full', 'c85': 'ch', 'c70': 'ch', 'c55': 'ch', 'b85': 'blk', 'b70': 'blk', 'd6': 'lay'}
    mk = {'full': ('D', FULL, FULL), 'ch': ('o', ORANGE, ORANGE), 'blk': ('s', 'none', ORANGE), 'lay': ('^', GREEN, GREEN)}
    lab_a = {'full': ('full model', -6, 4, 'right'), 'c85': ('85%', -6, -9, 'right'), 'c70': ('70%', 6, -7, 'left'),
             'c55': ('55%', 6, -7, 'left'), 'b85': ('85%, blocks', -6, 4, 'right'), 'b70': ('70%, blocks', -6, 4, 'right'),
             'd6': ('6 layers', 6, -8, 'left')}
    lab_b = {'full': ('full model', 0, 6, 'center'), 'c85': ('85%', -6, -4, 'right'), 'c70': ('70%', -6, -9, 'right'),
             'c55': ('55%', -6, 2, 'right'), 'b85': ('85%, blocks', 5, 9, 'left'), 'b70': ('70%, blocks', 6, 2, 'left'),
             'd6': ('6 layers', -6, -3, 'right')}
    fig = plt.figure(figsize=(COL, 3.8))
    a = fig.add_axes([0.15, 0.585, 0.835, 0.36])
    b = fig.add_axes([0.15, 0.095, 0.835, 0.36])

    def pts(ax, key, lab):
        for m, q in P.items():
            marker, fc, ec = mk[kind[m]]
            ax.scatter(q['gb'], q[key], s=24, marker=marker, facecolors=fc, edgecolors=ec, linewidths=1.0, zorder=4)
            t, dx, dy, ha = lab[m]
            ax.annotate(t, (q['gb'], q[key]), xytext=(dx, dy), textcoords='offset points', ha=ha, color=INK2, fontsize=6.0)
    for ax in (a, b):
        style(ax)
        ax.grid(True, axis='both', color=GRID, lw=0.5)
        ax.spines['left'].set_visible(True)
        ax.set_xlim(17.0, 24.4)
        ax.tick_params(labelsize=6.3)
        ax.set_xlabel('weights read per token (GB)', fontsize=6.4)
    fit = ['full', 'c85', 'c70', 'c55']
    k, c0 = np.polyfit([P[m]['gb'] for m in fit], [P[m]['ms'] for m in fit], 1)
    xs = np.linspace(17.2, 24.2, 50)
    a.plot(xs, k * xs + c0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    a.text(0.98, 0.05, f'fit: {k:.2f} ms per GB + {c0:.1f} ms', transform=a.transAxes, color=INK2, fontsize=6.0,
           ha='right', va='bottom')
    pts(a, 'ms', lab_a)
    a.set_ylim(19.5, 27.0)
    a.set_ylabel('decode ms per token', fontsize=6.4)
    a.set_title('Decode time follows the bytes', fontsize=7.2)
    pts(b, 'damage', lab_b)
    b.set_ylim(-0.04, 1.0)
    b.set_ylabel('damage (nats per agent token)', fontsize=6.4)
    b.set_title('Damage depends on what is cut', fontsize=7.2)
    hs = [plt.Line2D([], [], ls='', marker='D', ms=4.2, mfc=FULL, mec=FULL, label='full model'),
          plt.Line2D([], [], ls='', marker='o', ms=4.2, mfc=ORANGE, mec=ORANGE, label='single channels'),
          plt.Line2D([], [], ls='', marker='s', ms=4.2, mfc='none', mec=ORANGE, label='whole 32-channel blocks'),
          plt.Line2D([], [], ls='', marker='^', ms=4.2, mfc=GREEN, mec=GREEN, label='6 layers removed')]
    b.legend(handles=hs, loc='upper left', frameon=False, fontsize=6.0, handletextpad=0.3, borderaxespad=0.3)
    save(fig, 'fig_speed')


def speculation_col():
    sp = json.load(open(f'{V7}/spec_test.json'))['speedup']
    b5 = {n: json.load(open(f'{V7}/prune/serve-v7/bench_{n}.json'))['turns_s'] for n in ('full', 'full_b')}
    bars = [('synthetic, 2k context', sp['p2k']), ('synthetic, 16k context', sp['p16k']),
            ('synthetic, 8 streams', sp['streams8']), ('agent turns, 1 stream', sp['turns_streams1']),
            ('agent turns, 8 streams', sp['turns_streams8']), ('109-state pass, 8 streams', b5['full'] / b5['full_b'])]
    fig = plt.figure(figsize=(COL, 1.95))
    ax = fig.add_axes([0.40, 0.22, 0.565, 0.64])
    style(ax)
    ys = list(range(len(bars)))[::-1]
    for (lab, v), yy in zip(bars, ys):
        c = AXIS if lab.startswith('synthetic') else FULL
        ax.barh(yy, v, height=0.55, color=c)
        ax.text(v + 0.04, yy, f'{v:.2f}×', fontsize=6.2, color=INK, va='center')
    ax.axvline(1.0, color=INK2, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.4)
    ax.tick_params(axis='x', labelsize=6.3)
    ax.set_xlim(0, 4.0)
    ax.set_xlabel('speed with speculation / without', fontsize=6.4)
    ax.set_title('Speedup from n-gram speculation', fontsize=7.2)
    save(fig, 'fig_speculation')


COLFIGS = {'tasks': tasks_col, 'nextaction': nextaction_col, 'failure_moves': failure_moves_col, 'speed': speed_col,
           'speculation': speculation_col}

if __name__ == '__main__':
    names = [a for a in sys.argv[1:] if not a.startswith('--')] or list(FIGS)
    for name in names:
        (COLFIGS.get(name, FIGS[name]) if TWO else FIGS[name])()
