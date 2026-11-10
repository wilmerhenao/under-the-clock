"""Paper figure (pruning section): decode time follows the bytes removed; damage depends on what is removed.

Reads the speed test's measured files (B175, `runs/prune-serve-gpu/v3/prune/serve-v1/bench_*.json`; vLLM, W4A16,
4 x L4, one stream at a 2,000-token prompt) and writes `docs/paper/fig_prune_speed.png`.
usage: python3 kernels/analysis/fig_prune_speed.py [out.png]
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = 'runs/prune-serve-gpu/v3/prune/serve-v1/'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'docs/paper/fig_prune_speed.png'
GIB = 1.073741824
# weights file size in GiB (B175; the full model's is derived from its shards)
SIZE_GIB = {'full': 21.93, 'c85': 20.11, 'c70': 18.29, 'c55': 16.74, 'b85': 20.11, 'b70': 18.29, 'd6': 20.16}
rep = {r['model']: r for r in json.load(open(D + 'report.json'))['table']}
P = {}
for m, g in SIZE_GIB.items():
    b = json.load(open(D + f'bench_{m}.json'))
    P[m] = {'gb': g * GIB, 'ms': 1000 / b['speed']['p2k']['decode_tok_s'], 'damage': rep[m]['agent_damage']}

# reference palette (dataviz skill, light mode): categorical slots 1-3, text and axis tokens
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SER = {'channels': ('#2a78d6', 'o', 'single channels (agent scores)'),
       'blocks': ('#eb6834', 's', 'whole 32-channel blocks'),
       'layers': ('#1baf7a', '^', '6 layers removed (block influence)')}
KIND = {'c85': 'channels', 'c70': 'channels', 'c55': 'channels', 'b85': 'blocks', 'b70': 'blocks', 'd6': 'layers'}
# direct labels: (text, dx, dy in points, ha)
LAB_A = {'full': ('full model', -10, 5, 'right'), 'c85': ('85%', -10, -9, 'right'), 'c70': ('70%', 10, -7, 'left'),
         'c55': ('55%', 10, -7, 'left'), 'b85': ('85% blocks', -10, 5, 'right'), 'b70': ('70% blocks', -10, 5, 'right'),
         'd6': ('6 layers', 10, -12, 'left')}
LAB_B = {'full': ('full model', -8, 4, 'right'), 'c85': ('85%', 8, -12, 'left'), 'c70': ('70%', 8, -12, 'left'),
         'c55': ('55%', -9, 2, 'right'), 'b85': ('85% blocks', -9, 6, 'right'), 'b70': ('70% blocks', 9, 2, 'left'),
         'd6': ('6 layers', -9, -3, 'right')}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 12,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10})
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor=SURF)


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, pad=6)


def points(ax, key, lab):
    for m, p in P.items():
        if m == 'full':
            ax.scatter(p['gb'], p[key], s=70, c=INK, marker='D', edgecolors=SURF, linewidths=2, zorder=4)
        else:
            col, mk, _ = SER[KIND[m]]
            ax.scatter(p['gb'], p[key], s=80, c=col, marker=mk, edgecolors=SURF, linewidths=2, zorder=4)
        t, dx, dy, ha = lab[m]
        ax.annotate(t, (p['gb'], p[key]), xytext=(dx, dy), textcoords='offset points', ha=ha, color=INK2, fontsize=9.5)


# panel A: decode time vs bytes, with the fit through the full model and the single-channel cuts
ax = axs[0]
style(ax)
fit = ['full', 'c85', 'c70', 'c55']
a, b = np.polyfit([P[m]['gb'] for m in fit], [P[m]['ms'] for m in fit], 1)
xs = np.linspace(17.2, 24.2, 50)
ax.plot(xs, a * xs + b, color=MUTED, lw=1.6, ls=(0, (5, 4)), zorder=2)
ax.text(0.97, 0.05, f'fit: {a:.2f} ms per GB + {b:.1f} ms\n(about {1000 / a / 4:.0f} GB/s per L4)',
        transform=ax.transAxes, color=INK2, fontsize=9, ha='right', va='bottom')
points(ax, 'ms', LAB_A)
ax.set_xlim(17.0, 24.4)
ax.set_ylim(19.5, 27.0)
ax.set_xlabel('weights read per token (GB)')
ax.set_ylabel('decode time per token (ms)')
ax.set_title('Decode time follows the bytes removed')

# panel B: served damage vs bytes
ax = axs[1]
style(ax)
points(ax, 'damage', LAB_B)
ax.set_xlim(17.0, 24.4)
ax.set_ylim(-0.04, 1.0)
ax.set_xlabel('weights read per token (GB)')
ax.set_ylabel('damage (nats per agent-written token)')
ax.set_title('Damage depends on what is removed')

handles = [plt.Line2D([], [], ls='', marker='D', ms=7, mfc=INK, mec=SURF, label='full model')]
handles += [plt.Line2D([], [], ls='', marker=mk, ms=8, mfc=col, mec=SURF, label=lab) for col, mk, lab in SER.values()]
handles += [plt.Line2D([], [], color=MUTED, lw=1.6, ls=(0, (5, 4)), label='linear fit (full + single channels)')]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=9.5, labelcolor=INK2,
           bbox_to_anchor=(0.5, -0.01), handletextpad=0.4, columnspacing=1.6)
fig.text(0.01, 0.955, 'Gemma 4 31B, W4A16, vLLM on 4 x L4, one stream, 2,000-token prompt; damage = rise in held-out '
         'nats per token on the agent\'s own text (40 trajectories, 25,896 tokens)', color=MUTED, fontsize=8.8)
fig.tight_layout(rect=(0, 0.07, 1, 0.94), w_pad=3)
fig.savefig(OUT, dpi=200, facecolor=SURF)
print(OUT, {m: (round(p['gb'], 2), round(p['ms'], 2), p['damage']) for m, p in P.items()}, round(a, 3), round(b, 2))
