"""Banner for the public repository's README (docs/assets/banner.png): title, the step-time model and three measured
results, in the paper figures' palette. usage: <python with matplotlib> kernels/analysis/fig_banner.py
Numbers (measured): 70% cut decodes 1.15x faster (B175); healed and merged it solves 32 of 112 against 32 and 34 for
the full model (-1 task, B197); malformed tool calls 24% for the cut, 3% healed (B197)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Wedge  # noqa: E402

OUT = 'docs/assets/banner.png'
W, H = 12.8, 3.2
BG0, BG1 = '#0d1626', '#18304f'
WHITE, SOFT, DIM = '#ffffff', '#c9d6ea', '#8ea3c2'
BLUE, ORANGE, GREEN = '#4a90e2', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig = plt.figure(figsize=(W, H), dpi=250)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_alpha(0)

# card with a left-to-right gradient, clipped to rounded corners
card = FancyBboxPatch((0.02, 0.02), W - 0.04, H - 0.04, boxstyle='round,pad=0,rounding_size=0.22', fc='none', ec='none')
ax.add_patch(card)
grad = np.linspace(0, 1, 512)[None, :]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('bg', [BG0, BG1])
im = ax.imshow(grad, extent=(0, W, 0, H), cmap=cmap, aspect='auto', zorder=0)
im.set_clip_path(card)
for k in range(1, 9):                                     # faint grid of "ticks" for texture
    ax.plot([k * 1.42, k * 1.42], [0.18, H - 0.18], color=WHITE, alpha=0.035, lw=1, zorder=1)

# the clock: 8 minutes of 10 spent (the ring), hands at 7:52
cx, cy, r = 1.62, H / 2, 1.02
ax.add_patch(Circle((cx, cy), r + 0.10, fc='#0a1220', ec='none', zorder=2))
ax.add_patch(Wedge((cx, cy), r + 0.06, 90 - 360 * 0.8, 90, width=0.13, fc=ORANGE, ec='none', zorder=3))
ax.add_patch(Wedge((cx, cy), r + 0.06, 90, 90 + 360 * 0.2, width=0.13, fc='#24395a', ec='none', zorder=3))
for k in range(12):
    a = np.pi / 2 - 2 * np.pi * k / 12
    r0 = r - (0.20 if k % 3 == 0 else 0.13)
    ax.plot([cx + r0 * np.cos(a), cx + (r - 0.06) * np.cos(a)], [cy + r0 * np.sin(a), cy + (r - 0.06) * np.sin(a)],
            color=SOFT if k % 3 == 0 else DIM, lw=2.2 if k % 3 == 0 else 1.3, solid_capstyle='round', zorder=4)
for frac, length, lw, col in ((7 / 12 + 52 / 720, 0.48, 4.0, WHITE), (52 / 60, 0.74, 2.6, SOFT)):
    a = np.pi / 2 - 2 * np.pi * frac
    ax.plot([cx, cx + length * np.cos(a)], [cy, cy + length * np.sin(a)], color=col, lw=lw, solid_capstyle='round',
            zorder=5)
ax.add_patch(Circle((cx, cy), 0.06, fc=ORANGE, ec='none', zorder=6))

# title, subtitle, the step-time model
x0 = 3.05
ax.text(x0, 2.30, 'Under the Clock', color=WHITE, fontsize=34, weight='bold', va='center', zorder=6)
ax.text(x0, 1.66, 'Shorter Steps and Fewer Bytes for Tool-Using Agents', color=SOFT, fontsize=15.5, va='center',
        zorder=6)
pill = FancyBboxPatch((x0, 0.55), 4.2, 0.62, boxstyle='round,pad=0,rounding_size=0.31', fc='#0a1220', ec='#2c4366',
                      lw=1.2, zorder=5)
ax.add_patch(pill)
parts = [('seconds per step  ≈  a  +  ', WHITE), ('n', GREEN), (' · ', WHITE), ('b', GREEN)]
ax.text(x0 + 0.30, 0.86, ''.join(p for p, _ in parts), color='none', fontsize=15, va='center')  # spacing reference
xx = x0 + 0.30
renderer = fig.canvas.get_renderer()
for text, col in parts:
    t = ax.text(xx, 0.86, text, color=col, fontsize=15, weight='bold', va='center', zorder=6)
    bb = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
    xx = bb.x1
ax.text(x0 + 0.32, 0.30, 'n = tokens per step,   b = seconds per token', color=DIM, fontsize=10, va='center', zorder=6)

# three measured results
stats = [('1.15×', 'faster decoding, 70% of MLP channels kept', ORANGE),
         ('−1 task', 'of 112 after healing, vs the full model', BLUE),
         ('24% → 3%', 'broken tool calls, cut vs healed', GREEN)]
sx, sw, sh = 9.05, 3.45, 0.70
for k, (big, small, col) in enumerate(stats):
    y = 2.42 - k * 0.86
    ax.add_patch(FancyBboxPatch((sx, y - sh / 2), sw, sh, boxstyle='round,pad=0,rounding_size=0.12', fc='#0a1220',
                                ec='#24395a', lw=1.0, zorder=5))
    ax.add_patch(FancyBboxPatch((sx, y - sh / 2), 0.09, sh, boxstyle='round,pad=0,rounding_size=0.04', fc=col,
                                ec='none', zorder=6))
    ax.text(sx + 0.28, y + 0.11, big, color=WHITE, fontsize=16, weight='bold', va='center', zorder=6)
    ax.text(sx + 0.28, y - 0.19, small, color=DIM, fontsize=9.4, va='center', zorder=6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=250, transparent=True)
print('wrote', OUT)
