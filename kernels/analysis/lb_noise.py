#!/usr/bin/env python3
"""How much of a public-leaderboard score is luck: binomial noise on the 58 public tasks.

Inputs are our own scored submissions of one budget family (8 min / 28 calls): sub-v5 7, sub-v5b 11, sub-v7 6 of 58.
Prints the per-run sd, P(a run >= 11), the sd of a difference of two runs, the power to see a true +3-task gain with
one pair of submissions, and the expected best of N runs (Kaggle shows a team's best public score).
"""
import math
from statistics import NormalDist

runs, n = [7, 11, 6], 58
mean = sum(runs) / len(runs)
p = mean / n
sd = math.sqrt(n * p * (1 - p))
sample_sd = math.sqrt(sum((x - mean) ** 2 for x in runs) / (len(runs) - 1))
tail = sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(11, n + 1))
diff_sd = sd * math.sqrt(2)
power = 1 - NormalDist().cdf(1.645 - 3 / diff_sd)
print(f'family mean {mean:.1f}/58 = {p:.3f}; binomial sd {sd:.2f} tasks; sample sd of the 3 runs {sample_sd:.2f}')
print(f'P(one run >= 11 | p={p:.3f}) = {tail:.3f}')
print(f'sd of a difference of two runs {diff_sd:.2f} tasks; power to detect +3 tasks (one-sided 5%) {power:.2f}')
for N in (2, 5, 10, 20, 30):
    import random
    rng = random.Random(0)
    best = sum(max(sum(rng.random() < p for _ in range(n)) for _ in range(N)) for _ in range(4000)) / 4000
    print(f'expected best of {N:>2} runs: {best:.1f}/58 = {best / n:.3f}')
