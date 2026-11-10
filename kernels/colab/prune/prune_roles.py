# %% [markdown]
# ## Part 1b (score-v4): what each kind of agent token uses
#
# The first measurement averaged over every token of the agent's runs. Most of those tokens (93%) are text the agent
# *reads*: files, test output, the issue. Only a few are its own tool calls, and very few are the small token that
# closes a string argument (`<|"|>`). When we cut 45% of the channels by that average, the model started to forget
# to close its strings: a file name swallowed the next argument (`a.py",start_line:350` became one file name) and the
# tool call broke. On real tasks 43% of that model's tool calls were broken (B200).
#
# So the model reads the same 300 examples again, and we keep **separate counts for each role of the token it is
# about to write**: text it reads, its own thoughts, argument values, call syntax, and the closing quote (the roles
# are explained in the cell above). The new channel score `role_ff` gives **each role an equal vote**: a channel
# that matters only for closing quotes is kept, although closing quotes are under 1% of the tokens.
#
# Checks before trusting it: the plain average is measured again in the same pass and must rank the channels like
# the first run (rank correlation of at least 0.999 in every layer), on exactly the same examples and token counts.
# The result file keeps every number of the first run, so the earlier cuts can be rebuilt exactly; the next notebook
# (serving) cuts the model both ways and compares the tool calls it writes.

# %%
torch.backends.cuda.matmul.allow_tf32 = False     # exact float32 sums, as in the first run
MULTIPLE = 512 if not TINY else 32                # channels kept per layer in the served cuts: a multiple of this
ROLE_KEEPS = [0.85, 0.70, 0.55]                   # shares of channels kept, for comparing the two scores
prev_dir = os.environ.get('PRUNE_PREV') or os.path.dirname(find('**/scores.npz') or '')
assert prev_dir and os.path.exists(f'{prev_dir}/report.json'), 'needs the earlier run (score-v3) attached as an input'
_z = np.load(f'{prev_dir}/scores.npz')
V3 = {k: _z[k] for k in _z.files}
V3_REP = json.load(open(f'{prev_dir}/report.json'))
same = {k: [V3_REP['settings'][k], REPORT['settings'][k]] for k in ('CAL_ROWS', 'HELDOUT_ROWS', 'MAX_TOKENS', 'SEED')}
same.update({k: [V3_REP['data'][k], REPORT['data'][k]]
             for k in ('agent_rows_total', 'cal_rows', 'cal_tokens', 'cal_own_tokens', 'heldout_own_tokens')})
REPORT['roles'] = {'from_run': V3_REP['run'], 'from_dir': prev_dir, 'same_examples': same,
                   'same_examples_ok': all(a == b for a, b in same.values())}
save_report()
if not SMOKE:
    assert REPORT['roles']['same_examples_ok'], f"not the same examples as {V3_REP['run']}: {same}"


def roles_of(r, n):
    """Role of the token each of the first n positions of example r predicts (-1: none), and its first n token ids."""
    enc = tok(r['text'], add_special_tokens=False, return_offsets_mapping=True)
    pr = position_roles([o[0] for o in enc['offset_mapping']], char_roles(r['text']), n)
    return torch.from_numpy(pr), enc['input_ids'][:n]


t_r = time.time()
CAL_R = []
for (ids, own), r in zip(CAL, CAL_SRC):
    pr, ids2 = roles_of(r, len(ids))
    assert ids.tolist() == list(ids2), r['instance_id']
    CAL_R.append((ids, own, pr))
REPORT['roles']['positions_per_role'] = {n: int(sum(int((pr == j).sum()) for _, _, pr in CAL_R))
                                         for j, n in enumerate(ROLE_NAMES)}
REPORT['roles']['tagging_s'] = round(time.time() - t_r, 1)
log('positions per role of the token they predict:', REPORT['roles']['positions_per_role'])
save_report()

# %% [markdown]
# ### Measuring, per role
#
# For every position, the probes add its feed-forward activity to three counts at once: all agent tokens and the
# agent's own tokens (both exactly as in the first run, for the check) and the role of the token it is about to write.

# %%
def role_weights(ids, own, pr):
    W = torch.zeros(len(ids), 2 + len(RG))
    W[:, 0] = 1                                    # every position: 'agent'
    W[:, 1] = own.float()                          # the agent's own tokens (the rows' spans): 'agent_own'
    ok = torch.nonzero(pr >= 0).squeeze(1)
    W[ok, 2 + pr[ok]] = 1                          # the role of the next token
    return W


def measure_roles(items):
    t, n_tok = time.time(), 0
    for k, (ids, own, pr) in enumerate(items):
        STATE['w'] = (role_weights(ids, own, pr), ['agent', 'agent_own', *RG])
        forward_only(ids)
        ACC['agent']['n'] += len(ids)
        ACC['agent_own']['n'] += int(own.sum())
        for j, g in enumerate(RG):
            ACC[g]['n'] += int((pr == j).sum())
        n_tok += len(ids)
        if k % 25 == 0:
            log(f'agent example {k + 1}/{len(items)}, {n_tok:,} tokens, {n_tok / max(1e-6, time.time() - t):,.0f} tok/s')
    STATE['w'] = None
    return n_tok, time.time() - t


with torch.no_grad():
    down_norm = torch.stack([lyr.mlp.down_proj.weight.float().norm(dim=0).to(DEV0) for lyr in LAYERS])     # (L, I)


def role_scores():
    """Per group: sqrt(mean square activity) x size of the outgoing weights (the first run's score). role_ff: each role's
    mean square divided by its median in that layer, averaged over the roles with an equal weight, then the same."""
    MS = {g: ACC[g]['ff_sq'] / max(1, ACC[g]['n']) for g in ('agent', 'agent_own', *RG)}
    S = {f'{g}_ff': (MS[g].sqrt() * down_norm).cpu().numpy() for g in MS}
    used = [g for g in RG if ACC[g]['n'] > 0]
    rel = sum(MS[g] / MS[g].median(dim=1, keepdim=True).values.clamp_min(1e-30) for g in used) / max(1, len(used))
    S['role_ff'] = (rel.sqrt() * down_norm).cpu().numpy()
    return S, used


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind='stable'), kind='stable')
    rb = np.argsort(np.argsort(b, kind='stable'), kind='stable')
    return float(np.corrcoef(ra, rb)[0, 1])


def kept_idx(S, keep):
    """Channels a served cut keeps per layer: the best round(keep x I / MULTIPLE) x MULTIPLE (slice_model.make_plan)."""
    k = max(MULTIPLE, min(I, int(round(keep * I / MULTIPLE)) * MULTIPLE))
    return [np.argsort(-np.asarray(S[i], np.float64), kind='stable')[:k] for i in range(L)], k


def sanity(S):
    rc = [spearman(S['agent_ff'][i], V3['agent_ff'][i]) for i in range(L)]
    rc_own = [spearman(S['agent_own_ff'][i], V3['agent_own_ff'][i]) for i in range(L)]
    dn = np.abs(down_norm.cpu().numpy() - V3['down_norm']).max() / max(1e-30, float(np.abs(V3['down_norm']).max()))
    counts = {'agent': [ACC['agent']['n'], int(V3['n_agent'])], 'agent_own': [ACC['agent_own']['n'], int(V3['n_agent_own'])]}
    return {'agent_ff_rank_corr_min': round(min(rc), 6), 'agent_ff_rank_corr_mean': round(float(np.mean(rc)), 6),
            'agent_own_ff_rank_corr_min': round(min(rc_own), 6), 'down_norm_max_rel_diff': float(dn), 'counts': counts,
            'ok': bool(min(rc) >= 0.999 and dn < 1e-4 and all(a == b for a, b in counts.values()))}


def compare(S):
    """How different the role score is from the first one, and which channels each cut keeps."""
    names = {'role_ff': 'role_ff', **{g: f'{g}_ff' for g in RG}}
    out = {'rank_corr_with_agent_ff_mean': {n: round(float(np.mean([spearman(S[k][i], V3['agent_ff'][i])
                                                                    for i in range(L)])), 4) for n, k in names.items()}}
    top_n = max(1, round(0.05 * I))
    tops = {g: [np.argsort(-np.asarray(S[f'{g}_ff'][i], np.float64), kind='stable')[:top_n] for i in range(L)]
            for g in RG}
    for keep in ROLE_KEEPS:
        A, k = kept_idx(V3['agent_ff'], keep)
        B, _ = kept_idx(S['role_ff'], keep)
        d = {'k': k, 'kept_overlap_agent_ff_vs_role_ff': round(float(np.mean([np.isin(B[i], A[i]).mean()
                                                                               for i in range(L)])), 4)}
        for g in RG:   # each role's own top 5% channels per layer: which share does each cut keep?
            d[f'{g}_top5pct_kept'] = {'agent_ff': round(float(np.mean([np.isin(tops[g][i], A[i]).mean() for i in range(L)])), 4),
                                      'role_ff': round(float(np.mean([np.isin(tops[g][i], B[i]).mean() for i in range(L)])), 4)}
        out[f'keep_{keep}'] = d
    return out


# a first run of every step on one example (a problem shows up now, not after an hour); then the counts are reset
t_pf = time.time()
measure_roles(CAL_R[:1])
S_pf, used_pf = role_scores()
REPORT['roles']['preflight'] = {'roles_used': used_pf, 'sanity': sanity(S_pf), 'compare': compare(S_pf),
                                's': round(time.time() - t_pf, 1)}
for g in GROUPS:
    for k in ('ff_abs', 'ff_sq', 'head_sq', 'bi'):
        ACC[g][k].zero_()
    ACC[g]['n'] = 0
log('preflight', {k: v for k, v in REPORT['roles']['preflight'].items() if k != 'compare'})
save_report()

# %%
n_tok, dt = measure_roles(CAL_R)
REPORT['timings_min']['measure'] = round(dt / 60, 1)
REPORT['measure_tok_per_s'] = round(n_tok / dt, 1)
if torch.cuda.is_available():
    REPORT['peak_gpu_memory_gb'] = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                                    for i in range(torch.cuda.device_count())]
S4, used = role_scores()
REPORT['roles'].update(roles_used=used, counts={g: ACC[g]['n'] for g in ('agent', 'agent_own', *RG)},
                       sanity=sanity(S4), compare=compare(S4))
log(f'measured {n_tok:,} tokens in {dt / 60:.1f} min; check', REPORT['roles']['sanity'])
save_report()

# %% [markdown]
# ### Save the scores (and the first run's)
#
# `scores.npz` holds every array of the first run unchanged (`agent_ff` and the rest: the earlier cuts stay exactly
# the same), plus `role_ff`, each role's own score `role_<name>_ff`, and the plain scores measured again
# (`agent_ff_v4`, `agent_own_ff_v4`).

# %%
out = dict(V3)
out.update({'agent_ff_v4': S4['agent_ff'], 'agent_own_ff_v4': S4['agent_own_ff'], 'role_ff': S4['role_ff'],
            **{f'{g}_ff': S4[f'{g}_ff'] for g in RG}, **{f'n_{g}': np.array(ACC[g]['n']) for g in RG},
            'n_agent_v4': np.array(ACC['agent']['n']), 'n_agent_own_v4': np.array(ACC['agent_own']['n'])})
np.savez_compressed(f'{OUT}/scores.npz', **out)
if V3_REP.get('layer_sweep'):   # the first runs' layer test, for cuts that remove layers in its order
    REPORT['layer_sweep'] = dict(V3_REP['layer_sweep'], reused_from=V3_REP['run'])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmp4 = REPORT['roles']['compare']
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for g in RG:
    ax[0].plot([spearman(S4[f'{g}_ff'][i], V3['agent_ff'][i]) for i in range(L)], label=g.replace('role_', ''))
ax[0].set(title='Does each role use the channels of the plain score?', xlabel='layer',
          ylabel='rank correlation with agent_ff')
ax[0].legend()
kk = f'keep_{ROLE_KEEPS[-1]}'
xs = np.arange(len(RG))
ax[1].bar(xs - 0.2, [cmp4[kk][f'{g}_top5pct_kept']['agent_ff'] for g in RG], 0.4, label='cut by agent_ff')
ax[1].bar(xs + 0.2, [cmp4[kk][f'{g}_top5pct_kept']['role_ff'] for g in RG], 0.4, label='cut by role_ff')
ax[1].set_xticks(xs, [g.replace('role_', '') for g in RG])
ax[1].set(title=f"Each role's top 5% channels kept at {ROLE_KEEPS[-1]:.0%}", ylabel='share kept', ylim=(0, 1.05))
ax[1].legend()
plt.tight_layout()
plt.savefig(f'{OUT}/scores.png', dpi=120)
REPORT['timings_min']['total'] = round((time.time() - T0) / 60, 1)
save_report()
print(json.dumps({k: REPORT['roles'][k] for k in ('positions_per_role', 'counts', 'sanity')}, indent=1))
print(json.dumps(cmp4, indent=1))
if not SMOKE:
    assert REPORT['roles']['sanity']['ok'], 'the plain scores measured again do not match the first run: do not use role_ff'
log('done')
