"""Pass rate of agent episodes by when they end (same-session runs B197 / B200): episodes that end late are the ones a
faster model could rescue from the clock. usage: python3 kernels/analysis/late_episodes.py (reads runs/eval-rg-same4-gpu and
runs/eval-rg-same55-gpu task results)."""
import json,glob,os
RUNS=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','runs')   # the repository's runs/
to=lambda r:'timeout' in (r.get('error') or '').lower()
def rows(run,pat):
    out=[]
    for f in sorted(glob.glob(f'{RUNS}/{run}/train_eval/repograph_{pat}_b*/results/task_results.jsonl')):
        out+=[json.loads(l) for l in open(f) if l.strip()]
    return out
for label,R in [('full x4 (same4+same55)',rows('eval-rg-same4-gpu','full*')+rows('eval-rg-same55-gpu','full*')),
                ('c70kd',rows('eval-rg-same4-gpu','c70kd')),('c55kd',rows('eval-rg-same55-gpu','c55kd'))]:
    n=len(R)
    def band(a,b):
        s=[r for r in R if not to(r) and a<=r['duration_seconds']<b]; return len(s),sum(r['resolved'] for r in s)
    e=band(0,300); m=band(300,420); l1=band(420,480); l2=band(480,10**6)
    print(f'{label}: n {n}; normal endings <5 min {e[1]}/{e[0]}, 5-7 min {m[1]}/{m[0]}, 7-8 min {l1[1]}/{l1[0]}, after 8 min (final call in flight) {l2[1]}/{l2[0]}; timeouts {sum(to(r) for r in R)}')
