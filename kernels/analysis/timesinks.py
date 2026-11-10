import json,glob,re,collections,statistics as st
from pathlib import Path
ok=set(json.load(open('runs/reverify-gemma-v5d/resolved_ids.json')))
agg={True:collections.Counter(),False:collections.Counter()}; n={True:0,False:0}
slow=[]; rep={True:[],False:[]}; think={True:[],False:[]}; rows=[]
for p in glob.glob('runs/gen-gpu-v1/gen/gemma-4-31b-qat/pass0/results/traces/*.json'):
    iid=p.split('trace_')[-1][:-5]; d=json.load(open(p)); good=iid in ok; n[good]+=1
    prev=None; seen=collections.Counter(); decode=0; tooltime=collections.Counter(); calls=collections.Counter(); bigthink=0; fails=0
    for s in d['steps']:
        if s.get('source')!='agent': continue
        c=(s.get('metrics') or {}).get('completion_tokens',0) or 0
        e=(s.get('extra') or {}).get('elapsed_s'); dt=(e-prev) if (e is not None and prev is not None) else (e or 0); prev=e if e is not None else prev
        decode+=c/26.0; nd=min(max(dt-c/8.5,0.0),120.0)
        if c>800: bigthink+=1
        tcs=s.get('tool_calls') or []
        fn=tcs[0]['function_name'] if tcs else 'no_tool'
        arg=json.dumps(tcs[0].get('arguments'),sort_keys=True)[:300] if tcs else ''
        kind=fn
        if fn=='run_command':
            a=arg.lower()
            kind='run:pytest' if 'pytest' in a else ('run:python' if re.search(r'python3? ',a) else ('run:grep/find' if re.search(r'grep|find |ls |cat |sed |head|tail',a) else 'run:other'))
        tooltime[kind]+=nd; calls[kind]+=1; seen[(fn,arg)]+=1
        obs=(s.get('observation') or {}).get('content','') or ''
        if '"status": "error"' in obs[:60] or 'not found' in obs[:200]: fails+=1
        if nd>45: slow.append((round(nd),good,iid,kind,arg[:140]))
    repeats=sum(v-1 for v in seen.values() if v>1)
    rep[good].append(repeats); think[good].append(bigthink)
    for k,v in tooltime.items(): agg[good]['t:'+k]+=v
    for k,v in calls.items(): agg[good]['n:'+k]+=v
    agg[good]['decode']+=decode
    rows.append((iid,good,decode,sum(tooltime.values()),repeats,calls['run:pytest'],calls['read_file'],bigthink,fails))
for good in (True,False):
    N=n[good]; a=agg[good]
    print(f"\n{'RESOLVED' if good else 'UNRESOLVED'} (n={N}): per-task scorer-time estimate: decode {a['decode']/N:.0f}s; non-decode by kind:")
    for k in sorted([k for k in a if k.startswith('t:')], key=lambda k:-a[k]):
        kk=k[2:]; print(f"   {kk:14s} {a[k]/N:6.0f}s over {a['n:'+kk]/N:4.1f} calls  ({a[k]/max(1,a['n:'+kk]):.0f}s per call)")
    print(f"   repeated identical calls per task: median {st.median(rep[good])}, mean {st.mean(rep[good]):.1f}; steps with >800 completion tokens: mean {st.mean(think[good]):.1f}")
print('\nSlowest single steps (non-decode seconds, resolved?, task, kind, args):')
for x in sorted(slow,reverse=True)[:14]: print('  ',x)
print('\nslow steps (>45s):',len(slow),' in resolved:',sum(1 for x in slow if x[1]),' unresolved:',sum(1 for x in slow if not x[1]))
out = Path(__file__).resolve().parents[2] / 'runs/analysis/timesinks_rows.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows))
