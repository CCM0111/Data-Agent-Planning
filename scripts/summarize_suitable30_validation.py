"""Summarize complete, paired node comparisons; action availability is conditional."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
from statistics import mean

def summarize(root):
    groups=defaultdict(list);details=[]
    for p in sorted(root.glob('*/nodes/*/result.json')):
        r=json.loads(p.read_text());s=json.loads(p.with_name('screening.json').read_text())
        policies=[('baseline','baseline'),('skill',r['chosen'])]
        if 'comparison_region' in r:
            policies.extend((s['action_types'][a],a) for a in r['metrics'] if a!='baseline')
        for name,a in policies:
            m=r['metrics'][a]
            if m is None:
                continue
            effects=r.get('action_effects',{}).get(a,{})
            rec=dict(comparison_region=r.get('comparison_region'),added=effects.get('added'),removed=effects.get('removed'),
                target_before=effects.get('target_before'),target_after=effects.get('target_after'),task=p.parents[2].name,depth=r['depth'],batch=p.parent.name,policy=name,
                action=a,top1=m['top1_correct'],retained=m['champion_retained'],pairs=m['pair_accuracy'],
                cutoff_tie=m['cutoff_tie'],distinct_samples=r['distinct_samples'],best=(a in r['best_actions'] if r.get('skill_matches_best') is not None else None),
                decision_status=r.get('decision_status'))
            details.append(rec);groups[(r['depth'],name)].append(rec)
    if not details:return
    def save(name,rows):
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
    save('paired_nodes.csv',details)
    summary=[]
    for (depth,policy),rows in sorted(groups.items()):
        informative=[r for r in rows if r['distinct_samples']>1 and r['best'] is not None and r['decision_status']!='fallback']
        summary.append(dict(depth=depth,policy=policy,n=len(rows),top1=mean(r['top1'] for r in rows),
            champion_retained=mean(r['retained'] for r in rows),multiple_samples=len(informative),
            best_action_rate_among_choices=mean(r['best'] for r in informative) if informative else ''))
    save('validation_summary.csv',summary)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);summarize(ap.parse_args().root)
