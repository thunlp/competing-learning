"""Plot retrieval attribution, gap mixtures, gradients, positional loss, and heads."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def mixture_scores(root, gap, step):
    """Sum same-family layer scores per seed, then average seeds (Figure 3)."""
    files=sorted(root.glob(f'seed*/short*/eval_{step}.json'))
    if not files:
        raise ValueError(f'No seed*/short*/eval_{step}.json results under {root}')
    grouped={}
    for file in files:
        percent=int(file.parent.name.removeprefix('short'))
        row=json.loads(file.read_text())
        scores=row['contribution'][gap]
        grouped.setdefault(percent,[]).append([
            sum(scores[layer]['mean'] for layer in layers)
            for layers in (('0','2'),('1','3'))
        ])
    return {percent:np.mean(values,axis=0) for percent,values in sorted(grouped.items())}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('kind',choices=['attribution','mixture','gradients','position','heads'])
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--labels',nargs='+')
    p.add_argument('--gap',default='8192')
    p.add_argument('--step',type=int,default=1500,help='Checkpoint update for the gap-mixture plot')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    fig,ax=plt.subplots(figsize=(6,4))
    for index,root in enumerate(args.input):
        label=args.labels[index] if args.labels else root.name
        if args.kind=='attribution':
            rows=sorted([json.loads(x.read_text()) for x in root.glob('eval_*.json')],key=lambda x:x['step'])
            for family,layers in [('KDA',('0','2')),('Attention',('1','3'))]:
                ax.plot([r['step'] for r in rows],[np.mean([r['contribution'][args.gap][l]['mean'] for l in layers]) for r in rows],label=f'{label}: {family}')
            ax.set(xlabel='Training update',ylabel='Signed retrieval score')
        elif args.kind=='mixture':
            scores=mixture_scores(root,args.gap,args.step)
            for family,column in [('KDA',0),('Attention',1)]:
                ax.plot(list(scores),[value[column] for value in scores.values()],
                        marker='o',label=f'{label}: {family}')
            ax.set(xlabel='Short-distance pattern fraction (%)',ylabel='Summed retrieval ability')
        elif args.kind=='gradients':
            # Group seed trajectories, use geometric means of paired ratios per mixer.
            files=list(root.glob('**/gradients.jsonl'))
            grouped={}
            for file in files:
                for line in file.read_text().splitlines():
                    row=json.loads(line)
                    for branch,ratios in row['ratios'].items():
                        if branch!='baseline':grouped.setdefault((branch,row['step']),[]).append([ratios['1'],ratios['3']])
            for branch in ('frozen','unfrozen'):
                steps=sorted(s for b,s in grouped if b==branch)
                y=[float(np.exp(np.log(np.maximum(grouped[branch,s],1e-30)).mean(axis=0)).mean()) for s in steps]
                ax.plot(steps,y,label=f'{label}: {branch}')
            ax.set(xlabel='Training update',ylabel='Attention gradient / baseline',yscale='log')
        elif args.kind=='position':
            rows=[r for r in csv.DictReader((root/'buckets.csv').open()) if r['corpus']=='Combined']
            ax.plot([(int(r['start'])+int(r['end']))/2 for r in rows],[float(r['ce']) for r in rows],label=label)
            ax.set(xlabel='Prediction position',ylabel='Cross-entropy')
        else:
            rows=json.loads((root/'summary.json').read_text())
            ax.plot([r['actual_percent'] for r in rows],[100*r['first_token_accuracy'] for r in rows],label=label)
            ax.set(xlabel='KV heads using streaming attention (%)',ylabel='First-token retrieval accuracy (%)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output)


if __name__=='__main__':main()
