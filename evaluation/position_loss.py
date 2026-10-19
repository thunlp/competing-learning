"""Token-weighted loss by prediction position on local book JSONL files."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .longppl import load_model, load_tokenizer, log_probabilities, verify_scorer


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--books',type=Path,nargs='+',required=True,help='Books3.jsonl and BookCorpus2.jsonl')
    p.add_argument('--documents',type=int,default=50,help='First nonempty documents per corpus')
    p.add_argument('--max-length',type=int,default=16384)
    p.add_argument('--bin-size',type=int,default=512)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=load_tokenizer(args.model)
    model=load_model(args.model,args.device)
    values=[]
    for source in args.books:
        count=0
        with source.open() as f:
            for line in f:
                text=json.loads(line)['text']
                if not text.strip():continue
                ids=tokenizer(text,add_special_tokens=False,truncation=True,max_length=args.max_length+1)['input_ids']
                if len(ids)<2:continue
                if not values:verify_scorer(model,ids,args.device)
                loss=-np.asarray(log_probabilities(model,ids,args.device)[1:],dtype=np.float64)
                if not np.isfinite(loss).all():raise ValueError('Nonfinite positional loss')
                np.save(args.output/f'{source.stem}-{count}.npy',loss)
                values.append((source.stem,loss))
                count+=1
                if count==args.documents:break
        if count!=args.documents:raise ValueError(f'Insufficient documents in {source}')
    rows=[]
    for corpus in [*[p.stem for p in args.books],'Combined']:
        arrays=[v for name,v in values if name==corpus or corpus=='Combined']
        for start in range(0,args.max_length,args.bin_size):
            chunks=[a[start:start+args.bin_size] for a in arrays]
            n=sum(len(x) for x in chunks)
            if n:rows.append(dict(corpus=corpus,start=start+1,end=start+args.bin_size,tokens=n,
                                  ce=sum(float(x.sum()) for x in chunks)/n))
    with (args.output/'buckets.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
