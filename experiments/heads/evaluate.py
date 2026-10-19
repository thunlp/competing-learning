"""Evaluate binary streaming masks in increasing learned-alpha order."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from data.multi_passkey import build_dataset
from modeling.kda import KDAForCausalLM
from . import masking


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--alpha',type=Path,required=True)
    p.add_argument('--booksum',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,default=64)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.alpha.read_text())
    if not manifest['metadata']['attention_only']:
        raise ValueError('Expected attention-only alpha identification')
    ranking=sorted([dict(layer=int(re.search(r'layers\.(\d+)',row['name'])[1]),head=h,alpha=float(value))
        for row in manifest['layers'] for h,value in enumerate(row['values'])],
        key=lambda row:(row['alpha'],row['layer'],row['head']))
    model=KDAForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    masking.LAYERS=tuple(model.config.attn['layers'])
    masking.HEADS=len(ranking)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    dataset=build_dataset(tokenizer,data_dir=args.booksum,seq_len=16384,gap=300,key_len=32,
        num_passkeys=10,num_samples=args.examples,seed=20260925,min_depth_ratio=.05,
        max_depth_ratio=.95,text_field='chapter',passkey_format='words',shift_labels=False,data_start_index=10000)
    fingerprint=hashlib.sha256()
    for sample in dataset:fingerprint.update(sample['input_ids'].numpy().tobytes())
    (args.output/'dataset.sha256').write_text(fingerprint.hexdigest()+'\n')
    controller=masking.MaskController(model)
    try:
        results={}
        for n in sorted(set(masking.count_for_percent(p) for p in range(0,101,10))):
            results[n]=masking.evaluate(model,controller,dataset,tokenizer,ranking,n,args.output/f'heads-{n}.json')
        masking.atomic(args.output/'summary.json',[dict(requested_percent=p,
            actual_percent=100*masking.count_for_percent(p)/masking.HEADS,
            **{k:results[masking.count_for_percent(p)][k] for k in ('n_keys','accuracy','first_token_accuracy')})
            for p in range(0,101,10)])
    finally:controller.close()


if __name__=='__main__':main()
