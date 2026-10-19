"""Learn attention-only KV-head gates with reference-answer hidden-state MSE."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.multi_passkey import build_dataset
from modeling.kda import KDAForCausalLM
from .objective import _duo_forward, duo_distillation_loss, freeze_for_duo_distillation, save_duo_alpha_artifacts


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--booksum',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    model=KDAForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16).cuda()
    model.enable_duo_attention(attention_only=True,sink_size=128,recent_size=256,
        alpha_init=1.0,kda_use_sink=True,attention_backend='flex')
    freeze_for_duo_distillation(model)
    model.gradient_checkpointing_enable()
    alphas=tuple(p for _,p in model.named_duo_alpha_parameters())
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    dataset=build_dataset(tokenizer,data_dir=args.booksum,seq_len=16384,gap=300,
        key_len=32,num_passkeys=10,num_samples=args.steps,seed=args.seed,
        min_depth_ratio=.05,max_depth_ratio=.95,text_field='chapter',passkey_format='words',shift_labels=False)
    optimizer=torch.optim.AdamW(alphas,lr=.02,weight_decay=0)
    model.train()
    for step,batch in enumerate(DataLoader(dataset,batch_size=1),1):
        lr=.02 * min(1.0,max(step/200,.1),max((1001-step)/200,.1))
        for group in optimizer.param_groups:group['lr']=lr
        ids,labels=batch['input_ids'].cuda(),batch['labels'].cuda()
        optimizer.zero_grad(set_to_none=True)
        result=duo_distillation_loss(_duo_forward(model,ids),labels,alphas,.05)
        result.total.backward()
        optimizer.step()
        model.clamp_duo_alpha_()
        print(f'{step}/{args.steps} loss={result.total.item():.6f}',flush=True)
    save_duo_alpha_artifacts(model,args.output,dict(attention_only=True,steps=args.steps,seed=args.seed))


if __name__=='__main__':main()
