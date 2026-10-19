"""Paired gradient trajectories with shared baseline clipping (Figure 5)."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import core
from .run import apply_update, backward, batch_rows, frozen_names, parser, setup


def probe(model, rows, device):
    model.train()
    model.zero_grad(set_to_none=True)
    inputs, positions, targets = core.recall_batch(rows, 8192, device)
    inputs.pop('labels')
    logits = core.logits_at(model, inputs, positions[:, 0])
    F.cross_entropy(logits, targets[:, 0]).backward()
    result = {}
    for layer in (1,3):
        gradients = [p.grad.detach().float().flatten() for p in model.model.layers[layer].attn.parameters() if p.grad is not None]
        result[str(layer)] = torch.cat(gradients).norm().item()
    model.zero_grad(set_to_none=True)
    return result


def main():
    p = parser()
    p.description = __doc__
    p.set_defaults(steps=425, freeze_step=175, unfreeze_step=225, eval_examples=2000)
    args = p.parse_args()
    if args.resume:
        raise ValueError('Paired gradient branches start from the same initialization')
    if not 0 <= args.freeze_step < args.unfreeze_step < args.steps:
        raise ValueError('Require freeze-step < unfreeze-step < steps')
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, optimizer, scheduler, data, orders = setup(args)
    # Full backward passes remain enabled even when an update is masked.
    branches = {'baseline': (baseline, optimizer, scheduler)}
    fixed_rows = core.select_rows(data['train'], slice(0,32))
    frozen = frozen_names(baseline, mixers_only=True)
    for step in range(1, args.steps + 1):
        if step == args.freeze_step + 1:
            branches['frozen'] = copy.deepcopy(branches['baseline'])
        if step == args.unfreeze_step + 1:
            branches['unfrozen'] = copy.deepcopy(branches['frozen'])
        clip = None
        for name, (model, opt, sched) in branches.items():
            loss = backward(model, batch_rows(data, orders, step, args.short_percent), args.device)
            if name == 'baseline':
                norm = torch.sqrt(sum(p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None))
                clip = min(1.0, 1.0 / (norm.item() + 1e-6))
            apply_update(model, opt, sched, frozen if name == 'frozen' else (), clip_factor=clip)
        if step >= args.freeze_step:
            norms = {name:probe(model, fixed_rows, args.device) for name,(model,_,_) in branches.items()}
            ratios = {name:{layer:value / max(norms['baseline'][layer],1e-30) for layer,value in values.items()}
                      for name,values in norms.items()}
            row = dict(seed=args.seed,step=step,norms=norms,ratios=ratios)
            with (args.output/'gradients.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
