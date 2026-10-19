"""Train Spaced Retrieval with mixture, freezing, unfreezing, and attribution."""

import argparse
import copy
import json
from pathlib import Path

import torch
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from . import core


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-config', default='configs/model/retrieval.json')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-seed', type=int, default=20260910)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--short-percent', type=int, default=50)
    p.add_argument('--freeze-step', type=int, default=-1, help='Completed updates before freezing; -1 disables it')
    p.add_argument('--unfreeze-step', type=int, default=-1, help='Completed updates before resuming frozen updates')
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-examples', type=int, default=2000)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', default='cuda')
    return p


def setup(args):
    if not 0 <= args.short_percent <= 100:
        raise ValueError('short-percent must be in [0, 100]')
    if args.steps < 1 or args.steps * 32 > 50000:
        raise ValueError('Require 1..1562 updates for a single pass through 50,000 keys')
    config = dict(model_config=args.model_config, data_seed=args.data_seed,
                  train_examples=50000, eval_examples=args.eval_examples)
    data = core.build_data(config)
    model = core.initialize_model(config, args.seed, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=1500, min_lr=1e-5,
    )
    generator = torch.Generator().manual_seed(args.seed + 1000)
    orders = torch.stack([torch.randperm(50000, generator=generator) for _ in range(2)])
    return model, optimizer, scheduler, data, orders


def batch_rows(data, orders, step, percent):
    # Integer quotas preserve the requested proportion across successive updates.
    before = (step - 1) * 32 * percent // 100
    after = step * 32 * percent // 100
    short = after - before
    long_before = (step - 1) * 32 - before
    return [(64, core.select_rows(data['train'], orders[0, before:after]), short),
            (8192, core.select_rows(data['train'], orders[1, long_before:long_before + 32 - short]), 32 - short)]


def backward(model, batches, device):
    model.train()
    model.zero_grad(set_to_none=True)
    total = 0.0
    for gap, rows, count in batches:
        if count == 0:
            continue
        inputs, _, _ = core.recall_batch(rows, gap, device)
        loss = model(**inputs, use_cache=False, return_loss_only=True)[0]
        (loss * (count / 32)).backward()
        total += loss.item() * count / 32
    return total


def frozen_names(model, *, mixers_only=False):
    prefixes = tuple(f'model.layers.{i}.' + ('attn.' if mixers_only else '') for i in (0, 2))
    return {n for n, _ in model.named_parameters() if n.startswith(prefixes)}


def apply_update(model, optimizer, scheduler, frozen=(), clip_factor=None):
    # The optimizer keeps all parameters and paused Adam moments across interventions.
    if clip_factor is None:
        for name, param in model.named_parameters():
            if name in frozen:
                param.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    else:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(clip_factor)
        for name, param in model.named_parameters():
            if name in frozen:
                param.grad = None
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def save_checkpoint(path, model, optimizer, scheduler, step, args):
    core.save_tensor(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), step=step, args=vars(args),
        cpu_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))


def restore(path, model, optimizer, scheduler, args):
    state = torch.load(path, map_location='cpu', weights_only=False)
    for name in ('seed', 'data_seed', 'short_percent', 'model_config'):
        if state['args'][name] != getattr(args, name):
            raise ValueError(f'Resume changes {name}')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    scheduler.load_state_dict(state['scheduler'])
    torch.random.set_rng_state(state['cpu_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']


def main():
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    core.write_json(args.output / 'protocol.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    model, optimizer, scheduler, data, orders = setup(args)
    start = restore(args.resume, model, optimizer, scheduler, args) if args.resume else 0
    frozen = frozen_names(model)
    for step in range(start + 1, args.steps + 1):
        loss = backward(model, batch_rows(data, orders, step, args.short_percent), args.device)
        active = args.freeze_step >= 0 and step > args.freeze_step and (args.unfreeze_step < 0 or step <= args.unfreeze_step)
        apply_update(model, optimizer, scheduler, frozen if active else ())
        print(json.dumps(dict(step=step, loss=loss, kda_frozen=active)), flush=True)
        if step == args.steps or step % args.eval_every == 0:
            attribution = core.causal_contributions(model, data['evaluation'], [64,8192], args.device, args.eval_batch_size)
            native = core.native_accuracy(model, data['evaluation'], [64,8192], args.device, args.eval_batch_size)
            core.save_tensor(args.output / f'attribution_{step}.pt', attribution)
            core.write_json(args.output / f'eval_{step}.json', dict(step=step,native=native,
                contribution={gap:{layer:core.summarize_values(v) for layer,v in item['effect'].items()}
                              for gap,item in attribution.items()}))
            save_checkpoint(args.output / f'checkpoint_{step}.pt', model, optimizer, scheduler, step, args)


if __name__ == '__main__':
    main()
